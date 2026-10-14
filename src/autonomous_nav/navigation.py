"""Navigation state machine connecting PathPlanner and MotionController.

This module orchestrates the full autonomous navigation lifecycle for the
simulated 2D robot: planning a collision-free route, executing waypoint-by-
waypoint movement, tracking mission progress, signalling terminal states,
and performing dynamic replanning when newly observed obstacles invalidate
the active route.

State Definitions
-----------------
IDLE
    No active mission.  The robot is stationary and awaiting a new goal.

PLANNING
    ``start_mission()`` has been called.  The planner is computing a path.
    This state is transient; it resolves to NAVIGATING, REACHED, or FAILED
    before ``start_mission()`` returns.  It also occurs transiently inside
    ``update()`` during replanning.

NAVIGATING
    A valid path has been found and the robot is moving toward successive
    waypoints.  ``update()`` advances the robot and the waypoint index.

BLOCKED
    A newly observed obstacle has been found on the remaining route.  The
    manager enters BLOCKED, attempts an immediate replan, and then either
    resumes NAVIGATING (if a replacement path exists) or transitions to
    FAILED.  BLOCKED is therefore a transient state within ``update()``;
    it is never the final state returned from ``update()`` unless it cannot
    resolve — in that case it resolves to FAILED.

    Difference from FAILED
    ~~~~~~~~~~~~~~~~~~~~~~
    BLOCKED is recoverable — it means "replanning attempted".
    FAILED is terminal — it means "no viable route exists with current map."

REACHED
    The robot has arrived at the final goal waypoint (within arrival
    tolerance).  No further movement commands are accepted until a new
    mission is started.

FAILED
    No valid path to the goal exists under the current occupancy map.
    This may happen at mission start (unreachable goal) or during navigation
    (replanning exhausted all options).  No movement is attempted.

Valid State Transitions
-----------------------
IDLE      ──start_mission()──► PLANNING ──path found──► NAVIGATING
IDLE      ──start_mission()──► PLANNING ──start==goal──► REACHED
IDLE      ──start_mission()──► PLANNING ──no path─────► FAILED
NAVIGATING ──update()─────────────────────────────────► NAVIGATING (step)
NAVIGATING ──update(), goal reached────────────────────► REACHED
NAVIGATING ──update(), path blocked────────────────────► BLOCKED
BLOCKED   ──replan ok──────────────────────────────────► NAVIGATING
BLOCKED   ──replan failed──────────────────────────────► FAILED
REACHED   ──start_mission()──► PLANNING ──...
FAILED    ──start_mission()──► PLANNING ──...
any       ──reset()──────────────────────────────────── ► IDLE

Coordinate Conventions
----------------------
* Grid space  : (row, col) integer tuples — used by PathPlanner and
  OccupancyGrid.
* Pixel space : (x, y) float tuples — used by MotionController.  Origin is
  top-left; x increases right, y increases down.
* Conversion  : ``OccupancyGrid.grid_to_pixel_center(row, col)`` → (x, y).

Waypoint Advancement
--------------------
The planner returns the full path including the start cell as waypoint 0.
``start_mission()`` immediately skips waypoint 0 (the starting cell) so that
the first ``update()`` drives the robot toward waypoint 1.  This prevents a
false REACHED signal when start == first waypoint in pixel space.

If the path has a single cell (start == goal), the mission completes
immediately with state REACHED.  (The planner itself handles start==goal by
returning [start] with cost=0.)

Path Validity Checking
----------------------
Before every movement step, ``update()`` calls ``_is_remaining_path_valid()``
which inspects every cell from the current waypoint index to the end of the
path.  A cell is invalid if its state in the live OccupancyGrid is
``CellState.OCCUPIED``.  Unknown cells are treated as non-traversable by
default, matching the PathPlanner cautious policy.  The robot's own current
cell is not checked because the occupancy grid is updated by the sensor
subsystem — if the robot's cell becomes marked OCCUPIED after the robot has
left it, replanning is still triggered conservatively.

Replanning
----------
When the remaining route is found invalid, the manager:
1. Marks state BLOCKED (transient).
2. Derives the robot's current grid cell from ``robot_pixel_pos``.
3. Calls ``PathPlanner.plan()`` with the live OccupancyGrid.
4. On success → installs the new path (index set to 1), resumes NAVIGATING.
5. On failure → sets state FAILED; no further movement.
6. Increments ``replan_count`` each attempt.
7. The replacement path is guaranteed to use only FREE cells under the
   current map (unknown cells remain non-traversable).

Sensor Integration
------------------
``ObstacleDetector.integrate_observations(observations, grid)`` mutates the
same ``OccupancyGrid`` instance held by the manager.  Because the manager
holds a live reference, it automatically sees all map updates between calls
to ``update()``.  No additional wiring is required.

Dynamic replanning is triggered lazily — only when the route is checked
before each movement step.

Known Limitations (Phase 5C)
-----------------------------
* Replanning is attempted only once per call to ``update()``.  A loop that
  keeps generating obstacles and calling ``update()`` with the same position
  will fail after the first replan attempt fails.
* If the robot's own current cell becomes OCCUPIED in the map (e.g., mis-
  classified by the sensor), replanning will treat it as blocked.  The
  manager converts the pixel position to a grid cell for replanning; if that
  cell is marked OCCUPIED the replan will immediately fail because the planner
  rejects blocked starts.  The sensor subsystem should avoid misclassifying
  the robot's current cell.
* No multi-step lookahead — the check scans all remaining waypoints but the
  robot may still be physically moving between them.
"""

from __future__ import annotations

from enum import Enum, auto
from typing import Any, List, Optional, Tuple

try:
    from autonomous_nav.controller import MotionController, MoveResult
    from autonomous_nav.mapping import CellState, OccupancyGrid
    from autonomous_nav.planner import PathPlanner
    from autonomous_nav.config import SimulationConfig
except ImportError:  # pragma: no cover
    MotionController = Any  # type: ignore
    MoveResult = Any        # type: ignore
    OccupancyGrid = Any     # type: ignore
    CellState = Any         # type: ignore
    PathPlanner = Any       # type: ignore
    SimulationConfig = Any  # type: ignore


class NavigationState(Enum):
    """High-level operating states for autonomous navigation.

    See module docstring for full state definitions and transition table.
    """
    IDLE = auto()
    PLANNING = auto()
    NAVIGATING = auto()
    BLOCKED = auto()
    REACHED = auto()
    FAILED = auto()


class NavigationManager:
    """Mission lifecycle manager integrating PathPlanner and MotionController.

    The manager keeps the robot's current pixel position, the active grid
    path, and the current waypoint index consistent across calls.  It also
    monitors path validity and replans when newly observed obstacles
    invalidate the remaining route.

    Args:
        planner: A configured ``PathPlanner`` instance.
        controller: A configured ``MotionController`` instance.
        grid: The robot's ``OccupancyGrid`` used for planning, path-validity
            checking, and grid-to-pixel conversion.  The ground-truth
            ``Environment`` is intentionally not accessed here.
        allow_unknown: Whether UNKNOWN cells are treated as traversable
            during path-validity checking and replanning.  Must match the
            planner's ``allow_unknown`` parameter.  Default False (cautious).

    Attributes:
        state: Current NavigationState.
        active_path: Current planned path as a list of (row, col) tuples.
        current_waypoint_index: Index into active_path of the current target.
        replan_count: Number of replanning attempts made during the mission.
    """

    def __init__(
        self,
        planner: PathPlanner,
        controller: MotionController,
        grid: OccupancyGrid,
        allow_unknown: bool = False,
    ) -> None:
        self._planner = planner
        self._controller = controller
        self._grid = grid
        self._allow_unknown = allow_unknown

        self.state: NavigationState = NavigationState.IDLE
        self.active_path: List[Tuple[int, int]] = []
        self.current_waypoint_index: int = 0
        self._goal_grid: Optional[Tuple[int, int]] = None
        self.replan_count: int = 0

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(
        cls,
        config: SimulationConfig,
        grid: OccupancyGrid,
        allow_unknown: bool = False,
    ) -> NavigationManager:
        """Constructs a NavigationManager from a validated SimulationConfig.

        Args:
            config: Validated SimulationConfig instance.
            grid: The robot's OccupancyGrid.
            allow_unknown: Whether UNKNOWN cells are traversable during
                planning and path-validity checking.

        Returns:
            NavigationManager configured with a PathPlanner and
            MotionController derived from the config.
        """
        planner = PathPlanner.from_config(config, allow_unknown=allow_unknown)
        controller = MotionController.from_config(config)
        return cls(planner=planner, controller=controller, grid=grid,
                   allow_unknown=allow_unknown)

    # ------------------------------------------------------------------
    # Mission control (backward-compatible public API preserved)
    # ------------------------------------------------------------------

    def set_path(self, path: List[Tuple[int, int]]) -> None:
        """Sets a new active path and resets waypoint progress.

        Preserved for backward compatibility.  Prefer ``start_mission()``
        for full lifecycle management.

        Args:
            path: Ordered list of (row, col) waypoints.  An empty list
                resets to IDLE.
        """
        self.active_path = list(path)
        self.current_waypoint_index = 0
        self._goal_grid = path[-1] if path else None
        self.replan_count = 0
        self.state = NavigationState.NAVIGATING if path else NavigationState.IDLE

    def start_mission(
        self,
        start_grid: Tuple[int, int],
        goal_grid: Tuple[int, int],
        robot_pixel_pos: Tuple[float, float],
    ) -> None:
        """Plans a path and begins navigation from start to goal.

        Handles the special case where start equals goal: the mission is
        considered immediately REACHED and no planning or movement occurs.

        Transitions:
            IDLE / REACHED / FAILED → PLANNING → NAVIGATING  (valid path)
            IDLE / REACHED / FAILED → PLANNING → REACHED     (start == goal)
            IDLE / REACHED / FAILED → PLANNING → FAILED      (unreachable)

        Args:
            start_grid:     (row, col) grid cell where the robot currently is.
            goal_grid:      (row, col) grid cell the robot should reach.
            robot_pixel_pos: Current robot position in pixel space (x, y).

        Raises:
            TypeError:    If arguments are not properly typed tuples.
            ValueError:   If start or goal are blocked.
            IndexError:   If start or goal are out of grid bounds.
            RuntimeError: If called while the state is NAVIGATING (use
                          ``reset()`` first).
        """
        if self.state == NavigationState.NAVIGATING:
            raise RuntimeError(
                "Cannot start a new mission while NAVIGATING. Call reset() first."
            )

        self._validate_grid_cell(start_grid, "start_grid")
        self._validate_grid_cell(goal_grid, "goal_grid")

        self.state = NavigationState.PLANNING
        self.replan_count = 0

        # Special case: start == goal → immediate REACHED
        if start_grid == goal_grid:
            self.active_path = [start_grid]
            self.current_waypoint_index = 0
            self._goal_grid = goal_grid
            self.state = NavigationState.REACHED
            return

        # Run the planner
        try:
            path = self._planner.plan(
                start=start_grid, goal=goal_grid, grid=self._grid
            )
        except (ValueError, IndexError, TypeError):
            # Blocked or out-of-bounds start/goal — propagate to caller
            self.active_path = []
            self.current_waypoint_index = 0
            self._goal_grid = None
            self.state = NavigationState.FAILED
            raise

        if not path:
            self.active_path = []
            self.current_waypoint_index = 0
            self._goal_grid = None
            self.state = NavigationState.FAILED
            return

        self.active_path = path
        self._goal_grid = goal_grid

        # Skip waypoint 0 (the start cell); path is guaranteed length >= 2
        # because start != goal
        self.current_waypoint_index = 1
        self.state = NavigationState.NAVIGATING

    def reset(self) -> None:
        """Resets the manager to IDLE, clearing all mission state.

        Safe to call from any state.  Also resets ``replan_count``.
        """
        self.state = NavigationState.IDLE
        self.active_path = []
        self.current_waypoint_index = 0
        self._goal_grid = None
        self.replan_count = 0

    # ------------------------------------------------------------------
    # Waypoint access
    # ------------------------------------------------------------------

    def get_current_target(self) -> Optional[Tuple[int, int]]:
        """Returns the current target waypoint (grid) if navigating.

        Returns:
            (row, col) of the current waypoint, or None if not NAVIGATING.
        """
        if (
            self.state == NavigationState.NAVIGATING
            and 0 <= self.current_waypoint_index < len(self.active_path)
        ):
            return self.active_path[self.current_waypoint_index]
        return None

    def get_current_target_pixel(self) -> Optional[Tuple[float, float]]:
        """Returns the current target waypoint in pixel space (x, y).

        Returns:
            (x, y) pixel center of the current waypoint, or None if not
            NAVIGATING.
        """
        waypoint = self.get_current_target()
        if waypoint is None:
            return None
        row, col = waypoint
        return self._grid.grid_to_pixel_center(row, col)

    def advance_waypoint(self) -> None:
        """Advances to the next waypoint after the current one is reached.

        Transitions to REACHED when the last waypoint is consumed.
        Only valid while NAVIGATING; silently ignored in other states.
        """
        if self.state != NavigationState.NAVIGATING:
            return
        self.current_waypoint_index += 1
        if self.current_waypoint_index >= len(self.active_path):
            self.state = NavigationState.REACHED
    def _validate_grid_cell(
        self,
        cell: Tuple[int, int],
        name: str,
    ) -> None:
        """Validate a grid cell and ensure it is traversable."""
        if not isinstance(cell, tuple) or len(cell) != 2:
            raise TypeError(
                f"{name} must be a tuple containing (row, col)."
            )

        row, col = cell

        if (
            isinstance(row, bool)
            or isinstance(col, bool)
            or not isinstance(row, int)
            or not isinstance(col, int)
        ):
            raise TypeError(
                f"{name} row and column must be integers."
            )

        if not self._grid.is_in_bounds(row, col):
            raise IndexError(
                f"{name} {cell} is outside the occupancy grid bounds."
            )

        state = self._grid.get_cell(row, col)

        if state == CellState.OCCUPIED:
            raise ValueError(
                f"{name} {cell} is occupied and cannot be used."
            )

        if state == CellState.UNKNOWN and not self._allow_unknown:
            raise ValueError(
                f"{name} {cell} is UNKNOWN and is not traversable."
            )
    # ------------------------------------------------------------------
    # Path validity and replanning
    # ------------------------------------------------------------------

    def _is_remaining_path_valid(self) -> bool:
        """Checks whether all remaining waypoints are traversable in the live grid.

        Scans all cells from ``current_waypoint_index`` to the end of
        ``active_path``.  A cell is invalid if its OccupancyGrid state is
        OCCUPIED, or UNKNOWN when ``allow_unknown`` is False.

        Returns:
            True if every remaining waypoint is traversable, False otherwise.
        """
        for i in range(self.current_waypoint_index, len(self.active_path)):
            row, col = self.active_path[i]
            if not self._grid.is_in_bounds(row, col):
                return False
            state = self._grid.get_cell(row, col)
            if state == CellState.OCCUPIED:
                return False
            if state == CellState.UNKNOWN and not self._allow_unknown:
                return False
        return True

    def _attempt_replan(self, robot_pixel_pos: Tuple[float, float]) -> bool:
        """Attempts to replan from the robot's current position to the goal.

        Derives the robot's current grid cell from ``robot_pixel_pos``.
        Calls the planner with the live OccupancyGrid.  On success, installs
        the new path with waypoint index set to 1.  On failure, does not
        modify ``active_path`` (caller sets FAILED).

        Args:
            robot_pixel_pos: Current robot pixel position (x, y).

        Returns:
            True if a replacement path was found, False otherwise.
        """
        self.replan_count += 1

        # Derive current grid cell from pixel position
        try:
            current_grid = self._grid.pixel_to_grid(
                robot_pixel_pos[0], robot_pixel_pos[1]
            )
        except (IndexError, TypeError):
            # Robot is outside grid bounds — cannot replan
            return False

        if self._goal_grid is None:
            return False

        # Immediate success if robot is already at the goal
        if current_grid == self._goal_grid:
            self.active_path = [current_grid]
            self.current_waypoint_index = 0
            return True

        try:
            new_path = self._planner.plan(
                start=current_grid, goal=self._goal_grid, grid=self._grid
            )
        except (ValueError, IndexError, TypeError):
            return False

        if not new_path:
            return False

        self.active_path = new_path
        self.current_waypoint_index = 1
        return True

    # ------------------------------------------------------------------
    # Update loop
    # ------------------------------------------------------------------

    def update(
        self,
        robot_pixel_pos: Tuple[float, float],
        dt: float = 1.0,
    ) -> MoveResult:
        """Advances the robot one simulation step toward the current waypoint.

        This method should be called once per simulation tick while NAVIGATING.

        On each call it:
        1. Validates the remaining path against the live OccupancyGrid.
        2. If the path is blocked, enters BLOCKED and attempts replanning.
        3. On successful replan, resumes NAVIGATING toward the new route.
        4. On failed replan, sets FAILED and returns a zero-displacement result.
        5. Converts the current grid waypoint to a pixel target.
        6. Calls ``MotionController.move_toward()`` to compute the new position.
        7. If the controller signals arrival, calls ``advance_waypoint()``.
        8. Returns the MoveResult so the caller can update the robot's
           rendered pixel position.

        No movement is performed when the state is not NAVIGATING.  In
        terminal states (REACHED, FAILED) a zero-displacement MoveResult
        is returned.

        Args:
            robot_pixel_pos: Current robot position in pixel space (x, y).
            dt: Time-step multiplier passed to ``MotionController.move_toward()``.
                Default 1.0 = one full simulation step.

        Returns:
            MoveResult(new_position, displacement, arrived).
            ``arrived`` reflects arrival at the *current waypoint*.
            Check ``self.state == NavigationState.REACHED`` for mission end.

        Raises:
            TypeError:  If robot_pixel_pos is not a valid (x, y) numeric tuple.
        """
        from autonomous_nav.controller import MoveResult as _MoveResult

        # Validate position type
        MotionController._validate_position(robot_pixel_pos, "robot_pixel_pos")

        _noop = _MoveResult(
            new_position=(float(robot_pixel_pos[0]), float(robot_pixel_pos[1])),
            displacement=0.0,
            arrived=False,
        )

        # Terminal / non-navigating states: return a no-op result
        if self.state != NavigationState.NAVIGATING:
            return _noop

        # ── Path validity check and replanning ──────────────────────────
        if not self._is_remaining_path_valid():
            self.state = NavigationState.BLOCKED
            if not self._attempt_replan(robot_pixel_pos):
                self.state = NavigationState.FAILED
                return _noop
            # Replan succeeded
            self.state = NavigationState.NAVIGATING

        # ── Resolve current waypoint ─────────────────────────────────────
        target_pixel = self.get_current_target_pixel()
        if target_pixel is None:
            self.state = NavigationState.REACHED
            return _MoveResult(
                new_position=(float(robot_pixel_pos[0]), float(robot_pixel_pos[1])),
                displacement=0.0,
                arrived=True,
            )

        # ── Execute one movement step ─────────────────────────────────────
        result = self._controller.move_toward(robot_pixel_pos, target_pixel, dt=dt)

        if result.arrived:
            self.advance_waypoint()

        return result

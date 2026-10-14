"""Motion control and waypoint following.

This module translates geometric waypoint paths into kinematic velocity
and steering commands for the simulated robot.

Coordinate System & Conventions:
    - All positions are in pixel space: (x, y) tuples of floats.
      Origin (0, 0) is at the top-left.
      x increases to the right, y increases downward.
    - Movement is computed as a displacement vector scaled by max_speed and
      the provided time-step scalar ``dt``.
    - ``max_speed`` is expressed in **pixels per simulation step** (matching
      the ``robot.max_speed`` field in ``configs/default.json``).
    - ``dt`` is a dimensionless step-fraction multiplier (default 1.0 =
      one full simulation step).  Pass ``simulation.time_step * fps`` if you
      want SI-unit integration, but the default keeps units consistent with
      the rest of the project.

Movement Guarantee (no overshoot):
    If the Euclidean distance to the target is smaller than
    ``max_speed * dt``, the returned position is clamped to the exact
    target rather than overshooting past it.

Independence from Pygame:
    This module has no dependency on Pygame or any rendering subsystem.
    It operates purely on numeric pixel coordinates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Tuple

try:
    from autonomous_nav.config import SimulationConfig
except ImportError:  # pragma: no cover
    SimulationConfig = Any  # type: ignore


# ---------------------------------------------------------------------------
# Movement result
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class MoveResult:
    """Result of a single controller movement update.

    Attributes:
        new_position: Updated (x, y) pixel position after clamped movement.
        displacement: Euclidean distance actually moved (pixels).
        arrived: True if the robot is now within arrival_tolerance of the target.
    """

    new_position: Tuple[float, float]
    displacement: float
    arrived: bool


# ---------------------------------------------------------------------------
# Controller
# ---------------------------------------------------------------------------

class MotionController:
    """Computes motion commands to steer the robot toward targeted waypoints.

    Movement Convention:
        Each call to :meth:`move_toward` advances the robot at most
        ``max_speed * dt`` pixels along the straight-line vector toward
        the target.  If the remaining distance is smaller than this limit
        the robot is placed exactly at the target (no overshoot).

    Args:
        max_speed: Maximum linear speed in pixels per simulation step.
            Must be positive.
        arrival_tolerance: Euclidean distance threshold in pixels at which
            the current waypoint is considered reached.  Must be positive.

    Raises:
        ValueError: If ``max_speed`` or ``arrival_tolerance`` are not positive.
        TypeError: If ``max_speed`` or ``arrival_tolerance`` are not numeric.
    """

    def __init__(self, max_speed: float, arrival_tolerance: float = 2.0) -> None:
        """Initializes controller constraints."""
        if not isinstance(max_speed, (int, float)) or isinstance(max_speed, bool):
            raise TypeError(
                f"max_speed must be a numeric value, got {type(max_speed).__name__}"
            )
        if not isinstance(arrival_tolerance, (int, float)) or isinstance(
            arrival_tolerance, bool
        ):
            raise TypeError(
                f"arrival_tolerance must be a numeric value, got {type(arrival_tolerance).__name__}"
            )

        if max_speed <= 0:
            raise ValueError("max_speed must be positive.")
        if arrival_tolerance <= 0:
            raise ValueError("arrival_tolerance must be positive.")

        self.max_speed: float = float(max_speed)
        self.arrival_tolerance: float = float(arrival_tolerance)

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_config(cls, config: SimulationConfig) -> MotionController:
        """Constructs a MotionController from a validated SimulationConfig.

        Uses ``config.robot.max_speed`` for the speed limit and a fixed
        arrival tolerance of half the grid cell size so that the robot
        reliably snaps to cell centers.

        Args:
            config: Validated SimulationConfig instance.

        Returns:
            MotionController configured for the loaded simulation parameters.
        """
        arrival_tolerance = config.grid.cell_size / 2.0
        return cls(
            max_speed=config.robot.max_speed,
            arrival_tolerance=arrival_tolerance,
        )

    # ------------------------------------------------------------------
    # Arrival check  (preserved from Phase 4 scaffold)
    # ------------------------------------------------------------------

    def has_arrived(
        self, current_pos: Tuple[float, float], target_pos: Tuple[float, float]
    ) -> bool:
        """Checks if current position is within arrival tolerance of target.

        Args:
            current_pos: Current (x, y) pixel position.
            target_pos: Target (x, y) pixel position.

        Returns:
            True if Euclidean distance <= arrival_tolerance.

        Raises:
            TypeError: If positions are not 2-tuples of numeric values.
        """
        self._validate_position(current_pos, "current_pos")
        self._validate_position(target_pos, "target_pos")
        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        return math.hypot(dx, dy) <= self.arrival_tolerance

    # ------------------------------------------------------------------
    # Core movement
    # ------------------------------------------------------------------

    def move_toward(
        self,
        current_pos: Tuple[float, float],
        target_pos: Tuple[float, float],
        dt: float = 1.0,
    ) -> MoveResult:
        """Advances the robot one simulation step toward a target position.

        The robot moves along the straight-line vector from ``current_pos``
        to ``target_pos`` by at most ``max_speed * dt`` pixels.  If the
        remaining distance is less than this limit the robot is placed exactly
        at ``target_pos`` (no overshoot).

        Args:
            current_pos: Current (x, y) pixel position as floats.
            target_pos: Target (x, y) pixel position as floats.
            dt: Time-step multiplier (dimensionless, default 1.0 = one full
                simulation step).  Must be positive.

        Returns:
            MoveResult with the updated position, displacement actually moved,
            and whether the robot has arrived within arrival_tolerance.

        Raises:
            TypeError: If any position is not a 2-tuple of numeric values,
                or ``dt`` is not numeric.
            ValueError: If ``dt`` is not positive.
        """
        self._validate_position(current_pos, "current_pos")
        self._validate_position(target_pos, "target_pos")

        if not isinstance(dt, (int, float)) or isinstance(dt, bool):
            raise TypeError(f"dt must be a numeric value, got {type(dt).__name__}")
        if dt <= 0:
            raise ValueError("dt must be positive.")

        dx = target_pos[0] - current_pos[0]
        dy = target_pos[1] - current_pos[1]
        distance = math.hypot(dx, dy)

        # Already at target (zero-distance guard)
        if distance == 0.0:
            return MoveResult(
                new_position=(float(current_pos[0]), float(current_pos[1])),
                displacement=0.0,
                arrived=True,
            )

        max_step = self.max_speed * dt

        if distance <= max_step:
            # Clamp to exact target — no overshoot
            new_x = float(target_pos[0])
            new_y = float(target_pos[1])
            displacement = distance
        else:
            # Normalise direction and scale to max_step
            scale = max_step / distance
            new_x = current_pos[0] + dx * scale
            new_y = current_pos[1] + dy * scale
            displacement = max_step

        arrived = math.hypot(new_x - target_pos[0], new_y - target_pos[1]) <= self.arrival_tolerance

        return MoveResult(
            new_position=(new_x, new_y),
            displacement=displacement,
            arrived=arrived,
        )

    # ------------------------------------------------------------------
    # Internal validation helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_position(pos: Tuple[float, float], name: str = "position") -> None:
        """Validates that a position is a 2-tuple of numeric (non-bool) values.

        Args:
            pos: Value to validate.
            name: Parameter name for error messages.

        Raises:
            TypeError: If pos is not a 2-tuple of numeric values.
        """
        if not isinstance(pos, tuple) or len(pos) != 2:
            raise TypeError(
                f"{name} must be a 2-tuple of numeric values, got {type(pos).__name__}"
            )
        for i, coord in enumerate(pos):
            if not isinstance(coord, (int, float)) or isinstance(coord, bool):
                raise TypeError(
                    f"{name}[{i}] must be numeric, got {type(coord).__name__}"
                )

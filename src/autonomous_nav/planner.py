"""Path planning algorithms (A* and Dijkstra).

This module provides deterministic, discrete 2D grid pathfinding solvers operating
over the robot's OccupancyGrid belief.

Coordinate System & Movement Conventions:
    - Grid coordinates: (row, column) as integer tuples.
    - Movement Model: Strict 4-connected orthogonal movements:
        1. Up:    (-1, 0)
        2. Right: (0, 1)
        3. Down:  (1, 0)
        4. Left:  (0, -1)
    - Movement Cost: Unit cost (1.0) per step. Diagonal movement is not used.
    - Neighbor Ordering: Fixed deterministic order [Up, Right, Down, Left].
    - Tie-Breaking Policy: Priority queue orders items by (f_score, h_score, counter, cell).
      Ties in f_score are broken by lower h_score (favoring proximity to goal), and
      further ties are resolved stably by insertion sequence counter (FIFO).

Occupancy & Unknown-Cell Policy:
    - CellState.FREE: Traversable.
    - CellState.OCCUPIED: Non-traversable (blocked).
    - CellState.UNKNOWN: Non-traversable by default (cautious policy).
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

from autonomous_nav.mapping import CellState, OccupancyGrid

try:
    from autonomous_nav.config import SimulationConfig
except ImportError:  # pragma: no cover
    SimulationConfig = Any  # type: ignore


# 4-connected orthogonal movement directions in deterministic evaluation order
ORTHOGONAL_DIRECTIONS: Tuple[Tuple[int, int], ...] = (
    (-1, 0),  # Up
    (0, 1),   # Right
    (1, 0),   # Down
    (0, -1),  # Left
)


@dataclass(frozen=True)
class PlanResult:
    """Detailed result and search telemetry from path planning.

    Attributes:
        path: Ordered list of (row, col) grid cells from start to goal inclusive,
            or None if the goal is unreachable.
        success: True if an optimal path was found, False otherwise.
        algorithm: Name of algorithm used ('astar' or 'dijkstra').
        cost: Total path cost (number of unit moves), or inf if unreachable.
        nodes_expanded: Number of unique nodes popped from priority queue whose neighbors
            were evaluated.
        nodes_discovered: Number of unique nodes added to the open set during search.
    """

    path: Optional[List[Tuple[int, int]]]
    success: bool
    algorithm: str
    cost: float
    nodes_expanded: int
    nodes_discovered: int


class PathPlanner:
    """Path planner implementing deterministic A* and Dijkstra graph search."""

    def __init__(
        self,
        algorithm: str = "astar",
        heuristic: str = "manhattan",
        allow_diagonal: bool = False,
        allow_unknown: bool = False,
    ) -> None:
        """Initializes path planner configuration.

        Args:
            algorithm: Pathfinding algorithm ('astar' or 'dijkstra').
            heuristic: Heuristic for A* ('manhattan' or 'euclidean').
            allow_diagonal: Whether 8-connected diagonal movement is allowed
                (must be False in Phase 4).
            allow_unknown: Whether UNKNOWN cells are considered traversable
                (default: False, cautious policy).

        Raises:
            ValueError: If algorithm or heuristic are unsupported.
        """
        algo_normalized = algorithm.strip().lower()
        if algo_normalized not in {"astar", "dijkstra"}:
            raise ValueError(
                f"Unsupported planner algorithm '{algorithm}'. Must be 'astar' or 'dijkstra'."
            )

        heur_normalized = heuristic.strip().lower()
        if heur_normalized not in {"manhattan", "euclidean"}:
            raise ValueError(
                f"Unsupported heuristic '{heuristic}'. Must be 'manhattan' or 'euclidean'."
            )

        self.algorithm: str = algo_normalized
        self.heuristic: str = heur_normalized
        self.allow_diagonal: bool = allow_diagonal
        self.allow_unknown: bool = allow_unknown
        self.last_result: Optional[PlanResult] = None

    @classmethod
    def from_config(
        cls,
        config: SimulationConfig,
        allow_unknown: bool = False,
    ) -> PathPlanner:
        """Constructs a PathPlanner using parameters from SimulationConfig."""
        return cls(
            algorithm=config.planner.algorithm,
            heuristic=config.planner.heuristic,
            allow_diagonal=config.planner.allow_diagonal,
            allow_unknown=allow_unknown,
        )

    # -------------------------------------------------------------------------
    # Heuristic Functions
    # -------------------------------------------------------------------------

    def compute_heuristic(self, cell: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Calculates heuristic distance estimate from cell to goal."""
        dr = abs(cell[0] - goal[0])
        dc = abs(cell[1] - goal[1])
        if self.heuristic == "manhattan":
            return float(dr + dc)
        elif self.heuristic == "euclidean":
            return math.hypot(dr, dc)
        return float(dr + dc)

    # -------------------------------------------------------------------------
    # Cell Traversability & Validation
    # -------------------------------------------------------------------------

    @staticmethod
    def _get_grid_dimensions(grid: Union[OccupancyGrid, List[List[int]]]) -> Tuple[int, int]:
        """Extracts (rows, cols) from OccupancyGrid or 2D matrix."""
        if isinstance(grid, OccupancyGrid):
            return grid.rows, grid.cols
        if isinstance(grid, list):
            rows = len(grid)
            if rows == 0:
                raise ValueError("Grid must not be empty.")
            cols = len(grid[0])
            if cols == 0:
                raise ValueError("Grid rows must not be empty.")
            return rows, cols
        raise TypeError(f"grid must be OccupancyGrid or List[List[int]], got {type(grid).__name__}")

    def _is_traversable(
        self,
        grid: Union[OccupancyGrid, List[List[int]]],
        row: int,
        col: int,
        rows: int,
        cols: int,
    ) -> bool:
        """Checks if a cell is inside bounds and traversable under current policy."""
        # Strictly reject negative indices and out-of-bounds indices
        if row < 0 or row >= rows or col < 0 or col >= cols:
            return False

        if isinstance(grid, OccupancyGrid):
            state = grid.get_cell(row, col)
            if state == CellState.FREE:
                return True
            if state == CellState.UNKNOWN:
                return self.allow_unknown
            return False  # CellState.OCCUPIED

        # Raw 2D list matrix representation
        val = grid[row][col]
        if val == CellState.FREE.value or val == 0:
            return True
        if val == CellState.UNKNOWN.value or val == -1:
            return self.allow_unknown
        return False

    def _validate_endpoints(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid: Union[OccupancyGrid, List[List[int]]],
        rows: int,
        cols: int,
    ) -> None:
        """Validates that start and goal are valid coordinates and traversable."""
        for name, pt in [("Start", start), ("Goal", goal)]:
            if not isinstance(pt, tuple) or len(pt) != 2:
                raise TypeError(f"{name} must be a (row, col) tuple of integers, got {pt}")
            r, c = pt
            if not isinstance(r, int) or isinstance(r, bool) or not isinstance(c, int) or isinstance(c, bool):
                raise TypeError(f"{name} coordinates must be integers, got ({type(r).__name__}, {type(c).__name__})")
            if r < 0 or r >= rows or c < 0 or c >= cols:
                raise IndexError(
                    f"{name} coordinate {pt} is outside grid boundaries (rows={rows}, cols={cols}). "
                    f"Negative indices are not permitted."
                )

        if not self._is_traversable(grid, start[0], start[1], rows, cols):
            raise ValueError(
                f"Start coordinate {start} is blocked or non-traversable under current policy."
            )
        if not self._is_traversable(grid, goal[0], goal[1], rows, cols):
            raise ValueError(
                f"Goal coordinate {goal} is blocked or non-traversable under current policy."
            )

    # -------------------------------------------------------------------------
    # Planning Algorithms (A* and Dijkstra)
    # -------------------------------------------------------------------------

    def plan_with_stats(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid: Union[OccupancyGrid, List[List[int]]],
        algorithm: Optional[str] = None,
    ) -> PlanResult:
        """Executes graph search and returns path along with search telemetry.

        Args:
            start: Starting grid coordinate (row, col).
            goal: Destination grid coordinate (row, col).
            grid: OccupancyGrid or 2D matrix of cell states.
            algorithm: Optional algorithm override ('astar' or 'dijkstra').

        Returns:
            PlanResult containing path, success status, cost, and expansion statistics.

        Raises:
            TypeError: If start, goal, or grid have invalid types.
            IndexError: If start or goal are outside grid bounds.
            ValueError: If start or goal are blocked.
        """
        algo_name = (algorithm or self.algorithm).strip().lower()
        if algo_name not in {"astar", "dijkstra"}:
            raise ValueError(f"Unsupported algorithm '{algo_name}'. Must be 'astar' or 'dijkstra'.")

        rows, cols = self._get_grid_dimensions(grid)
        self._validate_endpoints(start, goal, grid, rows, cols)

        # Trivial case: start equals goal
        if start == goal:
            result = PlanResult(
                path=[start],
                success=True,
                algorithm=algo_name,
                cost=0.0,
                nodes_expanded=1,
                nodes_discovered=1,
            )
            self.last_result = result
            return result

        is_astar = (algo_name == "astar")

        # Priority Queue min-heap entries: (f_score, h_score, tie_breaker_counter, cell)
        # Ties in f_score are broken by lower h_score, then by FIFO counter
        open_set: List[Tuple[float, float, int, Tuple[int, int]]] = []
        counter = 0

        start_h = self.compute_heuristic(start, goal) if is_astar else 0.0
        start_f = start_h
        heapq.heappush(open_set, (start_f, start_h, counter, start))

        g_score: Dict[Tuple[int, int], float] = {start: 0.0}
        parent: Dict[Tuple[int, int], Optional[Tuple[int, int]]] = {start: None}
        closed_set: Set[Tuple[int, int]] = set()

        nodes_expanded = 0
        discovered_set: Set[Tuple[int, int]] = {start}

        while open_set:
            f, h, _, current = heapq.heappop(open_set)

            # Stale entry pruning
            if current in closed_set:
                continue

            closed_set.add(current)
            nodes_expanded += 1

            # Termination condition: Goal reached
            if current == goal:
                # Reconstruct path from goal to start
                path_reversed: List[Tuple[int, int]] = []
                curr_node: Optional[Tuple[int, int]] = goal
                while curr_node is not None:
                    path_reversed.append(curr_node)
                    curr_node = parent[curr_node]
                path_reversed.reverse()

                result = PlanResult(
                    path=path_reversed,
                    success=True,
                    algorithm=algo_name,
                    cost=g_score[goal],
                    nodes_expanded=nodes_expanded,
                    nodes_discovered=len(discovered_set),
                )
                self.last_result = result
                return result

            # Explore 4-connected neighbors in fixed deterministic order
            for dr, dc in ORTHOGONAL_DIRECTIONS:
                neighbor = (current[0] + dr, current[1] + dc)

                if neighbor in closed_set:
                    continue

                if not self._is_traversable(grid, neighbor[0], neighbor[1], rows, cols):
                    continue

                tentative_g = g_score[current] + 1.0

                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    g_score[neighbor] = tentative_g
                    parent[neighbor] = current
                    h_val = self.compute_heuristic(neighbor, goal) if is_astar else 0.0
                    f_val = tentative_g + h_val
                    counter += 1

                    discovered_set.add(neighbor)
                    heapq.heappush(open_set, (f_val, h_val, counter, neighbor))

        # Goal is unreachable
        result = PlanResult(
            path=None,
            success=False,
            algorithm=algo_name,
            cost=float("inf"),
            nodes_expanded=nodes_expanded,
            nodes_discovered=len(discovered_set),
        )
        self.last_result = result
        return result

    def plan(
        self,
        start: Tuple[int, int],
        goal: Tuple[int, int],
        grid: Union[OccupancyGrid, List[List[int]]],
        algorithm: Optional[str] = None,
    ) -> Optional[List[Tuple[int, int]]]:
        """Calculates a collision-free path from start to goal.

        Args:
            start: Starting grid coordinate (row, col).
            goal: Destination grid coordinate (row, col).
            grid: OccupancyGrid or 2D matrix of cell states.
            algorithm: Optional algorithm override ('astar' or 'dijkstra').

        Returns:
            Ordered list of (row, col) coordinates from start to goal inclusive,
            or None if the goal is unreachable.
        """
        result = self.plan_with_stats(start, goal, grid, algorithm=algorithm)
        return result.path

"""Obstacle detection and observation integration into the robot's occupancy map.

This module processes structured sensor observations (RayObservation) and updates
the robot's internal OccupancyGrid belief.

Integration Rules:
    - Traversed cells along a ray beam are marked FREE (unless already confirmed OCCUPIED).
    - If an obstacle hit is detected, the hit cell is marked OCCUPIED.
    - If no obstacle was detected (no-hit), no endpoint is marked OCCUPIED; only
      the traversed path within range is marked FREE.
    - Cells outside observed ray paths remain completely untouched (UNKNOWN).
    - Separation of Concerns: This module never inspects or copies the ground-truth
      Environment directly.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Tuple, Union

from autonomous_nav.mapping import CellState, OccupancyGrid
from autonomous_nav.perception import RayObservation

try:
    from autonomous_nav.config import SimulationConfig
except ImportError:  # pragma: no cover
    SimulationConfig = Any  # type: ignore


class ObstacleDetector:
    """Interprets range sensor observations and updates the robot's occupancy grid."""

    def __init__(self, cell_size: int) -> None:
        """Initializes detector with grid cell dimensions.

        Args:
            cell_size: Resolution of grid cells in pixels (must be > 0).

        Raises:
            TypeError: If cell_size is not an integer.
            ValueError: If cell_size is non-positive.
        """
        if not isinstance(cell_size, int) or isinstance(cell_size, bool):
            raise TypeError(f"cell_size must be an integer, got {type(cell_size).__name__}")
        if cell_size <= 0:
            raise ValueError(f"cell_size must be positive, got {cell_size}")
        self.cell_size = cell_size

    @classmethod
    def from_config(cls, config: SimulationConfig) -> ObstacleDetector:
        """Constructs an ObstacleDetector using a validated SimulationConfig."""
        return cls(cell_size=config.grid.cell_size)

    # -------------------------------------------------------------------------
    # Spatial Point Interpretation
    # -------------------------------------------------------------------------

    def to_grid_cell(self, x: float, y: float) -> Tuple[int, int]:
        """Converts continuous world coordinates into discrete grid cell indices (row, col).

        Args:
            x: Horizontal continuous pixel coordinate.
            y: Vertical continuous pixel coordinate.

        Returns:
            Tuple of (row, col) grid indices.

        Raises:
            TypeError: If x or y are non-numeric.
        """
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            raise TypeError(f"x must be numeric, got {type(x).__name__}")
        if not isinstance(y, (int, float)) or isinstance(y, bool):
            raise TypeError(f"y must be numeric, got {type(y).__name__}")

        col = int(x // self.cell_size)
        row = int(y // self.cell_size)
        return row, col

    def extract_occupied_cells(
        self, hit_points: List[Tuple[float, float]]
    ) -> List[Tuple[int, int]]:
        """Converts raw Cartesian hit points into discrete occupied grid cells (row, col)."""
        if not isinstance(hit_points, list):
            raise TypeError(f"hit_points must be a list, got {type(hit_points).__name__}")
        return [self.to_grid_cell(x, y) for (x, y) in hit_points]

    # -------------------------------------------------------------------------
    # Observation-to-Map Integration
    # -------------------------------------------------------------------------

    def integrate_observations(
        self,
        observations: Union[RayObservation, Iterable[RayObservation]],
        grid: OccupancyGrid,
    ) -> Dict[str, int]:
        """Updates the robot's OccupancyGrid with range sensor observations.

        Behavior:
            1. Cells in each observation's traversed_cells are marked FREE (preserving
               any cells that were already confirmed OCCUPIED).
            2. If hit is True, hit_cell is marked OCCUPIED.
            3. If hit is False, no endpoint cell is marked OCCUPIED.
            4. Unobserved cells remain in their previous state (UNKNOWN).

        Args:
            observations: Single RayObservation or sequence of RayObservation objects.
            grid: Robot's OccupancyGrid to be updated.

        Returns:
            Dictionary with counts of newly marked free and occupied cells:
            {"freed_cells": int, "occupied_cells": int}

        Raises:
            TypeError: If grid or observations are of invalid types.
            ValueError: If observations are malformed or grid dimensions are incompatible.
        """
        if not isinstance(grid, OccupancyGrid):
            raise TypeError(f"grid must be an OccupancyGrid instance, got {type(grid).__name__}")
        if grid.cell_size != self.cell_size:
            raise ValueError(
                f"Incompatible grid resolution: grid cell_size ({grid.cell_size}) "
                f"does not match detector cell_size ({self.cell_size})."
            )

        if isinstance(observations, RayObservation):
            obs_list = [observations]
        elif isinstance(observations, Iterable):
            obs_list = list(observations)
        else:
            raise TypeError(
                f"observations must be a RayObservation or iterable of RayObservation, "
                f"got {type(observations).__name__}"
            )

        newly_freed = 0
        newly_occupied = 0

        for obs in obs_list:
            if not isinstance(obs, RayObservation):
                raise TypeError(
                    f"Observation item must be a RayObservation instance, got {type(obs).__name__}"
                )

            # Validate internal observation consistency
            if obs.distance < 0.0:
                raise ValueError(f"Observation distance cannot be negative, got {obs.distance}")
            if obs.hit and obs.hit_cell is None:
                raise ValueError("Malformed observation: hit is True but hit_cell is None.")
            if not obs.hit and obs.hit_cell is not None:
                raise ValueError("Malformed observation: hit is False but hit_cell is not None.")

            # Step 1: Mark traversed beam cells as FREE
            for cell in obs.traversed_cells:
                if not isinstance(cell, tuple) or len(cell) != 2:
                    raise ValueError(f"Malformed traversed cell entry: {cell}")
                r, c = cell
                if grid.is_in_bounds(r, c):
                    # Never overwrite an already confirmed obstacle cell with FREE
                    if grid.get_cell(r, c) != CellState.OCCUPIED:
                        if grid.get_cell(r, c) != CellState.FREE:
                            grid.mark_free(r, c)
                            newly_freed += 1

            # Step 2: Mark detected obstacle cell as OCCUPIED
            if obs.hit and obs.hit_cell is not None:
                r, c = obs.hit_cell
                if grid.is_in_bounds(r, c):
                    if grid.get_cell(r, c) != CellState.OCCUPIED:
                        grid.mark_occupied(r, c)
                        newly_occupied += 1

        return {"freed_cells": newly_freed, "occupied_cells": newly_occupied}

    def update_map(
        self,
        observations: Union[RayObservation, Iterable[RayObservation]],
        grid: OccupancyGrid,
    ) -> Dict[str, int]:
        """Convenience alias for integrate_observations."""
        return self.integrate_observations(observations, grid)

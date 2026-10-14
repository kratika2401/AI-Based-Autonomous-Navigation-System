"""Occupancy grid representation and internal robot map.

This module maintains the robot's subjective belief of the 2D world.
It tracks which cells are known free, known occupied, or unexplored (unknown),
serving as the internal spatial representation and search domain for path planning.

Coordinate System & Conventions:
    - Pixel space: (x, y) coordinates with floating-point or integer values.
      Origin (0.0, 0.0) is at the top-left.
      x increases to the right: valid range [0.0, width).
      y increases downward: valid range [0.0, height).
      Coordinates exactly on the outer boundaries (x == width or y == height)
      are strictly out of bounds (half-open interval [0, dimension)).
    - Grid space: (row, column) coordinates with integer values.
      row increases downward: valid range [0, rows - 1].
      col increases to the right: valid range [0, cols - 1].
    - Grid-to-pixel conversion maps a grid cell (row, col) to its geometric center:
      center_x = col * cell_size + cell_size / 2.0
      center_y = row * cell_size + cell_size / 2.0

Separation of Concerns:
    The OccupancyGrid is independent of the ground-truth Environment.
    It represents only what the robot has observed or been explicitly given.
"""

from __future__ import annotations

from enum import IntEnum
from typing import Any, List, Optional, Tuple, Union

try:
    from autonomous_nav.config import SimulationConfig
except ImportError:  # pragma: no cover
    SimulationConfig = Any  # type: ignore


class CellState(IntEnum):
    """Explicit enumeration of valid discrete occupancy grid cell states."""
    UNKNOWN = -1
    FREE = 0
    OCCUPIED = 1


class OccupancyGrid:
    """Robot's internal discrete occupancy grid belief."""

    # Backward-compatible class-level references to CellState
    UNKNOWN: int = CellState.UNKNOWN
    FREE: int = CellState.FREE
    OCCUPIED: int = CellState.OCCUPIED

    def __init__(
        self,
        rows: int,
        cols: int,
        cell_size: int,
        initial_state: Union[CellState, int] = CellState.UNKNOWN,
    ) -> None:
        """Initializes the occupancy grid.

        Args:
            rows: Number of vertical grid rows.
            cols: Number of horizontal grid columns.
            cell_size: Dimension of each square cell in pixels.
            initial_state: Initial state for all cells (default: UNKNOWN).

        Raises:
            TypeError: If rows, cols, or cell_size are not integers.
            ValueError: If rows, cols, or cell_size are non-positive, or if initial_state is invalid.
        """
        if not isinstance(rows, int) or isinstance(rows, bool):
            raise TypeError(f"rows must be an integer, got {type(rows).__name__}")
        if not isinstance(cols, int) or isinstance(cols, bool):
            raise TypeError(f"cols must be an integer, got {type(cols).__name__}")
        if not isinstance(cell_size, int) or isinstance(cell_size, bool):
            raise TypeError(f"cell_size must be an integer, got {type(cell_size).__name__}")

        if rows <= 0 or cols <= 0 or cell_size <= 0:
            raise ValueError(
                f"OccupancyGrid dimensions and cell_size must be positive. "
                f"Got rows={rows}, cols={cols}, cell_size={cell_size}"
            )

        validated_initial = self._validate_state(initial_state)

        self.rows: int = rows
        self.cols: int = cols
        self.cell_size: int = cell_size
        self.width: int = cols * cell_size
        self.height: int = rows * cell_size
        self.shape: Tuple[int, int] = (rows, cols)

        # 2D representation indexed as [row][col]
        self._grid: List[List[CellState]] = [
            [validated_initial for _ in range(cols)] for _ in range(rows)
        ]

    @classmethod
    def from_config(
        cls,
        config: SimulationConfig,
        initial_state: Union[CellState, int] = CellState.UNKNOWN,
    ) -> OccupancyGrid:
        """Constructs an OccupancyGrid using a validated SimulationConfig."""
        return cls(
            rows=config.grid.height_cells,
            cols=config.grid.width_cells,
            cell_size=config.grid.cell_size,
            initial_state=initial_state,
        )

    # -------------------------------------------------------------------------
    # Internal Helpers
    # -------------------------------------------------------------------------

    @staticmethod
    def _validate_state(state: Union[CellState, int]) -> CellState:
        """Validates and coerces an input state to a CellState enum."""
        if not isinstance(state, (CellState, int)) or isinstance(state, bool):
            raise TypeError(f"Cell state must be an integer or CellState, got {type(state).__name__}")
        try:
            return CellState(state)
        except ValueError:
            raise ValueError(
                f"Invalid occupancy state: {state}. "
                f"Must be one of CellState.UNKNOWN (-1), CellState.FREE (0), or CellState.OCCUPIED (1)."
            )

    # -------------------------------------------------------------------------
    # Boundary Checks
    # -------------------------------------------------------------------------

    def is_in_bounds(self, row: int, col: int) -> bool:
        """Checks if discrete grid coordinates lie within grid bounds.

        Args:
            row: Vertical grid index (0-indexed, increasing downward).
            col: Horizontal grid index (0-indexed, increasing rightward).

        Returns:
            True if 0 <= row < rows and 0 <= col < cols, False otherwise.
        """
        if not isinstance(row, int) or isinstance(row, bool):
            return False
        if not isinstance(col, int) or isinstance(col, bool):
            return False
        return 0 <= row < self.rows and 0 <= col < self.cols

    def is_point_in_bounds(self, x: float, y: float) -> bool:
        """Checks if continuous pixel coordinates lie within grid bounds.

        Outer Boundary Convention:
            Half-open region [0.0, width) and [0.0, height). Coordinates exactly
            on width or height are out of bounds.

        Args:
            x: Horizontal pixel coordinate.
            y: Vertical pixel coordinate.

        Returns:
            True if within [0.0, width) and [0.0, height), False otherwise.
        """
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            return False
        if not isinstance(y, (int, float)) or isinstance(y, bool):
            return False
        return 0.0 <= float(x) < float(self.width) and 0.0 <= float(y) < float(self.height)

    # -------------------------------------------------------------------------
    # Coordinate Conversions
    # -------------------------------------------------------------------------

    def pixel_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Converts continuous pixel coordinates (x, y) to discrete grid (row, col).

        Args:
            x: Horizontal pixel coordinate.
            y: Vertical pixel coordinate.

        Returns:
            Tuple of (row, col) grid coordinates.

        Raises:
            TypeError: If x or y are non-numeric.
            IndexError: If (x, y) is out of bounds [0.0, width) x [0.0, height).
        """
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            raise TypeError(f"x must be numeric, got {type(x).__name__}")
        if not isinstance(y, (int, float)) or isinstance(y, bool):
            raise TypeError(f"y must be numeric, got {type(y).__name__}")

        if not self.is_point_in_bounds(x, y):
            raise IndexError(
                f"Pixel coordinate ({x}, {y}) is outside grid bounds "
                f"[0, {self.width}) x [0, {self.height})."
            )

        col = int(x // self.cell_size)
        row = int(y // self.cell_size)
        return row, col

    def grid_to_pixel_center(self, row: int, col: int) -> Tuple[float, float]:
        """Converts discrete grid coordinates (row, col) to the pixel center (x, y).

        Args:
            row: Vertical grid index.
            col: Horizontal grid index.

        Returns:
            Tuple of (center_x, center_y) in pixels.

        Raises:
            TypeError: If row or col are not integers.
            IndexError: If (row, col) is outside grid boundaries.
        """
        if not isinstance(row, int) or isinstance(row, bool):
            raise TypeError(f"row must be an integer, got {type(row).__name__}")
        if not isinstance(col, int) or isinstance(col, bool):
            raise TypeError(f"col must be an integer, got {type(col).__name__}")

        if not self.is_in_bounds(row, col):
            raise IndexError(
                f"Grid cell ({row}, {col}) is out of grid bounds ({self.rows}, {self.cols})."
            )

        center_x = col * self.cell_size + (self.cell_size / 2.0)
        center_y = row * self.cell_size + (self.cell_size / 2.0)
        return center_x, center_y

    # -------------------------------------------------------------------------
    # State Access & Mutation
    # -------------------------------------------------------------------------

    def get_cell(self, row: int, col: int) -> CellState:
        """Returns the belief state of the specified cell.

        Args:
            row: Vertical grid index.
            col: Horizontal grid index.

        Returns:
            CellState (UNKNOWN, FREE, or OCCUPIED).

        Raises:
            TypeError: If row or col are not integers.
            IndexError: If (row, col) is out of bounds.
        """
        if not isinstance(row, int) or isinstance(row, bool):
            raise TypeError(f"row must be an integer, got {type(row).__name__}")
        if not isinstance(col, int) or isinstance(col, bool):
            raise TypeError(f"col must be an integer, got {type(col).__name__}")

        if not self.is_in_bounds(row, col):
            raise IndexError(
                f"Cell ({row}, {col}) is outside grid bounds ({self.rows}, {self.cols})."
            )
        return self._grid[row][col]

    def set_cell(self, row: int, col: int, state: Union[CellState, int]) -> None:
        """Updates the belief state of the specified cell.

        Args:
            row: Vertical grid index.
            col: Horizontal grid index.
            state: New state (CellState or integer: -1, 0, or 1).

        Raises:
            TypeError: If row/col are not integers, or state is invalid type.
            IndexError: If (row, col) is out of bounds.
            ValueError: If state is not a valid CellState value.
        """
        if not isinstance(row, int) or isinstance(row, bool):
            raise TypeError(f"row must be an integer, got {type(row).__name__}")
        if not isinstance(col, int) or isinstance(col, bool):
            raise TypeError(f"col must be an integer, got {type(col).__name__}")

        if not self.is_in_bounds(row, col):
            raise IndexError(
                f"Cell ({row}, {col}) is outside grid bounds ({self.rows}, {self.cols})."
            )

        validated_state = self._validate_state(state)
        self._grid[row][col] = validated_state

    def mark_free(self, row: int, col: int) -> None:
        """Convenience method to set a cell to FREE."""
        self.set_cell(row, col, CellState.FREE)

    def mark_occupied(self, row: int, col: int) -> None:
        """Convenience method to set a cell to OCCUPIED."""
        self.set_cell(row, col, CellState.OCCUPIED)

    def mark_unknown(self, row: int, col: int) -> None:
        """Convenience method to reset a cell to UNKNOWN."""
        self.set_cell(row, col, CellState.UNKNOWN)

    def is_free(self, row: int, col: int) -> bool:
        """Queries if cell is known FREE."""
        return self.get_cell(row, col) == CellState.FREE

    def is_occupied(self, row: int, col: int) -> bool:
        """Queries if cell is known OCCUPIED."""
        return self.get_cell(row, col) == CellState.OCCUPIED

    def is_unknown(self, row: int, col: int) -> bool:
        """Queries if cell is UNKNOWN."""
        return self.get_cell(row, col) == CellState.UNKNOWN

    def to_matrix(self) -> List[List[int]]:
        """Returns a snapshot of the grid as a 2D list of integer states."""
        return [[int(cell) for cell in row] for row in self._grid]

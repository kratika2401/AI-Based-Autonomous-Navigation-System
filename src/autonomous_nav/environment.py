"""Ground-truth simulation environment and obstacle representation.

This module maintains the physical ground truth of the 2D simulated world,
including dimensional boundaries, static obstacle geometries, collision queries,
and start/goal validation.

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
    The Environment represents the objective ground truth of the physical simulation.
    The autonomous agent does NOT possess direct access to this world model; it must
    observe features via perception sensors and construct its own OccupancyGrid.
"""

from __future__ import annotations

from typing import Any, List, Optional, Set, Tuple, Union

try:
    from autonomous_nav.config import SimulationConfig
except ImportError:  # pragma: no cover
    SimulationConfig = Any  # type: ignore


class Environment:
    """Represents the ground-truth 2D simulation world."""

    def __init__(self, width: int, height: int, cell_size: int) -> None:
        """Initializes the ground truth environment boundaries.

        Args:
            width: Total width of the simulation world in pixels. Must be positive
                and divisible by cell_size.
            height: Total height of the simulation world in pixels. Must be positive
                and divisible by cell_size.
            cell_size: Resolution of each discrete grid cell in pixels. Must be positive.

        Raises:
            TypeError: If width, height, or cell_size are not integers.
            ValueError: If dimensions are non-positive or do not divide evenly by cell_size.
        """
        if not isinstance(width, int) or isinstance(width, bool):
            raise TypeError(f"width must be an integer, got {type(width).__name__}")
        if not isinstance(height, int) or isinstance(height, bool):
            raise TypeError(f"height must be an integer, got {type(height).__name__}")
        if not isinstance(cell_size, int) or isinstance(cell_size, bool):
            raise TypeError(f"cell_size must be an integer, got {type(cell_size).__name__}")

        if width <= 0 or height <= 0 or cell_size <= 0:
            raise ValueError(
                f"Environment dimensions and cell_size must be positive. "
                f"Got width={width}, height={height}, cell_size={cell_size}"
            )

        if width % cell_size != 0:
            raise ValueError(
                f"width ({width}) must be an exact multiple of cell_size ({cell_size})."
            )
        if height % cell_size != 0:
            raise ValueError(
                f"height ({height}) must be an exact multiple of cell_size ({cell_size})."
            )

        self.width: int = width
        self.height: int = height
        self.cell_size: int = cell_size
        self.cols: int = width // cell_size
        self.rows: int = height // cell_size

        # Set of ground-truth occupied grid coordinates: (row, col)
        self.static_obstacles: Set[Tuple[int, int]] = set()
        # Preserved for future phases:
        self.dynamic_obstacles: Set[Tuple[int, int]] = set()

    @classmethod
    def from_config(cls, config: SimulationConfig) -> Environment:
        """Constructs an Environment instance using a validated SimulationConfig."""
        return cls(
            width=config.window.width,
            height=config.window.height,
            cell_size=config.grid.cell_size,
        )

    # -------------------------------------------------------------------------
    # Boundary Checks
    # -------------------------------------------------------------------------

    def is_within_bounds(self, row: int, col: int) -> bool:
        """Checks if a discrete grid coordinate falls within world boundaries.

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

    def is_point_within_bounds(self, x: float, y: float) -> bool:
        """Checks if a continuous pixel position falls inside the world boundaries.

        Outer Boundary Convention:
            The valid coordinate region is the half-open interval [0.0, width)
            and [0.0, height). Coordinates exactly on the right or bottom boundary
            (x == width or y == height) or negative coordinates are outside bounds.

        Args:
            x: Continuous horizontal coordinate in pixels.
            y: Continuous vertical coordinate in pixels.

        Returns:
            True if 0.0 <= x < width and 0.0 <= y < height, False otherwise.
        """
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            return False
        if not isinstance(y, (int, float)) or isinstance(y, bool):
            return False
        return 0.0 <= float(x) < float(self.width) and 0.0 <= float(y) < float(self.height)

    # -------------------------------------------------------------------------
    # Coordinate Conversions
    # -------------------------------------------------------------------------

    def point_to_grid(self, x: float, y: float) -> Tuple[int, int]:
        """Converts continuous pixel coordinates (x, y) to discrete grid (row, col).

        Args:
            x: Horizontal pixel coordinate.
            y: Vertical pixel coordinate.

        Returns:
            Tuple of (row, col) grid coordinates.

        Raises:
            TypeError: If x or y are not numeric.
            IndexError: If (x, y) is outside world boundaries [0, width) x [0, height).
        """
        if not isinstance(x, (int, float)) or isinstance(x, bool):
            raise TypeError(f"x must be numeric, got {type(x).__name__}")
        if not isinstance(y, (int, float)) or isinstance(y, bool):
            raise TypeError(f"y must be numeric, got {type(y).__name__}")

        if not self.is_point_within_bounds(x, y):
            raise IndexError(
                f"Pixel coordinate ({x}, {y}) is outside world bounds "
                f"[0, {self.width}) x [0, {self.height})."
            )

        col = int(x // self.cell_size)
        row = int(y // self.cell_size)
        return row, col

    def grid_to_point(self, row: int, col: int) -> Tuple[float, float]:
        """Converts discrete grid cell (row, col) to its continuous pixel center (x, y).

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

        if not self.is_within_bounds(row, col):
            raise IndexError(
                f"Grid cell ({row}, {col}) is out of world bounds ({self.rows}, {self.cols})."
            )

        center_x = col * self.cell_size + (self.cell_size / 2.0)
        center_y = row * self.cell_size + (self.cell_size / 2.0)
        return center_x, center_y

    # -------------------------------------------------------------------------
    # Obstacle Management
    # -------------------------------------------------------------------------

    def add_static_obstacle(self, row: int, col: int) -> None:
        """Adds a static obstacle cell to the ground truth environment.

        Args:
            row: Vertical grid coordinate.
            col: Horizontal grid coordinate.

        Raises:
            TypeError: If row or col are not integers.
            IndexError: If (row, col) is outside world boundaries.
        """
        if not isinstance(row, int) or isinstance(row, bool):
            raise TypeError(f"row must be an integer, got {type(row).__name__}")
        if not isinstance(col, int) or isinstance(col, bool):
            raise TypeError(f"col must be an integer, got {type(col).__name__}")

        if not self.is_within_bounds(row, col):
            raise IndexError(
                f"Cell ({row}, {col}) is out of world bounds (rows={self.rows}, cols={self.cols})."
            )
        self.static_obstacles.add((row, col))

    def add_static_obstacle_rect(
        self, top_row: int, left_col: int, num_rows: int, num_cols: int
    ) -> None:
        """Adds a rectangular block of static obstacle cells atomically.

        Rectangle Convention:
            Covers rows in range [top_row, top_row + num_rows - 1] (inclusive)
            and cols in range [left_col, left_col + num_cols - 1] (inclusive).

        Args:
            top_row: Topmost row index of the rectangle.
            left_col: Leftmost column index of the rectangle.
            num_rows: Number of rows in height (must be > 0).
            num_cols: Number of columns in width (must be > 0).

        Raises:
            TypeError: If arguments are not integers.
            ValueError: If num_rows or num_cols are non-positive.
            IndexError: If any cell in the rectangle lies outside world bounds.
        """
        for name, val in [
            ("top_row", top_row),
            ("left_col", left_col),
            ("num_rows", num_rows),
            ("num_cols", num_cols),
        ]:
            if not isinstance(val, int) or isinstance(val, bool):
                raise TypeError(f"{name} must be an integer, got {type(val).__name__}")

        if num_rows <= 0 or num_cols <= 0:
            raise ValueError(
                f"Rectangle dimensions must be positive, got {num_rows}x{num_cols}"
            )

        # Validate that all candidate cells fall within bounds before insertion
        candidate_cells = []
        for r in range(top_row, top_row + num_rows):
            for c in range(left_col, left_col + num_cols):
                if not self.is_within_bounds(r, c):
                    raise IndexError(
                        f"Obstacle rectangle [{top_row}:{top_row+num_rows}, {left_col}:{left_col+num_cols}] "
                        f"exceeds world bounds (rows={self.rows}, cols={self.cols}) at cell ({r}, {c})."
                    )
                candidate_cells.append((r, c))

        for cell in candidate_cells:
            self.static_obstacles.add(cell)

    def remove_static_obstacle(self, row: int, col: int) -> bool:
        """Removes a static obstacle cell if present.

        Returns:
            True if an obstacle was removed, False if the cell was not occupied.

        Raises:
            TypeError: If row or col are not integers.
            IndexError: If (row, col) is outside world bounds.
        """
        if not isinstance(row, int) or isinstance(row, bool):
            raise TypeError(f"row must be an integer, got {type(row).__name__}")
        if not isinstance(col, int) or isinstance(col, bool):
            raise TypeError(f"col must be an integer, got {type(col).__name__}")

        if not self.is_within_bounds(row, col):
            raise IndexError(
                f"Cell ({row}, {col}) is out of world bounds (rows={self.rows}, cols={self.cols})."
            )
        if (row, col) in self.static_obstacles:
            self.static_obstacles.remove((row, col))
            return True
        return False

    def clear_obstacles(self) -> None:
        """Clears all static obstacles from the environment."""
        self.static_obstacles.clear()

    # -------------------------------------------------------------------------
    # Occupancy Queries
    # -------------------------------------------------------------------------

    def is_occupied(self, row: int, col: int) -> bool:
        """Queries whether a grid location is occupied by a static or dynamic obstacle.

        Args:
            row: Vertical grid index.
            col: Horizontal grid index.

        Returns:
            True if occupied, False if free.

        Raises:
            TypeError: If row or col are not integers.
            IndexError: If (row, col) is outside world boundaries.
        """
        if not isinstance(row, int) or isinstance(row, bool):
            raise TypeError(f"row must be an integer, got {type(row).__name__}")
        if not isinstance(col, int) or isinstance(col, bool):
            raise TypeError(f"col must be an integer, got {type(col).__name__}")

        if not self.is_within_bounds(row, col):
            raise IndexError(
                f"Cell ({row}, {col}) is out of world bounds (rows={self.rows}, cols={self.cols})."
            )
        return (row, col) in self.static_obstacles or (row, col) in self.dynamic_obstacles

    def is_point_occupied(self, x: float, y: float) -> bool:
        """Queries whether a continuous pixel point falls into an occupied grid cell.

        Args:
            x: Horizontal pixel coordinate.
            y: Vertical pixel coordinate.

        Returns:
            True if the enclosing cell is occupied, False otherwise.

        Raises:
            TypeError: If x or y are not numeric.
            IndexError: If (x, y) is outside world bounds.
        """
        row, col = self.point_to_grid(x, y)
        return self.is_occupied(row, col)

    # -------------------------------------------------------------------------
    # Validation & Utilities
    # -------------------------------------------------------------------------

    def validate_position(self, row: int, col: int, label: str = "Position") -> None:
        """Validates that a grid position is inside world bounds and unoccupied.

        Args:
            row: Vertical grid index.
            col: Horizontal grid index.
            label: Descriptive label for error reporting (e.g. "Start", "Goal").

        Raises:
            TypeError: If row or col are not integers.
            IndexError: If position is outside world bounds.
            ValueError: If position is occupied by an obstacle.
        """
        if not self.is_within_bounds(row, col):
            raise IndexError(
                f"{label} ({row}, {col}) is out of world bounds (rows={self.rows}, cols={self.cols})."
            )
        if self.is_occupied(row, col):
            raise ValueError(
                f"{label} ({row}, {col}) is occupied by an obstacle."
            )

    def validate_start_goal(
        self, start: Tuple[int, int], goal: Tuple[int, int]
    ) -> None:
        """Validates both start and goal positions for reachability constraints.

        Args:
            start: (row, col) tuple for start point.
            goal: (row, col) tuple for goal point.

        Raises:
            TypeError: If start or goal are not 2-tuples of integers.
            IndexError: If start or goal are out of bounds.
            ValueError: If start or goal are occupied, or if start equals goal.
        """
        if not isinstance(start, tuple) or len(start) != 2:
            raise TypeError(f"start must be a (row, col) tuple, got {start}")
        if not isinstance(goal, tuple) or len(goal) != 2:
            raise TypeError(f"goal must be a (row, col) tuple, got {goal}")

        self.validate_position(start[0], start[1], label="Start")
        self.validate_position(goal[0], goal[1], label="Goal")

        if start == goal:
            raise ValueError(f"Start position ({start}) cannot be identical to goal position ({goal}).")

    def get_static_obstacles(self) -> Set[Tuple[int, int]]:
        """Returns a copy of the static obstacles set."""
        return set(self.static_obstacles)

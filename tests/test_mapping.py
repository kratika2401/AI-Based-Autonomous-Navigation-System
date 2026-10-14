"""Unit tests for occupancy grid mapping and state management."""

import pytest

from autonomous_nav.config import load_config
from autonomous_nav.mapping import CellState, OccupancyGrid


# =============================================================================
# 1. Construction and Dimension Tests
# =============================================================================

def test_occupancy_grid_valid_construction():
    """Verify OccupancyGrid initializes with expected dimensions and default UNKNOWN state."""
    grid = OccupancyGrid(rows=30, cols=40, cell_size=20)
    assert grid.rows == 30
    assert grid.cols == 40
    assert grid.cell_size == 20
    assert grid.width == 800
    assert grid.height == 600
    assert grid.shape == (30, 40)

    # All cells must start in UNKNOWN state
    for r in range(grid.rows):
        for c in range(grid.cols):
            assert grid.get_cell(r, c) == CellState.UNKNOWN
            assert grid.is_unknown(r, c) is True


def test_occupancy_grid_construction_from_config():
    """Verify OccupancyGrid.from_config creates grid matching configuration parameters."""
    cfg = load_config()
    grid = OccupancyGrid.from_config(cfg)
    assert grid.rows == cfg.grid.height_cells
    assert grid.cols == cfg.grid.width_cells
    assert grid.cell_size == cfg.grid.cell_size
    assert grid.width == cfg.window.width
    assert grid.height == cfg.window.height


def test_occupancy_grid_custom_initial_state():
    """Verify OccupancyGrid can initialize with custom initial state like FREE."""
    grid = OccupancyGrid(rows=5, cols=5, cell_size=10, initial_state=CellState.FREE)
    assert grid.get_cell(0, 0) == CellState.FREE
    assert grid.is_free(0, 0) is True


@pytest.mark.parametrize(
    "rows, cols, cell_size, error_type, match_str",
    [
        (0, 40, 20, ValueError, "must be positive"),
        (30, 0, 20, ValueError, "must be positive"),
        (30, 40, 0, ValueError, "must be positive"),
        (-5, 40, 20, ValueError, "must be positive"),
        (30, -5, 20, ValueError, "must be positive"),
        (30, 40, -20, ValueError, "must be positive"),
        ("30", 40, 20, TypeError, "rows must be an integer"),
        (30, "40", 20, TypeError, "cols must be an integer"),
        (30, 40, 20.0, TypeError, "cell_size must be an integer"),
        (True, 40, 20, TypeError, "rows must be an integer"),
    ],
)
def test_occupancy_grid_invalid_dimensions(rows, cols, cell_size, error_type, match_str):
    """Verify invalid grid dimensions raise ValueError or TypeError."""
    with pytest.raises(error_type, match=match_str):
        OccupancyGrid(rows=rows, cols=cols, cell_size=cell_size)


# =============================================================================
# 2. State Access and Mutation Tests
# =============================================================================

def test_occupancy_grid_state_updates():
    """Verify mutating and reading cell states using direct and convenience methods."""
    grid = OccupancyGrid(rows=10, cols=10, cell_size=20)

    # Set to FREE
    grid.set_cell(2, 3, CellState.FREE)
    assert grid.get_cell(2, 3) == CellState.FREE
    assert grid.is_free(2, 3) is True
    assert grid.is_occupied(2, 3) is False
    assert grid.is_unknown(2, 3) is False

    # Mark OCCUPIED
    grid.mark_occupied(4, 5)
    assert grid.get_cell(4, 5) == CellState.OCCUPIED
    assert grid.is_occupied(4, 5) is True

    # Mark FREE via helper
    grid.mark_free(4, 5)
    assert grid.get_cell(4, 5) == CellState.FREE

    # Mark UNKNOWN via helper
    grid.mark_unknown(4, 5)
    assert grid.get_cell(4, 5) == CellState.UNKNOWN


def test_occupancy_grid_accepts_integer_state_values():
    """Verify integer constants matching CellState are accepted."""
    grid = OccupancyGrid(rows=5, cols=5, cell_size=10)
    grid.set_cell(0, 0, 0)   # FREE
    assert grid.get_cell(0, 0) == CellState.FREE
    grid.set_cell(0, 1, 1)   # OCCUPIED
    assert grid.get_cell(0, 1) == CellState.OCCUPIED
    grid.set_cell(0, 2, -1)  # UNKNOWN
    assert grid.get_cell(0, 2) == CellState.UNKNOWN


def test_occupancy_grid_rejects_invalid_state():
    """Verify that invalid state values raise ValueError or TypeError."""
    grid = OccupancyGrid(rows=5, cols=5, cell_size=10)
    with pytest.raises(ValueError, match="Invalid occupancy state"):
        grid.set_cell(1, 1, 999)
    with pytest.raises(ValueError, match="Invalid occupancy state"):
        grid.set_cell(1, 1, -2)
    with pytest.raises(TypeError, match="Cell state must be an integer"):
        grid.set_cell(1, 1, "FREE")
    with pytest.raises(TypeError, match="Cell state must be an integer"):
        grid.set_cell(1, 1, True)


# =============================================================================
# 3. Boundary & Indexing Error Handling Tests
# =============================================================================

def test_occupancy_grid_bounds_checking():
    """Verify is_in_bounds handles edge cells and out-of-bounds cells."""
    grid = OccupancyGrid(rows=30, cols=40, cell_size=20)
    assert grid.is_in_bounds(0, 0) is True
    assert grid.is_in_bounds(29, 39) is True

    # Negative indices must be rejected (no Python negative wrap-around)
    assert grid.is_in_bounds(-1, 0) is False
    assert grid.is_in_bounds(0, -1) is False

    # Outer indices
    assert grid.is_in_bounds(30, 0) is False
    assert grid.is_in_bounds(0, 40) is False


def test_occupancy_grid_out_of_bounds_access_raises():
    """Verify get_cell and set_cell raise IndexError on out-of-bounds indices."""
    grid = OccupancyGrid(rows=10, cols=10, cell_size=20)

    with pytest.raises(IndexError, match="outside grid bounds"):
        grid.get_cell(10, 5)
    with pytest.raises(IndexError, match="outside grid bounds"):
        grid.get_cell(-1, 0)
    with pytest.raises(IndexError, match="outside grid bounds"):
        grid.set_cell(0, 10, CellState.FREE)
    with pytest.raises(IndexError, match="outside grid bounds"):
        grid.set_cell(0, -1, CellState.FREE)


# =============================================================================
# 4. Coordinate Conversion Tests
# =============================================================================

def test_pixel_to_grid_conversion():
    """Verify continuous pixel coordinates convert to expected discrete grid indices."""
    grid = OccupancyGrid(rows=30, cols=40, cell_size=20)

    # Origin
    assert grid.pixel_to_grid(0.0, 0.0) == (0, 0)

    # Near boundary of first cell
    assert grid.pixel_to_grid(19.99, 19.99) == (0, 0)

    # Exactly at boundary of second cell
    assert grid.pixel_to_grid(20.0, 0.0) == (0, 1)
    assert grid.pixel_to_grid(0.0, 20.0) == (1, 0)

    # Bottom-right valid cell (row 29, col 39)
    assert grid.pixel_to_grid(799.9, 599.9) == (29, 39)


def test_pixel_to_grid_boundary_rejection():
    """Verify points on right/bottom edges or outside grid raise IndexError."""
    grid = OccupancyGrid(rows=30, cols=40, cell_size=20)  # width=800, height=600

    # Exact outer boundary edges must be rejected
    with pytest.raises(IndexError, match="outside grid bounds"):
        grid.pixel_to_grid(800.0, 300.0)  # x == width
    with pytest.raises(IndexError, match="outside grid bounds"):
        grid.pixel_to_grid(400.0, 600.0)  # y == height
    with pytest.raises(IndexError, match="outside grid bounds"):
        grid.pixel_to_grid(800.0, 600.0)

    # Negative coordinates
    with pytest.raises(IndexError, match="outside grid bounds"):
        grid.pixel_to_grid(-0.01, 10.0)
    with pytest.raises(IndexError, match="outside grid bounds"):
        grid.pixel_to_grid(10.0, -0.01)


def test_grid_to_pixel_center_conversion():
    """Verify discrete grid cells convert to geometric pixel centers."""
    grid = OccupancyGrid(rows=30, cols=40, cell_size=20)

    # Cell (0, 0) center: x = 0*20 + 10 = 10.0, y = 0*20 + 10 = 10.0
    assert grid.grid_to_pixel_center(0, 0) == (10.0, 10.0)

    # Cell (row 5, col 10) center: x = 10*20 + 10 = 210.0, y = 5*20 + 10 = 110.0
    assert grid.grid_to_pixel_center(5, 10) == (210.0, 110.0)

    # Bottom-right cell (29, 39) center: x = 39*20 + 10 = 790.0, y = 29*20 + 10 = 590.0
    assert grid.grid_to_pixel_center(29, 39) == (790.0, 590.0)


def test_grid_to_pixel_center_out_of_bounds_raises():
    """Verify grid_to_pixel_center raises IndexError on invalid indices."""
    grid = OccupancyGrid(rows=30, cols=40, cell_size=20)
    with pytest.raises(IndexError, match="out of grid bounds"):
        grid.grid_to_pixel_center(30, 0)
    with pytest.raises(IndexError, match="out of grid bounds"):
        grid.grid_to_pixel_center(0, 40)
    with pytest.raises(IndexError, match="out of grid bounds"):
        grid.grid_to_pixel_center(-1, 0)


# =============================================================================
# 5. Serialization & Export Tests
# =============================================================================

def test_occupancy_grid_to_matrix():
    """Verify to_matrix produces 2D matrix matching cell states."""
    grid = OccupancyGrid(rows=2, cols=3, cell_size=10)
    grid.mark_free(0, 0)
    grid.mark_occupied(0, 1)
    # Remaining cells UNKNOWN (-1)

    mat = grid.to_matrix()
    assert mat == [
        [0, 1, -1],
        [-1, -1, -1],
    ]

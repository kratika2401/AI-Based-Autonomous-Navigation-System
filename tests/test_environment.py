"""Unit tests for ground-truth environment representation and spatial mechanics."""

import pytest

from autonomous_nav.config import load_config
from autonomous_nav.environment import Environment


# =============================================================================
# 1. Construction and Dimension Validation Tests
# =============================================================================

def test_environment_valid_construction():
    """Verify that environment initializes with valid dimensions and grid calculation."""
    env = Environment(width=800, height=600, cell_size=20)
    assert env.width == 800
    assert env.height == 600
    assert env.cell_size == 20
    assert env.cols == 40
    assert env.rows == 30
    assert len(env.static_obstacles) == 0


def test_environment_construction_from_config():
    """Verify Environment.from_config correctly initializes from loaded configuration."""
    cfg = load_config()
    env = Environment.from_config(cfg)
    assert env.width == cfg.window.width
    assert env.height == cfg.window.height
    assert env.cell_size == cfg.grid.cell_size
    assert env.cols == cfg.grid.width_cells
    assert env.rows == cfg.grid.height_cells


@pytest.mark.parametrize(
    "width, height, cell_size, error_type, match_str",
    [
        (0, 600, 20, ValueError, "must be positive"),
        (800, 0, 20, ValueError, "must be positive"),
        (800, 600, 0, ValueError, "must be positive"),
        (-800, 600, 20, ValueError, "must be positive"),
        (800, -600, 20, ValueError, "must be positive"),
        (800, 600, -20, ValueError, "must be positive"),
        (805, 600, 20, ValueError, "must be an exact multiple of cell_size"),
        (800, 605, 20, ValueError, "must be an exact multiple of cell_size"),
        ("800", 600, 20, TypeError, "width must be an integer"),
        (800, "600", 20, TypeError, "height must be an integer"),
        (800, 600, 20.0, TypeError, "cell_size must be an integer"),
        (True, 600, 20, TypeError, "width must be an integer"),
    ],
)
def test_environment_invalid_dimensions(width, height, cell_size, error_type, match_str):
    """Verify that invalid dimensions or non-integer types are rejected with clear errors."""
    with pytest.raises(error_type, match=match_str):
        Environment(width=width, height=height, cell_size=cell_size)


# =============================================================================
# 2. Boundary Checking Tests
# =============================================================================

def test_environment_grid_bounds():
    """Verify is_within_bounds correctly validates discrete grid indices."""
    env = Environment(width=800, height=600, cell_size=20)  # 30 rows, 40 cols
    assert env.is_within_bounds(0, 0) is True
    assert env.is_within_bounds(29, 39) is True
    assert env.is_within_bounds(0, 39) is True
    assert env.is_within_bounds(29, 0) is True

    # Out-of-bounds cases
    assert env.is_within_bounds(-1, 0) is False
    assert env.is_within_bounds(0, -1) is False
    assert env.is_within_bounds(30, 0) is False  # row == rows
    assert env.is_within_bounds(0, 40) is False  # col == cols
    assert env.is_within_bounds(100, 200) is False

    # Type validation
    assert env.is_within_bounds("0", 0) is False
    assert env.is_within_bounds(True, 0) is False


def test_environment_point_bounds_and_outer_edge_semantics():
    """Verify pixel boundaries enforce the half-open interval [0, width) x [0, height)."""
    env = Environment(width=800, height=600, cell_size=20)
    assert env.is_point_within_bounds(0.0, 0.0) is True
    assert env.is_point_within_bounds(799.9, 599.9) is True
    assert env.is_point_within_bounds(400.0, 300.0) is True

    # Exact boundary edges must be rejected as out of bounds
    assert env.is_point_within_bounds(800.0, 300.0) is False  # x == width
    assert env.is_point_within_bounds(400.0, 600.0) is False  # y == height
    assert env.is_point_within_bounds(800.0, 600.0) is False  # bottom-right corner

    # Negative coordinates
    assert env.is_point_within_bounds(-0.1, 10.0) is False
    assert env.is_point_within_bounds(10.0, -0.1) is False


# =============================================================================
# 3. Coordinate Conversion Tests
# =============================================================================

def test_environment_point_to_grid_conversion():
    """Verify conversion of continuous pixel coordinates to discrete grid cells."""
    env = Environment(width=800, height=600, cell_size=20)

    # Top-left cell
    assert env.point_to_grid(0.0, 0.0) == (0, 0)
    assert env.point_to_grid(19.9, 19.9) == (0, 0)

    # Specific cell: row 5, col 10 => x in [200, 220), y in [100, 120)
    assert env.point_to_grid(210.0, 110.0) == (5, 10)

    # Bottom-rightmost valid cell (row 29, col 39)
    assert env.point_to_grid(799.9, 599.9) == (29, 39)


def test_environment_point_to_grid_out_of_bounds_raises():
    """Verify that points outside world boundaries raise IndexError."""
    env = Environment(width=800, height=600, cell_size=20)
    with pytest.raises(IndexError, match="outside world bounds"):
        env.point_to_grid(800.0, 300.0)
    with pytest.raises(IndexError, match="outside world bounds"):
        env.point_to_grid(400.0, 600.0)
    with pytest.raises(IndexError, match="outside world bounds"):
        env.point_to_grid(-1.0, 100.0)


def test_environment_grid_to_point_conversion():
    """Verify conversion of discrete grid cells to cell center pixel positions."""
    env = Environment(width=800, height=600, cell_size=20)

    # (0, 0) center: x = 0*20 + 10 = 10.0, y = 0*20 + 10 = 10.0
    center_x, center_y = env.grid_to_point(0, 0)
    assert center_x == 10.0
    assert center_y == 10.0

    # (row 5, col 10) center: x = 10*20 + 10 = 210.0, y = 5*20 + 10 = 110.0
    center_x, center_y = env.grid_to_point(5, 10)
    assert center_x == 210.0
    assert center_y == 110.0


def test_environment_grid_to_point_out_of_bounds_raises():
    """Verify grid_to_point on out-of-bounds indices raises IndexError."""
    env = Environment(width=800, height=600, cell_size=20)
    with pytest.raises(IndexError, match="out of world bounds"):
        env.grid_to_point(30, 10)
    with pytest.raises(IndexError, match="out of world bounds"):
        env.grid_to_point(10, 40)
    with pytest.raises(IndexError, match="out of world bounds"):
        env.grid_to_point(-1, 0)


# =============================================================================
# 4. Obstacle Management & Query Tests
# =============================================================================

def test_environment_add_and_query_single_obstacle():
    """Verify adding static obstacles and querying occupancy in grid and pixel space."""
    env = Environment(width=800, height=600, cell_size=20)

    # Unoccupied by default
    assert env.is_occupied(5, 5) is False
    assert env.is_point_occupied(110.0, 110.0) is False

    # Add obstacle
    env.add_static_obstacle(5, 5)
    assert env.is_occupied(5, 5) is True
    # Pixel (105, 105) falls inside row 5, col 5
    assert env.is_point_occupied(105.0, 105.0) is True

    # Adjacent cells remain unoccupied
    assert env.is_occupied(5, 6) is False
    assert env.is_occupied(6, 5) is False


def test_environment_add_obstacle_out_of_bounds_raises():
    """Verify adding obstacle outside bounds raises IndexError."""
    env = Environment(width=800, height=600, cell_size=20)
    with pytest.raises(IndexError, match="out of world bounds"):
        env.add_static_obstacle(30, 0)
    with pytest.raises(IndexError, match="out of world bounds"):
        env.add_static_obstacle(-1, 5)


def test_environment_add_static_obstacle_rect():
    """Verify adding rectangular obstacles and checking all contained cells."""
    env = Environment(width=800, height=600, cell_size=20)
    # Add 3 rows by 4 cols starting at (row 10, col 15)
    env.add_static_obstacle_rect(top_row=10, left_col=15, num_rows=3, num_cols=4)

    # Check that all 12 cells in the rectangular block are occupied
    for r in range(10, 13):
        for c in range(15, 19):
            assert env.is_occupied(r, c) is True

    # Check boundaries of rectangle are not occupied
    assert env.is_occupied(9, 15) is False
    assert env.is_occupied(13, 15) is False
    assert env.is_occupied(10, 14) is False
    assert env.is_occupied(10, 19) is False


def test_environment_add_obstacle_rect_atomic_failure():
    """Verify adding an invalid rectangle does not partially pollute obstacle set."""
    env = Environment(width=800, height=600, cell_size=20)
    # Rectangle extends beyond rows=30 (rows 28..31)
    with pytest.raises(IndexError, match="exceeds world bounds"):
        env.add_static_obstacle_rect(top_row=28, left_col=0, num_rows=4, num_cols=2)

    # None of the cells should have been added
    assert len(env.static_obstacles) == 0


def test_environment_remove_and_clear_obstacles():
    """Verify removing individual obstacles and clearing entire environment."""
    env = Environment(width=800, height=600, cell_size=20)
    env.add_static_obstacle(2, 2)
    assert env.is_occupied(2, 2) is True

    assert env.remove_static_obstacle(2, 2) is True
    assert env.is_occupied(2, 2) is False
    assert env.remove_static_obstacle(2, 2) is False  # Already removed

    env.add_static_obstacle(1, 1)
    env.add_static_obstacle(2, 2)
    env.clear_obstacles()
    assert len(env.static_obstacles) == 0


# =============================================================================
# 5. Start and Goal Position Validation Tests
# =============================================================================

def test_environment_validate_start_goal_valid():
    """Verify validate_start_goal succeeds for valid, distinct, free locations."""
    env = Environment(width=800, height=600, cell_size=20)
    env.validate_start_goal(start=(0, 0), goal=(29, 39))


def test_environment_validate_start_goal_same_position_raises():
    """Verify validate_start_goal raises ValueError if start and goal are identical."""
    env = Environment(width=800, height=600, cell_size=20)
    with pytest.raises(ValueError, match="cannot be identical"):
        env.validate_start_goal(start=(5, 5), goal=(5, 5))


def test_environment_validate_start_goal_occupied_raises():
    """Verify validate_start_goal raises ValueError if either start or goal is in an obstacle."""
    env = Environment(width=800, height=600, cell_size=20)
    env.add_static_obstacle(5, 5)

    with pytest.raises(ValueError, match="Start.*is occupied by an obstacle"):
        env.validate_start_goal(start=(5, 5), goal=(10, 10))

    with pytest.raises(ValueError, match="Goal.*is occupied by an obstacle"):
        env.validate_start_goal(start=(0, 0), goal=(5, 5))


def test_environment_validate_start_goal_out_of_bounds_raises():
    """Verify validate_start_goal raises IndexError if position is out of bounds."""
    env = Environment(width=800, height=600, cell_size=20)
    with pytest.raises(IndexError, match="out of world bounds"):
        env.validate_start_goal(start=(-1, 0), goal=(10, 10))
    with pytest.raises(IndexError, match="out of world bounds"):
        env.validate_start_goal(start=(0, 0), goal=(30, 40))

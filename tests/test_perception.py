"""Unit tests for simulated range sensing and raycasting mechanics."""

import math
import pytest

from autonomous_nav.config import load_config
from autonomous_nav.environment import Environment
from autonomous_nav.perception import RangeSensor, RayObservation


# =============================================================================
# 1. Construction and Configuration Tests
# =============================================================================

def test_range_sensor_valid_construction():
    """Verify RangeSensor initializes with valid parameters and defaults."""
    sensor = RangeSensor(max_range=100.0, num_rays=32)
    assert sensor.max_range == 100.0
    assert sensor.num_rays == 32
    assert math.isclose(sensor.fov_rad, 2.0 * math.pi, rel_tol=1e-6)


def test_range_sensor_construction_from_config():
    """Verify RangeSensor.from_config creates sensor matching robot configuration."""
    cfg = load_config()
    sensor = RangeSensor.from_config(cfg, num_rays=24)
    assert sensor.max_range == cfg.robot.sensor_range
    assert sensor.num_rays == 24


@pytest.mark.parametrize(
    "max_range, num_rays, fov_rad, error_type, match_str",
    [
        (0.0, 16, None, ValueError, "must be positive"),
        (-50.0, 16, None, ValueError, "must be positive"),
        (float("nan"), 16, None, ValueError, "must be positive and finite"),
        (float("inf"), 16, None, ValueError, "must be positive and finite"),
        ("100", 16, None, TypeError, "max_range must be a numeric value"),
        (True, 16, None, TypeError, "max_range must be a numeric value"),
        (100.0, 0, None, ValueError, "num_rays must be positive"),
        (100.0, -5, None, ValueError, "num_rays must be positive"),
        (100.0, 16.5, None, TypeError, "num_rays must be an integer"),
        (100.0, True, None, TypeError, "num_rays must be an integer"),
        (100.0, 16, 0.0, ValueError, "fov_rad must be in range"),
        (100.0, 16, -1.0, ValueError, "fov_rad must be in range"),
        (100.0, 16, 7.0, ValueError, "fov_rad must be in range"),
        (100.0, 16, "pi", TypeError, "fov_rad must be a numeric value or None"),
    ],
)
def test_range_sensor_invalid_parameters_rejected(
    max_range, num_rays, fov_rad, error_type, match_str
):
    """Verify invalid constructor parameters raise appropriate errors."""
    with pytest.raises(error_type, match=match_str):
        RangeSensor(max_range=max_range, num_rays=num_rays, fov_rad=fov_rad)


# =============================================================================
# 2. Obstacle Detection and Distance Calculation Tests
# =============================================================================

def test_ray_encounters_obstacle_at_expected_distance():
    """Verify a horizontal ray encounters an obstacle at the exact geometric distance."""
    # 40 cols x 30 rows, cell size 20px
    env = Environment(width=800, height=600, cell_size=20)
    # Robot at center of cell (row 5, col 5) => (x=110.0, y=110.0)
    origin_x, origin_y = env.grid_to_point(5, 5)
    # Add obstacle at (row 5, col 8) => left boundary at x = 8 * 20 = 160.0
    env.add_static_obstacle(5, 8)

    sensor = RangeSensor(max_range=100.0)
    # Cast ray pointing right (angle = 0.0 rad)
    obs = sensor.cast_ray(origin_x, origin_y, angle_rad=0.0, environment=env)

    assert obs.hit is True
    # Distance from x=110.0 to obstacle boundary x=160.0 is 50.0 pixels
    assert math.isclose(obs.distance, 50.0, rel_tol=1e-5)
    assert obs.hit_cell == (5, 8)
    assert math.isclose(obs.hit_point[0], 160.0, rel_tol=1e-5)
    assert math.isclose(obs.hit_point[1], 110.0, rel_tol=1e-5)
    # Traversed free cells must be (5, 5), (5, 6), (5, 7)
    assert obs.traversed_cells == ((5, 5), (5, 6), (5, 7))


def test_ray_no_obstacle_within_range_returns_explicit_no_hit():
    """Verify that a ray with no obstacle within max_range returns an explicit no-hit."""
    env = Environment(width=800, height=600, cell_size=20)
    # Obstacle is at col 8 (distance 50px away)
    env.add_static_obstacle(5, 8)

    # Sensor max range is 30px (shorter than 50px distance to obstacle)
    sensor = RangeSensor(max_range=30.0)
    origin_x, origin_y = env.grid_to_point(5, 5)  # (110.0, 110.0)

    obs = sensor.cast_ray(origin_x, origin_y, angle_rad=0.0, environment=env)

    assert obs.hit is False
    assert math.isclose(obs.distance, 30.0, rel_tol=1e-5)
    assert obs.hit_point is None
    assert obs.hit_cell is None
    # Cells reached within 30px: starts at (5, 5) [10px to right boundary], enters (5, 6) [20px],
    # reaches total 30px at boundary of (5, 6) and (5, 7)
    assert (5, 5) in obs.traversed_cells
    assert (5, 8) not in obs.traversed_cells


def test_ray_reports_first_obstacle_not_behind():
    """Verify that a ray reports the first obstacle and does not see obstacles behind it."""
    env = Environment(width=800, height=600, cell_size=20)
    origin_x, origin_y = env.grid_to_point(5, 5)  # (110.0, 110.0)

    # First obstacle at col 7 (x=140), second obstacle at col 8 (x=160)
    env.add_static_obstacle(5, 7)
    env.add_static_obstacle(5, 8)

    sensor = RangeSensor(max_range=200.0)
    obs = sensor.cast_ray(origin_x, origin_y, angle_rad=0.0, environment=env)

    assert obs.hit is True
    # First obstacle is at x=140.0, distance = 140.0 - 110.0 = 30.0px
    assert math.isclose(obs.distance, 30.0, rel_tol=1e-5)
    assert obs.hit_cell == (5, 7)
    assert (5, 8) not in obs.traversed_cells
    assert obs.traversed_cells == ((5, 5), (5, 6))


def test_rays_pointing_in_four_cardinal_directions():
    """Verify rays cast horizontally and vertically in all 4 cardinal directions."""
    env = Environment(width=800, height=600, cell_size=20)
    origin_x, origin_y = env.grid_to_point(10, 10)  # (210.0, 210.0)

    # Surround origin with obstacles 2 cells away in all 4 directions
    env.add_static_obstacle(10, 12)  # Right: col 12 (left edge x = 240, dist = 30px)
    env.add_static_obstacle(12, 10)  # Down: row 12 (top edge y = 240, dist = 30px)
    env.add_static_obstacle(10, 8)   # Left: col 8 (right edge x = 180, dist = 30px)
    env.add_static_obstacle(8, 10)   # Up: row 8 (bottom edge y = 180, dist = 30px)

    sensor = RangeSensor(max_range=100.0)

    # Right (0 rad)
    obs_r = sensor.cast_ray(origin_x, origin_y, 0.0, env)
    assert obs_r.hit is True
    assert math.isclose(obs_r.distance, 30.0, rel_tol=1e-5)
    assert obs_r.hit_cell == (10, 12)

    # Down (pi/2 rad)
    obs_d = sensor.cast_ray(origin_x, origin_y, math.pi / 2.0, env)
    assert obs_d.hit is True
    assert math.isclose(obs_d.distance, 30.0, rel_tol=1e-5)
    assert obs_d.hit_cell == (12, 10)

    # Left (pi rad)
    obs_l = sensor.cast_ray(origin_x, origin_y, math.pi, env)
    assert obs_l.hit is True
    assert math.isclose(obs_l.distance, 30.0, rel_tol=1e-5)
    assert obs_l.hit_cell == (10, 8)

    # Up (-pi/2 rad)
    obs_u = sensor.cast_ray(origin_x, origin_y, -math.pi / 2.0, env)
    assert obs_u.hit is True
    assert math.isclose(obs_u.distance, 30.0, rel_tol=1e-5)
    assert obs_u.hit_cell == (8, 10)


def test_ray_diagonal_direction_and_distance():
    """Verify diagonal ray distance matches Euclidean theorem and hits correct corner."""
    env = Environment(width=800, height=600, cell_size=20)
    origin_x, origin_y = env.grid_to_point(0, 0)  # (10.0, 10.0)
    # Obstacle at (1, 1) => corner at (20.0, 20.0)
    env.add_static_obstacle(1, 1)

    sensor = RangeSensor(max_range=100.0)
    # 45 degrees diagonal ray: angle = pi / 4
    obs = sensor.cast_ray(origin_x, origin_y, angle_rad=math.pi / 4.0, environment=env)

    assert obs.hit is True
    # Distance from (10, 10) to corner (20, 20): sqrt(10^2 + 10^2) = 10 * sqrt(2)
    expected_dist = 10.0 * math.sqrt(2.0)
    assert math.isclose(obs.distance, expected_dist, rel_tol=1e-5)
    assert obs.hit_cell == (1, 1)
    assert math.isclose(obs.hit_point[0], 20.0, rel_tol=1e-5)
    assert math.isclose(obs.hit_point[1], 20.0, rel_tol=1e-5)
    assert obs.traversed_cells == ((0, 0),)


# =============================================================================
# 3. Boundary & Error Handling Tests
# =============================================================================

def test_ray_starts_inside_occupied_cell():
    """Verify policy: A ray starting inside an occupied cell terminates immediately at distance 0.0."""
    env = Environment(width=800, height=600, cell_size=20)
    origin_x, origin_y = env.grid_to_point(3, 3)  # (70.0, 70.0)
    env.add_static_obstacle(3, 3)

    sensor = RangeSensor(max_range=50.0)
    obs = sensor.cast_ray(origin_x, origin_y, angle_rad=0.0, environment=env)

    assert obs.hit is True
    assert obs.distance == 0.0
    assert obs.hit_point == (70.0, 70.0)
    assert obs.hit_cell == (3, 3)
    assert obs.traversed_cells == ()


def test_ray_starting_near_world_boundary_exits_gracefully():
    """Verify ray shooting out of the world terminates as a no-hit without error."""
    env = Environment(width=800, height=600, cell_size=20)
    # Position in bottom-right cell (29, 39) => (790.0, 590.0)
    origin_x, origin_y = env.grid_to_point(29, 39)

    sensor = RangeSensor(max_range=100.0)
    # Shoot right towards world exit (x=800 is only 10px away)
    obs = sensor.cast_ray(origin_x, origin_y, angle_rad=0.0, environment=env)

    assert obs.hit is False
    assert math.isclose(obs.distance, 100.0, rel_tol=1e-5)
    assert obs.hit_point is None
    assert obs.hit_cell is None
    assert obs.traversed_cells == ((29, 39),)


def test_ray_out_of_bounds_origin_raises_index_error():
    """Verify casting a ray from outside world bounds raises IndexError."""
    env = Environment(width=800, height=600, cell_size=20)
    sensor = RangeSensor(max_range=50.0)

    with pytest.raises(IndexError, match="outside world bounds"):
        sensor.cast_ray(800.0, 300.0, angle_rad=0.0, environment=env)
    with pytest.raises(IndexError, match="outside world bounds"):
        sensor.cast_ray(-1.0, 100.0, angle_rad=0.0, environment=env)
    with pytest.raises(IndexError, match="outside world bounds"):
        sensor.cast_ray(100.0, 600.0, angle_rad=0.0, environment=env)


def test_ray_invalid_origin_or_angle_types_raise():
    """Verify non-numeric origins, non-numeric angles, or non-finite angles raise errors."""
    env = Environment(width=800, height=600, cell_size=20)
    sensor = RangeSensor(max_range=50.0)

    with pytest.raises(TypeError, match="origin_x must be numeric"):
        sensor.cast_ray("100", 100.0, 0.0, env)
    with pytest.raises(TypeError, match="origin_y must be numeric"):
        sensor.cast_ray(100.0, True, 0.0, env)
    with pytest.raises(TypeError, match="angle_rad must be numeric"):
        sensor.cast_ray(100.0, 100.0, "0.0", env)
    with pytest.raises(ValueError, match="angle_rad must be finite"):
        sensor.cast_ray(100.0, 100.0, float("nan"), env)
    with pytest.raises(TypeError, match="environment must be an Environment instance"):
        sensor.cast_ray(100.0, 100.0, 0.0, "not_an_environment")


# =============================================================================
# 4. Multi-Ray Scan Tests
# =============================================================================

def test_sensor_scan_generates_configured_number_of_rays():
    """Verify scan() returns exactly num_rays observations covering omnidirectional circle."""
    env = Environment(width=800, height=600, cell_size=20)
    origin_x, origin_y = env.grid_to_point(10, 10)

    sensor = RangeSensor(max_range=80.0, num_rays=8)
    observations = sensor.scan(origin_x, origin_y, environment=env, heading_rad=0.0)

    assert len(observations) == 8
    for i, obs in enumerate(observations):
        expected_angle = i * (2.0 * math.pi / 8.0)
        assert math.isclose(obs.angle_rad, expected_angle, rel_tol=1e-5)


def test_sensor_scan_directional_fov():
    """Verify sector scan respects fov_rad centered around heading."""
    env = Environment(width=800, height=600, cell_size=20)
    origin_x, origin_y = env.grid_to_point(10, 10)

    # 90-degree forward cone with 3 rays: [-45 deg, 0 deg, +45 deg]
    fov = math.pi / 2.0
    sensor = RangeSensor(max_range=80.0, num_rays=3, fov_rad=fov)
    observations = sensor.scan(origin_x, origin_y, environment=env, heading_rad=0.0)

    assert len(observations) == 3
    assert math.isclose(observations[0].angle_rad, -math.pi / 4.0, rel_tol=1e-5)
    assert math.isclose(observations[1].angle_rad, 0.0, rel_tol=1e-5)
    assert math.isclose(observations[2].angle_rad, math.pi / 4.0, rel_tol=1e-5)

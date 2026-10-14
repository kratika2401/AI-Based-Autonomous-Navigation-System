"""Unit tests for obstacle detection and observation-to-map integration."""

import pytest

from autonomous_nav.config import load_config
from autonomous_nav.mapping import CellState, OccupancyGrid
from autonomous_nav.obstacle_detection import ObstacleDetector
from autonomous_nav.perception import RayObservation


# =============================================================================
# 1. Construction and Configuration Tests
# =============================================================================

def test_detector_valid_construction():
    """Verify ObstacleDetector initializes with valid cell_size."""
    det = ObstacleDetector(cell_size=20)
    assert det.cell_size == 20


def test_detector_construction_from_config():
    """Verify ObstacleDetector.from_config creates detector matching config cell_size."""
    cfg = load_config()
    det = ObstacleDetector.from_config(cfg)
    assert det.cell_size == cfg.grid.cell_size


@pytest.mark.parametrize(
    "cell_size, error_type, match_str",
    [
        (0, ValueError, "must be positive"),
        (-20, ValueError, "must be positive"),
        (20.0, TypeError, "must be an integer"),
        ("20", TypeError, "must be an integer"),
        (True, TypeError, "must be an integer"),
    ],
)
def test_detector_invalid_cell_size_rejected(cell_size, error_type, match_str):
    """Verify non-positive or non-integer cell sizes raise appropriate errors."""
    with pytest.raises(error_type, match=match_str):
        ObstacleDetector(cell_size=cell_size)


# =============================================================================
# 2. Observation-to-Map Integration Tests
# =============================================================================

def test_traversed_cells_marked_free_and_hit_marked_occupied():
    """Verify traversed cells are marked FREE and the hit cell is marked OCCUPIED."""
    grid = OccupancyGrid(rows=10, cols=10, cell_size=20)  # All UNKNOWN (-1)
    detector = ObstacleDetector(cell_size=20)

    # Ray starts at (2, 2), traverses (2, 3) and (2, 4), and hits obstacle at (2, 5)
    obs = RayObservation(
        angle_rad=0.0,
        distance=60.0,
        hit=True,
        hit_point=(100.0, 50.0),
        hit_cell=(2, 5),
        traversed_cells=((2, 2), (2, 3), (2, 4)),
    )

    stats = detector.integrate_observations(obs, grid)

    assert stats["freed_cells"] == 3
    assert stats["occupied_cells"] == 1

    # Traversed cells must be FREE
    assert grid.get_cell(2, 2) == CellState.FREE
    assert grid.get_cell(2, 3) == CellState.FREE
    assert grid.get_cell(2, 4) == CellState.FREE

    # Hit cell must be OCCUPIED
    assert grid.get_cell(2, 5) == CellState.OCCUPIED


def test_cells_beyond_first_obstacle_remain_unknown():
    """Verify cells behind a detected obstacle remain UNKNOWN (no x-ray vision)."""
    grid = OccupancyGrid(rows=10, cols=10, cell_size=20)
    detector = ObstacleDetector(cell_size=20)

    obs = RayObservation(
        angle_rad=0.0,
        distance=40.0,
        hit=True,
        hit_point=(60.0, 10.0),
        hit_cell=(0, 2),
        traversed_cells=((0, 0), (0, 1)),
    )

    detector.integrate_observations(obs, grid)

    # Hit cell is (0, 2)
    assert grid.get_cell(0, 2) == CellState.OCCUPIED
    # Cells behind obstacle (0, 3), (0, 4), etc. must remain UNKNOWN
    assert grid.get_cell(0, 3) == CellState.UNKNOWN
    assert grid.get_cell(0, 4) == CellState.UNKNOWN
    assert grid.get_cell(0, 5) == CellState.UNKNOWN


def test_no_hit_ray_does_not_mark_endpoint_occupied():
    """Verify a no-hit ray marks traversed path FREE, but does not mark any cell OCCUPIED."""
    grid = OccupancyGrid(rows=10, cols=10, cell_size=20)
    detector = ObstacleDetector(cell_size=20)

    # No-hit ray: traveled through (1, 1), (1, 2), (1, 3)
    obs = RayObservation(
        angle_rad=0.0,
        distance=60.0,
        hit=False,
        hit_point=None,
        hit_cell=None,
        traversed_cells=((1, 1), (1, 2), (1, 3)),
    )

    stats = detector.integrate_observations(obs, grid)

    assert stats["freed_cells"] == 3
    assert stats["occupied_cells"] == 0

    assert grid.get_cell(1, 1) == CellState.FREE
    assert grid.get_cell(1, 2) == CellState.FREE
    assert grid.get_cell(1, 3) == CellState.FREE
    # Endpoint cell (1, 4) was outside range, must remain UNKNOWN
    assert grid.get_cell(1, 4) == CellState.UNKNOWN


def test_cells_outside_observed_ray_paths_remain_unknown():
    """Verify unobserved regions of the grid remain UNKNOWN after integration."""
    grid = OccupancyGrid(rows=5, cols=5, cell_size=20)
    detector = ObstacleDetector(cell_size=20)

    obs = RayObservation(
        angle_rad=0.0,
        distance=20.0,
        hit=True,
        hit_point=(20.0, 10.0),
        hit_cell=(0, 1),
        traversed_cells=((0, 0),),
    )

    detector.integrate_observations(obs, grid)

    assert grid.get_cell(0, 0) == CellState.FREE
    assert grid.get_cell(0, 1) == CellState.OCCUPIED

    # All other rows must be entirely UNKNOWN
    for r in range(1, 5):
        for c in range(5):
            assert grid.get_cell(r, c) == CellState.UNKNOWN


def test_multiple_observations_combine_without_corrupting_obstacles():
    """Verify multiple observations combine, and a free ray never clears a known obstacle."""
    grid = OccupancyGrid(rows=5, cols=5, cell_size=20)
    detector = ObstacleDetector(cell_size=20)

    # Observation 1: detects obstacle at (2, 2)
    obs1 = RayObservation(
        angle_rad=0.0,
        distance=40.0,
        hit=True,
        hit_point=(40.0, 50.0),
        hit_cell=(2, 2),
        traversed_cells=((2, 0), (2, 1)),
    )

    # Observation 2: an erroneous or grazing ray listing (2, 2) in traversed cells
    obs2 = RayObservation(
        angle_rad=1.57,
        distance=60.0,
        hit=False,
        hit_point=None,
        hit_cell=None,
        traversed_cells=((0, 2), (1, 2), (2, 2), (3, 2)),
    )

    detector.integrate_observations(obs1, grid)
    assert grid.get_cell(2, 2) == CellState.OCCUPIED

    detector.integrate_observations(obs2, grid)
    # Crucial: cell (2, 2) must remain OCCUPIED and not be overwritten as FREE
    assert grid.get_cell(2, 2) == CellState.OCCUPIED
    assert grid.get_cell(0, 2) == CellState.FREE
    assert grid.get_cell(1, 2) == CellState.FREE
    assert grid.get_cell(3, 2) == CellState.FREE


def test_hit_cell_not_overwritten_by_same_ray():
    """Verify that even within a single observation, the hit cell is firmly OCCUPIED."""
    grid = OccupancyGrid(rows=5, cols=5, cell_size=20)
    detector = ObstacleDetector(cell_size=20)

    # Observation hitting (1, 1)
    obs = RayObservation(
        angle_rad=0.0,
        distance=20.0,
        hit=True,
        hit_point=(20.0, 30.0),
        hit_cell=(1, 1),
        traversed_cells=((1, 0),),
    )

    detector.integrate_observations(obs, grid)
    assert grid.get_cell(1, 1) == CellState.OCCUPIED


# =============================================================================
# 3. Validation and Error Handling Tests
# =============================================================================

def test_malformed_observations_rejected():
    """Verify malformed observations raise appropriate ValueError or TypeError."""
    grid = OccupancyGrid(rows=5, cols=5, cell_size=20)
    detector = ObstacleDetector(cell_size=20)

    # Inconsistent: hit is True but hit_cell is None
    malformed1 = RayObservation(
        angle_rad=0.0,
        distance=20.0,
        hit=True,
        hit_point=(20.0, 20.0),
        hit_cell=None,
        traversed_cells=(),
    )
    with pytest.raises(ValueError, match="hit is True but hit_cell is None"):
        detector.integrate_observations(malformed1, grid)

    # Inconsistent: hit is False but hit_cell is provided
    malformed2 = RayObservation(
        angle_rad=0.0,
        distance=20.0,
        hit=False,
        hit_point=None,
        hit_cell=(1, 1),
        traversed_cells=(),
    )
    with pytest.raises(ValueError, match="hit is False but hit_cell is not None"):
        detector.integrate_observations(malformed2, grid)

    # Negative distance
    malformed3 = RayObservation(
        angle_rad=0.0,
        distance=-5.0,
        hit=False,
        hit_point=None,
        hit_cell=None,
        traversed_cells=(),
    )
    with pytest.raises(ValueError, match="cannot be negative"):
        detector.integrate_observations(malformed3, grid)

    # Non-RayObservation item
    with pytest.raises(TypeError, match="Observation item must be a RayObservation instance"):
        detector.integrate_observations(["not_an_observation"], grid)


def test_incompatible_grid_dimensions_rejected():
    """Verify grid with incompatible cell_size raises ValueError."""
    detector = ObstacleDetector(cell_size=20)
    incompatible_grid = OccupancyGrid(rows=10, cols=10, cell_size=25)  # cell_size 25 != 20

    obs = RayObservation(
        angle_rad=0.0,
        distance=20.0,
        hit=False,
        hit_point=None,
        hit_cell=None,
        traversed_cells=(),
    )

    with pytest.raises(ValueError, match="Incompatible grid resolution"):
        detector.integrate_observations(obs, incompatible_grid)


def test_legacy_methods_extract_occupied_cells():
    """Verify to_grid_cell and extract_occupied_cells return (row, col) coordinates."""
    detector = ObstacleDetector(cell_size=20)

    # Continuous points: (x=25, y=45) => col = 25//20 = 1, row = 45//20 = 2 => (2, 1)
    assert detector.to_grid_cell(25.0, 45.0) == (2, 1)

    points = [(10.0, 10.0), (30.0, 50.0)]
    cells = detector.extract_occupied_cells(points)
    assert cells == [(0, 0), (2, 1)]

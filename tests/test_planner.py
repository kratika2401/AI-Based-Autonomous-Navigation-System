"""Unit tests for deterministic A* and Dijkstra path planning algorithms."""

import pytest

from autonomous_nav.config import load_config
from autonomous_nav.mapping import CellState, OccupancyGrid
from autonomous_nav.planner import ORTHOGONAL_DIRECTIONS, PathPlanner, PlanResult


# =============================================================================
# Helper Utilities
# =============================================================================

def create_free_grid(rows: int = 10, cols: int = 10, cell_size: int = 20) -> OccupancyGrid:
    """Helper creating an OccupancyGrid where all cells are initialized to FREE."""
    return OccupancyGrid(rows=rows, cols=cols, cell_size=cell_size, initial_state=CellState.FREE)


def verify_path_properties(path, start, goal, grid, expected_cost=None):
    """Verifies fundamental path invariants: endpoints, 4-connectivity, and validity."""
    assert path is not None
    assert len(path) > 0
    assert path[0] == start
    assert path[-1] == goal

    for i, cell in enumerate(path):
        r, c = cell
        assert grid.is_in_bounds(r, c)
        assert grid.get_cell(r, c) == CellState.FREE

        if i > 0:
            prev_r, prev_c = path[i - 1]
            step_dist = abs(r - prev_r) + abs(c - prev_c)
            # Consecutive cells must be strictly 4-connected (distance of 1)
            assert step_dist == 1

    if expected_cost is not None:
        assert len(path) - 1 == expected_cost


# =============================================================================
# 1. Construction and Configuration Tests
# =============================================================================

def test_planner_interface_initialization():
    """Verify PathPlanner initializes with supported parameters and defaults."""
    planner = PathPlanner(algorithm="astar", heuristic="manhattan", allow_diagonal=True)
    assert planner.algorithm == "astar"
    assert planner.heuristic == "manhattan"
    assert planner.allow_diagonal is True
    assert planner.allow_unknown is False


def test_planner_construction_from_config():
    """Verify PathPlanner.from_config correctly creates planner from SimulationConfig."""
    cfg = load_config()
    planner = PathPlanner.from_config(cfg)
    assert planner.algorithm == cfg.planner.algorithm
    assert planner.heuristic == cfg.planner.heuristic


@pytest.mark.parametrize(
    "algorithm, heuristic, error_type, match_str",
    [
        ("genetic_algorithm", "manhattan", ValueError, "Unsupported planner algorithm"),
        ("bfs", "manhattan", ValueError, "Unsupported planner algorithm"),
        ("astar", "chebyshev", ValueError, "Unsupported heuristic"),
    ],
)
def test_planner_invalid_config_raises(algorithm, heuristic, error_type, match_str):
    """Verify unsupported algorithm or heuristic raises ValueError."""
    with pytest.raises(error_type, match=match_str):
        PathPlanner(algorithm=algorithm, heuristic=heuristic)


# =============================================================================
# 2. Basic Pathfinding & Optimality Tests (A* and Dijkstra)
# =============================================================================

@pytest.mark.parametrize("algorithm", ["astar", "dijkstra"])
def test_pathfinding_in_empty_grid(algorithm):
    """Verify A* and Dijkstra find the exact optimal Manhattan path in an open grid."""
    grid = create_free_grid(rows=10, cols=10)
    planner = PathPlanner(algorithm=algorithm)

    start = (1, 1)
    goal = (7, 8)
    expected_cost = abs(7 - 1) + abs(8 - 1)  # 6 + 7 = 13 steps

    res = planner.plan_with_stats(start, goal, grid)
    assert res.success is True
    assert res.cost == expected_cost
    verify_path_properties(res.path, start, goal, grid, expected_cost=expected_cost)


def test_astar_and_dijkstra_return_identical_optimal_cost():
    """Verify A* and Dijkstra yield the exact same optimal cost on the same map."""
    grid = create_free_grid(rows=15, cols=15)
    # Add an obstacle barrier forcing a detour
    for r in range(2, 10):
        grid.mark_occupied(r, 7)

    start = (5, 2)
    goal = (5, 12)

    planner_astar = PathPlanner(algorithm="astar")
    planner_dijkstra = PathPlanner(algorithm="dijkstra")

    res_a = planner_astar.plan_with_stats(start, goal, grid)
    res_d = planner_dijkstra.plan_with_stats(start, goal, grid)

    assert res_a.success is True
    assert res_d.success is True
    assert res_a.cost == res_d.cost  # Both must achieve the exact optimal cost
    verify_path_properties(res_a.path, start, goal, grid, expected_cost=res_a.cost)
    verify_path_properties(res_d.path, start, goal, grid, expected_cost=res_d.cost)


def test_astar_expands_fewer_or_equal_nodes_than_dijkstra():
    """Verify goal-directed heuristic allows A* to expand fewer nodes than Dijkstra."""
    grid = create_free_grid(rows=20, cols=20)
    start = (2, 2)
    goal = (18, 18)

    planner = PathPlanner()
    res_a = planner.plan_with_stats(start, goal, grid, algorithm="astar")
    res_d = planner.plan_with_stats(start, goal, grid, algorithm="dijkstra")

    assert res_a.success is True
    assert res_d.success is True
    assert res_a.cost == res_d.cost
    # A* directed search must expand significantly fewer nodes than blind Dijkstra
    assert res_a.nodes_expanded <= res_d.nodes_expanded


def test_astar_navigates_around_concave_wall():
    """Verify A* navigates around a U-shaped obstacle barrier without collision."""
    grid = create_free_grid(rows=10, cols=10)
    # Build U-shaped wall
    for c in range(2, 8):
        grid.mark_occupied(3, c)
    grid.mark_occupied(4, 2)
    grid.mark_occupied(4, 7)

    start = (4, 4)  # Inside the cup
    goal = (1, 4)   # Outside the cup

    planner = PathPlanner(algorithm="astar")
    path = planner.plan(start, goal, grid)

    verify_path_properties(path, start, goal, grid)


# =============================================================================
# 3. Unreachable Goals & Blocked Paths
# =============================================================================

def test_unreachable_goal_returns_none_and_failure():
    """Verify that a completely enclosed goal returns None path and failure status."""
    grid = create_free_grid(rows=8, cols=8)
    goal = (3, 3)
    # Fully enclose goal cell with obstacles
    grid.mark_occupied(2, 3)
    grid.mark_occupied(4, 3)
    grid.mark_occupied(3, 2)
    grid.mark_occupied(3, 4)

    start = (0, 0)
    planner = PathPlanner(algorithm="astar")
    res = planner.plan_with_stats(start, goal, grid)

    assert res.success is False
    assert res.path is None
    assert res.cost == float("inf")
    assert res.nodes_expanded > 0

    # plan() convenience wrapper must also return None
    assert planner.plan(start, goal, grid) is None


def test_wall_dividing_grid_returns_unreachable():
    """Verify search returns None when a continuous vertical barrier divides the grid."""
    grid = create_free_grid(rows=6, cols=6)
    # Impassable barrier down column 3
    for r in range(6):
        grid.mark_occupied(r, 3)

    planner = PathPlanner(algorithm="dijkstra")
    res = planner.plan_with_stats(start=(2, 1), goal=(2, 4), grid=grid)

    assert res.success is False
    assert res.path is None


# =============================================================================
# 4. Occupancy Policy & Unknown Cells
# =============================================================================

def test_unknown_cells_are_blocked_under_default_cautious_policy():
    """Verify that by default, UNKNOWN cells cannot be traversed."""
    grid = OccupancyGrid(rows=5, cols=5, cell_size=20)  # Starts all UNKNOWN (-1)
    # Only carve a specific free path
    grid.mark_free(0, 0)
    grid.mark_free(0, 1)
    grid.mark_free(0, 2)
    # (0, 3) is UNKNOWN

    planner = PathPlanner(allow_unknown=False)
    # Trying to reach (0, 4) through UNKNOWN (0, 3)
    grid.mark_free(0, 4)
    res = planner.plan_with_stats(start=(0, 0), goal=(0, 4), grid=grid)

    # Path is blocked because (0, 3) is UNKNOWN
    assert res.success is False
    assert res.path is None


def test_unknown_cells_traversable_when_allow_unknown_is_true():
    """Verify UNKNOWN cells can be traversed when allow_unknown=True."""
    grid = OccupancyGrid(rows=5, cols=5, cell_size=20)  # Starts all UNKNOWN
    planner = PathPlanner(allow_unknown=True)

    res = planner.plan_with_stats(start=(0, 0), goal=(4, 4), grid=grid)
    assert res.success is True
    assert res.cost == 8.0  # (0,0) to (4,4) is 4 + 4 = 8 steps


# =============================================================================
# 5. Endpoint Validation & Edge Cases
# =============================================================================

def test_start_equals_goal_returns_single_node_path():
    """Verify when start equals goal, a 1-node path of zero cost is returned."""
    grid = create_free_grid(rows=5, cols=5)
    planner = PathPlanner()

    res = planner.plan_with_stats(start=(2, 2), goal=(2, 2), grid=grid)
    assert res.success is True
    assert res.path == [(2, 2)]
    assert res.cost == 0.0


def test_occupied_start_or_goal_raises_value_error():
    """Verify blocked start or goal cells raise ValueError."""
    grid = create_free_grid(rows=5, cols=5)
    grid.mark_occupied(0, 0)
    grid.mark_occupied(4, 4)

    planner = PathPlanner()
    with pytest.raises(ValueError, match="Start coordinate.*is blocked"):
        planner.plan(start=(0, 0), goal=(2, 2), grid=grid)

    with pytest.raises(ValueError, match="Goal coordinate.*is blocked"):
        planner.plan(start=(2, 2), goal=(4, 4), grid=grid)


def test_out_of_bounds_and_negative_indices_raise_index_error():
    """Verify out-of-bounds and negative indices raise IndexError (no Python negative wrap)."""
    grid = create_free_grid(rows=5, cols=5)
    planner = PathPlanner()

    # Negative coordinates
    with pytest.raises(IndexError, match="outside grid boundaries"):
        planner.plan(start=(-1, 0), goal=(2, 2), grid=grid)
    with pytest.raises(IndexError, match="outside grid boundaries"):
        planner.plan(start=(0, 0), goal=(2, -1), grid=grid)

    # Exceeding bounds
    with pytest.raises(IndexError, match="outside grid boundaries"):
        planner.plan(start=(5, 0), goal=(2, 2), grid=grid)
    with pytest.raises(IndexError, match="outside grid boundaries"):
        planner.plan(start=(0, 0), goal=(2, 5), grid=grid)


def test_deterministic_tie_breaking_repeatability():
    """Verify repeated runs on identical maps yield identical paths and expansion counts."""
    grid = create_free_grid(rows=12, cols=12)
    # Add symmetric obstacles
    grid.mark_occupied(4, 4)
    grid.mark_occupied(4, 5)

    planner = PathPlanner(algorithm="astar")
    res1 = planner.plan_with_stats(start=(1, 1), goal=(8, 8), grid=grid)
    res2 = planner.plan_with_stats(start=(1, 1), goal=(8, 8), grid=grid)

    assert res1.path == res2.path
    assert res1.nodes_expanded == res2.nodes_expanded
    assert res1.nodes_discovered == res2.nodes_discovered
    assert res1.cost == res2.cost


def test_search_statistics_internal_consistency():
    """Verify search telemetry metrics are internally consistent."""
    grid = create_free_grid(rows=8, cols=8)
    planner = PathPlanner(algorithm="astar")
    res = planner.plan_with_stats(start=(0, 0), goal=(5, 5), grid=grid)

    assert res.success is True
    assert res.algorithm == "astar"
    assert res.cost == len(res.path) - 1
    assert res.nodes_expanded > 0
    assert res.nodes_discovered >= res.nodes_expanded

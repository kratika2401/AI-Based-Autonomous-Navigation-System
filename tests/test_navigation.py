"""Unit tests for NavigationManager state machine (Phase 5B + 5C).

Test grid convention
--------------------
Unless stated otherwise, tests use a small 5x5 FREE occupancy grid with
cell_size=10.  Pixel centers follow:
  (row, col) → (col*10 + 5, row*10 + 5)

Examples:
  (0,0) → (5.0, 5.0)
  (0,1) → (15.0, 5.0)
  (1,0) → (5.0, 15.0)
  (4,4) → (45.0, 45.0)
"""

from __future__ import annotations

import math
from typing import Tuple

import pytest

from autonomous_nav.controller import MotionController
from autonomous_nav.mapping import CellState, OccupancyGrid
from autonomous_nav.navigation import NavigationManager, NavigationState
from autonomous_nav.planner import PathPlanner


# =============================================================================
# Helpers / fixtures
# =============================================================================

CELL_SIZE = 10
ROWS = 5
COLS = 5


def make_free_grid(rows: int = ROWS, cols: int = COLS, cell_size: int = CELL_SIZE) -> OccupancyGrid:
    """Returns a fully-FREE occupancy grid."""
    grid = OccupancyGrid(rows=rows, cols=cols, cell_size=cell_size)
    for r in range(rows):
        for c in range(cols):
            grid.mark_free(r, c)
    return grid


def make_manager(
    max_speed: float = 100.0,
    arrival_tolerance: float = 3.0,
    algorithm: str = "astar",
    grid: OccupancyGrid | None = None,
    allow_unknown: bool = False,
) -> Tuple[NavigationManager, OccupancyGrid]:
    """Creates a NavigationManager with a fast controller for deterministic tests."""
    if grid is None:
        grid = make_free_grid()
    planner = PathPlanner(algorithm=algorithm)
    controller = MotionController(max_speed=max_speed, arrival_tolerance=arrival_tolerance)
    return NavigationManager(planner=planner, controller=controller, grid=grid,
                             allow_unknown=allow_unknown), grid


def pixel_center(row: int, col: int, cell_size: int = CELL_SIZE) -> Tuple[float, float]:
    """Expected pixel center for a grid cell."""
    return (col * cell_size + cell_size / 2.0, row * cell_size + cell_size / 2.0)


def run_to_completion(
    nav: NavigationManager, start_pixel: Tuple, max_steps: int = 5000
) -> Tuple:
    """Drives nav.update() until non-NAVIGATING, returns final pos."""
    pos = start_pixel
    for _ in range(max_steps):
        if nav.state != NavigationState.NAVIGATING:
            break
        result = nav.update(pos)
        pos = result.new_position
    return pos


# =============================================================================
# 1. Initial State
# =============================================================================

class TestInitialState:
    def test_initial_state_is_idle(self):
        nav, _ = make_manager()
        assert nav.state == NavigationState.IDLE

    def test_initial_path_is_empty(self):
        nav, _ = make_manager()
        assert nav.active_path == []

    def test_initial_waypoint_index_is_zero(self):
        nav, _ = make_manager()
        assert nav.current_waypoint_index == 0

    def test_initial_target_is_none(self):
        nav, _ = make_manager()
        assert nav.get_current_target() is None

    def test_initial_target_pixel_is_none(self):
        nav, _ = make_manager()
        assert nav.get_current_target_pixel() is None

    def test_initial_replan_count_is_zero(self):
        nav, _ = make_manager()
        assert nav.replan_count == 0


# =============================================================================
# 2. start_mission() — start equals goal (Phase 5C fix)
# =============================================================================

class TestStartEqualsGoal:
    def test_start_equals_goal_sets_reached(self):
        nav, _ = make_manager()
        nav.start_mission((2, 2), (2, 2), robot_pixel_pos=pixel_center(2, 2))
        assert nav.state == NavigationState.REACHED

    def test_start_equals_goal_single_cell_path(self):
        nav, _ = make_manager()
        nav.start_mission((2, 2), (2, 2), robot_pixel_pos=pixel_center(2, 2))
        assert nav.active_path == [(2, 2)]

    def test_start_equals_goal_no_movement_on_update(self):
        nav, _ = make_manager()
        pos = pixel_center(2, 2)
        nav.start_mission((2, 2), (2, 2), robot_pixel_pos=pos)
        result = nav.update(pos)
        assert result.displacement == 0.0
        assert nav.state == NavigationState.REACHED

    def test_start_equals_goal_goal_recorded(self):
        nav, _ = make_manager()
        nav.start_mission((1, 1), (1, 1), robot_pixel_pos=pixel_center(1, 1))
        assert nav._goal_grid == (1, 1)

    def test_start_equals_goal_replan_count_zero(self):
        nav, _ = make_manager()
        nav.start_mission((0, 0), (0, 0), robot_pixel_pos=pixel_center(0, 0))
        assert nav.replan_count == 0

    def test_start_equals_goal_at_corner(self):
        nav, _ = make_manager()
        nav.start_mission((4, 4), (4, 4), robot_pixel_pos=pixel_center(4, 4))
        assert nav.state == NavigationState.REACHED


# =============================================================================
# 3. start_mission() — normal valid mission
# =============================================================================

class TestStartMission:
    def test_valid_mission_reaches_navigating(self):
        nav, _ = make_manager()
        nav.start_mission((0, 0), (4, 4), robot_pixel_pos=pixel_center(0, 0))
        assert nav.state == NavigationState.NAVIGATING

    def test_valid_mission_sets_active_path(self):
        nav, _ = make_manager()
        nav.start_mission((0, 0), (2, 0), robot_pixel_pos=pixel_center(0, 0))
        assert len(nav.active_path) > 0

    def test_valid_mission_path_starts_at_start(self):
        nav, _ = make_manager()
        nav.start_mission((0, 0), (3, 0), robot_pixel_pos=pixel_center(0, 0))
        assert nav.active_path[0] == (0, 0)

    def test_valid_mission_path_ends_at_goal(self):
        nav, _ = make_manager()
        nav.start_mission((0, 0), (3, 0), robot_pixel_pos=pixel_center(0, 0))
        assert nav.active_path[-1] == (3, 0)

    def test_waypoint_index_skips_start_cell(self):
        nav, _ = make_manager()
        nav.start_mission((0, 0), (3, 0), robot_pixel_pos=pixel_center(0, 0))
        assert nav.current_waypoint_index == 1

    def test_current_target_is_second_waypoint(self):
        nav, _ = make_manager()
        nav.start_mission((0, 0), (3, 0), robot_pixel_pos=pixel_center(0, 0))
        assert nav.get_current_target() == nav.active_path[1]

    def test_unreachable_goal_sets_failed(self):
        grid = make_free_grid()
        for r, c in [(1, 2), (3, 2), (2, 1), (2, 3)]:
            grid.mark_occupied(r, c)
        planner = PathPlanner()
        controller = MotionController(max_speed=10.0)
        nav = NavigationManager(planner=planner, controller=controller, grid=grid)
        nav.start_mission((0, 0), (2, 2), robot_pixel_pos=pixel_center(0, 0))
        assert nav.state == NavigationState.FAILED

    def test_failed_state_has_empty_path(self):
        grid = make_free_grid()
        for r, c in [(1, 2), (3, 2), (2, 1), (2, 3)]:
            grid.mark_occupied(r, c)
        nav, _ = make_manager(grid=grid)
        nav.start_mission((0, 0), (2, 2), robot_pixel_pos=pixel_center(0, 0))
        assert nav.active_path == []

    def test_blocked_start_raises(self):
        grid = make_free_grid()
        grid.mark_occupied(0, 0)
        nav, _ = make_manager(grid=grid)
        with pytest.raises((ValueError, IndexError)):
            nav.start_mission((0, 0), (3, 3), robot_pixel_pos=pixel_center(0, 0))

    def test_blocked_goal_raises(self):
        grid = make_free_grid()
        grid.mark_occupied(3, 3)
        nav, _ = make_manager(grid=grid)
        with pytest.raises((ValueError, IndexError)):
            nav.start_mission((0, 0), (3, 3), robot_pixel_pos=pixel_center(0, 0))

    def test_start_while_navigating_raises_runtime_error(self):
        nav, _ = make_manager()
        nav.start_mission((0, 0), (4, 4), robot_pixel_pos=pixel_center(0, 0))
        with pytest.raises(RuntimeError, match="NAVIGATING"):
            nav.start_mission((0, 0), (4, 4), robot_pixel_pos=pixel_center(0, 0))

    def test_invalid_start_type_raises_type_error(self):
        nav, _ = make_manager()
        with pytest.raises(TypeError):
            nav.start_mission("0,0", (4, 4), robot_pixel_pos=pixel_center(0, 0))  # type: ignore

    def test_invalid_goal_type_raises_type_error(self):
        nav, _ = make_manager()
        with pytest.raises(TypeError):
            nav.start_mission((0, 0), [4, 4], robot_pixel_pos=pixel_center(0, 0))  # type: ignore

    def test_invalid_float_grid_coords_raise_type_error(self):
        nav, _ = make_manager()
        with pytest.raises(TypeError):
            nav.start_mission((0.0, 0.0), (4, 4), robot_pixel_pos=pixel_center(0, 0))  # type: ignore

    def test_replan_count_reset_on_new_mission(self):
        nav, _ = make_manager()
        nav.start_mission((0, 0), (4, 4), robot_pixel_pos=pixel_center(0, 0))
        nav.reset()
        nav.start_mission((0, 0), (4, 4), robot_pixel_pos=pixel_center(0, 0))
        assert nav.replan_count == 0


# =============================================================================
# 4. Grid-to-Pixel Conversion
# =============================================================================

class TestGridToPixelConversion:
    def test_pixel_center_origin_cell(self):
        _, grid = make_manager()
        assert grid.grid_to_pixel_center(0, 0) == (5.0, 5.0)

    def test_pixel_center_col_offset(self):
        _, grid = make_manager()
        x, y = grid.grid_to_pixel_center(0, 2)
        assert x == pytest.approx(25.0)
        assert y == pytest.approx(5.0)

    def test_pixel_center_row_offset(self):
        _, grid = make_manager()
        x, y = grid.grid_to_pixel_center(3, 0)
        assert x == pytest.approx(5.0)
        assert y == pytest.approx(35.0)

    def test_get_current_target_pixel_matches_conversion(self):
        nav, grid = make_manager()
        nav.start_mission((0, 0), (3, 0), robot_pixel_pos=pixel_center(0, 0))
        target_grid = nav.get_current_target()
        target_pixel = nav.get_current_target_pixel()
        expected = grid.grid_to_pixel_center(*target_grid)
        assert target_pixel == expected

    def test_pixel_x_increases_with_col(self):
        _, grid = make_manager()
        x0, _ = grid.grid_to_pixel_center(0, 0)
        x1, _ = grid.grid_to_pixel_center(0, 1)
        assert x1 > x0

    def test_pixel_y_increases_with_row(self):
        _, grid = make_manager()
        _, y0 = grid.grid_to_pixel_center(0, 0)
        _, y1 = grid.grid_to_pixel_center(1, 0)
        assert y1 > y0


# =============================================================================
# 5. Valid Route — No Unnecessary Replanning
# =============================================================================

class TestValidRouteNoReplan:
    def test_clear_path_does_not_replan(self):
        nav, _ = make_manager(max_speed=100.0, arrival_tolerance=3.0)
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        pos = pixel_center(0, 0)
        run_to_completion(nav, pos)
        assert nav.replan_count == 0

    def test_clear_path_completes_to_reached(self):
        nav, _ = make_manager(max_speed=100.0, arrival_tolerance=3.0)
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        pos = pixel_center(0, 0)
        run_to_completion(nav, pos)
        assert nav.state == NavigationState.REACHED

    def test_path_validity_true_on_clear_grid(self):
        nav, _ = make_manager()
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        assert nav._is_remaining_path_valid()


# =============================================================================
# 6. Obstacle Appears on Next Waypoint — Immediate Replan
# =============================================================================

class TestObstacleOnNextWaypoint:
    def _setup_blocked_next(self):
        grid = make_free_grid()
        nav, _ = make_manager(max_speed=2.0, arrival_tolerance=1.0, grid=grid)
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        # Block the next planned waypoint
        next_wp = nav.active_path[nav.current_waypoint_index]
        grid.mark_occupied(next_wp[0], next_wp[1])
        return nav, grid

    def test_path_validity_false_after_block(self):
        nav, grid = self._setup_blocked_next()
        assert not nav._is_remaining_path_valid()

    def test_update_triggers_replan(self):
        nav, grid = self._setup_blocked_next()
        pos = pixel_center(0, 0)
        nav.update(pos)
        assert nav.replan_count == 1

    def test_after_replan_state_is_navigating_or_failed(self):
        nav, grid = self._setup_blocked_next()
        pos = pixel_center(0, 0)
        nav.update(pos)
        assert nav.state in (NavigationState.NAVIGATING, NavigationState.FAILED)

    def test_replacement_path_excludes_occupied_cell(self):
        grid = make_free_grid()
        nav, _ = make_manager(max_speed=2.0, arrival_tolerance=1.0, grid=grid)
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        next_wp = nav.active_path[nav.current_waypoint_index]
        grid.mark_occupied(next_wp[0], next_wp[1])
        pos = pixel_center(0, 0)
        nav.update(pos)
        if nav.state == NavigationState.NAVIGATING:
            for wp in nav.active_path:
                assert grid.get_cell(wp[0], wp[1]) != CellState.OCCUPIED


# =============================================================================
# 7. Obstacle Farther Along the Remaining Path
# =============================================================================

class TestObstacleFartherAlong:
    def test_path_blocked_at_later_waypoint_triggers_replan(self):
        grid = make_free_grid()
        nav, _ = make_manager(max_speed=2.0, arrival_tolerance=1.0, grid=grid)
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        # Block the LAST waypoint (the goal cell)
        last_wp = nav.active_path[-1]
        grid.mark_occupied(last_wp[0], last_wp[1])
        pos = pixel_center(0, 0)
        nav.update(pos)
        assert nav.replan_count >= 1

    def test_farther_block_transitions_to_failed_when_no_route(self):
        grid = make_free_grid()
        nav, _ = make_manager(max_speed=2.0, arrival_tolerance=1.0, grid=grid)
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        # Block all non-start cells in row 0 except the first, making goal unreachable
        for c in range(1, 5):
            grid.mark_occupied(0, c)
        pos = pixel_center(0, 0)
        nav.update(pos)
        assert nav.state == NavigationState.FAILED


# =============================================================================
# 8. Successful Replan Around an Obstacle
# =============================================================================

class TestSuccessfulReplan:
    def test_replan_around_obstacle_completes_mission(self):
        """Block (0,1) in a straight-line path; robot must detour via row 1."""
        grid = make_free_grid()
        nav, _ = make_manager(max_speed=100.0, arrival_tolerance=3.0, grid=grid)
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        # Block cell on the planned straight route
        grid.mark_occupied(0, 1)
        pos = pixel_center(0, 0)
        # update() should detect blocked path, replan, then navigate
        final_pos = run_to_completion(nav, pos)
        assert nav.state == NavigationState.REACHED

    def test_replan_count_increments(self):
        grid = make_free_grid()
        nav, _ = make_manager(max_speed=100.0, arrival_tolerance=3.0, grid=grid)
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        grid.mark_occupied(0, 1)
        pos = pixel_center(0, 0)
        run_to_completion(nav, pos)
        assert nav.replan_count >= 1

    def test_waypoint_index_valid_after_replan(self):
        grid = make_free_grid()
        nav, _ = make_manager(max_speed=100.0, arrival_tolerance=3.0, grid=grid)
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        grid.mark_occupied(0, 1)
        pos = pixel_center(0, 0)
        nav.update(pos)
        if nav.state == NavigationState.NAVIGATING:
            assert 0 <= nav.current_waypoint_index < len(nav.active_path)

    def test_replacement_path_ends_at_goal(self):
        grid = make_free_grid()
        nav, _ = make_manager(max_speed=100.0, arrival_tolerance=3.0, grid=grid)
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        grid.mark_occupied(0, 1)
        pos = pixel_center(0, 0)
        nav.update(pos)
        if nav.state == NavigationState.NAVIGATING:
            assert nav.active_path[-1] == (0, 4)


# =============================================================================
# 9. Replanning Failure → FAILED
# =============================================================================

class TestReplanFailure:
    def _make_sealed_grid(self):
        """Full wall around the goal makes it permanently unreachable."""
        grid = make_free_grid()
        nav, _ = make_manager(max_speed=2.0, arrival_tolerance=1.0, grid=grid)
        # Plan path to (0,4) first
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        # Now block all routes: cover entire right column and row 0 col 1..4
        for r in range(5):
            grid.mark_occupied(r, 4)
        for c in range(1, 5):
            grid.mark_occupied(0, c)
        return nav, grid

    def test_replan_failure_sets_failed(self):
        nav, grid = self._make_sealed_grid()
        pos = pixel_center(0, 0)
        nav.update(pos)
        assert nav.state == NavigationState.FAILED

    def test_no_movement_after_replan_failure(self):
        nav, grid = self._make_sealed_grid()
        pos = pixel_center(0, 0)
        nav.update(pos)
        assert nav.state == NavigationState.FAILED
        result = nav.update(pos)
        assert result.displacement == 0.0

    def test_replan_count_increments_on_failure(self):
        nav, grid = self._make_sealed_grid()
        pos = pixel_center(0, 0)
        nav.update(pos)
        assert nav.replan_count >= 1


# =============================================================================
# 10. No Movement in Terminal States
# =============================================================================

class TestTerminalStates:
    def test_no_movement_after_reached(self):
        nav, _ = make_manager(max_speed=100.0, arrival_tolerance=5.0)
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        pos = pixel_center(0, 0)
        pos = run_to_completion(nav, pos)
        assert nav.state == NavigationState.REACHED
        result = nav.update(pos)
        assert result.displacement == 0.0

    def test_no_movement_after_failed(self):
        grid = make_free_grid()
        for r, c in [(1, 2), (3, 2), (2, 1), (2, 3)]:
            grid.mark_occupied(r, c)
        nav, _ = make_manager(grid=grid)
        nav.start_mission((0, 0), (2, 2), robot_pixel_pos=pixel_center(0, 0))
        assert nav.state == NavigationState.FAILED
        result = nav.update(pixel_center(0, 0))
        assert result.displacement == 0.0

    def test_no_movement_after_start_equals_goal(self):
        nav, _ = make_manager()
        pos = pixel_center(2, 2)
        nav.start_mission((2, 2), (2, 2), robot_pixel_pos=pos)
        result = nav.update(pos)
        assert result.displacement == 0.0


# =============================================================================
# 11. Movement and Waypoint Progression
# =============================================================================

class TestMovementAndProgression:
    def test_update_when_idle_returns_same_position(self):
        nav, _ = make_manager()
        pos = (5.0, 5.0)
        result = nav.update(pos)
        assert result.new_position == pos
        assert result.displacement == 0.0

    def test_update_moves_robot_toward_target(self):
        nav, _ = make_manager(max_speed=2.0, arrival_tolerance=0.5)
        nav.start_mission((0, 0), (0, 2), robot_pixel_pos=pixel_center(0, 0))
        pos = pixel_center(0, 0)
        result = nav.update(pos)
        assert result.displacement > 0.0
        assert result.new_position != pos

    def test_mission_completes_to_reached(self):
        nav, _ = make_manager(max_speed=100.0, arrival_tolerance=5.0)
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        run_to_completion(nav, pixel_center(0, 0))
        assert nav.state == NavigationState.REACHED

    def test_repeated_updates_at_same_position_converge(self):
        nav, _ = make_manager(max_speed=100.0, arrival_tolerance=5.0)
        nav.start_mission((0, 0), (0, 2), robot_pixel_pos=pixel_center(0, 0))
        run_to_completion(nav, pixel_center(0, 0))
        assert nav.state == NavigationState.REACHED


# =============================================================================
# 12. advance_waypoint()
# =============================================================================

class TestAdvanceWaypoint:
    def test_advance_increments_index(self):
        nav, _ = make_manager()
        nav.set_path([(0, 0), (1, 0), (2, 0)])
        nav.advance_waypoint()
        assert nav.current_waypoint_index == 1

    def test_advance_past_last_sets_reached(self):
        nav, _ = make_manager()
        nav.set_path([(0, 0), (1, 0)])
        nav.advance_waypoint()
        nav.advance_waypoint()
        assert nav.state == NavigationState.REACHED

    def test_advance_when_idle_is_noop(self):
        nav, _ = make_manager()
        nav.advance_waypoint()
        assert nav.state == NavigationState.IDLE
        assert nav.current_waypoint_index == 0

    def test_advance_when_reached_is_noop(self):
        nav, _ = make_manager()
        nav.set_path([(0, 0)])
        nav.advance_waypoint()
        assert nav.state == NavigationState.REACHED
        nav.advance_waypoint()
        assert nav.state == NavigationState.REACHED


# =============================================================================
# 13. Reset and New Mission After Terminal State
# =============================================================================

class TestResetAndRestart:
    def test_reset_from_navigating_to_idle(self):
        nav, _ = make_manager()
        nav.start_mission((0, 0), (4, 4), robot_pixel_pos=pixel_center(0, 0))
        nav.reset()
        assert nav.state == NavigationState.IDLE

    def test_reset_clears_path(self):
        nav, _ = make_manager()
        nav.start_mission((0, 0), (4, 4), robot_pixel_pos=pixel_center(0, 0))
        nav.reset()
        assert nav.active_path == []

    def test_reset_clears_waypoint_index(self):
        nav, _ = make_manager()
        nav.start_mission((0, 0), (4, 4), robot_pixel_pos=pixel_center(0, 0))
        nav.reset()
        assert nav.current_waypoint_index == 0

    def test_reset_clears_replan_count(self):
        nav, _ = make_manager()
        nav.replan_count = 5
        nav.reset()
        assert nav.replan_count == 0

    def test_new_mission_after_reached(self):
        nav, _ = make_manager(max_speed=100.0, arrival_tolerance=5.0)
        nav.start_mission((0, 0), (0, 2), robot_pixel_pos=pixel_center(0, 0))
        pos = pixel_center(0, 0)
        for _ in range(500):
            if nav.state != NavigationState.NAVIGATING:
                break
            result = nav.update(pos)
            pos = result.new_position
        assert nav.state == NavigationState.REACHED
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        assert nav.state == NavigationState.NAVIGATING

    def test_new_mission_after_failed(self):
        grid = make_free_grid()
        for r, c in [(1, 2), (3, 2), (2, 1), (2, 3)]:
            grid.mark_occupied(r, c)
        nav, _ = make_manager(grid=grid)
        nav.start_mission((0, 0), (2, 2), robot_pixel_pos=pixel_center(0, 0))
        assert nav.state == NavigationState.FAILED
        grid.mark_free(1, 2)  # open a gap
        nav.start_mission((0, 0), (2, 2), robot_pixel_pos=pixel_center(0, 0))
        assert nav.state == NavigationState.NAVIGATING

    def test_reset_from_failed_to_idle(self):
        nav, _ = make_manager()
        nav.state = NavigationState.FAILED
        nav.reset()
        assert nav.state == NavigationState.IDLE

    def test_new_mission_after_start_equals_goal(self):
        nav, _ = make_manager()
        nav.start_mission((2, 2), (2, 2), robot_pixel_pos=pixel_center(2, 2))
        assert nav.state == NavigationState.REACHED
        nav.start_mission((2, 2), (0, 0), robot_pixel_pos=pixel_center(2, 2))
        assert nav.state == NavigationState.NAVIGATING


# =============================================================================
# 14. Unknown-Cell Policy
# =============================================================================

class TestUnknownCellPolicy:
    def test_unknown_cell_on_path_treated_as_blocked_by_default(self):
        grid = OccupancyGrid(rows=5, cols=5, cell_size=10)
        # All cells UNKNOWN initially
        # Mark start and goal free only
        grid.mark_free(0, 0)
        grid.mark_free(0, 4)
        planner = PathPlanner(allow_unknown=False)
        controller = MotionController(max_speed=10.0)
        nav = NavigationManager(planner=planner, controller=controller, grid=grid,
                                allow_unknown=False)
        # Plan will fail because UNKNOWN cells are not traversable
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        assert nav.state == NavigationState.FAILED

    def test_unknown_cell_traversable_when_allow_unknown_true(self):
        grid = OccupancyGrid(rows=5, cols=5, cell_size=10)
        grid.mark_free(0, 0)
        grid.mark_free(0, 4)
        # All intermediate cells remain UNKNOWN — traversable if allow_unknown=True
        planner = PathPlanner(allow_unknown=True)
        controller = MotionController(max_speed=10.0)
        nav = NavigationManager(planner=planner, controller=controller, grid=grid,
                                allow_unknown=True)
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        assert nav.state == NavigationState.NAVIGATING

    def test_validity_check_respects_allow_unknown_false(self):
        grid = make_free_grid()
        nav, _ = make_manager(grid=grid)
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        # Flip a remaining waypoint to UNKNOWN
        wp = nav.active_path[nav.current_waypoint_index]
        grid.mark_unknown(wp[0], wp[1])
        assert not nav._is_remaining_path_valid()

    def test_validity_check_respects_allow_unknown_true(self):
        grid = make_free_grid()
        planner = PathPlanner(allow_unknown=True)
        controller = MotionController(max_speed=10.0)
        nav = NavigationManager(planner=planner, controller=controller, grid=grid,
                                allow_unknown=True)
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        # Flip a remaining waypoint to UNKNOWN — should still be valid
        wp = nav.active_path[nav.current_waypoint_index]
        grid.mark_unknown(wp[0], wp[1])
        assert nav._is_remaining_path_valid()


# =============================================================================
# 15. from_config() Factory
# =============================================================================

class TestFromConfig:
    def test_from_config_returns_navigation_manager(self):
        from autonomous_nav.config import load_config
        cfg = load_config()
        grid = OccupancyGrid.from_config(cfg, initial_state=CellState.FREE)
        nav = NavigationManager.from_config(cfg, grid)
        assert isinstance(nav, NavigationManager)
        assert nav.state == NavigationState.IDLE

    def test_from_config_controller_speed_matches(self):
        from autonomous_nav.config import load_config
        cfg = load_config()
        grid = OccupancyGrid.from_config(cfg, initial_state=CellState.FREE)
        nav = NavigationManager.from_config(cfg, grid)
        assert nav._controller.max_speed == cfg.robot.max_speed

    def test_from_config_planner_algorithm_matches(self):
        from autonomous_nav.config import load_config
        cfg = load_config()
        grid = OccupancyGrid.from_config(cfg, initial_state=CellState.FREE)
        nav = NavigationManager.from_config(cfg, grid)
        assert nav._planner.algorithm == cfg.planner.algorithm.lower()


# =============================================================================
# 16. Determinism
# =============================================================================

class TestDeterminism:
    def test_same_mission_produces_same_path(self):
        nav1, _ = make_manager()
        nav2, _ = make_manager()
        nav1.start_mission((0, 0), (4, 4), robot_pixel_pos=pixel_center(0, 0))
        nav2.start_mission((0, 0), (4, 4), robot_pixel_pos=pixel_center(0, 0))
        assert nav1.active_path == nav2.active_path

    def test_same_update_produces_same_result(self):
        nav1, _ = make_manager(max_speed=4.0)
        nav2, _ = make_manager(max_speed=4.0)
        nav1.start_mission((0, 0), (4, 4), robot_pixel_pos=pixel_center(0, 0))
        nav2.start_mission((0, 0), (4, 4), robot_pixel_pos=pixel_center(0, 0))
        pos = pixel_center(0, 0)
        r1 = nav1.update(pos)
        r2 = nav2.update(pos)
        assert r1.new_position == r2.new_position
        assert r1.displacement == r2.displacement

    def test_dijkstra_produces_valid_path(self):
        nav, _ = make_manager(algorithm="dijkstra")
        nav.start_mission((0, 0), (0, 4), robot_pixel_pos=pixel_center(0, 0))
        assert nav.state == NavigationState.NAVIGATING
        assert nav.active_path[-1] == (0, 4)


# =============================================================================
# 17. Backward-Compatible set_path() API
# =============================================================================

class TestSetPathCompat:
    def test_set_path_transitions_to_navigating(self):
        nav, _ = make_manager()
        nav.set_path([(0, 0), (1, 0)])
        assert nav.state == NavigationState.NAVIGATING

    def test_set_path_exposes_first_waypoint(self):
        nav, _ = make_manager()
        nav.set_path([(0, 0), (1, 0)])
        assert nav.get_current_target() == (0, 0)

    def test_set_path_empty_transitions_to_idle(self):
        nav, _ = make_manager()
        nav.set_path([])
        assert nav.state == NavigationState.IDLE
        assert nav.get_current_target() is None

    def test_set_path_stores_path_length(self):
        nav, _ = make_manager()
        path = [(0, 0), (1, 0), (2, 0)]
        nav.set_path(path)
        assert len(nav.active_path) == 3


# =============================================================================
# 18. Invalid update() Inputs
# =============================================================================

class TestUpdateInvalidInputs:
    @pytest.mark.parametrize("bad_pos", [
        None, "string", 5.0, (1,), (1, 2, 3), (True, 1.0), (1.0, "y")
    ])
    def test_bad_robot_pos_raises_type_error(self, bad_pos):
        nav, _ = make_manager()
        with pytest.raises(TypeError):
            nav.update(bad_pos)

    def test_invalid_dt_raises_value_error(self):
        nav, _ = make_manager()
        nav.start_mission((0, 0), (4, 4), robot_pixel_pos=pixel_center(0, 0))
        with pytest.raises(ValueError):
            nav.update(pixel_center(0, 0), dt=-1.0)


# =============================================================================
# 19. Preserved Phase 5B original tests
# =============================================================================

def test_navigation_manager_initial_state():
    """Verify that manager begins in IDLE state without active target."""
    nav, _ = make_manager()
    assert nav.state == NavigationState.IDLE
    assert nav.get_current_target() is None


def test_navigation_manager_set_path():
    """Verify setting an active path transitions state to NAVIGATING and sets target."""
    nav, _ = make_manager()
    path = [(0, 0), (1, 0), (2, 0)]
    nav.set_path(path)
    assert nav.state == NavigationState.NAVIGATING
    assert nav.get_current_target() == (0, 0)
    assert len(nav.active_path) == 3


def test_navigation_manager_empty_path():
    """Verify setting an empty path sets state to IDLE."""
    nav, _ = make_manager()
    nav.set_path([])
    assert nav.state == NavigationState.IDLE
    assert nav.get_current_target() is None


@pytest.mark.skip(reason="Phase 5C implemented dynamic replanning inline in update(). "
                          "Dedicated event-based replanning triggers reserved for future phases.")
def test_replanning_trigger_planned():
    """Placeholder documenting event-based replanning trigger tests."""
    pass

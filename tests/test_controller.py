"""Unit tests for MotionController motion commands (Phase 5A).

Covers:
    - Valid and invalid construction
    - from_config() factory
    - has_arrived() edge cases
    - move_toward() in multiple directions
    - Maximum movement distance per step
    - No-overshoot clamping when target is close
    - Current position equals target (zero-distance)
    - dt parameter validation and scaling
    - Invalid position inputs
    - Deterministic / reproducible results
"""

import math

import pytest

from autonomous_nav.config import load_config
from autonomous_nav.controller import MoveResult, MotionController


# =============================================================================
# 1. Construction and Validation Tests
# =============================================================================

class TestConstruction:
    def test_valid_default_tolerance(self):
        ctrl = MotionController(max_speed=4.0)
        assert ctrl.max_speed == 4.0
        assert ctrl.arrival_tolerance == 2.0

    def test_valid_custom_tolerance(self):
        ctrl = MotionController(max_speed=10.0, arrival_tolerance=5.0)
        assert ctrl.max_speed == 10.0
        assert ctrl.arrival_tolerance == 5.0

    def test_speed_stored_as_float(self):
        ctrl = MotionController(max_speed=3)  # int input
        assert isinstance(ctrl.max_speed, float)

    def test_tolerance_stored_as_float(self):
        ctrl = MotionController(max_speed=4.0, arrival_tolerance=1)
        assert isinstance(ctrl.arrival_tolerance, float)

    @pytest.mark.parametrize("bad_speed", [0, -1, -0.001])
    def test_invalid_zero_or_negative_speed(self, bad_speed):
        with pytest.raises(ValueError, match="max_speed must be positive"):
            MotionController(max_speed=bad_speed)

    @pytest.mark.parametrize("bad_tol", [0, -0.5])
    def test_invalid_zero_or_negative_tolerance(self, bad_tol):
        with pytest.raises(ValueError, match="arrival_tolerance must be positive"):
            MotionController(max_speed=4.0, arrival_tolerance=bad_tol)

    @pytest.mark.parametrize("bad_speed", ["fast", None, True, [4.0]])
    def test_invalid_speed_type(self, bad_speed):
        with pytest.raises((TypeError, ValueError)):
            MotionController(max_speed=bad_speed)

    @pytest.mark.parametrize("bad_tol", ["close", None, True])
    def test_invalid_tolerance_type(self, bad_tol):
        with pytest.raises((TypeError, ValueError)):
            MotionController(max_speed=4.0, arrival_tolerance=bad_tol)


# =============================================================================
# 2. from_config() Factory
# =============================================================================

class TestFromConfig:
    def test_from_config_speed_matches(self):
        cfg = load_config()
        ctrl = MotionController.from_config(cfg)
        assert ctrl.max_speed == cfg.robot.max_speed

    def test_from_config_tolerance_half_cell(self):
        cfg = load_config()
        ctrl = MotionController.from_config(cfg)
        assert ctrl.arrival_tolerance == cfg.grid.cell_size / 2.0

    def test_from_config_returns_controller_instance(self):
        cfg = load_config()
        ctrl = MotionController.from_config(cfg)
        assert isinstance(ctrl, MotionController)


# =============================================================================
# 3. has_arrived() Tests
# =============================================================================

class TestHasArrived:
    def setup_method(self):
        self.ctrl = MotionController(max_speed=4.0, arrival_tolerance=2.0)

    def test_same_position_is_arrived(self):
        assert self.ctrl.has_arrived((10.0, 10.0), (10.0, 10.0)) is True

    def test_exactly_at_tolerance_is_arrived(self):
        # distance == arrival_tolerance exactly => arrived
        assert self.ctrl.has_arrived((10.0, 10.0), (12.0, 10.0)) is True

    def test_inside_tolerance_is_arrived(self):
        assert self.ctrl.has_arrived((10.0, 10.0), (11.0, 10.0)) is True

    def test_beyond_tolerance_not_arrived(self):
        assert self.ctrl.has_arrived((10.0, 10.0), (15.0, 10.0)) is False

    def test_diagonal_within_tolerance(self):
        # distance = sqrt(2) ≈ 1.414 < 2.0
        assert self.ctrl.has_arrived((0.0, 0.0), (1.0, 1.0)) is True

    def test_diagonal_beyond_tolerance(self):
        # distance = sqrt(18) ≈ 4.24 > 2.0
        assert self.ctrl.has_arrived((0.0, 0.0), (3.0, 3.0)) is False

    @pytest.mark.parametrize("bad_pos", [
        "str", None, (1,), (1, 2, 3), (True, 1.0), (1.0, "y")
    ])
    def test_invalid_current_pos_raises(self, bad_pos):
        with pytest.raises(TypeError):
            self.ctrl.has_arrived(bad_pos, (0.0, 0.0))

    @pytest.mark.parametrize("bad_pos", [
        "str", None, (1,), (1, 2, 3), (True, 1.0)
    ])
    def test_invalid_target_pos_raises(self, bad_pos):
        with pytest.raises(TypeError):
            self.ctrl.has_arrived((0.0, 0.0), bad_pos)


# =============================================================================
# 4. move_toward() Direction Tests
# =============================================================================

class TestMovementDirection:
    def setup_method(self):
        self.ctrl = MotionController(max_speed=4.0, arrival_tolerance=2.0)

    def test_move_rightward(self):
        result = self.ctrl.move_toward((0.0, 0.0), (100.0, 0.0))
        assert result.new_position[0] == pytest.approx(4.0)
        assert result.new_position[1] == pytest.approx(0.0)

    def test_move_leftward(self):
        result = self.ctrl.move_toward((100.0, 0.0), (0.0, 0.0))
        assert result.new_position[0] == pytest.approx(96.0)
        assert result.new_position[1] == pytest.approx(0.0)

    def test_move_downward(self):
        result = self.ctrl.move_toward((0.0, 0.0), (0.0, 100.0))
        assert result.new_position[0] == pytest.approx(0.0)
        assert result.new_position[1] == pytest.approx(4.0)

    def test_move_upward(self):
        result = self.ctrl.move_toward((0.0, 100.0), (0.0, 0.0))
        assert result.new_position[0] == pytest.approx(0.0)
        assert result.new_position[1] == pytest.approx(96.0)

    def test_move_diagonal_45_degrees(self):
        # Pure diagonal: (0,0) -> (100,100)
        # Unit vector: (1/sqrt2, 1/sqrt2), step = 4.0
        result = self.ctrl.move_toward((0.0, 0.0), (100.0, 100.0))
        expected = 4.0 / math.sqrt(2)
        assert result.new_position[0] == pytest.approx(expected, abs=1e-9)
        assert result.new_position[1] == pytest.approx(expected, abs=1e-9)

    def test_move_negative_quadrant(self):
        result = self.ctrl.move_toward((10.0, 10.0), (6.0, 6.0))
        # distance ~5.66, max_step=4 => scale = 4/5.66
        dist = math.hypot(4.0, 4.0)
        scale = 4.0 / dist
        assert result.new_position[0] == pytest.approx(10.0 - 4.0 * scale)
        assert result.new_position[1] == pytest.approx(10.0 - 4.0 * scale)


# =============================================================================
# 5. Maximum Movement Distance per Step
# =============================================================================

class TestMaxMovementDistance:
    def setup_method(self):
        self.ctrl = MotionController(max_speed=4.0, arrival_tolerance=2.0)

    def test_displacement_capped_at_max_speed(self):
        result = self.ctrl.move_toward((0.0, 0.0), (1000.0, 0.0))
        assert result.displacement == pytest.approx(4.0)

    def test_displacement_less_than_max_speed_when_close(self):
        result = self.ctrl.move_toward((0.0, 0.0), (1.0, 0.0))
        # target is 1 px away, well within max_speed=4
        assert result.displacement == pytest.approx(1.0)

    def test_speed_scales_with_dt(self):
        result_half = self.ctrl.move_toward((0.0, 0.0), (100.0, 0.0), dt=0.5)
        assert result_half.displacement == pytest.approx(2.0)

    def test_speed_scales_with_dt_double(self):
        result_double = self.ctrl.move_toward((0.0, 0.0), (100.0, 0.0), dt=2.0)
        assert result_double.displacement == pytest.approx(8.0)

    def test_different_speeds_give_proportional_displacement(self):
        slow = MotionController(max_speed=2.0)
        fast = MotionController(max_speed=8.0)
        r_slow = slow.move_toward((0.0, 0.0), (100.0, 0.0))
        r_fast = fast.move_toward((0.0, 0.0), (100.0, 0.0))
        assert r_fast.displacement == pytest.approx(4 * r_slow.displacement)


# =============================================================================
# 6. No-Overshoot Tests
# =============================================================================

class TestNoOvershoot:
    def setup_method(self):
        self.ctrl = MotionController(max_speed=10.0, arrival_tolerance=2.0)

    def test_target_closer_than_max_speed_clamps_to_target(self):
        # target is 3 px right; max_speed=10 would overshoot without clamping
        result = self.ctrl.move_toward((0.0, 0.0), (3.0, 0.0))
        assert result.new_position == (3.0, 0.0)
        assert result.arrived is True

    def test_target_exactly_at_max_speed_snaps(self):
        result = self.ctrl.move_toward((0.0, 0.0), (10.0, 0.0))
        assert result.new_position == (10.0, 0.0)

    def test_displacement_equals_distance_when_close(self):
        result = self.ctrl.move_toward((0.0, 0.0), (5.0, 0.0))
        assert result.displacement == pytest.approx(5.0)

    def test_position_does_not_exceed_target(self):
        result = self.ctrl.move_toward((0.0, 0.0), (2.0, 0.0))
        # With max_speed=10 the robot must not go past x=2
        assert result.new_position[0] <= 2.0 + 1e-9


# =============================================================================
# 7. Zero-Distance (Current == Target) Tests
# =============================================================================

class TestZeroDistance:
    def setup_method(self):
        self.ctrl = MotionController(max_speed=4.0, arrival_tolerance=2.0)

    def test_same_position_returns_same_coords(self):
        result = self.ctrl.move_toward((50.0, 50.0), (50.0, 50.0))
        assert result.new_position == (50.0, 50.0)

    def test_same_position_zero_displacement(self):
        result = self.ctrl.move_toward((50.0, 50.0), (50.0, 50.0))
        assert result.displacement == 0.0

    def test_same_position_marked_arrived(self):
        result = self.ctrl.move_toward((50.0, 50.0), (50.0, 50.0))
        assert result.arrived is True

    def test_integer_equal_positions(self):
        # Integer coordinates should still work
        result = self.ctrl.move_toward((10, 10), (10, 10))
        assert result.arrived is True
        assert result.displacement == 0.0


# =============================================================================
# 8. Invalid Input Tests
# =============================================================================

class TestInvalidInputs:
    def setup_method(self):
        self.ctrl = MotionController(max_speed=4.0, arrival_tolerance=2.0)

    @pytest.mark.parametrize("bad_pos", [
        None, "string", 5.0, (1,), (1, 2, 3), [], {}, (True, 1.0), (1.0, None)
    ])
    def test_bad_current_pos_raises_type_error(self, bad_pos):
        with pytest.raises(TypeError):
            self.ctrl.move_toward(bad_pos, (10.0, 10.0))

    @pytest.mark.parametrize("bad_pos", [
        None, "string", 5.0, (1,), (1, 2, 3), [], (1.0, "x")
    ])
    def test_bad_target_pos_raises_type_error(self, bad_pos):
        with pytest.raises(TypeError):
            self.ctrl.move_toward((0.0, 0.0), bad_pos)

    @pytest.mark.parametrize("bad_dt", [0, -1.0, -0.001])
    def test_non_positive_dt_raises_value_error(self, bad_dt):
        with pytest.raises(ValueError, match="dt must be positive"):
            self.ctrl.move_toward((0.0, 0.0), (10.0, 0.0), dt=bad_dt)

    @pytest.mark.parametrize("bad_dt", ["1.0", None, True, [1.0]])
    def test_non_numeric_dt_raises_type_error(self, bad_dt):
        with pytest.raises(TypeError):
            self.ctrl.move_toward((0.0, 0.0), (10.0, 0.0), dt=bad_dt)


# =============================================================================
# 9. Determinism Tests
# =============================================================================

class TestDeterminism:
    def setup_method(self):
        self.ctrl = MotionController(max_speed=4.0, arrival_tolerance=2.0)

    def test_identical_inputs_produce_identical_outputs(self):
        r1 = self.ctrl.move_toward((5.0, 5.0), (50.0, 80.0))
        r2 = self.ctrl.move_toward((5.0, 5.0), (50.0, 80.0))
        assert r1.new_position == r2.new_position
        assert r1.displacement == r2.displacement
        assert r1.arrived == r2.arrived

    def test_repeated_steps_converge(self):
        """Step robot from (0,0) toward (100, 0) until arrived; must terminate."""
        pos = (0.0, 0.0)
        target = (100.0, 0.0)
        max_iterations = 200
        for _ in range(max_iterations):
            result = self.ctrl.move_toward(pos, target)
            pos = result.new_position
            if result.arrived:
                break
        else:
            pytest.fail("Robot did not reach target within expected steps.")
        assert self.ctrl.has_arrived(pos, target)

    def test_move_result_is_frozen_dataclass(self):
        result = self.ctrl.move_toward((0.0, 0.0), (10.0, 0.0))
        with pytest.raises(Exception):
            result.displacement = 999.0  # type: ignore[misc]


# =============================================================================
# 10. MoveResult Dataclass
# =============================================================================

class TestMoveResult:
    def test_move_result_fields_accessible(self):
        r = MoveResult(new_position=(1.0, 2.0), displacement=1.0, arrived=False)
        assert r.new_position == (1.0, 2.0)
        assert r.displacement == 1.0
        assert r.arrived is False

    def test_arrived_flag_set_correctly_when_at_tolerance(self):
        ctrl = MotionController(max_speed=10.0, arrival_tolerance=3.0)
        # Move to exactly 3 px away; with max_speed=10 it snaps to target → arrived
        result = ctrl.move_toward((0.0, 0.0), (3.0, 0.0))
        assert result.arrived is True

    def test_arrived_flag_false_when_still_moving(self):
        ctrl = MotionController(max_speed=4.0, arrival_tolerance=2.0)
        # 100 px away; one step moves 4 px, still 96 px from target
        result = ctrl.move_toward((0.0, 0.0), (100.0, 0.0))
        assert result.arrived is False

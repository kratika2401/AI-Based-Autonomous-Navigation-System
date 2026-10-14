"""Unit tests for simulation configuration loading and validation."""

import copy
import pytest

from autonomous_nav.config import (
    ConfigurationError,
    load_config,
    load_raw_config,
    parse_and_validate_config,
)


def test_load_default_config_succeeds():
    """Verify that the default JSON configuration file loads and passes validation."""
    config = load_config()
    assert config.window.width == 800
    assert config.window.height == 600
    assert config.grid.cell_size == 20
    assert config.grid.width_cells == 40
    assert config.grid.height_cells == 30
    assert config.simulation.fps == 60
    assert config.robot.radius > 0
    assert config.robot.max_speed > 0
    assert config.planner.algorithm == "astar"


def test_config_validation_detects_dimension_mismatch():
    """Verify that a mismatch between grid cells and window dimensions raises ConfigurationError."""
    raw = load_raw_config()
    corrupted = copy.deepcopy(raw)
    # 50 cells * 20px = 1000px, but window width is 800px
    corrupted["grid"]["width_cells"] = 50

    with pytest.raises(ConfigurationError, match="Grid resolution mismatch"):
        parse_and_validate_config(corrupted)


def test_config_validation_rejects_negative_parameters():
    """Verify that negative numerical values raise ConfigurationError."""
    raw = load_raw_config()

    corrupted_fps = copy.deepcopy(raw)
    corrupted_fps["simulation"]["fps"] = -10
    with pytest.raises(ConfigurationError, match="Simulation fps must be positive"):
        parse_and_validate_config(corrupted_fps)

    corrupted_robot = copy.deepcopy(raw)
    corrupted_robot["robot"]["radius"] = 0
    with pytest.raises(ConfigurationError, match="Robot radius must be positive"):
        parse_and_validate_config(corrupted_robot)


def test_config_validation_rejects_unsupported_planner():
    """Verify that unrecognized algorithms are rejected with an informative error."""
    raw = load_raw_config()
    corrupted = copy.deepcopy(raw)
    corrupted["planner"]["algorithm"] = "unsupported_genetic_algorithm"

    with pytest.raises(ConfigurationError, match="Unsupported planner algorithm"):
        parse_and_validate_config(corrupted)


def test_config_missing_section_raises_error():
    """Verify that missing a top-level section triggers ConfigurationError."""
    raw = load_raw_config()
    corrupted = copy.deepcopy(raw)
    del corrupted["planner"]

    with pytest.raises(ConfigurationError, match="Missing required configuration section: 'planner'"):
        parse_and_validate_config(corrupted)

"""Configuration management for the 2D simulation environment.

Provides utilities for loading, parsing, and validating simulation
parameters from JSON configuration files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict


class ConfigurationError(ValueError):
    """Raised when configuration values are missing, invalid, or inconsistent."""
    pass


@dataclass(frozen=True)
class WindowConfig:
    width: int
    height: int
    title: str = "AI Autonomous Navigation Simulation"

    def validate(self) -> None:
        if self.width <= 0 or self.height <= 0:
            raise ConfigurationError(f"Window dimensions must be positive, got {self.width}x{self.height}")


@dataclass(frozen=True)
class GridConfig:
    cell_size: int
    width_cells: int
    height_cells: int

    def validate(self) -> None:
        if self.cell_size <= 0:
            raise ConfigurationError(f"Grid cell_size must be positive, got {self.cell_size}")
        if self.width_cells <= 0 or self.height_cells <= 0:
            raise ConfigurationError(
                f"Grid cell dimensions must be positive, got {self.width_cells}x{self.height_cells}"
            )


@dataclass(frozen=True)
class SimulationConfigData:
    fps: int
    time_step: float

    def validate(self) -> None:
        if self.fps <= 0:
            raise ConfigurationError(f"Simulation fps must be positive, got {self.fps}")
        if self.time_step <= 0:
            raise ConfigurationError(f"Simulation time_step must be positive, got {self.time_step}")


@dataclass(frozen=True)
class RobotConfig:
    radius: float
    max_speed: float
    sensor_range: float

    def validate(self) -> None:
        if self.radius <= 0:
            raise ConfigurationError(f"Robot radius must be positive, got {self.radius}")
        if self.max_speed <= 0:
            raise ConfigurationError(f"Robot max_speed must be positive, got {self.max_speed}")
        if self.sensor_range <= 0:
            raise ConfigurationError(f"Robot sensor_range must be positive, got {self.sensor_range}")


@dataclass(frozen=True)
class PlannerConfig:
    algorithm: str
    allow_diagonal: bool
    heuristic: str = "manhattan"

    def validate(self) -> None:
        valid_algorithms = {"astar", "dijkstra"}
        if self.algorithm.lower() not in valid_algorithms:
            raise ConfigurationError(
                f"Unsupported planner algorithm '{self.algorithm}'. Must be one of {valid_algorithms}"
            )
        valid_heuristics = {"manhattan", "euclidean", "octile"}
        if self.heuristic.lower() not in valid_heuristics:
            raise ConfigurationError(
                f"Unsupported heuristic '{self.heuristic}'. Must be one of {valid_heuristics}"
            )


@dataclass(frozen=True)
class SimulationConfig:
    window: WindowConfig
    grid: GridConfig
    simulation: SimulationConfigData
    robot: RobotConfig
    planner: PlannerConfig
    raw_config: Dict[str, Any]

    def validate(self) -> None:
        """Validates all nested configuration blocks and internal consistency."""
        self.window.validate()
        self.grid.validate()
        self.simulation.validate()
        self.robot.validate()
        self.planner.validate()

        # Check consistency between window pixels and grid cell dimensions
        expected_width = self.grid.width_cells * self.grid.cell_size
        expected_height = self.grid.height_cells * self.grid.cell_size
        if expected_width != self.window.width or expected_height != self.window.height:
            raise ConfigurationError(
                f"Grid resolution mismatch: {self.grid.width_cells}x{self.grid.cell_size}px = {expected_width}px "
                f"(window width is {self.window.width}px), and {self.grid.height_cells}x{self.grid.cell_size}px = "
                f"{expected_height}px (window height is {self.window.height}px)."
            )


def get_default_config_path() -> Path:
    """Returns the default configuration file path relative to project root."""
    current_file = Path(__file__).resolve()
    # Path is src/autonomous_nav/config.py -> root is 3 levels up
    project_root = current_file.parents[2]
    return project_root / "configs" / "default.json"


def load_raw_config(config_path: Path | str | None = None) -> Dict[str, Any]:
    """Loads raw JSON dictionary from file."""
    path = Path(config_path) if config_path else get_default_config_path()
    if not path.is_file():
        raise FileNotFoundError(f"Configuration file not found at: {path}")

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict):
        raise ConfigurationError(f"Configuration file at {path} must contain a JSON object.")
    return data


def parse_and_validate_config(raw_dict: Dict[str, Any]) -> SimulationConfig:
    """Parses a dictionary into typed config structures and validates values."""
    required_sections = ["window", "grid", "simulation", "robot", "planner"]
    for section in required_sections:
        if section not in raw_dict or not isinstance(raw_dict[section], dict):
            raise ConfigurationError(f"Missing required configuration section: '{section}'")

    win_data = raw_dict["window"]
    grid_data = raw_dict["grid"]
    sim_data = raw_dict["simulation"]
    bot_data = raw_dict["robot"]
    plan_data = raw_dict["planner"]

    config = SimulationConfig(
        window=WindowConfig(
            width=int(win_data["width"]),
            height=int(win_data["height"]),
            title=str(win_data.get("title", "AI Autonomous Navigation Simulation")),
        ),
        grid=GridConfig(
            cell_size=int(grid_data["cell_size"]),
            width_cells=int(grid_data["width_cells"]),
            height_cells=int(grid_data["height_cells"]),
        ),
        simulation=SimulationConfigData(
            fps=int(sim_data["fps"]),
            time_step=float(sim_data["time_step"]),
        ),
        robot=RobotConfig(
            radius=float(bot_data["radius"]),
            max_speed=float(bot_data["max_speed"]),
            sensor_range=float(bot_data["sensor_range"]),
        ),
        planner=PlannerConfig(
            algorithm=str(plan_data["algorithm"]),
            allow_diagonal=bool(plan_data.get("allow_diagonal", False)),
            heuristic=str(plan_data.get("heuristic", "manhattan")),
        ),
        raw_config=raw_dict,
    )

    config.validate()
    return config


def load_config(config_path: Path | str | None = None) -> SimulationConfig:
    """Convenience helper to load and validate simulation configuration."""
    raw = load_raw_config(config_path)
    return parse_and_validate_config(raw)

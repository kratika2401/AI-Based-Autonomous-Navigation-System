"""Application orchestration and package entry point.

Initializes configuration, coordinates simulation setup, and launches
the interactive autonomous navigation simulation.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional

from autonomous_nav.config import ConfigurationError, load_config
from autonomous_nav.simulator import Simulator


def parse_args(args: Optional[List[str]] = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "AI-Based Autonomous Navigation System "
            "- 2D Simulation Platform"
        )
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help=(
            "Path to a JSON configuration file "
            "(default: configs/default.json)"
        ),
    )

    return parser.parse_args(args)


def main(args: Optional[List[str]] = None) -> int:
    """Load configuration and launch the simulation."""
    print("=" * 65)
    print("AI-Based Autonomous Navigation System")
    print("2D Autonomous Robot Navigation Simulation")
    print("=" * 65)

    try:
        parsed = parse_args(args)

        config_path = Path(parsed.config) if parsed.config else None
        config = load_config(config_path)

        print("[OK] Configuration loaded and validated.")
        print(
            f"     Window: {config.window.width} x "
            f"{config.window.height} pixels"
        )
        print(
            f"     Grid: {config.grid.width_cells} x "
            f"{config.grid.height_cells} cells"
        )
        print(f"     Cell size: {config.grid.cell_size} pixels")
        print(
            f"     Robot radius: {config.robot.radius} pixels"
        )
        print(
            f"     Robot speed: {config.robot.max_speed} pixels/step"
        )
        print(
            f"     Sensor range: {config.robot.sensor_range} pixels"
        )
        print(
            f"     Planner: {config.planner.algorithm.upper()}"
        )

        simulator = Simulator(config)

        print("\n[INFO] Initializing simulator...")
        simulator.initialize()

        print("[OK] Simulator initialized.")
        print("[INFO] Starting interactive simulation.")
        print("[INFO] Close the simulation window to exit.")

        simulator.run()

        print("\n[OK] Simulation closed successfully.")
        return 0

    except (ConfigurationError, FileNotFoundError) as err:
        print(f"\n[ERROR] Configuration error: {err}", file=sys.stderr)
        return 1

    except KeyboardInterrupt:
        print("\n[INFO] Simulation interrupted by user.")
        return 0

    except Exception as err:
        print(
            f"\n[ERROR] Simulation failed: {err}",
            file=sys.stderr,
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())
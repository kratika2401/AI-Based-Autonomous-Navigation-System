# AI-Based Autonomous Navigation System

A Python-based 2D autonomous robot navigation simulator featuring
occupancy-grid mapping, simulated range sensing, obstacle detection,
A* path planning, motion control, and interactive Pygame visualization.

> **Scope:** This project simulates robot navigation in a 2D environment.
> It does not control physical robots or perform real-world navigation.

## 1. Features

- Configurable simulation parameters.
- 2D grid-based environment with static obstacles.
- Occupancy-grid representation of the environment.
- Simulated range sensing and obstacle detection.
- A* path planning with configurable heuristic settings.
- Dijkstra-based path-planning support for comparison.
- Navigation state management and path following.
- Collision-aware movement and obstacle avoidance.
- Interactive Pygame visualization.
- Automated unit tests using pytest.

## 2. Technology Stack

- **Language:** Python 3.10+
- **Numerical computing:** NumPy
- **Visualization:** Pygame
- **Path planning:** A* and Dijkstra
- **Testing:** pytest
- **Packaging:** setuptools with `pyproject.toml`

## 3. Project Structure

```text
AI-Based Autonomous Navigation System/
├── configs/             # Configuration files
├── data/                # Project data
├── docs/                # Technical documentation
├── images/              # Project images
├── notebooks/           # Experimental notebooks
├── outputs/             # Generated outputs
├── scripts/             # Utility scripts
├── src/
│   └── autonomous_nav/
│       ├── config.py
│       ├── controller.py
│       ├── environment.py
│       ├── main.py
│       ├── mapping.py
│       ├── metrics.py
│       ├── navigation.py
│       ├── obstacle_detection.py
│       ├── perception.py
│       ├── planner.py
│       └── simulator.py
├── tests/               # Automated tests
├── main.py
├── pyproject.toml
└── README.md
```

## 4. Requirements

- Python 3.10 or newer
- pip
- Pygame-compatible desktop environment for graphical simulation

The project dependencies are declared in `pyproject.toml`.

## 5. Installation

Open a terminal in the project directory.

### Windows PowerShell

Create a virtual environment:

```powershell
py -3.13 -m venv .venv
```

Activate it:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install the project and development dependencies:

```powershell
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

If Python 3.13 is not installed, use another compatible Python version
for which the required dependencies can be installed.

### Linux or macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"
```

## 6. Running the Simulator

From the project root, activate the virtual environment and run:

```powershell
python -m autonomous_nav.main
```

Alternatively, after installing the project, use:

```powershell
autonomous-nav
```

The graphical simulator displays the robot, goal, environment, and
navigation information.

### Controls

- **R:** Reset the simulation.
- **SPACE:** Pause or resume the simulation.
- **ESC:** Exit the simulator.

The robot attempts to reach the goal while avoiding obstacles.
When it reaches the goal, the simulation displays the `REACHED` state.

## 7. Running Tests

Run the complete test suite from the project root:

```powershell
python -m pytest
```

For verbose test output:

```powershell
python -m pytest -v
```

The tests cover configuration, environment behavior, mapping, perception,
obstacle detection, path planning, navigation, motion control, and metrics.

Some tests may be skipped when explicitly marked as skipped.

## 8. Navigation Overview

The system is organized into cooperating components:

1. **Configuration:** Loads simulation and navigation settings.
2. **Environment:** Represents the 2D world and its obstacles.
3. **Perception:** Simulates range measurements.
4. **Mapping:** Maintains an occupancy-grid representation.
5. **Obstacle detection:** Integrates sensor observations into the map.
6. **Path planning:** Computes routes through the grid.
7. **Navigation:** Manages mission progress and navigation states.
8. **Motion control:** Updates robot movement along the planned route.
9. **Visualization:** Displays the simulation and its current state.

## 9. Limitations

- The environment and sensors are simulated.
- The robot uses simplified 2D kinematics rather than a complete
  physical vehicle dynamics model.
- Sensor observations and obstacle representations are approximations.
- Performance in simulation does not establish safety or reliability
  for real-world autonomous vehicles.
- Physical robot hardware and real-world sensors are not controlled
  by this project.

## 10. Documentation

Additional technical documents are available in the `docs/` directory.
These include architecture, installation, simulation, algorithms,
testing, and results documentation.
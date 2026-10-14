# 2D Simulation Architecture

## 1. Architectural Model

The AI-Based Autonomous Navigation System operates as a modular, 2D simulation-first application. It clearly separates the physical ground truth of the world from the internal representations maintained by the autonomous agent.

```
+-------------------------------------------------------------------+
|                        Simulation Engine                          |
|                                                                   |
|   +--------------------------+     +--------------------------+   |
|   |  Ground-Truth World      |     |  Robot Internal Belief   |   |
|   |  (Environment)           |     |  (Occupancy Grid Map)    |   |
|   |  - Absolute boundaries   |     |  - Discovered cells      |   |
|   |  - True obstacle states  |     |  - Known free space      |   |
|   |  - True robot pose       |     |  - Unknown regions       |   |
|   +------------+-------------+     +-------------^------------+   |
|                |                                 |                |
|                | [Simulated Sensing]             | [Map Update]   |
|                v                                 |                |
|   +--------------------------+     +-------------+------------+   |
|   |   Perception Sensor      +---->|    Obstacle Detector     |   |
|   |   (Range/Raycasting)     |     |   (Evidence Fusion)      |   |
|   +--------------------------+     +--------------------------+   |
|                                                  |                |
|                                                  v                |
|                                    +--------------------------+   |
|                                    |   Path Planner Engine    |   |
|                                    |   (A* / Dijkstra)        |   |
|                                    +-------------+------------+   |
|                                                  |                |
|                                                  v [Waypoints]    |
|   +--------------------------+     +-------------+------------+   |
|   |   Kinematics & Physics   |<----+    Motion Controller     |   |
|   |   (Position / Velocity)  |     |    (Heading / Speed)     |   |
|   +--------------------------+     +--------------------------+   |
+-------------------------------------------------------------------+
```

---

## 2. Distinction: Ground-Truth Environment vs. Robot Map

A central tenet of the simulation design is the strict separation between:
1. **Ground-Truth Environment (`environment.py`)**:
   - Represents the objective reality of the simulated 2D world.
   - Contains all true static walls, barriers, and dynamic obstacle entities.
   - Manages physical collision checks and true robot position.
   - The robot cannot directly query this structure for planning; it must observe it through sensors.
2. **Robot Internal Map (`mapping.py`)**:
   - Represents the robot's subjective belief of its surroundings.
   - Initialized either as unknown or with prior static geometry.
   - Incrementally updated based only on simulated sensor readings filtered through `perception.py` and `obstacle_detection.py`.
   - The path planner plans trajectories strictly on this internal map.

---

## 3. Subsystem Breakdown

- **Configuration Layer (`config.py`):** Central source of validated simulation constants and hyperparameters.
- **Perception Subsystem (`perception.py`, `obstacle_detection.py`):** Simulates range sensing from robot pose into the environment and translates hits into occupancy evidence.
- **Planning Subsystem (`planner.py`):** Solves for the shortest collision-free grid path between start and goal using A* or Dijkstra.
- **Navigation State Machine (`navigation.py`):** Orchestrates high-level states (`IDLE`, `PLANNING`, `NAVIGATING`, `BLOCKED`, `REACHED`) and triggers dynamic replanning when path obstruction is detected.
- **Control Subsystem (`controller.py`):** Computes linear motion and heading commands along trajectory waypoints.
- **Simulation Loop (`simulator.py`):** Renders the 2D visualization via Pygame and coordinates timestep updates.
- **Metrics Subsystem (`metrics.py`):** Records benchmark telemetry including path length, compute duration, node expansions, and collision count.

---

## 4. Reference
For broader end-to-end AI and deep learning roadmap documentation, refer to [system_architecture.md](./system_architecture.md) and [modules_specification.md](./modules_specification.md).

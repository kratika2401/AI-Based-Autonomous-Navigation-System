# Modules Specification & Interfaces

This document outlines the functional modules, their specific responsibilities, input/output schemas, and interfaces.

---

## 1. Perception Module
- **Role:** Extracts semantic context, dynamic agents, and static obstacles from sensor streams.
- **Inputs:** Raw camera frames, depth maps / LiDAR ranges.
- **Outputs:** Bounding boxes, segmentation masks, occupancy grid map.
- **Interfaces:** `PerceptionEngine.process_frame(sensor_data) -> PerceptionOutput`

---

## 2. Localization & State Estimation Module
- **Role:** Tracks ego-vehicle pose and velocity relative to reference frames.
- **Inputs:** IMU, wheel encoders, visual odometry/GPS (if applicable).
- **Outputs:** Ego state matrix `[x, y, z, roll, pitch, yaw, vx, vy, vz]`.
- **Interfaces:** `StateEstimator.update(sensor_packet) -> VehicleState`

---

## 3. Global & Local Planner Module
- **Role:** Calculates route topology (Global) and dynamic collision-free trajectories (Local).
- **Inputs:** Goal coordinates, Occupancy costmap, VehicleState.
- **Outputs:** Trajectory waypoints `[(x_i, y_i, target_speed_i)]` with timestamp bounds.
- **Interfaces:** `Planner.plan_trajectory(current_state, goal, costmap) -> Trajectory`

---

## 4. Control Module
- **Role:** Drives vehicle along planned trajectory minimizing cross-track and heading errors.
- **Inputs:** Target trajectory, Current vehicle state.
- **Outputs:** Control signals `[steering, throttle, brake]`.
- **Interfaces:** `Controller.compute_controls(state, trajectory) -> ControlCommand`

---

## 5. Safety & Supervisor Module
- **Role:** Continuous sanity checker, collision risk detector, emergency braking trigger.
- **Inputs:** All module outputs, distance thresholds, hardware heartbeats.
- **Outputs:** Safety status flag, emergency override command.
- **Interfaces:** `SafetySupervisor.evaluate_risk(state, perception, commands) -> SafetyDecision`

# Data Flow & Pipeline Architecture

## 1. End-to-End Data Pipeline

The data flows sequentially across five critical transformation stages:

```
[Sensory Ingestion] -> [Preprocessing & Fusion] -> [Feature & State Representation] -> [Inference / Planning] -> [Actuation Output]
```

---

## 2. Detailed Transformation Stages

### Stage 1: Ingestion
- **Input Channels:** RGB frames, Depth arrays / Point clouds, IMU telemetry (accelerometer, gyroscope), Odometry readings.
- **Sampling Rates:** Synchronized at specified frequencies (e.g., 30 Hz vision, 100 Hz IMU).

### Stage 2: Preprocessing & Fusion
- **Vision:** Normalization, resizing, color-space conversion, artifact filtering.
- **Spatial:** Point cloud downsampling (Voxel Grid), ground-plane removal, coordinate transformations (Camera-to-Base, Base-to-World).
- **Fusion:** Temporal alignment and Kalman filtering for fused state estimation.

### Stage 3: Feature & State Representation
- Generation of 2D/3D costmaps and Bird's-Eye-View (BEV) semantic representations.
- State vector formulation: `[x, y, theta, v_x, v_y, omega]`.

### Stage 4: Policy Inference & Planning
- Forward pass of the neural navigation policy or algorithmic path solver.
- Production of waypoint sequence with velocity profiles.

### Stage 5: Actuation & Feedback Loop
- PID / MPC tracking computes command inputs `(steering_angle, throttle, brake)`.
- Telemetry feedback is returned to refine consecutive state predictions.

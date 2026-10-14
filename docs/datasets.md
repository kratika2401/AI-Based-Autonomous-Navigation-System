# Dataset Specifications & Preprocessing

## 1. Dataset Overview
This document specifies the datasets utilized for training, validation, and benchmarking the autonomous navigation algorithms.

---

## 2. Dataset Schema & Structure
```
dataset/
├── train/
│   ├── images/          # Synchronized camera frames (RGB / Grayscale)
│   ├── depth/           # Depth maps / disparity arrays
│   ├── telemetry.csv    # Timestamps, steer, throttle, brake, velocity, IMU
│   └── annotations.json # Bounding boxes, labels, route coordinates
├── val/
└── test/
```

---

## 3. Data Ingestion & Preprocessing
- **Resizing & Normalization:** Images normalized via ImageNet statistics (`mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`).
- **Telemetry Standardization:** Angle normalization to `[-1.0, 1.0]`, throttle/speed scaled to physical SI units (`m/s`).
- **Augmentation Techniques:**
  - Random brightness/contrast jittering to simulate varying lighting conditions.
  - Perspective shifts and horizontal mirroring (with inverted steering labels).
  - Synthetic shadow and weather injection.

---

## 4. Benchmark & Synthetic Sources
- **Simulated Environments:** CARLA, AirSim, Gazebo, or custom 2D grid/kinematic simulators.
- **Real-world Datasets (Optional/Reference):** nuScenes, KITTI, Udacity Self-Driving Car dataset.

# Model Architecture & Learning Dynamics

## 1. Learning Paradigms
The system supports multiple autonomous navigation paradigms depending on execution strategy:
1. **End-to-End Behavioral Cloning (Imitation Learning):** Mapping raw sensor inputs directly to control commands or waypoints using expert demonstration trajectories.
2. **Deep Reinforcement Learning (DRL):** Policy optimization in simulation (e.g., PPO / SAC) with reward functions penalizing collisions, jerk, and off-track deviation while rewarding target progress.
3. **Modular Deep Learning:** Specialized neural models for discrete tasks (e.g., YOLO/MobileNet for obstacle detection, semantic segmentation, and deep costmap forecasting).

---

## 2. Model Architecture
- **Backbone Network:** Convolutional / Vision Transformer (ViT) feature extractor for visual and spatial cues.
- **Temporal Modeling:** Recurrent (LSTM/GRU) or Temporal Convolutional layers (TCN) / Transformer blocks to preserve kinematic history across timesteps.
- **Action / Trajectory Head:** Fully connected layers with bounded activations (Tanh for normalized steering/throttle) or regression of future waypoints `(dx_t, dy_t)`.

---

## 3. Loss Functions & Objectives
- **Imitation / Trajectory Loss:**
  $$\mathcal{L}_{traj} = \frac{1}{N} \sum_{i=1}^N \| \hat{w}_i - w_i \|^2 + \lambda_{smooth} \|\Delta^2 \hat{w}\|_2$$
- **Safety / Proximity Penalty:**
  $$\mathcal{L}_{collision} = \max(0, d_{safe} - d_{obstacle})^2$$
- **Composite Objective:**
  $$\mathcal{L}_{total} = \mathcal{L}_{imitation} + \alpha \mathcal{L}_{safety} + \beta \mathcal{L}_{regularization}$$

---

## 4. Optimization & Training Loop
- **Optimizer:** AdamW with cosine learning rate scheduling and warmup.
- **Regularization:** Weight decay, Dropout, gradient clipping (max norm = 1.0).
- **Validation Checkpoint:** Evaluated on unseen validation tracks/environments against tracking error and zero-collision benchmarks.

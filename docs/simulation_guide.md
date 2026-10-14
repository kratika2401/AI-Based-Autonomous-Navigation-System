# 2D Simulation Guide

## 1. Overview
The autonomous navigation system runs as a 2D discrete-to-continuous simulation rendered using Pygame. The coordinate space translates between continuous pixel space and discrete grid cells.

---

## 2. Coordinate System & Geometry
- **Window Dimensions:** Default $800 \times 600$ pixels.
- **Grid Resolution:** $20 \times 20$ pixels per cell.
- **Grid Dimensions:** 40 columns $\times$ 30 rows.
- **Origin:** Top-left corner `(0, 0)` is index `[0, 0]`.
- **Conversion Equations:**
  $$\text{col} = \lfloor x / \text{cell\_size} \rfloor, \quad \text{row} = \lfloor y / \text{cell\_size} \rfloor$$
  $$x_{\text{center}} = \text{col} \times \text{cell\_size} + \frac{\text{cell\_size}}{2}, \quad y_{\text{center}} = \text{row} \times \text{cell\_size} + \frac{\text{cell\_size}}{2}$$

---

## 3. Visualization Elements (Planned)
When the Pygame simulator loop is executed:
- **Free Cells:** Light background color.
- **Obstacles (Static & Dynamic):** Dark gray / black rectangular blocks.
- **Robot:** Circular agent displaying current heading orientation ray.
- **Sensor Range:** Translucent arc or raycast lines representing range sensing.
- **Planned Path:** Colored line indicating planned waypoints.
- **Expanded Nodes (Debug Mode):** Tinted grid cells showing algorithm search frontier.

---

## 4. Simulation Execution
The simulation entry point is orchestrated through `autonomous_nav/main.py` and top-level `main.py`. Execution modes will support:
1. **Interactive Mode:** Pygame window with real-time visual feedback and manual obstacle toggling.
2. **Headless Batch Mode:** Fast-forward execution for batch testing and metrics aggregation without rendering overhead.

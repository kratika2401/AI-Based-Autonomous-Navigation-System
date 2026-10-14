# Path Planning & Navigation Algorithms

This document provides a beginner-friendly, comprehensive reference for the path planning algorithms in the **AI-Based Autonomous Navigation System**.

---

## 1. Overview & Planner Interface

The planning subsystem computes collision-free trajectories across the robot's [`OccupancyGrid`](file:///c:/Users/hp/OneDrive/Desktop/IIP/AI%20project/AI-Based%20Autonomous%20Navigation%20System/src/autonomous_nav/mapping.py#L48-L319). It operates strictly on the robot's internal knowledge without querying hidden ground-truth obstacles.

### Key API Methods
- **`plan(start, goal, grid, algorithm=None) -> Optional[List[Tuple[int, int]]]`**:
  Returns the ordered list of `(row, col)` cells from start to goal inclusive, or `None` if unreachable.
- **`plan_with_stats(start, goal, grid, algorithm=None) -> PlanResult`**:
  Returns a [`PlanResult`](file:///c:/Users/hp/OneDrive/Desktop/IIP/AI%20project/AI-Based%20Autonomous%20Navigation%20System/src/autonomous_nav/planner.py#L41-L59) containing the path, execution status, total path cost, nodes expanded, and nodes discovered.

---

## 2. Movement Model & Unit Costs

- **4-Connected Orthogonal Movement:**
  At every step, the agent can move only to its four direct neighbors:
  1. **Up:** `(-1, 0)`
  2. **Right:** `(0, 1)`
  3. **Down:** `(1, 0)`
  4. **Left:** `(0, -1)`
- **Unit Move Cost:** Every valid step incurs a uniform cost of `1.0`. Diagonal movement is not permitted in this phase.
- **Deterministic Neighbor Order:** Neighbors are always evaluated in the fixed order: `[Up, Right, Down, Left]`.

---

## 3. Supported Graph Search Algorithms

### A* (A-Star) Search
A* combines the accumulated path cost from the start with a heuristic estimating the remaining distance to the goal:

$$f(n) = g(n) + h(n)$$

- $g(n)$: Exact path cost accumulated from `start` to node $n$.
- $h(n)$: Estimated remaining distance from node $n$ to `goal`.
- **Manhattan Heuristic:** For 4-connected grid movement, the Manhattan distance is admissible and consistent:
  $$h(n) = |row_n - row_{goal}| + |col_n - col_{goal}|$$
  Because $h(n)$ never overestimates the true remaining step count, A* is mathematically guaranteed to return the **optimal shortest path**.

### Dijkstra's Algorithm
Dijkstra's algorithm evaluates nodes purely by their accumulated cost from the start:

$$f(n) = g(n) \quad (\text{equivalent to } h(n) = 0)$$

- It explores outward uniformly in all directions.
- While Dijkstra also guarantees finding the optimal path, it typically expands significantly more nodes than A* because it lacks directional goal guidance.

---

## 4. Occupancy & Unknown-Cell Policy

The planner uses the robot's [`OccupancyGrid`](file:///c:/Users/hp/OneDrive/Desktop/IIP/AI%20project/AI-Based%20Autonomous%20Navigation%20System/src/autonomous_nav/mapping.py#L48-L319) cell states:
1. **`CellState.FREE (0)`**: Fully traversable.
2. **`CellState.OCCUPIED (1)`**: Non-traversable (obstacle).
3. **`CellState.UNKNOWN (-1)`**: **Non-traversable by default** under the cautious policy (`allow_unknown=False`). The robot will not risk traversing unmapped territory unless explicitly configured with `allow_unknown=True`.

### Endpoint Validation
- `start` and `goal` coordinates must lie strictly within grid bounds ($0 \le row < rows$ and $0 \le col < cols$). Negative indices are rejected with `IndexError`.
- If `start` or `goal` are blocked by an obstacle or unknown cell under the active policy, a `ValueError` is raised immediately.
- If `start == goal`, the planner returns a trivial single-node path `[start]` with `cost = 0.0`.

---

## 5. Priority Queue & Deterministic Tie-Breaking

To guarantee repeatable, deterministic behavior across identical search queries, priority queue entries in `heapq` follow a 4-tuple:

```python
(f_score, h_score, insertion_counter, current_cell)
```

1. **Primary Key (`f_score`):** Nodes with lower total estimated cost are evaluated first.
2. **Secondary Key (`h_score`):** If two nodes share the same $f$-score, the node with lower $h$-score (closer to the goal) is prioritized.
3. **Tertiary Key (`insertion_counter`):** If both $f$ and $h$ are identical, a monotonically increasing FIFO counter preserves the exact discovery order without arbitrary comparisons.

---

## 6. Search Statistics & Telemetry

Each search returns a structured [`PlanResult`](file:///c:/Users/hp/OneDrive/Desktop/IIP/AI%20project/AI-Based%20Autonomous%20Navigation%20System/src/autonomous_nav/planner.py#L41-L59) with:
- **`path`**: Ordered list of coordinates `[(r0, c0), ..., (rG, cG)]`, or `None` if unreachable.
- **`success`**: `True` if a path was found; `False` if unreachable.
- **`cost`**: Total unit steps taken ($len(path) - 1$), or `inf` if unreachable.
- **`nodes_expanded`**: Number of unique nodes popped from the open set whose neighbors were evaluated.
- **`nodes_discovered`**: Total unique nodes visited and placed in the open set.

---

## 7. Unreachable-Goal Behavior

When obstacles completely surround the goal or divide the map, the priority queue empties without reaching the goal. The planner handles this cleanly:
- `path = None`
- `success = False`
- `cost = float("inf")`
- `nodes_expanded` records how many nodes were explored before exhaustion.
- No partial or corrupted paths are returned.

"""Navigation and path planning performance metrics collection.

This module aggregates objective performance metrics such as total path length,
computational planning duration, nodes expanded, execution steps, and collision counts.

Note:
    Automated benchmark export and aggregation across map batches are scheduled
    for implementation in Phase 7.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass
class NavigationMetrics:
    """Stores performance metrics for a navigation run."""

    path_length: float = 0.0
    planning_time_ms: float = 0.0
    nodes_expanded: int = 0
    steps_taken: int = 0
    collision_count: int = 0
    success: bool = False
    custom_metrics: Dict[str, float] = field(default_factory=dict)

    @staticmethod
    def calculate_path_length(path: List[Tuple[float, float]]) -> float:
        """Calculates total Euclidean distance along a series of continuous coordinates."""
        if len(path) < 2:
            return 0.0
        total_dist = 0.0
        for i in range(len(path) - 1):
            p1, p2 = path[i], path[i + 1]
            total_dist += math.hypot(p2[0] - p1[0], p2[1] - p1[1])
        return total_dist

    def to_dict(self) -> Dict[str, float | int | bool]:
        """Serializes metric records into a dictionary."""
        return {
            "path_length": self.path_length,
            "planning_time_ms": self.planning_time_ms,
            "nodes_expanded": self.nodes_expanded,
            "steps_taken": self.steps_taken,
            "collision_count": self.collision_count,
            "success": self.success,
        }

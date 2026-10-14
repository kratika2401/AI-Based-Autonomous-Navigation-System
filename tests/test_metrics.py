"""Unit tests for navigation performance metrics."""

import math
import pytest

from autonomous_nav.metrics import NavigationMetrics


def test_metrics_empty_path_length():
    """Verify that path length of empty or single-point paths is 0.0."""
    assert NavigationMetrics.calculate_path_length([]) == 0.0
    assert NavigationMetrics.calculate_path_length([(10.0, 10.0)]) == 0.0


def test_metrics_euclidean_path_length():
    """Verify accurate calculation of multi-point path length (3-4-5 right triangle)."""
    # Triangle: (0,0) -> (3,0) [length 3] -> (3,4) [length 4] -> (0,0) [length 5] => Total = 12.0
    path = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0), (0.0, 0.0)]
    calculated_length = NavigationMetrics.calculate_path_length(path)
    assert math.isclose(calculated_length, 12.0, rel_tol=1e-5)


def test_metrics_to_dict_structure():
    """Verify that metric records serialize to dictionary with expected keys."""
    metrics = NavigationMetrics(
        path_length=25.5,
        planning_time_ms=4.2,
        nodes_expanded=120,
        steps_taken=45,
        collision_count=0,
        success=True,
    )
    data = metrics.to_dict()
    assert data["path_length"] == 25.5
    assert data["planning_time_ms"] == 4.2
    assert data["nodes_expanded"] == 120
    assert data["steps_taken"] == 45
    assert data["collision_count"] == 0
    assert data["success"] is True


@pytest.mark.skip(reason="Planned for Phase 7: Automated metrics benchmarking and CSV/JSON exporter")
def test_metrics_batch_export_planned():
    """Placeholder documenting planned batch export tests in Phase 7."""
    pass

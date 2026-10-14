# Testing Strategy & Guidelines

## 1. Testing Philosophy
The testing strategy is designed to ensure rigorous verification without using trivial or fake passing assertions. Each test must assert concrete behavior, validation contracts, or mathematically verifiable outcomes.

---

## 2. Test Organization

Tests reside in the `tests/` directory and mirror the package architecture:

```
tests/
├── __init__.py
├── test_config.py        # Configuration loading, types, and schema validation
├── test_environment.py   # Ground-truth world boundaries, obstacle additions, collision queries
├── test_mapping.py       # Occupancy grid representation, bounds, coordinate conversions
├── test_planner.py       # A* and Dijkstra path correctness, optimality, unreachable goals
├── test_navigation.py    # State transitions, replanning triggers, waypoint sequences
└── test_metrics.py       # KPI calculations (path length, execution duration, metrics export)
```

---

## 3. Running Tests

Run all unit and integration tests:
```bash
pytest
```

Run tests with verbose output:
```bash
pytest -v
```

Run a specific test suite:
```bash
pytest tests/test_config.py
```

---

## 4. Test Guidelines
- **No Dummy Assertions:** Assertions such as `assert True` are strictly prohibited.
- **Unimplemented Modules:** Tests for modules that have not yet been implemented in the current phase must be explicitly marked with `@pytest.mark.skip(reason="Planned for Phase X: <Feature>")` rather than writing false passing tests.
- **Parametrization:** Use `@pytest.mark.parametrize` to test boundary conditions, invalid inputs, and corner cases across grid coordinates.

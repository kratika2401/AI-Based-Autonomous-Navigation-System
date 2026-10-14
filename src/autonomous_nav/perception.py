"""Simulated range sensing and raycasting observations.

This module provides a deterministic, simulation-only 2D range sensor (e.g. simulated
LiDAR / ultrasonic proximity scanner). The sensor casts directional rays across the
ground-truth Environment using 2D Digital Differential Analysis (DDA) grid traversal
and returns structured observations.

Coordinate System & Conventions:
    - Sensor origin: continuous pixel coordinates (origin_x, origin_y) in [0.0, width) x [0.0, height).
    - Ray angles: radians, measured clockwise from positive x-axis (0 rad = right, pi/2 rad = down).
    - Grid traversal: integer (row, col) cells checked for occupancy against Environment.
    - Start inside obstacle convention:
        If a ray originates inside an already occupied cell, it terminates immediately
        at distance 0.0, reporting hit=True, hit_point=(origin_x, origin_y),
        hit_cell=(start_row, start_col), and an empty traversed_cells tuple.

Separation of Concerns:
    The RangeSensor queries the ground-truth Environment to simulate physical sensing.
    It does not write to or mutate the robot's OccupancyGrid.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, List, Optional, Sequence, Tuple

from autonomous_nav.environment import Environment

try:
    from autonomous_nav.config import SimulationConfig
except ImportError:  # pragma: no cover
    SimulationConfig = Any  # type: ignore


@dataclass(frozen=True)
class RayObservation:
    """Structured result of a single range sensor raycast.

    Attributes:
        angle_rad: Direction angle in radians.
        distance: Measured distance in pixels to the obstacle surface, or max_range if no hit.
        hit: True if an obstacle was intercepted within max_range, False otherwise.
        hit_point: Continuous pixel coordinates (x, y) where the ray struck the obstacle,
            or None if no obstacle was detected.
        hit_cell: Grid indices (row, col) of the first occupied cell struck, or None if no hit.
        traversed_cells: Ordered sequence of (row, col) grid cells traversed through free
            space prior to the hit or range termination. Never includes hit_cell.
    """

    angle_rad: float
    distance: float
    hit: bool
    hit_point: Optional[Tuple[float, float]]
    hit_cell: Optional[Tuple[int, int]]
    traversed_cells: Tuple[Tuple[int, int], ...]


class RangeSensor:
    """Deterministic 2D range sensor simulating directional and radial raycasts."""

    def __init__(
        self,
        max_range: float,
        num_rays: int = 16,
        fov_rad: Optional[float] = None,
    ) -> None:
        """Initializes sensor specifications.

        Args:
            max_range: Maximum detection distance in pixels (must be > 0).
            num_rays: Number of rays in a full multi-ray scan (must be > 0).
            fov_rad: Field of view in radians. If None, defaults to full 360-degree circle (2*pi).

        Raises:
            TypeError: If arguments are of invalid types.
            ValueError: If numerical parameters are non-positive or non-finite.
        """
        if not isinstance(max_range, (int, float)) or isinstance(max_range, bool):
            raise TypeError(f"max_range must be a numeric value, got {type(max_range).__name__}")
        if not isinstance(num_rays, int) or isinstance(num_rays, bool):
            raise TypeError(f"num_rays must be an integer, got {type(num_rays).__name__}")

        if math.isnan(max_range) or math.isinf(max_range) or max_range <= 0.0:
            raise ValueError(f"max_range must be positive and finite, got {max_range}")
        if num_rays <= 0:
            raise ValueError(f"num_rays must be positive, got {num_rays}")

        if fov_rad is not None:
            if not isinstance(fov_rad, (int, float)) or isinstance(fov_rad, bool):
                raise TypeError(f"fov_rad must be a numeric value or None, got {type(fov_rad).__name__}")
            if math.isnan(fov_rad) or math.isinf(fov_rad) or fov_rad <= 0.0 or fov_rad > 2.0 * math.pi + 1e-6:
                raise ValueError(f"fov_rad must be in range (0.0, 2*pi], got {fov_rad}")
            self.fov_rad: float = float(fov_rad)
        else:
            self.fov_rad = 2.0 * math.pi

        self.max_range: float = float(max_range)
        self.num_rays: int = num_rays

    @classmethod
    def from_config(cls, config: SimulationConfig, num_rays: int = 16) -> RangeSensor:
        """Constructs a RangeSensor using parameters from SimulationConfig."""
        return cls(
            max_range=config.robot.sensor_range,
            num_rays=num_rays,
        )

    # -------------------------------------------------------------------------
    # Core Raycasting Engine
    # -------------------------------------------------------------------------

    def cast_ray(
        self,
        origin_x: float,
        origin_y: float,
        angle_rad: float,
        environment: Environment,
    ) -> RayObservation:
        """Casts a single ray in direction angle_rad through the environment.

        Uses 2D Digital Differential Analysis (DDA) to traverse grid cells along the line
        segment from (origin_x, origin_y) up to max_range.

        Args:
            origin_x: Continuous horizontal start coordinate in pixels.
            origin_y: Continuous vertical start coordinate in pixels.
            angle_rad: Direction angle in radians.
            environment: Ground-truth Environment instance to query for obstacles.

        Returns:
            Structured RayObservation containing hit status, distance, hit coordinates,
            hit cell, and traversed free cells.

        Raises:
            TypeError: If origin, angle, or environment are of invalid types.
            ValueError: If angle is non-finite.
            IndexError: If origin is outside environment boundaries.
        """
        if not isinstance(origin_x, (int, float)) or isinstance(origin_x, bool):
            raise TypeError(f"origin_x must be numeric, got {type(origin_x).__name__}")
        if not isinstance(origin_y, (int, float)) or isinstance(origin_y, bool):
            raise TypeError(f"origin_y must be numeric, got {type(origin_y).__name__}")
        if not isinstance(angle_rad, (int, float)) or isinstance(angle_rad, bool):
            raise TypeError(f"angle_rad must be numeric, got {type(angle_rad).__name__}")
        if math.isnan(angle_rad) or math.isinf(angle_rad):
            raise ValueError(f"angle_rad must be finite, got {angle_rad}")
        if not isinstance(environment, Environment):
            raise TypeError(f"environment must be an Environment instance, got {type(environment).__name__}")

        orig_x = float(origin_x)
        orig_y = float(origin_y)
        angle = float(angle_rad)

        if not environment.is_point_within_bounds(orig_x, orig_y):
            raise IndexError(
                f"Ray origin ({orig_x}, {orig_y}) is outside world bounds "
                f"[0.0, {environment.width}) x [0.0, {environment.height})."
            )

        start_row, start_col = environment.point_to_grid(orig_x, orig_y)

        # Policy: Ray starting inside an occupied cell terminates immediately at origin
        if environment.is_occupied(start_row, start_col):
            return RayObservation(
                angle_rad=angle,
                distance=0.0,
                hit=True,
                hit_point=(orig_x, orig_y),
                hit_cell=(start_row, start_col),
                traversed_cells=(),
            )

        # Traversed free space cells along the ray (excludes hit_cell)
        traversed: List[Tuple[int, int]] = [(start_row, start_col)]

        cell_size = float(environment.cell_size)
        dx = math.cos(angle)
        dy = math.sin(angle)

        eps = 1e-12
        if abs(dx) < eps:
            dx = 0.0
        if abs(dy) < eps:
            dy = 0.0

        curr_col = start_col
        curr_row = start_row

        # Setup step and boundary distances for horizontal axis
        if dx > 0.0:
            step_x = 1
            x_boundary = (curr_col + 1) * cell_size
            t_max_x = (x_boundary - orig_x) / dx
            t_delta_x = cell_size / dx
        elif dx < 0.0:
            step_x = -1
            x_boundary = curr_col * cell_size
            t_max_x = (x_boundary - orig_x) / dx
            t_delta_x = cell_size / (-dx)
        else:
            step_x = 0
            t_max_x = float("inf")
            t_delta_x = float("inf")

        # Setup step and boundary distances for vertical axis
        if dy > 0.0:
            step_y = 1
            y_boundary = (curr_row + 1) * cell_size
            t_max_y = (y_boundary - orig_y) / dy
            t_delta_y = cell_size / dy
        elif dy < 0.0:
            step_y = -1
            y_boundary = curr_row * cell_size
            t_max_y = (y_boundary - orig_y) / dy
            t_delta_y = cell_size / (-dy)
        else:
            step_y = 0
            t_max_y = float("inf")
            t_delta_y = float("inf")

        # Guard against slight numerical negative t_max
        if t_max_x < 0.0:
            t_max_x = 0.0
        if t_max_y < 0.0:
            t_max_y = 0.0

        hit_occurred = False
        hit_dist = self.max_range
        hit_pt: Optional[Tuple[float, float]] = None
        hit_grid_cell: Optional[Tuple[int, int]] = None

        # 2D DDA grid step loop
        while True:
            if t_max_x < t_max_y:
                t_next = t_max_x
                if t_next > self.max_range:
                    break
                curr_col += step_x
                t_max_x += t_delta_x
                t_event = t_next
            elif t_max_y < t_max_x:
                t_next = t_max_y
                if t_next > self.max_range:
                    break
                curr_row += step_y
                t_max_y += t_delta_y
                t_event = t_next
            else:
                # Diagonal boundary crossing
                t_next = t_max_x
                if t_next > self.max_range:
                    break
                curr_col += step_x
                curr_row += step_y
                t_max_x += t_delta_x
                t_max_y += t_delta_y
                t_event = t_next

            # If the ray steps outside the environment boundary, it terminates without a hit
            if not environment.is_within_bounds(curr_row, curr_col):
                break

            # If the cell entered is occupied, record hit and terminate ray
            if environment.is_occupied(curr_row, curr_col):
                hit_occurred = True
                hit_dist = t_event
                hit_grid_cell = (curr_row, curr_col)
                hit_pt = (orig_x + t_event * dx, orig_y + t_event * dy)
                break
            else:
                traversed.append((curr_row, curr_col))

        return RayObservation(
            angle_rad=angle,
            distance=hit_dist,
            hit=hit_occurred,
            hit_point=hit_pt,
            hit_cell=hit_grid_cell,
            traversed_cells=tuple(traversed),
        )

    # -------------------------------------------------------------------------
    # Multi-Ray Scanning
    # -------------------------------------------------------------------------

    def get_ray_angles(self, heading_rad: float = 0.0) -> List[float]:
        """Calculates ray angles distributed according to sensor FOV and ray count."""
        if not isinstance(heading_rad, (int, float)) or isinstance(heading_rad, bool):
            raise TypeError(f"heading_rad must be numeric, got {type(heading_rad).__name__}")
        if math.isnan(heading_rad) or math.isinf(heading_rad):
            raise ValueError(f"heading_rad must be finite, got {heading_rad}")

        heading = float(heading_rad)
        if self.num_rays == 1:
            return [heading]

        # Full 360-degree omnidirectional scan
        if abs(self.fov_rad - 2.0 * math.pi) < 1e-6:
            step = (2.0 * math.pi) / self.num_rays
            return [heading + i * step for i in range(self.num_rays)]

        # Sector / directional scan across [heading - fov/2, heading + fov/2]
        half_fov = self.fov_rad / 2.0
        step = self.fov_rad / (self.num_rays - 1)
        return [heading - half_fov + i * step for i in range(self.num_rays)]

    def scan(
        self,
        origin_x: float,
        origin_y: float,
        environment: Environment,
        heading_rad: float = 0.0,
    ) -> List[RayObservation]:
        """Performs a multi-ray scan across the environment.

        Args:
            origin_x: Horizontal robot sensor origin in pixels.
            origin_y: Vertical robot sensor origin in pixels.
            environment: Ground-truth Environment instance.
            heading_rad: Center heading angle in radians (default: 0.0).

        Returns:
            List of RayObservation objects, one per ray angle.
        """
        angles = self.get_ray_angles(heading_rad)
        return [
            self.cast_ray(origin_x, origin_y, angle, environment)
            for angle in angles
        ]

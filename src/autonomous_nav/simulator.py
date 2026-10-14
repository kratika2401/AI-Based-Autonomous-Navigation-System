"""Interactive 2D autonomous navigation simulator."""

from __future__ import annotations

import math
from typing import Any, Optional, Tuple

from autonomous_nav.config import SimulationConfig
from autonomous_nav.environment import Environment
from autonomous_nav.mapping import CellState, OccupancyGrid
from autonomous_nav.perception import RangeSensor
from autonomous_nav.obstacle_detection import ObstacleDetector
from autonomous_nav.planner import PathPlanner
from autonomous_nav.controller import MotionController
from autonomous_nav.navigation import NavigationManager, NavigationState


class Simulator:
    """Runs and displays the autonomous navigation simulation."""

    def __init__(self, config: SimulationConfig) -> None:
        self.config = config
        self.is_running = False
        self.is_paused = False
        self.step_count = 0
        self.collision_count = 0

        self.environment: Optional[Environment] = None
        self.grid: Optional[OccupancyGrid] = None
        self.sensor: Optional[RangeSensor] = None
        self.detector: Optional[ObstacleDetector] = None
        self.planner: Optional[PathPlanner] = None
        self.controller: Optional[MotionController] = None
        self.navigation: Optional[NavigationManager] = None

        self.robot_x = 0.0
        self.robot_y = 0.0
        self.start_cell: Tuple[int, int] = (0, 0)
        self.goal_cell: Tuple[int, int] = (0, 0)

        self.observations: list[Any] = []
        self.screen: Any = None
        self.clock: Any = None
        self.font: Any = None
        self.small_font: Any = None
        self._pygame: Any = None

    def initialize(self) -> None:
        """Initialize the environment and navigation components."""
        self.step_count = 0
        self.collision_count = 0
        self.is_paused = False

        self.environment = Environment.from_config(self.config)

        rows = self.environment.rows
        cols = self.environment.cols

        self.start_cell = (rows // 2, 2)
        self.goal_cell = (rows // 2, cols - 3)

        self._create_obstacles()

        self.environment.validate_start_goal(
            self.start_cell,
            self.goal_cell,
        )

        self.grid = OccupancyGrid.from_config(self.config)

        self.sensor = RangeSensor(
            max_range=self.config.robot.sensor_range,
            num_rays=32,
        )

        self.detector = ObstacleDetector.from_config(self.config)

        self.planner = PathPlanner.from_config(
            self.config,
            allow_unknown=True,
        )

        self.controller = MotionController.from_config(self.config)

        self.navigation = NavigationManager.from_config(
            self.config,
            self.grid,
            allow_unknown=True,
        )

        self.robot_x, self.robot_y = self.environment.grid_to_point(
            *self.start_cell
        )

        self.observations = []
        self._sense_and_update_map()

        self.navigation.start_mission(
            start_grid=self.start_cell,
            goal_grid=self.goal_cell,
            robot_pixel_pos=(self.robot_x, self.robot_y),
        )

        self.is_running = True

    def _create_obstacles(self) -> None:
        """Create a deterministic obstacle course."""
        assert self.environment is not None

        rows = self.environment.rows
        cols = self.environment.cols
        middle_row = rows // 2

        if rows < 12 or cols < 20:
            return

        first_col = cols // 3

        for row in range(3, rows - 3):
            if row not in (middle_row - 5, middle_row - 4):
                self.environment.add_static_obstacle(row, first_col)

        second_col = (2 * cols) // 3

        for row in range(2, rows - 2):
            if row not in (middle_row + 4, middle_row + 5):
                self.environment.add_static_obstacle(row, second_col)

        obstacle_row = min(rows - 4, middle_row + 5)
        obstacle_start = first_col + 2
        obstacle_end = min(cols - 4, second_col - 2)

        for col in range(obstacle_start, obstacle_end):
            if col not in (cols // 2, cols // 2 + 1):
                self.environment.add_static_obstacle(
                    obstacle_row,
                    col,
                )

        self.environment.static_obstacles.discard(self.start_cell)
        self.environment.static_obstacles.discard(self.goal_cell)

    def _sense_and_update_map(self) -> None:
        """Update the occupancy map from sensor observations."""
        assert self.environment is not None
        assert self.sensor is not None
        assert self.detector is not None
        assert self.grid is not None

        self.observations = self.sensor.scan(
            self.robot_x,
            self.robot_y,
            self.environment,
        )

        self.detector.integrate_observations(
            self.observations,
            self.grid,
        )

    def _position_is_safe(self, x: float, y: float) -> bool:
        """Check the robot footprint against ground-truth obstacles."""
        assert self.environment is not None

        radius = float(self.config.robot.radius)
        cell_size = float(self.config.grid.cell_size)

        # Keep the robot fully inside the window.
        if (
            x - radius < 0
            or y - radius < 0
            or x + radius > self.config.window.width
            or y + radius > self.config.window.height
        ):
            return False

        # Check circular robot footprint against obstacle rectangles.
        for row, col in self.environment.static_obstacles:
            left = col * cell_size
            right = left + cell_size
            top = row * cell_size
            bottom = top + cell_size

            closest_x = min(max(x, left), right)
            closest_y = min(max(y, top), bottom)

            dx = x - closest_x
            dy = y - closest_y

            if dx * dx + dy * dy < radius * radius:
                return False

        # Check grid bounds directly without calling unsupported
        # Environment methods.
        col = int(x // cell_size)
        row = int(y // cell_size)

        if not (
            0 <= row < self.environment.rows
            and 0 <= col < self.environment.cols
        ):
            return False

        return True

    def _move_with_collision_check(
        self,
        proposed_x: float,
        proposed_y: float,
    ) -> None:
        """Reject unsafe movement and try sliding along an obstacle."""
        old_x = self.robot_x
        old_y = self.robot_y

        dx = proposed_x - old_x
        dy = proposed_y - old_y
        distance = math.hypot(dx, dy)

        spacing = max(1.0, self.config.robot.radius / 2.0)
        sample_count = max(1, math.ceil(distance / spacing))

        for index in range(1, sample_count + 1):
            fraction = index / sample_count

            sample_x = old_x + dx * fraction
            sample_y = old_y + dy * fraction

            if not self._position_is_safe(sample_x, sample_y):
                self.collision_count += 1

                if self._position_is_safe(proposed_x, old_y):
                    self.robot_x = proposed_x
                    self.robot_y = old_y
                elif self._position_is_safe(old_x, proposed_y):
                    self.robot_x = old_x
                    self.robot_y = proposed_y
                else:
                    self.robot_x = old_x
                    self.robot_y = old_y

                return

        self.robot_x = proposed_x
        self.robot_y = proposed_y

    def step(self) -> None:
        """Advance the simulation by one step."""
        if not self.is_running:
            return

        assert self.navigation is not None

        self.step_count += 1

        if self.navigation.state in (
            NavigationState.REACHED,
            NavigationState.FAILED,
            NavigationState.BLOCKED,
        ):
            self._sense_and_update_map()
            return

        self._sense_and_update_map()

        result = self.navigation.update(
            (self.robot_x, self.robot_y),
            dt=1.0,
        )

        if result is None:
            return

        if hasattr(result, "new_position"):
            proposed_x, proposed_y = result.new_position
        else:
            proposed_x, proposed_y = result

        self._move_with_collision_check(
            float(proposed_x),
            float(proposed_y),
        )

    def run(self) -> None:
        """Launch Pygame and run the simulation."""
        if not self.is_running:
            self.initialize()

        try:
            import pygame
        except ImportError as exc:
            raise RuntimeError(
                "Pygame is required. Install it with "
                "'python -m pip install pygame'."
            ) from exc

        self._pygame = pygame
        pygame.init()

        self.screen = pygame.display.set_mode(
            (
                self.config.window.width,
                self.config.window.height,
            )
        )

        pygame.display.set_caption(self.config.window.title)

        self.clock = pygame.time.Clock()
        self.font = pygame.font.Font(None, 25)
        self.small_font = pygame.font.Font(None, 19)

        try:
            while self.is_running:
                self._handle_events()

                if not self.is_running:
                    break

                if not self.is_paused:
                    self.step()

                self._draw()
                self.clock.tick(self.config.simulation.fps)

        finally:
            pygame.quit()
            self.screen = None

    def _handle_events(self) -> None:
        """Handle window events and keyboard controls."""
        pygame = self._pygame

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.is_running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    self.is_running = False
                elif event.key == pygame.K_SPACE:
                    self.is_paused = not self.is_paused
                elif event.key == pygame.K_r:
                    self.initialize()

    def _draw(self) -> None:
        """Render the occupancy grid, route, sensor rays, and robot."""
        pygame = self._pygame

        assert self.screen is not None
        assert self.grid is not None
        assert self.navigation is not None
        assert self.environment is not None

        screen = self.screen
        cell_size = self.config.grid.cell_size
        rows = self.config.grid.height_cells
        cols = self.config.grid.width_cells

        screen.fill((24, 28, 36))

        # Occupancy grid.
        for row in range(rows):
            for col in range(cols):
                state = self.grid.get_cell(row, col)

                if state == CellState.UNKNOWN:
                    color = (49, 57, 70)
                elif state == CellState.FREE:
                    color = (224, 230, 235)
                else:
                    color = (220, 75, 75)

                rect = pygame.Rect(
                    col * cell_size,
                    row * cell_size,
                    cell_size,
                    cell_size,
                )

                pygame.draw.rect(screen, color, rect)
                pygame.draw.rect(
                    screen,
                    (115, 125, 137),
                    rect,
                    width=1,
                )

        # Planned path.
        path = self.navigation.active_path

        if path:
            points = [
                self.environment.grid_to_point(row, col)
                for row, col in path
            ]

            if len(points) >= 2:
                pygame.draw.lines(
                    screen,
                    (40, 125, 235),
                    False,
                    [(int(x), int(y)) for x, y in points],
                    width=3,
                )

        # Sensor rays.
        for observation in self.observations:
            end_x = (
                self.robot_x
                + math.cos(observation.angle_rad)
                * observation.distance
            )
            end_y = (
                self.robot_y
                + math.sin(observation.angle_rad)
                * observation.distance
            )

            color = (
                (240, 110, 90)
                if observation.hit
                else (80, 170, 190)
            )

            pygame.draw.line(
                screen,
                color,
                (int(self.robot_x), int(self.robot_y)),
                (int(end_x), int(end_y)),
                width=1,
            )

        # Goal.
        goal_x, goal_y = self.environment.grid_to_point(
            *self.goal_cell
        )

        pygame.draw.circle(
            screen,
            (245, 190, 55),
            (int(goal_x), int(goal_y)),
            max(5, cell_size // 3),
        )

        # Robot.
        pygame.draw.circle(
            screen,
            (45, 205, 120),
            (int(self.robot_x), int(self.robot_y)),
            self.config.robot.radius,
        )

        # Status panel.
        panel = pygame.Surface(
            (self.config.window.width, 76)
        )
        panel.set_alpha(225)
        panel.fill((20, 24, 32))
        screen.blit(panel, (0, 0))

        state_text = (
            "PAUSED"
            if self.is_paused
            else self.navigation.state.name
        )

        line1 = (
            f"Autonomous Navigation | State: {state_text} "
            f"| Steps: {self.step_count}"
        )

        unknown_count = 0
        free_count = 0
        occupied_count = 0

        for row in range(rows):
            for col in range(cols):
                state = self.grid.get_cell(row, col)

                if state == CellState.UNKNOWN:
                    unknown_count += 1
                elif state == CellState.FREE:
                    free_count += 1
                else:
                    occupied_count += 1

        line2 = (
            f"Map: {free_count} free | "
            f"{occupied_count} occupied | "
            f"{unknown_count} unknown"
        )

        line3 = (
            f"Blocked moves: {self.collision_count} "
            "| SPACE: Pause  R: Reset  ESC: Quit"
        )

        screen.blit(
            self.font.render(line1, True, (245, 245, 245)),
            (10, 7),
        )
        screen.blit(
            self.small_font.render(
                line2,
                True,
                (205, 215, 225),
            ),
            (10, 34),
        )
        screen.blit(
            self.small_font.render(
                line3,
                True,
                (205, 215, 225),
            ),
            (10, 53),
        )

        pygame.display.flip()
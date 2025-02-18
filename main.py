import pygame
import random
import numpy as np
import time


class RoombaEnv:
    def __init__(self, width=800, height=600):
        pygame.init()
        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Roomba Simulation")

        # Initialize font for display
        self.font = pygame.font.Font(None, 36)

        # Roomba properties
        self.roomba_radius = 20
        self.roomba_pos = np.array([width // 2, height // 2], dtype=float)
        self.roomba_speed = 5

        # Battery properties
        self.battery_level = 100
        self.steps_taken = 0
        self.steps_per_battery_decrease = 100
        self.battery_decrease_amount = 5
        self.charging_rate = 2  # 2% per 5 seconds
        self.last_charge_time = time.time()

        # Charging station
        self.charging_station_pos = np.array([50, 50])  # Top-left corner
        self.charging_station_size = 40

        # Walls
        self.walls = []
        self.generate_walls()

        # Dirt particles
        self.dirt_particles = []
        self.dirt_radius = 3
        self.generate_dirt(100)

        # Reward system
        self.score = 0
        self.points_per_dirt = 10

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.BROWN = (139, 69, 19)
        self.RED = (255, 0, 0)
        self.GRAY = (128, 128, 128)
        self.GREEN = (0, 255, 0)
        self.YELLOW = (255, 255, 0)

        self.clock = pygame.time.Clock()

    def generate_walls(self):
        """Generate walls in the room"""
        wall_width = 20

        # Outer walls
        self.walls = [
            pygame.Rect(0, 0, self.width, wall_width),  # Top
            pygame.Rect(0, self.height - wall_width, self.width, wall_width),  # Bottom
            pygame.Rect(0, 0, wall_width, self.height),  # Left
            pygame.Rect(self.width - wall_width, 0, wall_width, self.height),  # Right
        ]

        # Add some inner walls
        self.walls.extend(
            [
                pygame.Rect(200, 150, wall_width, 300),  # Vertical wall
                pygame.Rect(400, 200, 200, wall_width),  # Horizontal wall
                pygame.Rect(600, 400, wall_width, 200),  # Another vertical wall
            ]
        )

    def is_at_charging_station(self):
        """Check if Roomba is at the charging station"""
        distance = np.linalg.norm(self.roomba_pos - self.charging_station_pos)
        return distance < (self.roomba_radius + self.charging_station_size / 2)

    def update_battery(self):
        """Update battery level based on movement and charging"""
        # Decrease battery based on steps
        if self.steps_taken >= self.steps_per_battery_decrease:
            self.battery_level = max(
                0, self.battery_level - self.battery_decrease_amount
            )
            self.steps_taken = 0

        # Charge battery if at charging station
        if self.is_at_charging_station():
            current_time = time.time()
            if current_time - self.last_charge_time >= 5:  # Every 5 seconds
                self.battery_level = min(100, self.battery_level + self.charging_rate)
                self.last_charge_time = current_time

    def generate_dirt(self, num_particles):
        """Generate random dirt particles avoiding walls"""
        self.dirt_particles = []
        for _ in range(num_particles):
            while True:
                x = random.randint(self.dirt_radius, self.width - self.dirt_radius)
                y = random.randint(self.dirt_radius, self.height - self.dirt_radius)
                dirt_rect = pygame.Rect(
                    x - self.dirt_radius,
                    y - self.dirt_radius,
                    self.dirt_radius * 2,
                    self.dirt_radius * 2,
                )

                if not any(wall.colliderect(dirt_rect) for wall in self.walls):
                    self.dirt_particles.append([x, y])
                    break

    def check_collision(self, new_pos):
        """Check if new position collides with walls"""
        roomba_rect = pygame.Rect(
            new_pos[0] - self.roomba_radius,
            new_pos[1] - self.roomba_radius,
            self.roomba_radius * 2,
            self.roomba_radius * 2,
        )
        return any(wall.colliderect(roomba_rect) for wall in self.walls)

    def clean_dirt(self):
        """Clean dirt particles that the Roomba is over"""
        if self.battery_level > 0:  # Only clean if battery isn't dead
            roomba_rect = pygame.Rect(
                self.roomba_pos[0] - self.roomba_radius,
                self.roomba_pos[1] - self.roomba_radius,
                self.roomba_radius * 2,
                self.roomba_radius * 2,
            )

            initial_dirt_count = len(self.dirt_particles)
            self.dirt_particles = [
                dirt
                for dirt in self.dirt_particles
                if not roomba_rect.collidepoint(dirt[0], dirt[1])
            ]

            dirt_cleaned = initial_dirt_count - len(self.dirt_particles)
            self.score += dirt_cleaned * self.points_per_dirt

    def move_roomba(self, action):
        """Move the Roomba based on keyboard input"""
        if self.battery_level > 0:  # Only move if battery isn't dead
            new_pos = self.roomba_pos.copy()
            moved = False

            if action[pygame.K_LEFT]:
                new_pos[0] -= self.roomba_speed
                moved = True
            if action[pygame.K_RIGHT]:
                new_pos[0] += self.roomba_speed
                moved = True
            if action[pygame.K_UP]:
                new_pos[1] -= self.roomba_speed
                moved = True
            if action[pygame.K_DOWN]:
                new_pos[1] += self.roomba_speed
                moved = True

            if moved and not self.check_collision(new_pos):
                self.roomba_pos = new_pos
                self.steps_taken += 1

    def render(self):
        """Render the current state"""
        self.screen.fill(self.WHITE)

        # Draw charging station
        pygame.draw.rect(
            self.screen,
            self.YELLOW,
            (
                self.charging_station_pos[0] - self.charging_station_size / 2,
                self.charging_station_pos[1] - self.charging_station_size / 2,
                self.charging_station_size,
                self.charging_station_size,
            ),
        )

        # Draw walls
        for wall in self.walls:
            pygame.draw.rect(self.screen, self.GRAY, wall)

        # Draw dirt particles
        for dirt in self.dirt_particles:
            pygame.draw.circle(
                self.screen, self.BROWN, (int(dirt[0]), int(dirt[1])), self.dirt_radius
            )

        # Draw Roomba with color based on battery level
        if self.battery_level > 50:
            color = self.BLACK
        elif self.battery_level > 20:
            color = self.YELLOW
        else:
            color = self.RED

        pygame.draw.circle(
            self.screen,
            color,
            (int(self.roomba_pos[0]), int(self.roomba_pos[1])),
            self.roomba_radius,
        )

        # Draw score
        score_text = self.font.render(f"Score: {self.score}", True, self.BLACK)
        self.screen.blit(score_text, (10, 10))

        # Draw battery level
        battery_text = self.font.render(
            f"Battery: {int(self.battery_level)}%", True, self.BLACK
        )
        self.screen.blit(battery_text, (10, 50))

        # Draw remaining dirt count
        dirt_text = self.font.render(
            f"Dirt Remaining: {len(self.dirt_particles)}", True, self.BLACK
        )
        self.screen.blit(dirt_text, (10, 90))

        # Draw charging indicator if at station
        if self.is_at_charging_station():
            charging_text = self.font.render("CHARGING", True, self.GREEN)
            self.screen.blit(charging_text, (10, 130))

        pygame.display.flip()

    def run(self):
        """Main game loop"""
        running = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False

            # Get keyboard input
            keys = pygame.key.get_pressed()
            self.move_roomba(keys)

            # Clean dirt if space is pressed
            if keys[pygame.K_SPACE]:
                self.clean_dirt()

            # Update battery
            self.update_battery()

            self.render()
            self.clock.tick(60)

        pygame.quit()


if __name__ == "__main__":
    env = RoombaEnv()
    env.run()

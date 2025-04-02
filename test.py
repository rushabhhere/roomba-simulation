import numpy as np
import random
import pygame
import sys
import time
import os
import heapq
from collections import defaultdict
import threading

class RoombaEnvironment:
    """Environment for the Roomba vacuum cleaner simulation with room layouts and charging"""
    
    # Action space: 0: UP, 1: RIGHT, 2: DOWN, 3: LEFT
    ACTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]
    ACTION_NAMES = ['UP', 'RIGHT', 'DOWN', 'LEFT']
    
    # Cell types
    UNKNOWN = 0
    EMPTY = 1
    WALL = 2
    ROBOT = 3
    CHARGING_DOCK = 4
    DIRTY = 5
    CLEANED = 6
    
    def __init__(self, width=20, height=15, use_preset_map=True):
        self.width = width
        self.height = height
        
        # Create the house layout
        self.house_layout = np.ones((height, width)) * self.EMPTY
        
        # Add charging dock in top-left
        self.charging_dock_pos = (1, 1)
        self.house_layout[1, 1] = self.CHARGING_DOCK
        
        # Create room layout with walls
        if use_preset_map:
            self._create_preset_house_layout()
        else:
            self._create_random_house_layout()
        
        # Add dirt to the floor
        self._add_dirt()
        
        # Initialize robot position at the charging dock
        self.robot_pos = self.charging_dock_pos
        
        # Battery level (100% full)
        self.battery_level = 100.0
        self.is_charging = True
        self.battery_drain_rate = 0.1  # Battery drain per step
        self.charging_rate = 0.5      # Battery charging per step
        self.low_battery_threshold = 20.0
        
        # Robot's knowledge map (what it has discovered)
        self.knowledge_map = np.zeros((height, width))
        
        # Set the charging dock as known
        self.knowledge_map[self.charging_dock_pos[1], self.charging_dock_pos[0]] = self.CHARGING_DOCK
        
        # Traffic density map (how often each cell is visited)
        self.traffic_map = np.zeros((height, width))
        
        # Update the starting position in the knowledge map
        self.knowledge_map[self.robot_pos[1], self.robot_pos[0]] = self.EMPTY
        
        # Update traffic at starting position
        self.traffic_map[self.robot_pos[1], self.robot_pos[0]] += 1
        
        # Track the number of moves
        self.moves = 0
        self.cleaned_cells = 0
        
        # Track the coverage (percentage of non-wall cells cleaned)
        self.coverage = self.calculate_coverage()
        
        # Total number of dirty cells (for progress calculation)
        self.total_dirty_cells = np.sum(self.house_layout == self.DIRTY)
        
        # Direction the robot is facing (0: UP, 1: RIGHT, 2: DOWN, 3: LEFT)
        self.facing = 1  # Start facing right
        
        # Animation tracking
        self.last_action = None
        self.animation_progress = 0
        
        # Path history
        self.path_history = [(self.robot_pos[0], self.robot_pos[1])]
        
        # Optimal path data
        self.optimal_path = []
        self.target_position = None
        
        # For asynchronous path calculation
        self.path_calculation_thread = None
        self.calculation_lock = threading.Lock()
        
        # Start time
        self.start_time = time.time()
        self.elapsed_time = 0
        
        # Cleaning mode
        self.cleaning_mode = "Auto"  # Auto, Spot, Edge, or Manual
        self.returning_to_dock = False
        
    def _create_preset_house_layout(self):
        """Create a preset house layout with rooms and walls"""
        # Reset to empty
        self.house_layout = np.ones((self.height, self.width)) * self.EMPTY
        
        # Restore charging dock
        self.house_layout[self.charging_dock_pos[1], self.charging_dock_pos[0]] = self.CHARGING_DOCK
        
        # Create outer walls
        for x in range(self.width):
            self.house_layout[0, x] = self.WALL
            self.house_layout[self.height-1, x] = self.WALL
        for y in range(self.height):
            self.house_layout[y, 0] = self.WALL
            self.house_layout[y, self.width-1] = self.WALL
        
        # Create some room dividers
        
        # Living room / Kitchen divider
        for y in range(1, 9):
            self.house_layout[y, 10] = self.WALL
        # Door between living room and kitchen
        self.house_layout[5, 10] = self.EMPTY
        
        # Hallway
        for x in range(4, 18):
            self.house_layout[9, x] = self.WALL
        # Door to hallway
        self.house_layout[9, 7] = self.EMPTY
        
        # Bedroom 1
        for y in range(10, 14):
            self.house_layout[y, 8] = self.WALL
        # Door to bedroom 1
        self.house_layout[11, 8] = self.EMPTY
        
        # Bedroom 2
        for y in range(10, 14):
            self.house_layout[y, 15] = self.WALL
        # Door to bedroom 2
        self.house_layout[11, 15] = self.EMPTY
        
        # Bathroom wall
        for x in range(11, 18):
            self.house_layout[4, x] = self.WALL
        # Door to bathroom
        self.house_layout[4, 14] = self.EMPTY
        
        # Add some furniture (as walls)
        # Living room sofa
        for x in range(2, 6):
            self.house_layout[3, x] = self.WALL
        
        # Kitchen table
        for y in range(3, 5):
            for x in range(12, 14):
                self.house_layout[y, x] = self.WALL
        
        # Bed in bedroom 1
        for y in range(12, 14):
            for x in range(3, 7):
                self.house_layout[y, x] = self.WALL
        
        # Bed in bedroom 2
        for y in range(12, 14):
            for x in range(16, 19):
                self.house_layout[y, x] = self.WALL
    
    def _create_random_house_layout(self):
        """Create a random house layout with rooms and walls"""
        # Reset to empty
        self.house_layout = np.ones((self.height, self.width)) * self.EMPTY
        
        # Restore charging dock
        self.house_layout[self.charging_dock_pos[1], self.charging_dock_pos[0]] = self.CHARGING_DOCK
        
        # Create outer walls
        for x in range(self.width):
            self.house_layout[0, x] = self.WALL
            self.house_layout[self.height-1, x] = self.WALL
        for y in range(self.height):
            self.house_layout[y, 0] = self.WALL
            self.house_layout[y, self.width-1] = self.WALL
        
        # Add some random room dividers
        for _ in range(5):
            if random.random() < 0.5:  # Horizontal wall
                y = random.randint(3, self.height - 4)
                wall_length = random.randint(5, self.width - 6)
                start_x = random.randint(1, self.width - wall_length - 1)
                
                for x in range(start_x, start_x + wall_length):
                    self.house_layout[y, x] = self.WALL
                
                # Add a door
                door_pos = random.randint(start_x + 1, start_x + wall_length - 2)
                self.house_layout[y, door_pos] = self.EMPTY
            else:  # Vertical wall
                x = random.randint(3, self.width - 4)
                wall_length = random.randint(5, self.height - 6)
                start_y = random.randint(1, self.height - wall_length - 1)
                
                for y in range(start_y, start_y + wall_length):
                    self.house_layout[y, x] = self.WALL
                
                # Add a door
                door_pos = random.randint(start_y + 1, start_y + wall_length - 2)
                self.house_layout[door_pos, x] = self.EMPTY
        
        # Add some furniture (as walls)
        for _ in range(10):
            furniture_width = random.randint(1, 3)
            furniture_height = random.randint(1, 3)
            x = random.randint(1, self.width - furniture_width - 1)
            y = random.randint(1, self.height - furniture_height - 1)
            
            # Ensure we're not blocking the charging dock
            if not (x <= self.charging_dock_pos[0] + 1 and y <= self.charging_dock_pos[1] + 1):
                for fy in range(y, min(y + furniture_height, self.height - 1)):
                    for fx in range(x, min(x + furniture_width, self.width - 1)):
                        self.house_layout[fy, fx] = self.WALL
    
    def _add_dirt(self):
        """Add dirt to all empty cells"""
        for y in range(self.height):
            for x in range(self.width):
                if self.house_layout[y, x] == self.EMPTY:
                    self.house_layout[y, x] = self.DIRTY
        
        # Keep the charging dock clean
        self.house_layout[self.charging_dock_pos[1], self.charging_dock_pos[0]] = self.CHARGING_DOCK
    
    def calculate_coverage(self):
        """Calculate the percentage of dirty cells that have been cleaned"""
        total_dirty = np.sum(self.house_layout == self.DIRTY) + self.cleaned_cells
        if total_dirty == 0:
            return 1.0
        return self.cleaned_cells / total_dirty
    
    def get_state(self):
        """Return the current state for visualization"""
        # Create a combined map for visualization
        state = self.house_layout.copy()
        
        # Mark the robot's position
        state[self.robot_pos[1], self.robot_pos[0]] = self.ROBOT
        
        return {
            'visual_state': state,
            'battery_level': self.battery_level,
            'is_charging': self.is_charging,
            'coverage': self.coverage,
            'facing': self.facing,
            'cleaned_cells': self.cleaned_cells,
            'total_dirty_cells': self.total_dirty_cells,
            'moves': self.moves,
            'elapsed_time': self.elapsed_time,
            'cleaning_mode': self.cleaning_mode,
            'returning_to_dock': self.returning_to_dock
        }
    
    def update_battery(self, action_taken=True):
        """Update battery level based on charging status"""
        if self.is_at_charging_dock():
            self.is_charging = True
            self.battery_level = min(100.0, self.battery_level + self.charging_rate)
        else:
            self.is_charging = False
            if action_taken:
                self.battery_level = max(0.0, self.battery_level - self.battery_drain_rate)
    
    def is_at_charging_dock(self):
        """Check if robot is at the charging dock"""
        return self.robot_pos == self.charging_dock_pos
    
    def is_battery_low(self):
        """Check if battery level is low"""
        return self.battery_level < self.low_battery_threshold
    
    def step(self, action):
        """Take an action and return new state, reward, and done flag"""
        # Update the direction the robot is facing
        self.facing = action
        self.last_action = action
        self.animation_progress = 0  # Reset animation
        
        # Check if battery is depleted
        if self.battery_level <= 0:
            return self.get_state(), -5, True  # Game over if battery depleted
        
        # Calculate elapsed time
        self.elapsed_time = time.time() - self.start_time
        
        new_x = self.robot_pos[0] + self.ACTIONS[action][0]
        new_y = self.robot_pos[1] + self.ACTIONS[action][1]
        
        # Check if the move is valid
        if 0 <= new_x < self.width and 0 <= new_y < self.height:
            cell_type = self.house_layout[new_y, new_x]
            
            # Check if there's a wall
            if cell_type != self.WALL:
                # Valid move
                old_pos = self.robot_pos
                self.robot_pos = (new_x, new_y)
                
                # Update path history
                self.path_history.append((new_x, new_y))
                
                # Update traffic map with the new position
                self.traffic_map[new_y, new_x] += 1
                
                # Clean dirty cell if needed
                reward = 0
                if cell_type == self.DIRTY:
                    self.house_layout[new_y, new_x] = self.CLEANED
                    self.cleaned_cells += 1
                    reward = 1.0  # Reward for cleaning
                
                # Charge if at dock
                if self.is_at_charging_dock():
                    reward += 0.2  # Small reward for returning to dock
                
                # Update battery
                self.update_battery()
                
                # Update knowledge map with the new cell
                self.knowledge_map[new_y, new_x] = self.house_layout[new_y, new_x]
                
                # Increment move counter
                self.moves += 1
                
                # Calculate new coverage
                old_coverage = self.coverage
                self.coverage = self.calculate_coverage()
                
                # Check if cleaning is complete
                done = self.coverage >= 0.99  # Consider cleaning complete at 99% coverage
                
                return self.get_state(), reward, done
            else:
                # Hit a wall - invalid move
                self.update_battery(action_taken=False)  # No battery drain for invalid moves
                return self.get_state(), -0.2, False  # Small penalty for hitting a wall
        else:
            # Out of bounds
            self.update_battery(action_taken=False)  # No battery drain for invalid moves
            return self.get_state(), -0.2, False  # Small penalty for trying to go out of bounds
    
    def find_path_to_charging_dock(self):
        """Calculate path to the charging dock using A*"""
        if self.is_at_charging_dock():
            self.optimal_path = []
            return True
        
        # A* pathfinding
        start = self.robot_pos
        goal = self.charging_dock_pos
        
        # Initialize data structures
        open_set = []
        heapq.heappush(open_set, (0, start))
        came_from = {}
        g_score = {start: 0}
        f_score = {start: self._heuristic(start, goal)}
        
        open_set_hash = {start}
        
        while open_set:
            # Pop cell with lowest f_score
            current = heapq.heappop(open_set)[1]
            open_set_hash.remove(current)
            
            # Goal reached
            if current == goal:
                # Reconstruct path
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                
                self.optimal_path = path
                self.target_position = goal
                return True
            
            # Explore neighbors
            for dx, dy in self.ACTIONS:
                neighbor = (current[0] + dx, current[1] + dy)
                
                # Check if valid move
                if (0 <= neighbor[0] < self.width and 0 <= neighbor[1] < self.height and 
                    self.house_layout[neighbor[1], neighbor[0]] != self.WALL):
                    
                    tentative_g = g_score.get(current, float('inf')) + 1
                    
                    if tentative_g < g_score.get(neighbor, float('inf')):
                        # This path is better
                        came_from[neighbor] = current
                        g_score[neighbor] = tentative_g
                        f_score[neighbor] = tentative_g + self._heuristic(neighbor, goal)
                        
                        if neighbor not in open_set_hash:
                            heapq.heappush(open_set, (f_score[neighbor], neighbor))
                            open_set_hash.add(neighbor)
        
        # No path found
        self.optimal_path = []
        return False
    
    def get_next_move_to_charging_dock(self):
        """Get the next move to follow the path to charging dock"""
        if self.is_at_charging_dock():
            return None
        
        # If we don't have a path or need to recalculate
        if not self.optimal_path or len(self.optimal_path) <= 1:
            success = self.find_path_to_charging_dock()
            if not success or not self.optimal_path:
                # No path found, try a random move
                valid_actions = []
                for i, (dx, dy) in enumerate(self.ACTIONS):
                    nx, ny = self.robot_pos[0] + dx, self.robot_pos[1] + dy
                    if (0 <= nx < self.width and 0 <= ny < self.height and 
                        self.house_layout[ny, nx] != self.WALL):
                        valid_actions.append(i)
                
                if valid_actions:
                    return random.choice(valid_actions)
                else:
                    return None  # Stuck with no valid moves
        
        # Get next position from path
        if len(self.optimal_path) > 1:
            next_pos = self.optimal_path[1]
            
            # Calculate action to take
            dx = next_pos[0] - self.robot_pos[0]
            dy = next_pos[1] - self.robot_pos[1]
            
            for i, (action_dx, action_dy) in enumerate(self.ACTIONS):
                if dx == action_dx and dy == action_dy:
                    return i
        
        return None
    
    def _heuristic(self, a, b):
        """Manhattan distance heuristic for A*"""
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    def auto_clean(self):
        """Simple auto-cleaning algorithm"""
        # If battery low, return to charging dock
        if self.is_battery_low() and not self.is_at_charging_dock():
            self.returning_to_dock = True
            next_move = self.get_next_move_to_charging_dock()
            if next_move is not None:
                return next_move
        
        # If charging and battery full, resume cleaning
        if self.is_at_charging_dock() and self.battery_level > 90:
            self.returning_to_dock = False
        
        # If returning to dock, continue doing so
        if self.returning_to_dock:
            next_move = self.get_next_move_to_charging_dock()
            if next_move is not None:
                return next_move
        
        # Regular cleaning - prioritize dirty cells
        valid_actions = []
        dirty_actions = []
        
        for i, (dx, dy) in enumerate(self.ACTIONS):
            nx, ny = self.robot_pos[0] + dx, self.robot_pos[1] + dy
            if (0 <= nx < self.width and 0 <= ny < self.height and 
                self.house_layout[ny, nx] != self.WALL):
                valid_actions.append(i)
                if self.house_layout[ny, nx] == self.DIRTY:
                    dirty_actions.append(i)
        
        if dirty_actions:
            return random.choice(dirty_actions)
        elif valid_actions:
            # Prioritize less visited cells
            least_traffic = float('inf')
            best_actions = []
            
            for action in valid_actions:
                dx, dy = self.ACTIONS[action]
                nx, ny = self.robot_pos[0] + dx, self.robot_pos[1] + dy
                traffic = self.traffic_map[ny, nx]
                
                if traffic < least_traffic:
                    least_traffic = traffic
                    best_actions = [action]
                elif traffic == least_traffic:
                    best_actions.append(action)
            
            return random.choice(best_actions)
        
        # No valid moves, stay in place
        return None
    
    def spot_clean(self):
        """Clean in a spiral pattern around current location"""
        # If battery low, return to charging dock
        if self.is_battery_low() and not self.is_at_charging_dock():
            self.returning_to_dock = True
            next_move = self.get_next_move_to_charging_dock()
            if next_move is not None:
                return next_move
        
        # Prioritize dirty cells in immediate surroundings
        dirty_actions = []
        for i, (dx, dy) in enumerate(self.ACTIONS):
            nx, ny = self.robot_pos[0] + dx, self.robot_pos[1] + dy
            if (0 <= nx < self.width and 0 <= ny < self.height and 
                self.house_layout[ny, nx] == self.DIRTY):
                dirty_actions.append(i)
        
        if dirty_actions:
            return random.choice(dirty_actions)
        
        # If no dirty cells, try to move to least visited valid cell
        valid_actions = []
        for i, (dx, dy) in enumerate(self.ACTIONS):
            nx, ny = self.robot_pos[0] + dx, self.robot_pos[1] + dy
            if (0 <= nx < self.width and 0 <= ny < self.height and 
                self.house_layout[ny, nx] != self.WALL):
                valid_actions.append((i, self.traffic_map[ny, nx]))
        
        if valid_actions:
            # Sort by traffic (prefer less visited)
            valid_actions.sort(key=lambda x: x[1])
            return valid_actions[0][0]
        
        return None
    
    def edge_clean(self):
        """Follow walls/edges for cleaning"""
        # If battery low, return to charging dock
        if self.is_battery_low() and not self.is_at_charging_dock():
            self.returning_to_dock = True
            next_move = self.get_next_move_to_charging_dock()
            if next_move is not None:
                return next_move
        
        # Check if there's a wall on the right relative to current facing
        right_direction = (self.facing + 1) % 4
        dx, dy = self.ACTIONS[right_direction]
        right_x, right_y = self.robot_pos[0] + dx, self.robot_pos[1] + dy
        
        has_wall_on_right = (
            right_x < 0 or right_x >= self.width or
            right_y < 0 or right_y >= self.height or
            self.house_layout[right_y, right_x] == self.WALL
        )
        
        # Try to follow wall on right
        if has_wall_on_right:
            # Try to move forward
            forward_x = self.robot_pos[0] + self.ACTIONS[self.facing][0]
            forward_y = self.robot_pos[1] + self.ACTIONS[self.facing][1]
            
            if (0 <= forward_x < self.width and 0 <= forward_y < self.height and 
                self.house_layout[forward_y, forward_x] != self.WALL):
                return self.facing
            else:
                # Turn left if can't go forward
                return (self.facing - 1) % 4
        else:
            # No wall on right, turn right and continue
            return right_direction
    
    def get_action_auto(self):
        """Get the next action based on cleaning mode"""
        if self.cleaning_mode == "Auto":
            return self.auto_clean()
        elif self.cleaning_mode == "Spot":
            return self.spot_clean()
        elif self.cleaning_mode == "Edge":
            return self.edge_clean()
        else:  # Manual mode
            return None  # Let user control
    
    def toggle_cleaning_mode(self):
        """Toggle between cleaning modes"""
        modes = ["Auto", "Spot", "Edge", "Manual"]
        current_index = modes.index(self.cleaning_mode)
        self.cleaning_mode = modes[(current_index + 1) % len(modes)]
        return self.cleaning_mode


class RoombaVisualizer:
    """Visualizes the Roomba vacuum environment using Pygame"""
    
    # Colors
    WHITE = (255, 255, 255)
    BLACK = (0, 0, 0)
    GRAY = (200, 200, 200)
    DARK_GRAY = (100, 100, 100)
    RED = (255, 50, 50)
    BLUE = (50, 50, 255)
    GREEN = (50, 200, 50)
    YELLOW = (255, 255, 0)
    LIGHT_BROWN = (210, 180, 140)  # For dirty floor
    TEAL = (0, 128, 128)  # For charging dock
    
    # Button colors
    BUTTON_BLUE = (67, 127, 222)
    BUTTON_GREEN = (50, 200, 50)
    BUTTON_RED = (222, 67, 67)
    BUTTON_YELLOW = (222, 222, 67)
    
    def __init__(self, cell_size=40):
        # Initialize Pygame
        pygame.init()
        pygame.font.init()
        
        self.cell_size = cell_size
        self.font = pygame.font.SysFont('Arial', 24)
        self.small_font = pygame.font.SysFont('Arial', 16)
        self.button_font = pygame.font.SysFont('Arial', 16, bold=True)
        
        # Robot images based on direction
        self.robot_images = self.load_robot_images()
        
        # Placeholder for the screen
        self.screen = None
        self.width = 0
        self.height = 0
        
        # Path tracking
        self.path = []
        self.show_path = True
        
        # Button data
        self.buttons = {}
        
        # Animation variables
        self.animation_speed = 0.2  # Time (in seconds) for movement animation
        
        # Dock image
        self.dock_image = self.create_dock_image()
        
        # Control buttons
        self.control_buttons = {}

    
    
    def load_robot_images(self):
        """Create robot images for different directions"""
        images = []
        
        # Basic robot size
        robot_size = int(self.cell_size * 0.8)
        
        # Create a surface for each direction
        for i in range(4):  # UP, RIGHT, DOWN, LEFT
            surface = pygame.Surface((robot_size, robot_size), pygame.SRCALPHA)
            
            # Draw robot body (circle)
            pygame.draw.circle(surface, self.BLUE, (robot_size//2, robot_size//2), robot_size//2)
            
            # Draw direction indicator (triangle)
            if i == 0:  # UP
                points = [(robot_size//2, 0), (robot_size//4, robot_size//2), (3*robot_size//4, robot_size//2)]
            elif i == 1:  # RIGHT
                points = [(robot_size, robot_size//2), (robot_size//2, robot_size//4), (robot_size//2, 3*robot_size//4)]
            elif i == 2:  # DOWN
                points = [(robot_size//2, robot_size), (robot_size//4, robot_size//2), (3*robot_size//4, robot_size//2)]
            else:  # LEFT
                points = [(0, robot_size//2), (robot_size//2, robot_size//4), (robot_size//2, 3*robot_size//4)]
            
            pygame.draw.polygon(surface, self.YELLOW, points)
            images.append(surface)
        
        return images
    
    def create_dock_image(self):
        """Create an image for the charging dock"""
        dock_size = self.cell_size
        surface = pygame.Surface((dock_size, dock_size), pygame.SRCALPHA)
        
        # Base of the dock
        pygame.draw.rect(surface, self.TEAL, (0, dock_size//2, dock_size, dock_size//2))
        
        # Connector
        pygame.draw.rect(surface, self.YELLOW, (dock_size//4, dock_size//3, dock_size//2, dock_size//6))
        
        # Indicator light
        pygame.draw.circle(surface, self.GREEN, (dock_size//2, dock_size//4), dock_size//10)
        
        return surface
    
    def setup(self, env):
        """Initialize the visualizer with the environment"""
        # Calculate grid dimensions based on environment
        self.width = env.width
        self.height = env.height

        # Set up the window dimensions
        grid_width = self.width * self.cell_size
        grid_height = self.height * self.cell_size
        panel_width = 300  # Control panel width
        
        # Define window size
        window_width = grid_width + panel_width
        window_height = max(grid_height, 600)  # Ensure minimum height for control panel
        
        # Create the screen
        self.screen = pygame.display.set_mode((window_width, window_height))
        pygame.display.set_caption("Roomba Simulator")
        
        # Calculate grid position (centered vertically)
        self.grid_x = 0
        self.grid_y = (window_height - grid_height) // 2 if window_height > grid_height else 0
        
        # Control panel position
        self.panel_x = grid_width
        self.panel_y = 0
        
        # Set up control panel buttons
        self.setup_control_buttons(panel_width)
        
        # Initialize path with current robot position
        self.path = [(env.robot_pos[0], env.robot_pos[1])]
    
    def setup_control_buttons(self, panel_width):
        """Set up buttons for the control panel"""
        button_width = panel_width - 40  # 20px margin on each side
        button_height = 40
        button_spacing = 10
    
        y_pos = 20
    
    # Mode button
        self.buttons['mode'] = {
        'rect': pygame.Rect(self.panel_x + 20, y_pos, button_width, button_height),
        'color': self.BUTTON_BLUE,
        'text': 'Mode: Auto',
        'action': 'toggle_mode'
    }
        y_pos += button_height + button_spacing
        
        # Show/Hide Path button
        self.buttons['path'] = {
            'rect': pygame.Rect(self.panel_x + 20, y_pos, button_width, button_height),
            'color': self.BUTTON_GREEN,
            'text': 'Hide Path',
            'action': 'toggle_path'
        }
        y_pos += button_height + button_spacing
        
        # Reset button
        self.buttons['reset'] = {
            'rect': pygame.Rect(self.panel_x + 20, y_pos, button_width, button_height),
            'color': self.BUTTON_RED,
            'text': 'Reset Simulation',
            'action': 'reset'
        }
        y_pos += button_height + button_spacing
        
        # Direction control buttons (for manual mode)
        control_size = 60
        control_center_x = self.panel_x + panel_width // 2
        control_center_y = y_pos + control_size * 2
        
        # Up button
        self.control_buttons['up'] = {
            'rect': pygame.Rect(control_center_x - control_size//2, 
                            control_center_y - control_size, 
                            control_size, control_size),
            'color': self.BUTTON_BLUE,
            'text': '↑',
            'action': 0  # UP action
        }
        
        # Right button
        self.control_buttons['right'] = {
            'rect': pygame.Rect(control_center_x + control_size//2, 
                            control_center_y - control_size//2, 
                            control_size, control_size),
            'color': self.BUTTON_BLUE,
            'text': '→',
            'action': 1  # RIGHT action
        }
        
        # Down button
        self.control_buttons['down'] = {
            'rect': pygame.Rect(control_center_x - control_size//2, 
                            control_center_y + control_size//2, 
                            control_size, control_size),
            'color': self.BUTTON_BLUE,
            'text': '↓',
            'action': 2  # DOWN action
        }
        
        # Left button
        self.control_buttons['left'] = {
            'rect': pygame.Rect(control_center_x - control_size*3//2, 
                            control_center_y - control_size//2, 
                            control_size, control_size),
            'color': self.BUTTON_BLUE,
            'text': '←',
            'action': 3  # LEFT action
        }

        
    def draw(self, env_state, robot_pos, last_pos=None, animation_progress=0):
        """Draw the environment and control panel"""
        # Clear screen
        self.screen.fill(self.WHITE)
        
        # Draw the grid
        self.draw_grid(env_state['visual_state'], robot_pos, last_pos, animation_progress)
        
        # Draw control panel
        self.draw_control_panel(env_state)
        
        # Update display
        pygame.display.flip()

    def draw_grid(self, state, robot_pos, last_pos=None, animation_progress=0):
        """Draw the grid with cells and robot"""
        for y in range(self.height):
            for x in range(self.width):
                cell_x = self.grid_x + x * self.cell_size
                cell_y = self.grid_y + y * self.cell_size
                
                # Get cell value
                cell_value = state[y, x]
                
                # Draw cell based on type
                if cell_value == RoombaEnvironment.WALL:
                    pygame.draw.rect(self.screen, self.DARK_GRAY, 
                                    (cell_x, cell_y, self.cell_size, self.cell_size))
                elif cell_value == RoombaEnvironment.DIRTY:
                    pygame.draw.rect(self.screen, self.LIGHT_BROWN, 
                                    (cell_x, cell_y, self.cell_size, self.cell_size))
                elif cell_value == RoombaEnvironment.CLEANED:
                    pygame.draw.rect(self.screen, self.WHITE, 
                                    (cell_x, cell_y, self.cell_size, self.cell_size))
                elif cell_value == RoombaEnvironment.CHARGING_DOCK:
                    pygame.draw.rect(self.screen, self.WHITE, 
                                    (cell_x, cell_y, self.cell_size, self.cell_size))
                    # Draw charging dock
                    self.screen.blit(self.dock_image, (cell_x, cell_y))
                else:
                    pygame.draw.rect(self.screen, self.WHITE, 
                                    (cell_x, cell_y, self.cell_size, self.cell_size))
                
                # Draw grid lines
                pygame.draw.rect(self.screen, self.GRAY, 
                                (cell_x, cell_y, self.cell_size, self.cell_size), 1)
        
        # Draw path history if enabled
        if self.show_path:
            for i in range(1, len(self.path)):
                path_start = (
                    self.grid_x + self.path[i-1][0] * self.cell_size + self.cell_size // 2,
                    self.grid_y + self.path[i-1][1] * self.cell_size + self.cell_size // 2
                )
                path_end = (
                    self.grid_x + self.path[i][0] * self.cell_size + self.cell_size // 2,
                    self.grid_y + self.path[i][1] * self.cell_size + self.cell_size // 2
                )
                pygame.draw.line(self.screen, (255, 100, 100, 128), path_start, path_end, 2)
        
        # Draw robot with animation (if last_pos is provided)
        if animation_progress > 0 and last_pos:
            # Calculate interpolated position
            interp_x = last_pos[0] + (robot_pos[0] - last_pos[0]) * animation_progress
            interp_y = last_pos[1] + (robot_pos[1] - last_pos[1]) * animation_progress
            
            # Draw at interpolated position
            robot_x = self.grid_x + interp_x * self.cell_size + (self.cell_size - self.robot_images[0].get_width()) // 2
            robot_y = self.grid_y + interp_y * self.cell_size + (self.cell_size - self.robot_images[0].get_height()) // 2
            
            # Get facing direction - use facing directly instead of trying to get it from state matrix
            facing = int(state[robot_pos[1], robot_pos[0]]) % 4 if state[robot_pos[1], robot_pos[0]] < 4 else 0
            
            # Draw robot image for current direction
            self.screen.blit(self.robot_images[facing], (robot_x, robot_y))
        else:
            # Draw at exact position
            robot_x = self.grid_x + robot_pos[0] * self.cell_size + (self.cell_size - self.robot_images[0].get_width()) // 2
            robot_y = self.grid_y + robot_pos[1] * self.cell_size + (self.cell_size - self.robot_images[0].get_height()) // 2
            
            # Get robot facing direction
            facing = 0  # Default
            
            # The key change: make sure facing is an integer between 0-3
            if 'facing' in state:
                facing = int(state['facing']) % 4
            
            # Draw robot image
            self.screen.blit(self.robot_images[facing], (robot_x, robot_y))

    def draw_control_panel(self, state):
        """Draw the control panel with buttons and status information"""
        # Draw panel background
        panel_rect = pygame.Rect(self.panel_x, self.panel_y, 
                            self.screen.get_width() - self.panel_x, 
                            self.screen.get_height())
        pygame.draw.rect(self.screen, self.GRAY, panel_rect)
        
        # Draw separator line
        pygame.draw.line(self.screen, self.BLACK, 
                    (self.panel_x, 0), 
                    (self.panel_x, self.screen.get_height()), 2)
        
        # Draw title
        title_text = self.font.render("Roomba Simulator", True, self.BLACK)
        self.screen.blit(title_text, (self.panel_x + 20, 20))
        
        # Draw status information
        y_pos = 70
        
        # Battery level
        battery_level = state['battery_level']
        battery_text = f"Battery: {battery_level:.1f}%"
        battery_color = self.GREEN if battery_level > 20 else self.RED
        battery_surface = self.font.render(battery_text, True, battery_color)
        self.screen.blit(battery_surface, (self.panel_x + 20, y_pos))
        y_pos += 30
        
        # Battery bar
        battery_bar_width = 200
        battery_bar_height = 20
        pygame.draw.rect(self.screen, self.BLACK, 
                    (self.panel_x + 20, y_pos, battery_bar_width, battery_bar_height), 1)
        
        battery_fill_width = int((battery_bar_width - 2) * (battery_level / 100))
        if battery_fill_width > 0:
            pygame.draw.rect(self.screen, battery_color, 
                        (self.panel_x + 21, y_pos + 1, battery_fill_width, battery_bar_height - 2))
        y_pos += battery_bar_height + 10
        
        # Charging status
        if state['is_charging']:
            charging_text = "Status: Charging"
            charging_color = self.GREEN
        else:
            charging_text = "Status: Cleaning"
            charging_color = self.BLUE
        
        charging_surface = self.font.render(charging_text, True, charging_color)
        self.screen.blit(charging_surface, (self.panel_x + 20, y_pos))
        y_pos += 40
        
        # Coverage
        coverage = state['coverage'] * 100
        coverage_text = f"Coverage: {coverage:.1f}%"
        coverage_surface = self.font.render(coverage_text, True, self.BLACK)
        self.screen.blit(coverage_surface, (self.panel_x + 20, y_pos))
        y_pos += 30
        
        # Coverage bar
        pygame.draw.rect(self.screen, self.BLACK, 
                    (self.panel_x + 20, y_pos, battery_bar_width, battery_bar_height), 1)
        
        coverage_fill_width = int((battery_bar_width - 2) * (coverage / 100))
        if coverage_fill_width > 0:
            pygame.draw.rect(self.screen, self.BLUE, 
                        (self.panel_x + 21, y_pos + 1, coverage_fill_width, battery_bar_height - 2))
        y_pos += battery_bar_height + 20
        
        # Cells cleaned
        cleaned = state['cleaned_cells']
        total = state['total_dirty_cells']
        cleaned_text = f"Cells: {cleaned}/{total}"
        cleaned_surface = self.font.render(cleaned_text, True, self.BLACK)
        self.screen.blit(cleaned_surface, (self.panel_x + 20, y_pos))
        y_pos += 40
        
        # Moves count
        moves_text = f"Moves: {state['moves']}"
        moves_surface = self.font.render(moves_text, True, self.BLACK)
        self.screen.blit(moves_surface, (self.panel_x + 20, y_pos))
        y_pos += 30
        
        # Time elapsed
        time_text = f"Time: {state['elapsed_time']:.1f}s"
        time_surface = self.font.render(time_text, True, self.BLACK)
        self.screen.blit(time_surface, (self.panel_x + 20, y_pos))
        y_pos += 30
        
        # Mode
        mode_text = f"Mode: {state['cleaning_mode']}"
        mode_surface = self.font.render(mode_text, True, self.BLACK)
        self.screen.blit(mode_surface, (self.panel_x + 20, y_pos))
        y_pos += 40
        
        # Status message
        if state['returning_to_dock']:
            status_text = "Returning to dock..."
            status_color = self.YELLOW
        elif state['is_charging'] and state['battery_level'] < 100:
            status_text = "Charging..."
            status_color = self.GREEN
        elif state['is_charging'] and state['battery_level'] >= 100:
            status_text = "Fully charged"
            status_color = self.GREEN
        else:
            status_text = "Cleaning"
            status_color = self.BLUE
        
        status_surface = self.font.render(status_text, True, status_color)
        self.screen.blit(status_surface, (self.panel_x + 20, y_pos))
        y_pos += 60
        
        # Draw buttons
        for button_name, button_data in self.buttons.items():
            self.draw_button(button_data)
        
        # Draw control buttons if in manual mode
        if state['cleaning_mode'] == "Manual":
            for button_name, button_data in self.control_buttons.items():
                self.draw_button(button_data)

    def draw_button(self, button_data):
        """Draw a button with text"""
        pygame.draw.rect(self.screen, button_data['color'], button_data['rect'])
        pygame.draw.rect(self.screen, self.BLACK, button_data['rect'], 2)
        
        # Button text
        text_surface = self.button_font.render(button_data['text'], True, self.WHITE)
        text_rect = text_surface.get_rect(center=button_data['rect'].center)
        self.screen.blit(text_surface, text_rect)

    def handle_button_click(self, pos, env):
        """Handle mouse click on buttons"""
        for button_name, button_data in self.buttons.items():
            if button_data['rect'].collidepoint(pos):
                if button_data['action'] == 'toggle_mode':
                    # Toggle cleaning mode
                    mode = env.toggle_cleaning_mode()
                    button_data['text'] = f"Mode: {mode}"
                    return True
                elif button_data['action'] == 'toggle_path':
                    # Toggle path display
                    self.show_path = not self.show_path
                    button_data['text'] = "Show Path" if not self.show_path else "Hide Path"
                    return True
                elif button_data['action'] == 'reset':
                    # Signal to reset the simulation
                    return 'reset'
        
        # Check direction control buttons if in manual mode
        if env.cleaning_mode == "Manual":
            for button_name, button_data in self.control_buttons.items():
                if button_data['rect'].collidepoint(pos):
                    return button_data['action']  # Return the action to take
        
        return False

    def update_path(self, robot_pos):
        """Update the path history with new robot position"""
        self.path.append(robot_pos)


def main():
    """Main function to run the simulation"""
    # Initialize environment
    env = RoombaEnvironment(width=20, height=15, use_preset_map=True)
    
    # Initialize visualizer
    visualizer = RoombaVisualizer(cell_size=40)
    visualizer.setup(env)
    
    # Main loop
    clock = pygame.time.Clock()
    running = True
    done = False
    last_step_time = time.time()
    step_delay = 0.2  # Time between auto steps (seconds)
    
    # For animation
    last_pos = None
    action = None
    animation_progress = 0
    
    while running:
        current_time = time.time()
        
        # Get events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click
                    # Check if a button was clicked
                    button_result = visualizer.handle_button_click(event.pos, env)
                    
                    if button_result == 'reset':
                        # Reset the simulation
                        env = RoombaEnvironment(width=20, height=15, use_preset_map=True)
                        visualizer.path = [(env.robot_pos[0], env.robot_pos[1])]
                        done = False
                    elif isinstance(button_result, int):
                        # Button returns an action to take
                        action = button_result
                        last_pos = env.robot_pos
                        animation_progress = 0
                        
                        # Take the action
                        state, reward, done = env.step(action)
                        
                        # Update path
                        visualizer.update_path((env.robot_pos[0], env.robot_pos[1]))
            
            elif event.type == pygame.KEYDOWN:
                if env.cleaning_mode == "Manual":
                    if event.key == pygame.K_UP:
                        action = 0  # UP
                    elif event.key == pygame.K_RIGHT:
                        action = 1  # RIGHT
                    elif event.key == pygame.K_DOWN:
                        action = 2  # DOWN
                    elif event.key == pygame.K_LEFT:
                        action = 3  # LEFT
                    
                    if action is not None:
                        last_pos = env.robot_pos
                        animation_progress = 0
                        
                        # Take the action
                        state, reward, done = env.step(action)
                        
                        # Update path
                        visualizer.update_path((env.robot_pos[0], env.robot_pos[1]))
        
        # Handle automatic action if not in manual mode and not done
        if not done and env.cleaning_mode != "Manual":
            # Check if it's time for a new step
            if current_time - last_step_time >= step_delay:
                # Get auto action
                action = env.get_action_auto()
                
                if action is not None:
                    last_pos = env.robot_pos
                    animation_progress = 0
                    
                    # Take the action
                    state, reward, done = env.step(action)
                    
                    # Update path
                    visualizer.update_path((env.robot_pos[0], env.robot_pos[1]))
                
                last_step_time = current_time
        
        # Update animation progress
        if action is not None:
            animation_progress = min(1.0, (current_time - last_step_time) / visualizer.animation_speed)
        
        # Get current state
        state = env.get_state()
        
        # Draw everything
        visualizer.draw(state, (env.robot_pos[0], env.robot_pos[1]), last_pos, animation_progress)
        
        # Cap frame rate
        clock.tick(60)
    
    # Quit Pygame
    pygame.quit()
    sys.exit()


if __name__ == "__main__":
    main()
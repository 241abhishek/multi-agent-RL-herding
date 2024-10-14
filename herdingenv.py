import gym
from gym import spaces
import numpy as np
from herdingrobot import DifferentialDriveRobot
import pygame

class HerdingSimEnv(gym.Env):
    def __init__(self, arena_length, arena_width, num_sheep, num_sheepdogs, 
                 robot_distance_between_wheels, robot_wheel_radius, max_wheel_velocity, 
                 initial_positions=None):
        """
        Initialize the simulation environment.

        Args:
            arena_length (float): Length of the arena (meters)
            arena_width (float): Width of the arena (meters)
            num_sheep (int): Number of sheep in the simulation
            num_sheepdogs (int): Number of sheepdogs in the simulation
            robot_distance_between_wheels (float): Distance between the wheels of the robots (meters)
            robot_wheel_radius (float): Radius of the wheels of the robots (meters)
            max_wheel_velocity (float): Maximum velocity of the wheels of the robots (meters/second)
            initial_positions (List[List[float]]): Initial positions of the robots in the simulation. [x, y, theta].

        """
        super(HerdingSimEnv, self).__init__()
        self.arena_length = arena_length
        self.arena_width = arena_width
        self.num_sheep = num_sheep
        self.num_sheepdogs = num_sheepdogs
        self.distance_between_wheels = robot_distance_between_wheels
        self.wheel_radius = robot_wheel_radius
        self.max_wheel_velocity = max_wheel_velocity
        # generate random initial positions within the arena if not provided
        if initial_positions is None:
            initial_positions = []
            for _ in range(num_sheep + num_sheepdogs):
                x = np.random.uniform(0, self.arena_length)
                y = np.random.uniform(0, self.arena_width)
                theta = np.random.uniform(-np.pi, np.pi)
                initial_positions.append([x, y, theta])
        # hold the information of all robots in self.robots
        # the first num_sheepdogs robots are sheepdogs and the rest are sheep
        self.robots = self.init_robots(initial_positions) 

        # Action and observation space
        # Action space is wheel velocities for sheep-dogs
        self.action_space = spaces.Box(low=-max_wheel_velocity, high=max_wheel_velocity, 
                                       shape=(num_sheepdogs * 2,), dtype=np.float32)
        # Observation space is positions and orientations of all robots
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, 
                                            shape=(num_sheep + num_sheepdogs, 3), dtype=np.float32)


        # Convert to pygame units (pixels)
        self.scale_factor = 50  # 1 meter = 50 pixels, adjust as needed
        self.arena_length_px = self.arena_length * self.scale_factor
        self.arena_width_px = self.arena_width * self.scale_factor

        # save the frames at each step for future rendering
        self.frames = []

    def init_robots(self, initial_positions):
        robots = []
        for pos in initial_positions:
            robots.append(DifferentialDriveRobot(pos, self.distance_between_wheels, self.wheel_radius))
        return robots

    def step(self, action):
        # Update sheep-dogs using RL agent actions
        for i in range(self.num_sheepdogs):
            left_wheel_velocity = action[i * 2]
            right_wheel_velocity = action[i * 2 + 1]
            self.robots[i].update_position(left_wheel_velocity, right_wheel_velocity)

        # Update sheep positions using predefined behavior 
        self.compute_sheep_actions()

        # Gather new observations (positions, orientations)
        observations = self.get_observations()

        # Compute reward, done, and info
        reward = self.compute_reward()
        done = self.check_done()
        info = {}

        return observations, reward, done, info

    def render(self, mode="human"):
        # Initialize pygame if it hasn't been already
        if not hasattr(self, 'screen'):
            pygame.init()
            self.screen = pygame.display.set_mode((self.arena_length_px, self.arena_width_px))
            pygame.display.set_caption("Herding Simulation")
            self.clock = pygame.time.Clock()

        # Clear the previous frame
        self.screen.fill((0, 0, 0))  # Fill screen with black

        # Draw the arena border
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(0, 0, self.arena_length_px, self.arena_width_px), 2)

        # Plot robots
        for i, robot in enumerate(self.robots):
            if i < self.num_sheepdogs:
                color = (0, 0, 255)  # Blue for Sheep-dogs
            else:
                color = (0, 255, 0)  # Green for Sheep

            robot_x, robot_y, robot_theta = robot.get_state()  # Get robot position
            robot_x_px = int(robot_x * self.scale_factor)
            robot_y_px = self.arena_width_px - int(robot_y * self.scale_factor) # Flip y-axis

            # Draw robot as a circle
            robot_size = 7.5  # Size of the robot
            pygame.draw.circle(self.screen, color, (robot_x_px, robot_y_px), robot_size)

            # Draw robot orientation as a white triangle
            # Triangle points to indicate direction based on robot_theta
            triangle_size = 2.5  # Size of the triangle
            angle_offset = 2 * np.pi / 3  # Offset to adjust the triangle's base around the center

            # Compute the vertices of the triangle
            point1 = (robot_x_px + (triangle_size + 2) * np.cos(robot_theta),
                      robot_y_px - (triangle_size + 2) * np.sin(robot_theta))
            point2 = (robot_x_px + triangle_size * np.cos(robot_theta + angle_offset),
                        robot_y_px - triangle_size * np.sin(robot_theta + angle_offset))
            point3 = (robot_x_px + triangle_size * np.cos(robot_theta - angle_offset),
                        robot_y_px - triangle_size * np.sin(robot_theta - angle_offset))
            
            # Draw the triangle
            pygame.draw.polygon(self.screen, (255, 255, 255), [point1, point2, point3])

            # draw an arrow to indicate the direction of the robot
            arrow_length = 15
            arrow_end = (robot_x_px + arrow_length * np.cos(robot_theta),
                         robot_y_px - arrow_length * np.sin(robot_theta))
            pygame.draw.line(self.screen, (255, 255, 255), (robot_x_px, robot_y_px), arrow_end, 2)

        # Update the display for "human" mode
        if mode == "human":
            pygame.display.flip()
            self.clock.tick(1)  # Set frame rate to 10 FPS (1/10th of a second per frame)

        # Save frames if needed for video output
        frame = pygame.surfarray.array3d(self.screen)
        self.frames.append(np.rot90(frame))  # Convert Pygame surface to numpy array
        
    def reset(self):
        # Reset the environment
        # Generate random initial positions for all robots
        initial_positions = []
        for _ in range(self.num_sheep + self.num_sheepdogs):
            x = np.random.uniform(0, self.arena_length)
            y = np.random.uniform(0, self.arena_width)
            theta = np.random.uniform(-np.pi, np.pi)
            initial_positions.append([x, y, theta])
        self.robots = self.init_robots(initial_positions)

        # clear the frames
        self.frames = []

        # Return the initial observation
        return self.get_observations()

    def compute_sheep_actions(self):
        # update sheep positions based on predefined behavior
        pass

    def get_observations(self):
        # Return positions and orientations of all robots
        obs = []
        for robot in self.robots:
            obs.append(robot.get_state())
        return np.array(obs)

    def compute_reward(self):
        # Define reward function
        pass

    def check_done(self):
        # Check if the episode is over
        pass

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

        # simulation parameters
        # these are hardcoded parameters that define the behavior of the simulation
        self.max_sheep_wheel_vel = 5.0 # max wheel velocity of the sheep 
        self.arena_threshold = 0.5 # distance from the boundary at which the sheep will start moving away from the boundary
        self.point_dist = 0.1 # distance between the point that is controlled using the vectpr headings on the sheep and the center of the sheep
        self.arena_rep = 0.5 # repulsion force from the boundary


    def init_robots(self, initial_positions):
        robots = []
        for pos in initial_positions:
            robots.append(DifferentialDriveRobot(pos, self.distance_between_wheels, self.wheel_radius))
        return robots

    def step(self, action):
        # check if the action is valid
        assert len(action) == self.num_sheepdogs * 2, "Invalid action! Incorrect number of actions."
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

    def render(self, mode="human", fps=1):
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
            self.clock.tick(fps)  # Set frame rate 

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
        """
        Update the positions of the sheep based on predefined behavior.
        """

        # loop over all the sheep and update their positions
        for i in range(self.num_sheepdogs, len(self.robots)):
            # get the current sheep position
            x, y, theta = self.robots[i].get_state()

            # calculate current heading
            vec_curr = np.array([np.cos(theta), np.sin(theta)])
            vec_curr = vec_curr / np.linalg.norm(vec_curr) # normalize the vector

            # initialize the desired heading vector
            vec_desired = np.array([0.0, 0.0])

            # first check if the sheep is close to the boundary
            if x < 0.0 or x > self.arena_length or y < 0.0 or y > self.arena_width:
                # calculate the vector perpendicular to the arena boundary
                vec_boundary = np.array([0.0, 0.0])
                if x < 0.0 + self.arena_threshold:
                    vec_boundary[0] = 1.0
                elif x > self.arena_length - self.arena_threshold:
                    vec_boundary[0] = -1.0
                if y < 0.0 + self.arena_threshold:
                    vec_boundary[1] = 1.0
                elif y > self.arena_width - self.arena_threshold:
                    vec_boundary[1] = -1.0

                # cancel out the component of the current heading vector that is perpendicular to the boundary
                # vec_desired = np.subtract(vec_curr, np.dot(vec_curr, vec_boundary) * vec_boundary)
                # vec_desired = vec_desired / np.linalg.norm(vec_desired) # normalize the vector

                # add the boundary vector to the desired heading vector
                vec_desired = np.add(vec_desired, self.arena_rep*vec_boundary)
                vec_desired = vec_desired / np.linalg.norm(vec_desired) # normalize the vector

                # use the diff drive motion model to calculate the wheel velocities
                wheel_velocities = self.diff_drive_motion_model(vec_desired, [x, y, theta])

                # update the sheep position based on the wheel velocities
                self.robots[i].update_position(wheel_velocities[0], wheel_velocities[1])
                continue

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

    def diff_drive_motion_model(self, vec_desired, pose) -> np.array:
        """
        Compute the wheel velocities for the sheep based on the desired and current heading vectors.

        Args:
            vec_desired (np.array): Desired heading vector
            pose (np.array): Current position and orientation of the sheep

        Returns:
            np.array: Wheel velocities for the sheep
        """

        # calculate the angle for the desired heading vector
        vec_desired = vec_desired / np.linalg.norm(vec_desired) # normalize the vector
        des_angle = np.arctan2(vec_desired[1], vec_desired[0])

        # calculate the angle difference
        angle_diff = des_angle - pose[2]

        # normalize the angle difference
        angle_diff = (angle_diff + np.pi) % (2 * np.pi) - np.pi

        # calculate the body frame forward velocity 
        v_b = np.cos(angle_diff)

        # calculate the body frame angular velocity
        w_b = np.sin(angle_diff) / self.point_dist

        # calculate the wheel velocities
        left_wheel_velocity = (2 * v_b - w_b * self.distance_between_wheels) / (2 * self.wheel_radius)
        right_wheel_velocity = (2 * v_b + w_b * self.distance_between_wheels) / (2 * self.wheel_radius)

        wheel_velocities = np.array([left_wheel_velocity, right_wheel_velocity])

        # normalize and scale the wheel velocities
        max_wheel_velocity = max(abs(left_wheel_velocity), abs(right_wheel_velocity))
        wheel_velocities = wheel_velocities / max_wheel_velocity * self.max_sheep_wheel_vel

        return wheel_velocities
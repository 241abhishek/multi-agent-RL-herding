import gymnasium as gym
from gymnasium import spaces
import numpy as np
from herdingrobot import DifferentialDriveRobot
import pygame
from enum import Enum

class TrainState(Enum):
    """
    Track the training state of the environment.

    The environment reset method can be called in different states to reset the environment to a specific configuration. 

    The following states are defined to control the spawn location of the robots in the environment:
    - RANDOM: Random spawn locations for all robots
    - CLOSE: Sheep spawn close to the goal point
    - MEDIUM: Sheep spawn at a medium distance from the goal point
    - FAR: Sheep spawn far from the goal point
    """

    RANDOM = 0
    CLOSE = 1
    MEDIUM = 2
    FAR = 3
    

class HerdingSimEnv(gym.Env):
    def __init__(self, arena_length, arena_width, num_sheep, num_sheepdogs, 
                 robot_distance_between_wheels, robot_wheel_radius, max_wheel_velocity, 
                 initial_positions=None, goal_point=None, train_state=TrainState.MEDIUM):
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
            goal_point (List[float]): Goal point for the sheep herd to reach [x, y]

        """
        super(HerdingSimEnv, self).__init__()
        self.arena_length = arena_length
        self.arena_width = arena_width
        self.num_sheep = num_sheep
        self.num_sheepdogs = num_sheepdogs
        self.distance_between_wheels = robot_distance_between_wheels
        self.wheel_radius = robot_wheel_radius
        self.max_wheel_velocity = max_wheel_velocity
        self.train_state = train_state
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
        self.action_space = spaces.Box(low=-1, high=1, 
                                       shape=(num_sheepdogs * 2,), dtype=np.float32)
        # Observation space is positions and orientations of all robots plus the goal point
        self.observation_space = spaces.Box(low=-1, high=1, 
                                            shape=(num_sheep + num_sheepdogs + 1, 3), dtype=np.float32)

        # Convert to pygame units (pixels)
        self.scale_factor = 50  # 1 meter = 50 pixels, adjust as needed
        self.arena_length_px = self.arena_length * self.scale_factor
        self.arena_width_px = self.arena_width * self.scale_factor

        # save the frames at each step for future rendering
        self.frames = []

        # set max number of steps for each episode
        self.curr_iter = 0
        self.MAX_STEPS = 2500

        # simulation parameters
        # these are hardcoded parameters that define the behavior of the sheep
        self.max_sheep_wheel_vel = 5.0 # max wheel velocity of the sheep 
        self.n = 5 # number of nearest neighbours to consider for attraction
        self.r_s = 2.0 # sheep-dog detection distance
        self.r_a = 0.4 # sheep-sheep interaction distance 
        self.p_a = 5.0 # relative strength of repulsion from other agents
        self.c = 0.2 # relative strength of attraction to the n nearest neighbours
        self.p_s = 1.0 # relative strength of repulsion from the sheep-dogs

        # arena parameters
        self.point_dist = 0.1 # distance between the point that is controlled using the vector headings on the sheep and the center of the sheep
        self.arena_threshold = 0.5 # distance from the boundary at which the sheep will start moving away from the boundary
        self.arena_rep = 0.5 # repulsion force from the boundary

        # set goal point parameters for the sheep herd
        self.goal_tolreance = 1.5 # accepatable tolerance for the sheep to be considered at the goal point 
        self.goal_point = goal_point
        if self.goal_point is None:
            self.goal_point = [self.arena_length / 2, self.arena_width / 2]
        else:
            assert len(self.goal_point) == 2, "Invalid goal point! Please provide a valid goal point."
            assert 0 + self.arena_threshold <= self.goal_point[0] <= self.arena_length - self.arena_threshold, f"Invalid goal point! x-coordinate out of bounds. Coordinate should be within {self.arena_threshold} and {self.arena_length - self.arena_threshold}."
            assert 0 + self.arena_threshold <= self.goal_point[1] <= self.arena_width - self.arena_threshold, f"Invalid goal point! y-coordinate out of bounds. Coordinate should be within {self.arena_threshold} and {self.arena_width - self.arena_threshold}."

        self.prev_sheepdog_position = None
        self.prev_sheep_position = None

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
            left_wheel_velocity = action[i * 2] * self.max_wheel_velocity # scale the action to the max wheel velocity
            right_wheel_velocity = action[i * 2 + 1] * self.max_wheel_velocity # scale the action to the max wheel velocity
            self.robots[i].update_position(left_wheel_velocity, right_wheel_velocity)

            # clip the sheep-dog position if updated position is outside the arena
            x, y, _ = self.robots[i].get_state()
            x = np.clip(x, 0.0, self.arena_length)
            y = np.clip(y, 0.0, self.arena_width)
            self.robots[i].x = x
            self.robots[i].y = y

        # Update sheep positions using predefined behavior 
        self.compute_sheep_actions()

        # Gather new observations (positions, orientations)
        observations = self.get_observations()
        # normalize the observations
        observations = self.normalize_observation(observations)

        # Compute reward, terminated, and info
        reward, terminated, truncated = self.compute_reward_v5()
        info = {}

        return observations, reward, terminated, truncated, info

    def render(self, mode=None, fps=1):
        # Initialize pygame if it hasn't been already
        if mode == "human":
            if not hasattr(self, 'screen') or not isinstance(self.screen, pygame.display.get_surface().__class__):
                pygame.init()
                self.screen = pygame.display.set_mode((self.arena_length_px, self.arena_width_px))
                pygame.display.set_caption("Herding Simulation")
                self.clock = pygame.time.Clock()
        else:
            if not hasattr(self, 'screen') or not isinstance(self.screen, pygame.Surface):
                pygame.init()
                self.screen = pygame.Surface((self.arena_length_px, self.arena_width_px))
                self.clock = pygame.time.Clock()  # Clock can still be used for controlling frame rate

        # Clear the previous frame
        self.screen.fill((0, 0, 0))  # Fill screen with black

        # Draw the arena border
        pygame.draw.rect(self.screen, (255, 255, 255), pygame.Rect(0, 0, self.arena_length_px, self.arena_width_px), 2)

        # Draw the goal point with a red circel indicating the tolerance zone
        goal_x_px = int(self.goal_point[0] * self.scale_factor)
        goal_y_px = self.arena_width_px - int(self.goal_point[1] * self.scale_factor) # Flip y-axis
        pygame.draw.circle(self.screen, (255, 0, 0), (goal_x_px, goal_y_px), int(self.goal_tolreance * self.scale_factor), 1)

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
        frame = np.rot90(frame)  # Convert Pygame surface to numpy array
        # flip the frame
        frame = np.flip(frame, axis=0)
        # reorder the axes to match the expected format (channel, height, width)
        frame = np.moveaxis(frame, -1, 0)
        self.frames.append(frame)  # Convert Pygame surface to numpy array

    def generate_robot_positions(self, state, goal_point):
        """
        Generate robot positions based on the training state.
        - all generated positions are within a radius of X meters from the goal point and within the arena bounds
        - state == RANDOM -> Random spawn locations for all robots (exception)
        """

        initial_positions = []

        if state == TrainState.RANDOM:
            for _ in range(self.num_sheep + self.num_sheepdogs):
                x = np.random.uniform(0, self.arena_length)
                y = np.random.uniform(0, self.arena_width)
                theta = np.random.uniform(-np.pi, np.pi)
                initial_positions.append([x, y, theta])

        elif state == TrainState.CLOSE: # 2.5 meters from the goal point
            # print("Generating initial positions for the robots close to the goal point.")
            # Generate initial positions for all robots close to the goal point
            # all robots are within a radius of 2 meters from the goal point and within the arena bounds
            for _ in range(self.num_sheep + self.num_sheepdogs):
                x = np.random.uniform(goal_point[0] - 2.5, goal_point[0] + 2.5)
                y = np.random.uniform(goal_point[1] - 2.5, goal_point[1] + 2.5)
                # ensure the generated position is within the arena bounds
                x = np.clip(x, 0, self.arena_length)
                y = np.clip(y, 0, self.arena_width)
                theta = np.random.uniform(-np.pi, np.pi)
                initial_positions.append([x, y, theta])

        elif state == TrainState.MEDIUM: # 5 meters from the goal point
            # print("Generating initial positions for the robots at a medium distance from the goal point.")
            # Generate initial positions for all robots at a medium distance from the goal point
            # all robots are within a radius of 5 meters from the goal point and within the arena bounds
            for _ in range(self.num_sheep + self.num_sheepdogs):
                x = np.random.uniform(goal_point[0] - 5, goal_point[0] + 5)
                y = np.random.uniform(goal_point[1] - 5, goal_point[1] + 5)
                # ensure the generated position is within the arena bounds
                x = np.clip(x, 0, self.arena_length)
                y = np.clip(y, 0, self.arena_width)
                theta = np.random.uniform(-np.pi, np.pi)
                initial_positions.append([x, y, theta])

        elif state == TrainState.FAR: # 7.5 meters from the goal point
            # print("Generating initial positions for the robots far from the goal point.")
            # Generate initial positions for all robots far from the goal point
            # all robots are within a radius of 7.5 meters from the goal point and within the arena bounds
            for _ in range(self.num_sheep + self.num_sheepdogs):
                x = np.random.uniform(goal_point[0] - 7.5, goal_point[0] + 7.5)
                y = np.random.uniform(goal_point[1] - 7.5, goal_point[1] + 7.5)
                # ensure the generated position is within the arena bounds
                x = np.clip(x, 0, self.arena_length)
                y = np.clip(y, 0, self.arena_width)
                theta = np.random.uniform(-np.pi, np.pi)
                initial_positions.append([x, y, theta])

        return initial_positions
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # Reset the environment
        # Generate random goal point for the sheep herd
        goal_x = np.random.uniform(0 + self.arena_threshold, self.arena_length - self.arena_threshold)
        goal_y = np.random.uniform(0 + self.arena_threshold, self.arena_width - self.arena_threshold)
        self.goal_point = [goal_x, goal_y]

        # Generate initial positions for all robots based on the training state
        robots=[[2.0, 2.0, np.pi/4], [5.0, 5.0, np.pi/4]]
        if robots is not None:
            self.robots = self.init_robots(robots)
        else:
            initial_positions = self.generate_robot_positions(self.train_state, self.goal_point)
            self.robots = self.init_robots(initial_positions)

        # clear the frames
        self.frames = []

        # reset the step counter
        self.curr_iter = 0

        # Return the initial observation and empty info
        info = {}
        obs = self.get_observations()
        obs = self.normalize_observation(obs)

        return obs, info 

    def get_video_frames(self):
        frames = np.array(self.frames)
        # add a time dimension to the video
        frames = np.expand_dims(frames, axis=0)
        return frames

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
            if x < 0.0 + self.arena_threshold or x > self.arena_length - self.arena_threshold or \
            y < 0.0 + self.arena_threshold or y > self.arena_width - self.arena_threshold:
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

            # sort the sheep based on the distance from the current sheep
            closest_neighbors = []
            for j in range(self.num_sheepdogs, len(self.robots)):
                if i == j:
                    continue
                x_j, y_j, _ = self.robots[j].get_state()
                dist = np.linalg.norm(np.array([x - x_j, y - y_j]))
                closest_neighbors.append((dist, j))
            closest_neighbors.sort()

            # calculate the LCM (local center of mass) of the sheep withing the interaction distance
            sheep_within_r = 0
            for j in range(len(closest_neighbors)):
                if closest_neighbors[j][0] < self.r_a:
                    sheep_within_r += 1
            if sheep_within_r > 0:
                lcm = np.array([0.0, 0.0]) 
                for j in range(sheep_within_r):
                    x_j, y_j, _ = self.robots[closest_neighbors[j][1]].get_state()
                    lcm = np.add(lcm, np.array([x_j, y_j]))
                lcm = lcm / sheep_within_r

                # calculate the vector pointing away from the LCM
                vec_repulsion = np.subtract(np.array([x, y]), lcm)
                vec_repulsion = vec_repulsion / np.linalg.norm(vec_repulsion) # normalize the vector
                # add the repulsion vector to the desired heading vector
                vec_desired = np.add(vec_desired, self.p_a*vec_repulsion)
                vec_desired = vec_desired / np.linalg.norm(vec_desired) # normalize the vector

            # calculate the vector pointing away from the sheep-dogs
            sheepdog_within_r = 0
            for j in range(self.num_sheepdogs):
                x_j, y_j, _ = self.robots[j].get_state()
                dist = np.linalg.norm(np.array([x - x_j, y - y_j]))
                if dist < self.r_s:
                    sheepdog_within_r += 1
                    vec_repulsion = np.subtract(np.array([x, y]), np.array([x_j, y_j]))
                    vec_repulsion = vec_repulsion / np.linalg.norm(vec_repulsion)
                    vec_desired = np.add(vec_desired, self.p_s*vec_repulsion)
                    vec_desired = vec_desired / np.linalg.norm(vec_desired) # normalize the vector

            # calculate the vector pointing towards the n nearest neighbors
            if sheepdog_within_r > 0:
                for j in range(min(self.n, len(closest_neighbors))):
                    x_j, y_j, _ = self.robots[closest_neighbors[j][1]].get_state()
                    vec_attraction = np.subtract(np.array([x_j, y_j]), np.array([x, y]))
                    vec_attraction = vec_attraction / np.linalg.norm(vec_attraction)
                    vec_desired = np.add(vec_desired, self.c*vec_attraction)
                    vec_desired = vec_desired / np.linalg.norm(vec_desired)

            # use the diff drive motion model to calculate the wheel velocities
            if vec_desired[0] == 0.0 and vec_desired[1] == 0.0:
                wheel_velocities = np.array([0.0, 0.0])
            else:
                wheel_velocities = self.diff_drive_motion_model(vec_desired, [x, y, theta])

            # update the sheep position based on the wheel velocities
            self.robots[i].update_position(wheel_velocities[0], wheel_velocities[1])

    def get_observations(self):
        # Return positions and orientations of all robots with the goal point
        obs = []
        for robot in self.robots:
            obs.append(robot.get_state())
        # append the goal point to the observations
        goal = np.array(self.goal_point + [0.0]) # add a dummy orientation
        obs.append(goal)
        return np.array(obs)

    def normalize_observation(self, observation):
        # Normalize the observations (both positions and orientations)
        
        for i in range(len(observation)):
            # Normalize the position
            observation[i][0] = observation[i][0] / self.arena_length
            observation[i][1] = observation[i][1] / self.arena_width
            # Normalize the orientation (between -pi and pi to between -1 and 1)
            observation[i][2] = observation[i][2] / np.pi
            
        return observation

    def compute_reward_v1(self):
        """
        Compute the reward for the current state of the environment.

        Reward is calculated on the following basis:
        - Reward based on the distance of the sheep from the goal point
        - Negative reward for each time step to encourage faster convergence
        - Negative reward for allowing the sheep to spread to far from each other (not clustered)
        - Positive reward for the sheep-dogs being on the other side of the sheep herd and the goal point
        - Large positive reward for the sheep herd reaching the goal point

        Returns:
            float: Reward value
        """

        reward = 0.0

        # calculate the distance of each sheep from the goal point
        for i in range(self.num_sheepdogs, len(self.robots)):
            x, y, _ = self.robots[i].get_state()
            dist = np.linalg.norm(np.array([x, y]) - np.array(self.goal_point))
            reward += -dist

        # add negative reward for each time step
        reward += -1.0

        # calculate the Global Center of Mass (GCM) of the sheep herd
        sheep_gcm = np.array([0.0, 0.0])
        for i in range(self.num_sheepdogs, len(self.robots)):
            x, y, _ = self.robots[i].get_state()
            sheep_gcm = np.add(sheep_gcm, np.array([x, y]))
        sheep_gcm = sheep_gcm / self.num_sheep

        # calculate the distance of each sheep from the GCM
        for i in range(self.num_sheepdogs, len(self.robots)):
            x, y, _ = self.robots[i].get_state()
            dist = np.linalg.norm(np.array([x, y]) - sheep_gcm)
            reward += -dist + 1.5 # 1.5 is added as a buffer to specify the min distance at which no penalty in incurred 

        # determine if the sheep-dogs are on the other side of the sheep herd and the goal point
        sheepdog_gcm = np.array([0.0, 0.0])
        for i in range(self.num_sheepdogs):
            x, y, _ = self.robots[i].get_state()
            sheepdog_gcm = np.add(sheepdog_gcm, np.array([x, y]))
        sheepdog_gcm = sheepdog_gcm / self.num_sheepdogs

        # calculate the equation of the line perpendicular to the line connecting the sheep GCM and the goal point and passing through the sheep GCM
        m = (sheep_gcm[1] - self.goal_point[1]) / (sheep_gcm[0] - self.goal_point[0])
        m_perp = -1.0 / m
        c = sheep_gcm[1] - m_perp * sheep_gcm[0]

        # determine the side of the sheep-dogs with respect to the line
        sheep_dog_side = np.sign(sheepdog_gcm[1] - m_perp * sheepdog_gcm[0] - c)

        # determine the side of the goal point with respect to the line
        goal_side = np.sign(self.goal_point[1] - m_perp * self.goal_point[0] - c)

        # check if the sheep-dogs are on the other side of the sheep herd and the goal point
        if sheep_dog_side != goal_side:
            reward += 25.0

        # check if the sheep herd has reached the goal point
        terminated = self.check_terminated()
        if terminated:
            reward += 2500.0

        return reward, terminated
    
    def compute_reward_v2(self):
        """
        Compute the reward for the current state of the environment.

        Reward is calculated on the following basis:
        - Reward based on the distance of the sheep from the goal point
        - Negative reward for each time step to encourage faster convergence
        - Large positive reward for the sheep herd reaching the goal point

        Returns:
            float: Reward value
        """

        reward = 0.0

        # calculate the distance of each sheep from the goal point
        for i in range(self.num_sheepdogs, len(self.robots)):
            x, y, _ = self.robots[i].get_state()
            dist = np.linalg.norm(np.array([x, y]) - np.array(self.goal_point))
            reward += -dist

        # add negative reward for each time step
        reward += -25.0

        # check if the sheep herd has reached the goal point
        terminated = self.check_terminated()
        if terminated:
            reward += 2500.0

        return reward, terminated

    def compute_reward_v3(self):
        """
        Compute the reward for the current state of the environment.

        Reward is calculated on the following basis:
        - Reward based on the distance of the sheep from the goal point
        - Large positive reward for the sheep herd reaching the goal point
        - Negative reward for the sheep dog staying in the same position in two consecutive time steps
        - Positive reward for the sheep dog moving

        Returns:
            float: Reward value
        """

        reward = 0.0

        # calculate the distance of each sheep from the goal point
        for i in range(self.num_sheepdogs, len(self.robots)):
            x, y, _ = self.robots[i].get_state()
            dist = np.linalg.norm(np.array([x, y]) - np.array(self.goal_point))
            reward += -dist

        # check if the sheep herd has reached the goal point
        terminated = self.check_terminated()
        if terminated:
            reward += 2500.0

        # check if the sheep dogs have stayed in the same position in two consecutive time steps
        tolerance = 0.025
        if self.prev_sheepdog_position is not None:
            same_position = True
            for i in range(self.num_sheepdogs):
                x, y, _ = self.robots[i].get_state()
                prev_x, prev_y, _ = self.prev_sheepdog_position[i]
                dist = np.linalg.norm(np.array([x, y]) - np.array([prev_x, prev_y]))
                if dist > tolerance:
                    same_position = False
                    break
            if same_position:
                reward += -50.0
            else:
                reward += 50.0

        # save the current position for the next time step
        self.prev_sheepdog_position = []
        for i in range(self.num_sheepdogs):
            x, y, theta = self.robots[i].get_state()
            self.prev_sheepdog_position.append([x, y, theta])

        return reward, terminated

    def compute_reward_v4(self):
        """
        Compute the reward for the current state of the environment.

        Reward is calculated on the following basis:
        - Large positive reward for the sheep herd reaching the goal point
        - Negative reward for the sheep dog staying in the same position in two consecutive time steps
        - Positive reward for the sheep dog moving
        - Positive reward for moving the sheep closer to the goal point

        Returns:
            float: Reward value
        """

        reward = 0.0

        # check if the sheep herd has reached the goal point
        terminated = self.check_terminated()
        if terminated:
            reward += 2500.0
            return reward, terminated

        # check if the sheep dogs have stayed in the same position in two consecutive time steps
        tolerance = 0.025
        if self.prev_sheepdog_position is not None:
            same_position = True
            for i in range(self.num_sheepdogs):
                x, y, _ = self.robots[i].get_state()
                prev_x, prev_y, _ = self.prev_sheepdog_position[i]
                dist = np.linalg.norm(np.array([x, y]) - np.array([prev_x, prev_y]))
                if dist > tolerance:
                    same_position = False
                    break
            if not same_position:
                reward += 50.0

        # save the current sheepdog position for the next time step
        self.prev_sheepdog_position = []
        for i in range(self.num_sheepdogs):
            x, y, theta = self.robots[i].get_state()
            self.prev_sheepdog_position.append([x, y, theta])

        # check if any of the sheep have moved closer to the goal point
        if self.prev_sheep_position is not None:
            for i in range(self.num_sheepdogs, len(self.robots)):
                x, y, _ = self.robots[i].get_state()
                prev_x, prev_y, _ = self.prev_sheep_position[i - self.num_sheepdogs]
                dist = np.linalg.norm(np.array([x, y]) - np.array(self.goal_point))
                prev_dist = np.linalg.norm(np.array([prev_x, prev_y]) - np.array(self.goal_point))
                if dist < prev_dist:
                    reward += 25.0

        # save the current sheep position for the next time step
        self.prev_sheep_position = []
        for i in range(self.num_sheepdogs, len(self.robots)):
            x, y, theta = self.robots[i].get_state()
            self.prev_sheep_position.append([x, y, theta])

        return reward, terminated

    def compute_reward_v5(self):
        """
        Compute the reward for the current state of the environment.

        Reward is calculated on the following basis:
        - Large positive reward for the sheep herd reaching the goal point
        - Positive reward for moving the sheep closer to the goal point for sheep outside the goal zone

        Returns:
            float: Reward value
        """

        reward = 0.0

        # check if the sheep herd has reached the goal point
        terminated = self.check_terminated()
        truncated = self.check_truncated()
        if terminated:
            reward += 10000.0
            return reward, terminated, truncated

        if truncated:
            reward += -1.0
            return reward, terminated, truncated

        # add negative reward for each time step
        reward += -1.0

        # check if any of the sheep have moved closer to the goal point
        if self.prev_sheep_position is not None:
            for i in range(self.num_sheepdogs, len(self.robots)):
                x, y, _ = self.robots[i].get_state()
                prev_x, prev_y, _ = self.prev_sheep_position[i - self.num_sheepdogs]
                dist = np.linalg.norm(np.array([x, y]) - np.array(self.goal_point))
                prev_dist = np.linalg.norm(np.array([prev_x, prev_y]) - np.array(self.goal_point))
                # check if dist is less than the previous distance and that the sheep is outside the goal tolerance zone
                if dist < prev_dist and dist > self.goal_tolreance:
                    reward += 100.0

        # save the current sheep position for the next time step
        self.prev_sheep_position = []
        for i in range(self.num_sheepdogs, len(self.robots)):
            x, y, theta = self.robots[i].get_state()
            self.prev_sheep_position.append([x, y, theta])

        return reward, terminated, truncated

    def check_terminated(self):
        # check if the sheep herd has reached the goal point
        terminated = True
        for i in range(self.num_sheepdogs, len(self.robots)):
            x, y, _ = self.robots[i].get_state()
            dist = np.linalg.norm(np.array([x, y]) - np.array(self.goal_point))
            if dist > self.goal_tolreance:
                terminated = False
        return terminated

    def check_truncated(self):
        # check if the episode is truncated
        self.curr_iter += 1
        if self.curr_iter >= self.MAX_STEPS:
            return True
        
        return False

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
import gym
from gym import spaces
import numpy as np
from herdingrobot import DifferentialDriveRobot

class HerdingSimEnv(gym.Env):
    def __init__(self, arena_length, arena_width, num_sheep, num_sheepdogs, 
                 robot_distance_between_wheels, robot_wheel_radius, max_wheel_velocity, 
                 initial_positions):
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

    def reset(self):
        # Reset the environment to the initial state
        pass

    def render(self, mode="human"):
        pass

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

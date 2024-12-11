import numpy as np

class DifferentialDriveRobot:
    def __init__(self, initial_position, distance_between_wheels, wheel_radius):
        self.x, self.y, self.theta = initial_position # theta is limited to [-pi, pi]
        self.distance_between_wheels = distance_between_wheels
        self.wheel_radius = wheel_radius
        self.prev_velocity = np.array([0.0, 0.0])
        self.max_acceleration = float("inf") # m/s^2
        self.omega_acceleration = 7.5 # rad/s^2

    def update_position(self, left_wheel_velocity, right_wheel_velocity, dt=0.1):
        # Differential drive kinematics
        v = self.wheel_radius * (left_wheel_velocity + right_wheel_velocity) / 2.0
        omega = self.wheel_radius * (right_wheel_velocity - left_wheel_velocity) / self.distance_between_wheels

        # apply artificial acceleration limit on velocity changes
        v = np.clip(v, self.prev_velocity[0] - self.max_acceleration * dt, self.prev_velocity[0] + self.max_acceleration * dt)        
        omega = np.clip(omega, self.prev_velocity[1] - self.omega_acceleration * dt, self.prev_velocity[1] + self.omega_acceleration * dt)

        self.x += v * np.cos(self.theta) * dt
        self.y += v * np.sin(self.theta) * dt
        self.theta += omega * dt

        # Limit theta to [-pi, pi]
        if self.theta > np.pi:
            self.theta -= 2*np.pi
        if self.theta < -np.pi:
            self.theta += 2*np.pi

        # update previous velocity
        self.prev_velocity = np.array([v, omega])

    def get_state(self):
        return np.array([self.x, self.y, self.theta])
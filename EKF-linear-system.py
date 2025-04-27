""""
    .The code simulates an object moving in 1D (along a straight line) 
    with constant velocity.
    .It generates noisy measurements of the position of the object (only position, not velocity).
    .It uses an Extended Kalman Filter (EKF) to estimate the object's true position 
    and velocity over time based on those noisy measurements.

"""

import numpy as np
import matplotlib.pyplot as plt

# Define system dynamics function (motion model)
def f(x, u):     #f(Position , Velocity).
    dt = 1.0     # time step
    # State transition matrix for constant velocity model
    F = np.array([
        [1, dt],
        [0, 1]
    ])
    return F @ x  # Predict next state

# Define measurement function (how we observe the state)
def h(x):
    return np.array([x[0]])  # Only position is measured

# Jacobian of the motion model (constant for linear motion model)
def F_jacobian(x, u):
    dt = 1.0
    return np.array([
        [1, dt],
        [0, 1]
    ])

# Jacobian of the measurement model (constant for measuring position only)
def H_jacobian(x):
    return np.array([
        [1, 0]
    ])

# Extended Kalman Filter class
class ExtendedKalmanFilter:
    def __init__(self, f, h, F_jacobian, H_jacobian, Q, R, x0, P0):
        self.f = f  # motion model
        self.h = h  # measurement model
        self.F_jacobian = F_jacobian  # Jacobian of motion model
        self.H_jacobian = H_jacobian  # Jacobian of measurement model
        self.Q = Q  # process noise covariance
        self.R = R  # measurement noise covariance
        self.x = x0  # initial state estimate
        self.P = P0  # initial error covariance

    # Prediction step
    def predict(self, u=None):
        self.x = self.f(self.x, u)  # predict next state
        F = self.F_jacobian(self.x, u)  # compute Jacobian
        self.P = F @ self.P @ F.T + self.Q  # update error covariance

    # Update step
    def update(self, z):
        H = self.H_jacobian(self.x)  # compute Jacobian of measurement function
        y = z - self.h(self.x)  # Innovation (measurement residual)
        S = H @ self.P @ H.T + self.R  # Innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y  # update state estimate
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P  # update error covariance

    # Retrieve current state estimate
    def get_state(self):
        return self.x

# ====== Simulation setup ======

# Initial state: position = 0, velocity = 1
x0 = np.array([0, 1])
# Initial uncertainty: large variance
P0 = np.eye(2) * 500

# Process noise covariance: small noise
Q = np.eye(2) * 0.01
# Measurement noise covariance: more noisy measurements
R = np.eye(1) * 10

# Initialize EKF
ekf = ExtendedKalmanFilter(f, h, F_jacobian, H_jacobian, Q, R, x0, P0)

# Simulate ground truth motion (no noise)
true_positions = []
true_velocity = 1.0
position = 0.0

# Simulate noisy measurements
measurements = []
np.random.seed(0)  # for reproducibility

for i in range(20):
    position += true_velocity * 1.0  # Update true position
    true_positions.append(position)
    # Add measurement noise
    noisy_measurement = position + np.random.normal(0, np.sqrt(R[0, 0]))
    measurements.append(noisy_measurement)

# ====== Apply EKF to measurements ======

estimated_positions = []
estimated_velocities = []

for z in measurements:
    ekf.predict()  # Predict next state
    ekf.update(np.array([z]))  # Update with measurement
    state = ekf.get_state()  # Get current state estimate
    estimated_positions.append(state[0])
    estimated_velocities.append(state[1])

# ====== Plotting results ======

time_steps = np.arange(len(measurements))

plt.figure(figsize=(12,6))
plt.plot(time_steps, true_positions, label="True Position", color="g")
plt.plot(time_steps, measurements, 'rx', label="Measurements")
plt.plot(time_steps, estimated_positions, label="EKF Estimated Position", color="b")
plt.xlabel('Time Step')
plt.ylabel('Position')
plt.title('Extended Kalman Filter Tracking')
plt.legend()
plt.grid(True)
plt.show()

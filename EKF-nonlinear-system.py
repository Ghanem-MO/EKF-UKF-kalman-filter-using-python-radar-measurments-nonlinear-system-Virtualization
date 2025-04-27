""""
    The code simulates a scenario where a "radar" observes an object 
    moving in two dimensions (X, Y) with a constant velocity, 
    while there is noise in the measurements. Then,
    it uses an Extended Kalman Filter (EKF) to accurately estimate 
    the object's position from the noisy measurements.
"""

import numpy as np
import matplotlib.pyplot as plt

# Define the motion model: state x = [px, py, vx, vy] (position and velocity in 2D)
def f(x, u):
    """
    Motion model to predict the next state based on constant velocity model.
    x: state vector [px, py, vx, vy]
    u: control input (not used in this case)
    Returns: predicted state [px_new, py_new, vx, vy]
    """
    dt = 1.0  # time step
    F = np.array([
        [1, 0, dt, 0],  # update x position
        [0, 1, 0, dt],  # update y position
        [0, 0, 1, 0],   # velocity x remains same
        [0, 0, 0, 1]    # velocity y remains same
    ])
    return np.dot(F, x)  # predicted next state


def h(x):
    """
    Measurement model: radar measures range and bearing (r, theta).
    x: state vector [px, py, vx, vy]
    Returns: measurement [range, bearing]
    """
    px, py, vx, vy = x
    range_ = np.sqrt(px**2 + py**2)  # Euclidean distance
    bearing = np.arctan2(py, px)     # Angle relative to x-axis
    return np.array([range_, bearing])

# Jacobian of the motion model (* linear here, so constant)
def F_jacobian(x, u):
    """
    Jacobian of the motion model (constant for linear system).
    """
    dt = 1.0
    return np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

# Jacobian of the measurement model (nonlinear system)
def H_jacobian(x):
    """
    Jacobian of the measurement model for radar system.
    Handles the partial derivatives of range and bearing with respect to the state.
    """
    px, py, vx, vy = x
    range_sq = px**2 + py**2
    range_ = np.sqrt(range_sq)

    # Prevent division by zero if the position is at the origin
    if range_sq == 0:
        raise ValueError("Division by zero in Jacobian calculation")

    H = np.array([
        [px / range_, py / range_, 0, 0],         # partial derivatives for range
        [-py / range_sq, px / range_sq, 0, 0]     # partial derivatives for bearing
    ])
    return H

# Extended Kalman Filter class
class ExtendedKalmanFilter:
    def __init__(self, f, h, F_jacobian, H_jacobian, Q, R, x0, P0):
        """
        Extended Kalman Filter initialization.
        """
        self.f = f  # motion model
        self.h = h  # measurement model
        self.F_jacobian = F_jacobian  # Jacobian of the motion model
        self.H_jacobian = H_jacobian  # Jacobian of the measurement model
        self.Q = Q  # process noise covariance
        self.R = R  # measurement noise covariance
        self.x = x0  # initial state estimate
        self.P = P0  # initial covariance estimate

    def predict(self, u=None):
        """
        Prediction step: estimate next state based on current state and motion model.
        """
        self.x = self.f(self.x, u)  # predict next state
        F = self.F_jacobian(self.x, u)  # linearize the motion model
        self.P = np.dot(np.dot(F, self.P), F.T) + self.Q  # update covariance

    def update(self, z):
        """
        Update step: correct the state estimate using new measurement z.
        """
        H = self.H_jacobian(self.x)  # linearize the measurement model
        y = z - self.h(self.x)  # innovation (measurement residual)

        # Normalize angle between -pi and pi
        y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))

        S = np.dot(np.dot(H, self.P), H.T) + self.R  # innovation covariance
        K = np.dot(np.dot(self.P, H.T), np.linalg.inv(S))  # Kalman gain

        self.x = self.x + np.dot(K, y)  # update the state estimate
        self.P = np.dot(np.eye(len(self.x)) - np.dot(K, H), self.P)  # update covariance

    def get_state(self):
        """
        Return the current state estimate.
        """
        return self.x

# ------------------------------
# Setup simulation parameters
# ------------------------------

# Initial state: position (0,0) and velocity (1,1)
x0 = np.array([0, 0, 1, 1])

# Initial uncertainty: very high (high initial covariance)
P0 = np.eye(4) * 500

# Process noise covariance (small uncertainty in motion)
Q = np.eye(4) * 0.1

# Measurement noise covariance
R = np.diag([10, np.deg2rad(5)])  # noise in range and bearing

# Initialize the Extended Kalman Filter
ekf = ExtendedKalmanFilter(f, h, F_jacobian, H_jacobian, Q, R, x0, P0)

# Lists to store true positions and noisy measurements
true_positions = []
measurements = []

# Initialize random seed for reproducibility
np.random.seed(0)

# Start from the initial true state
state = np.array([0, 0, 1, 1])

# Simulate true trajectory and noisy radar measurements
for i in range(30):
    state = f(state, None)  # move the true state
    true_positions.append(state.copy())

    px, py, vx, vy = state
    range_true = np.sqrt(px**2 + py**2)  # True range
    bearing_true = np.arctan2(py, px)   # True bearing

    # Add noise to the true range and bearing
    range_measured = range_true + np.random.normal(0, np.sqrt(R[0, 0]))
    bearing_measured = bearing_true + np.random.normal(0, np.sqrt(R[1, 1]))
    measurements.append(np.array([range_measured, bearing_measured]))

# ------------------------------
# Run the filter on the measurements
# ------------------------------

# Lists to store estimated positions
estimated_positions = []

for z in measurements:
    ekf.predict()  # predict the next state
    ekf.update(z)  # correct with measurement
    est = ekf.get_state()
    estimated_positions.append(est.copy())

# ------------------------------
# Plotting the results
# ------------------------------

# Convert lists to arrays for easier plotting
true_positions = np.array(true_positions)
estimated_positions = np.array(estimated_positions)

# Plot true vs estimated trajectory
plt.figure(figsize=(10, 8))
plt.plot(true_positions[:, 0], true_positions[:, 1], 'g-', label='True Trajectory')
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'b--', label='EKF Estimated Trajectory')
plt.scatter(true_positions[:, 0], true_positions[:, 1], c='g', label='True Positions')
plt.scatter(estimated_positions[:, 0], estimated_positions[:, 1], c='b', label='Estimated Positions')
plt.legend()
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Extended Kalman Filter Tracking')
plt.grid(True)
plt.axis('equal')
plt.show()

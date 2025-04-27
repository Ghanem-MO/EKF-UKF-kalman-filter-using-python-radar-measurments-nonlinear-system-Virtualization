
import numpy as np
import matplotlib.pyplot as plt

# Define the motion model: state x = [px, py, vx, vy] (position and velocity in 2D)
def f(x, u):
    dt = 1.0  # time step
    F = np.array([
        [1, 0, dt, 0],  # update x position
        [0, 1, 0, dt],  # update y position
        [0, 0, 1, 0],   # velocity x remains same
        [0, 0, 0, 1]    # velocity y remains same
    ])
    return F @ x  # predicted next state

# Define the measurement model: radar measures range and bearing >>(r , theta)
def h(x):
    px, py, vx, vy = x
    range_ = np.sqrt(px**2 + py**2)           # Euclidean distance
    bearing = np.arctan2(py, px)              # Angle relative to x-axis
    return np.array([range_, bearing])

# Jacobian of the motion model (* linear here, so constant)
def F_jacobian(x, u):
    dt = 1.0
    return np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])

# Jacobian of the measurement model (>> nonlinear system)
def H_jacobian(x):
    px, py, vx, vy = x
    range_sq = px**2 + py**2
    range_ = np.sqrt(range_sq)
    
    if range_sq == 0:
        raise ValueError("Division by zero in Jacobian calculation")

    H = np.array([
        [px / range_, py / range_, 0, 0],          # partial derivatives for range
        [-py / range_sq, px / range_sq, 0, 0]      # partial derivatives for bearing
    ])
    return H

# Define the Extended Kalman Filter (EKF) class
class ExtendedKalmanFilter:
    def __init__(self, f, h, F_jacobian, H_jacobian, Q, R, x0, P0):
        self.f = f  # motion model
        self.h = h  # measurement model
        self.F_jacobian = F_jacobian  # motion model Jacobian
        self.H_jacobian = H_jacobian  # measurement model Jacobian
        self.Q = Q  # process noise covariance
        self.R = R  # measurement noise covariance
        self.x = x0  # initial state estimate
        self.P = P0  # initial covariance estimate

    # Predict the next state and covariance
    def predict(self, u=None):
        self.x = self.f(self.x, u)  # predict next state
        F = self.F_jacobian(self.x, u)  # linearize motion model
        self.P = F @ self.P @ F.T + self.Q  # predict covariance

    # Update step with a new measurement z
    def update(self, z):
        H = self.H_jacobian(self.x)  # linearize measurement model
        y = z - self.h(self.x)  # innovation (measurement residual)
        y[1] = np.arctan2(np.sin(y[1]), np.cos(y[1]))  # normalize the angle
        S = H @ self.P @ H.T + self.R  # innovation covariance
        K = self.P @ H.T @ np.linalg.inv(S)  # Kalman gain
        
        self.x = self.x + K @ y  # correct the state estimate
        self.P = (np.eye(len(self.x)) - K @ H) @ self.P  # correct the covariance estimate

    # Return the current estimated state
    def get_state(self):
        return self.x

# ------------------------------
# Setup simulation parameters
# ------------------------------

# Initial state: position (0,0) and velocity (1,1)
x0 = np.array([0, 0, 1, 1])
# Initial uncertainty: very high
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
    range_true = np.sqrt(px**2 + py**2)
    bearing_true = np.arctan2(py, px)

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

# Convert lists to arrays
true_positions = np.array(true_positions)
estimated_positions = np.array(estimated_positions)

# Plot true vs estimated trajectory
plt.figure(figsize=(10, 8))
plt.plot(true_positions[:, 0], true_positions[:, 1], 'g-', label='True Trajectory')
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'b--', label='EKF Estimated Trajectory')
plt.scatter(true_positions[:, 0], true_positions[:, 1], c='g')
plt.scatter(estimated_positions[:, 0], estimated_positions[:, 1], c='b')
plt.legend()
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Extended Kalman Filter on Nonlinear System')
plt.grid()
plt.axis('equal')
plt.show()

""""
    The code simulates a scenario where a "radar" observes an object 
    moving in two dimensions (X, Y) with a constant velocity, 
    while there is noise in the measurements. Then,
    it uses an unscented Kalman Filter (UKF) to accurately estimate 
    the object's position from the noisy measurements.
"""

import numpy as np
import matplotlib.pyplot as plt
from filterpy.kalman import UnscentedKalmanFilter as UKF
from filterpy.kalman import MerweScaledSigmaPoints

# ---------------------------------
# Define the models
# ---------------------------------

def fx(x, dt):
    """
    Motion model for constant velocity.
    x: state vector [px, py, vx, vy]
    dt: time step
    """
    F = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1]
    ])
    return np.dot(F, x)

def hx(x):
    """
    Measurement model: radar measures [range, bearing] from position.
    x: state vector [px, py, vx, vy]
    """
    px, py, vx, vy = x
    range_ = np.sqrt(px**2 + py**2)
    bearing = np.arctan2(py, px)
    return np.array([range_, bearing])

# ---------------------------------
# Initialize the UKF
# ---------------------------------

dt = 1.0  # time step

# Define sigma points
points = MerweScaledSigmaPoints(n=4, alpha=0.1, beta=2., kappa=0)

# Create UKF object
ukf = UKF(dim_x=4, dim_z=2, fx=fx, hx=hx, dt=dt, points=points)

# Initial state: position (0,0) and velocity (1,1)
ukf.x = np.array([0., 0., 1., 1.])

# Initial covariance
ukf.P = np.eye(4) * 500

# Process noise covariance
ukf.Q = np.eye(4) * 0.1

# Measurement noise covariance
ukf.R = np.diag([10, np.deg2rad(5)])  # [range noise, bearing noise]

# ---------------------------------
# Simulate true trajectory and measurements
# ---------------------------------

true_positions = []
measurements = []

np.random.seed(0)

# Start from initial true state
state = np.array([0., 0., 1., 1.])

for i in range(30):
    # Move true state
    state = fx(state, dt)
    true_positions.append(state.copy())

    # True measurements
    px, py, vx, vy = state
    range_true = np.sqrt(px**2 + py**2)
    bearing_true = np.arctan2(py, px)

    # Add noise to measurements
    range_measured = range_true + np.random.normal(0, np.sqrt(ukf.R[0, 0]))
    bearing_measured = bearing_true + np.random.normal(0, np.sqrt(ukf.R[1, 1]))
    measurements.append(np.array([range_measured, bearing_measured]))

true_positions = np.array(true_positions)
measurements = np.array(measurements)

# ---------------------------------
# Run UKF on measurements
# ---------------------------------

estimated_positions = []

for z in measurements:
    ukf.predict()
    ukf.update(z)
    estimated_positions.append(ukf.x.copy())

estimated_positions = np.array(estimated_positions)

# ---------------------------------
# Plot results
# ---------------------------------

plt.figure(figsize=(10, 8))
plt.plot(true_positions[:, 0], true_positions[:, 1], 'g-', label='True Trajectory')
plt.plot(estimated_positions[:, 0], estimated_positions[:, 1], 'b--', label='UKF Estimated Trajectory')
plt.scatter(true_positions[:, 0], true_positions[:, 1], c='g', label='True Positions')
plt.scatter(estimated_positions[:, 0], estimated_positions[:, 1], c='b', label='Estimated Positions')
plt.legend()
plt.xlabel('X Position')
plt.ylabel('Y Position')
plt.title('Unscented Kalman Filter Tracking (Radar Measurements)')
plt.grid(True)
plt.axis('equal')
plt.show()

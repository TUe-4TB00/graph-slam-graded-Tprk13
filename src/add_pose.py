
import math
import numpy as np
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate):
    # Robot motion from X(3): rotate 45 deg, move 2m forward, rotate 45 deg more.
    # Composed relative pose in X(3) local frame: (sqrt(2), sqrt(2), pi/2).
    relative_pose = gtsam.Pose2(math.sqrt(2.0), math.sqrt(2.0), math.pi / 2.0)
    graph.add(gtsam.BetweenFactorPose2(X(3), X(4), relative_pose, ODOMETRY_NOISE))

    # Initial estimate for X(4) using the nominal global pose (X(3) ~ origin frame at (4,0,0)).
    pose_4 = gtsam.Pose2(4.0 + math.sqrt(2.0), math.sqrt(2.0), math.pi / 2.0)
    initial_estimate.insert(X(4), pose_4)

    return graph, initial_estimate

import pytest
import math
import pickle
from src.add_pose import add_pose
import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X
PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

# Testing the answer to question 8.1 (adding 4th pose)
def test_add_pose():
    # Test that the add_pose function correctly adds a new pose to the graph and updates the initial estimate accordingly.
    # Load test graph and estimate files
    with open('tests/test_add_pose_graph.pkl', 'rb') as inp:
        graph = pickle.load(inp)
    with open('tests/test_add_pose_estimate.pkl', 'rb') as inp:
        initial_estimate = pickle.load(inp)        
    updated_graph, updated_estimate = add_pose(graph, initial_estimate)
    # Check that the graph and initial estimate have been updated correctly.
    pose4 = updated_estimate.atPose2(X(4))
    correct_pose = False
    if pose4 is not None:
        margin = 0.1
        correct_pose = abs(pose4.x() - (4.0 + math.sqrt(2))) < margin and abs(pose4.y() - math.sqrt(2)) < margin and abs(pose4.theta() - math.pi / 2) < margin
    last_added_factor = updated_graph.at(updated_graph.size() - 1)
    assert isinstance(last_added_factor, gtsam.BetweenFactorPose2), "The added factor should be a BetweenFactorPose2."
 
    assert pose4 is not None, "Pose X(4) should be in the initial estimate after adding a new pose."

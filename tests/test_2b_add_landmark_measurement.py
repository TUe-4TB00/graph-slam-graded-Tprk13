from src.add_landmark_measurement import add_landmark_measurement
import pickle
import math
import gtsam
import numpy as np
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

# Testing the answer to question 8.2 (adding measurement from 4th pose to landmark 2)
def optimize_graph(graph, initial_estimate):
    # Create an optimizer and optimize the graph to find the best estimates for the poses and landmarks.
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate)
    result = optimizer.optimize()
    return result

def test_add_landmark():
    # Load test graph and initial_estimate as file
    with open('tests/test_add_landmark_measurement_test_graph.pkl', 'rb') as inp:
        graph = pickle.load(inp)
    with open('tests/test_add_landmark_measurement_test_estimate.pkl', 'rb') as inp:
        initial_estimate = pickle.load(inp)
    result = optimize_graph(graph, initial_estimate)
    updated_graph = add_landmark_measurement(graph, initial_estimate, result)
    # Check that the graph and the measurements have been updated correctly.
    # Load correct_graph as file
    with open('tests/test_add_landmark_measurement_graph.pkl', 'rb') as inp:
        correct_graph = pickle.load(inp)    

    last_added_factor = updated_graph.at(updated_graph.size() - 1)
    last_added_factor_correct = correct_graph.at(correct_graph.size() - 1)
    bearing = last_added_factor.measured().bearing().theta()
    range = last_added_factor.measured().range()
    correct_bearing = last_added_factor_correct.measured().bearing().theta()
    correct_range = last_added_factor_correct.measured().range()
    assert math.isclose(bearing, correct_bearing, abs_tol=0.1), "The bearing of the added factor should be correct."
    assert math.isclose(range, correct_range, abs_tol=0.1), "The range of the added factor should be correct."


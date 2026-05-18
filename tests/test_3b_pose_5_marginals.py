
import math
import gtsam
import pickle
from gtsam.symbol_shorthand import L, X
from src.pose_5 import minimize_marginals, minimize_errors


def test_minimize_margins():
    # Load test graph and estimate files
    with open('tests/test_pose_5_marginals_graph.pkl', 'rb') as inp:
        graph = pickle.load(inp)
    with open('tests/test_pose_5_marginals_estimate.pkl', 'rb') as inp:
        initial_estimate = pickle.load(inp)  
    with open('tests/test_pose_5_marginals_correct_pose.pkl', 'rb') as inp:
        CORRECT_POSE = pickle.load(inp)
    with open('tests/test_pose_5_marginals_correct_landmark.pkl', 'rb') as inp:
        CORRECT_LANDMARK = pickle.load(inp)
    pose_4 = initial_estimate.atPose2(X(4))
    pose_options = {
        "a": gtsam.Pose2(0.0, 3.0, pose_4.theta()),
        "b": gtsam.Pose2(0.0, 0.0, pose_4.theta()),
        "c": gtsam.Pose2(4.0, 3.0, pose_4.theta()),
        "d": gtsam.Pose2(2.0, 3.0, pose_4.theta()),
    }
    chosen_pose, chosen_landmark, marginals = minimize_marginals(graph, initial_estimate, pose_options)


    correct_marginals = 0.15697937979143664
    assert math.isclose(correct_marginals,marginals, abs_tol=0.02), "Sum of marginals is incorrect"

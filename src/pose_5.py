import numpy as np
from helperfunctions import add_pose_from_global, add_landmark_measurement_from_global
import gtsam
from gtsam.symbol_shorthand import L, X

PRIOR_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.1, 0.1, 0.05]))  # (x, y, theta)
ODOMETRY_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.2, 0.2, 0.1]))  # (dx, dy, dtheta)
MEASUREMENT_NOISE = gtsam.noiseModel.Diagonal.Sigmas(np.array([0.05, 0.1]))  # (bearing, range)

def add_pose(graph, initial_estimate, pose_5):
    # Adding the initial estimate for the 5th pose using our helper function `add_pose_from_global` which also adds the odometry factor between X(4) and X(5).
    pose_4 = initial_estimate.atPose2(X(4))
    graph, initial_estimate = add_pose_from_global(
        graph=graph,
        initial_estimate=initial_estimate,
        prev_key=X(4),
        new_key=X(5),
        prev_pose=pose_4,
        new_pose_global=pose_5,
        odom_noise=ODOMETRY_NOISE
    )
    return graph, initial_estimate

def add_landmark_measurement(graph, result, pose_5, landmark):
    # Adding the measurement from X(5) to the chosen landmark using our helper function `add_landmark_measurement_from_global` which calculates the correct bearing and range from the global poses.``
    landmark_point = result.atPoint2(L(landmark))
    graph = add_landmark_measurement_from_global(
        graph=graph,
        pose_key=X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # Initialize the Levenberg-Marquardt optimizer with default parameters.
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # Perform the optimization and return the result.
    result = optimizer.optimize()
    return result

def _evaluate_option(graph, initial_estimate, pose_5, landmark):
    """Build a candidate graph/estimate copy, add X(5) + landmark measurement, and optimize."""
    graph_copy = gtsam.NonlinearFactorGraph(graph)
    estimate_copy = gtsam.Values(initial_estimate)
    graph_copy, estimate_copy = add_pose(graph_copy, estimate_copy, pose_5)
    result = optimize(graph_copy, estimate_copy)
    graph_copy = add_landmark_measurement(graph_copy, result, pose_5, landmark)
    result = optimize(graph_copy, estimate_copy)
    return graph_copy, result

def minimize_marginals(graph, initial_estimate, pose_options):
    # For each (pose, landmark) combination, evaluate how well the *observed* landmark
    # is pinned down (its marginal covariance sum). The combination with the lowest
    # observed-landmark covariance wins. The returned sum_of_marginals aggregates
    # the marginal covariance sums of both landmarks for the winning combination.
    best_pose = None
    best_landmark = None
    best_observed = float("inf")
    best_total = None

    for pose_key, pose_5 in pose_options.items():
        for landmark in (1, 2):
            graph_copy, result = _evaluate_option(graph, initial_estimate, pose_5, landmark)
            marginals = gtsam.Marginals(graph_copy, result)
            observed_metric = marginals.marginalCovariance(L(landmark)).sum()
            total_sum = (
                marginals.marginalCovariance(L(1)).sum()
                + marginals.marginalCovariance(L(2)).sum()
            )
            if observed_metric < best_observed:
                best_observed = observed_metric
                best_total = total_sum
                best_pose = pose_key
                best_landmark = landmark

    return best_pose, best_landmark, best_total

def minimize_errors(graph, initial_estimate, pose_options):
    # Compare each (pose_5, landmark) choice on the accuracy of the first three poses.
    # Ground-truth values match the noise-free measurement chain.
    truths = {
        1: (0.0, 0.0, 0.0),
        2: (2.0, 0.0, 0.0),
        3: (4.0, 0.0, 0.0),
    }

    best_pose = None
    best_landmark = None
    best_sum_err = float("inf")

    for pose_key, pose_5 in pose_options.items():
        for landmark in (1, 2):
            _, result = _evaluate_option(graph, initial_estimate, pose_5, landmark)
            list_of_errors = []
            for pose_idx in (1, 2, 3):
                p = result.atPose2(X(pose_idx))
                tx, ty, tt = truths[pose_idx]
                list_of_errors.append(abs(p.x() - tx) + abs(p.y() - ty) + abs(p.theta() - tt))
            sum_of_errors = sum(list_of_errors)
            if sum_of_errors < best_sum_err:
                best_sum_err = sum_of_errors
                best_pose = pose_key
                best_landmark = landmark

    return best_pose, best_landmark, best_sum_err

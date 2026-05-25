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
    # TODO: Initialize the optimizer 
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)

    # TODO: Perform the optimization and print the result
    result = optimizer.optimize()
    print("\nFinal Result:\n{}".format(result))

    return result

def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_sum_of_marginals = float("inf")
    best_return_sum = None

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            test_graph = gtsam.NonlinearFactorGraph()
            for i in range(graph.size()):
                test_graph.push_back(graph.at(i))

            test_initial_estimate = gtsam.Values()
            for key in initial_estimate.keys():
                try:
                    test_initial_estimate.insert(key, initial_estimate.atPose2(key))
                except RuntimeError:
                    test_initial_estimate.insert(key, initial_estimate.atPoint2(key))

            test_graph, test_initial_estimate = add_pose(
                test_graph,
                test_initial_estimate,
                pose_5
            )

            result = optimize(test_graph, test_initial_estimate)

            test_graph = add_landmark_measurement(
                test_graph,
                result,
                pose_5,
                landmark
            )

            result = optimize(test_graph, test_initial_estimate)

            marginals = gtsam.Marginals(test_graph, result)
            score = marginals.marginalCovariance(L(landmark)).sum()  

            sum_of_marginals = (marginals.marginalCovariance(L(1)).sum()
                                + marginals.marginalCovariance(L(2)).sum()
               
            )

            if score < best_sum_of_marginals:
               best_sum_of_marginals = score
               best_pose = pose_name
               best_landmark = landmark
               best_return_sum = sum_of_marginals

    sum_of_marginals = best_return_sum

    return best_pose, best_landmark, sum_of_marginals

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
        pose_key= X(5),
        pose=pose_5,
        landmark_key=L(landmark),
        landmark_point=landmark_point,
        measurement_noise=MEASUREMENT_NOISE
    )
    return graph

def optimize(graph, initial_estimate):
    # TODO: Initialize the optimizer 
    params = gtsam.LevenbergMarquardtParams()
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()
    
    print("\nFinal Result:\n{}".format(result))
    # TODO: Perform the optimization and print the result

    return result


def copy_graph_and_estimate(graph, initial_estimate):
    copied_graph = gtsam.NonlinearFactorGraph()
    for i in range(graph.size()):
        copied_graph.push_back(graph.at(i))

    copied_estimate = gtsam.Values()
    for key in initial_estimate.keys():
        try:
            copied_estimate.insert(key, initial_estimate.atPose2(key))
        except RuntimeError:
            copied_estimate.insert(key, initial_estimate.atPoint2(key))

    return copied_graph, copied_estimate


def minimize_marginals(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_score = float("inf")
    best_sum_of_marginals = None

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            test_graph, test_initial_estimate = copy_graph_and_estimate(
                graph,
                initial_estimate
            )

            test_graph, test_initial_estimate = add_pose(
                test_graph,
                test_initial_estimate,
                pose_5
            )

            result = optimize(test_graph, test_initial_estimate)

            test_graph = add_landmark_measurement(
                test_graph,
                result,
                pose_5,
                landmark
            )

            result = optimize(test_graph, test_initial_estimate)

            marginals = gtsam.Marginals(test_graph, result)

            score = marginals.marginalCovariance(L(landmark)).sum()

            sum_of_marginals = (
                marginals.marginalCovariance(L(1)).sum()
                + marginals.marginalCovariance(L(2)).sum()
            )

            if score < best_score:
                best_score = score
                best_pose = pose_name
                best_landmark = landmark
                best_sum_of_marginals = sum_of_marginals

    return best_pose, best_landmark, best_sum_of_marginals

def minimize_errors(graph, initial_estimate, pose_options):
    best_pose = None
    best_landmark = None
    best_sum_of_errors = float("inf")

    for pose_name, pose_5 in pose_options.items():
        for landmark in [1, 2]:
            test_graph, test_initial_estimate = copy_graph_and_estimate(
                graph,
                initial_estimate
            )

            test_graph, test_initial_estimate = add_pose(
                test_graph,
                test_initial_estimate,
                pose_5
            )

            result = optimize(test_graph, test_initial_estimate)

            test_graph = add_landmark_measurement(
                test_graph,
                result,
                pose_5,
                landmark
            )

            result = optimize(test_graph, test_initial_estimate)

            list_of_errors = [
                np.linalg.norm(
                    result.atPose2(X(i)).localCoordinates(
                        gtsam.Pose2((i - 1) * 2, 0, 0)
                    )
                )
                for i in range(1, 4)
            ]

            sum_of_errors = sum(list_of_errors)

            if sum_of_errors < best_sum_of_errors:
                best_sum_of_errors = sum_of_errors
                best_pose = pose_name
                best_landmark = landmark

    return best_pose, best_landmark, best_sum_of_errors
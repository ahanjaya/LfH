#!/usr/bin/env python3
import cv2
import scipy.signal as signal
import numpy as np
from typing import Tuple

from copy import deepcopy
import copy as copy_module

from rosbags.rosbag2 import Reader
from rosbags.serde import deserialize_cdr
from geometry_msgs.msg import Point, Pose2D
from nav_msgs.msg import Path

from scipy.spatial import distance
from scipy.spatial.transform import Rotation as R

####################
## Data conversion.
def rosbag_load_odometry(file_name: str, topic: str) -> Tuple[np.ndarray]:
    """TODO:"""
    pose_data = []
    vel_data = []
    odom_msg = None

    with Reader(file_name) as reader:
        for connection, time_stamp, raw_data in reader.messages():
            if connection.topic == topic:
                odom_msg = deserialize_cdr(raw_data, connection.msgtype)

                pose_data.append(
                    [
                        odom_msg.pose.pose.position.x,
                        odom_msg.pose.pose.position.y,
                        odom_msg.pose.pose.orientation.x,
                        odom_msg.pose.pose.orientation.y,
                        odom_msg.pose.pose.orientation.z,
                        odom_msg.pose.pose.orientation.w,
                    ]
                )

                vel_data.append(
                    [odom_msg.twist.twist.linear.x, odom_msg.twist.twist.angular.z]
                )

    _np_pose_data = np.array(pose_data)
    _np_vel_data = np.array(vel_data)
    print(f"Total pose: {_np_pose_data.shape}, vel: {_np_vel_data.shape}")

    return (_np_pose_data, _np_vel_data)


def convert_poses_quarternion_to_poses_2d(poses: np.ndarray) -> np.ndarray:
    """TODO:"""
    xy_poses = poses[:, :2]
    quarternion_ori = poses[:, 2:]
    euler_ori = R.from_quat(quarternion_ori).as_euler("xyz")
    yaw_thetas = np.expand_dims(euler_ori[:, -1], axis=1)

    # concatenate X, Y, Yaw Theta
    poses_2d = np.append(xy_poses, yaw_thetas, axis=1)
    return poses_2d


def window_normalization(points: np.ndarray) -> np.ndarray:
    """TODO:"""
    tx, ty = -points[0, :2]
    trans_matrix = translation_matrix(tx, ty)
    rot_matrix = inverse_rotation_matrix(angle=points[0, 2])
    refl_matrix = reflection_matrix()

    local_path = points.copy()
    local_path[:, 2] = 1
    tranform_path = rot_matrix.dot(trans_matrix).dot(local_path.T)
    # tranform_path = refl_matrix.dot(rot_matrix).dot(trans_matrix).dot(local_path.T)
    norm_path = np.asarray(tranform_path.T)

    # Calculate angle between 2 points.
    diff_points = norm_path[1:] - norm_path[:-1]
    norm_path[:-1, 2] = np.arctan2(diff_points[:, 1], diff_points[:, 0])

    return norm_path[:-1]


####################
## Inference.
def convert_pose_stamped_to_array(poses: Path) -> np.ndarray:
    """TODO:"""
    # return np.array([[p.pose.position.x, p.pose.position.y] for p in poses])

    gp = []
    for pose in poses:
        gp.append([pose.pose.position.x, pose.pose.position.y])
    gp = np.array(gp)
    x = gp[:, 0]
    y = gp[:, 1]

    try:
        xhat = signal.savgol_filter(x, 19, 3)
    except:
        xhat = x
    try:
        yhat = signal.savgol_filter(y, 19, 3)
    except:
        yhat = y

    return np.column_stack((xhat, yhat))
 

def filter_global_path_based_robot_dist(
    xy_global_path: np.ndarray, robot_pos: Pose2D
) -> np.ndarray:
    """Consume paths that closest to robot distance."""
    np_robot_pos = np.array([robot_pos.x, robot_pos.y])
    l2_robot_to_path = np.linalg.norm(np_robot_pos - xy_global_path, axis=1)
    min_idx = np.argmin(l2_robot_to_path)

    return xy_global_path[min_idx:]


def norm_error_theta(error_th: float) -> float:
    """Normalizes angle error to between [-pi, pi].

    Normalizes the argument `error_th` according to the shortest path around
    pi. Returns the normalized value bound to -pi to pi. Commonly used for
    calculating rotation directions for controllers.

    Parameters
    ----------
    error_th : float
        The unnormalized angle error.

    Returns
    -------
    float
        Normalized angle bounded to [-pi, pi].
    """
    if error_th < -np.pi:
        return error_th % np.pi
    elif error_th > np.pi:
        return -np.pi + (error_th % np.pi)

    return error_th


def robot_window_norm(
    local_xy_path: np.ndarray, robot_pos: Pose2D, len_window: int, weight_filter: float
) -> np.ndarray:
    """Translate and rotate global path relative to robot frame"""
    np_robot_pos = np.array([robot_pos.x, robot_pos.y])

    diff_translate = local_xy_path[0] - np_robot_pos
    tx, ty = -np_robot_pos - diff_translate
    trans_matrix = translation_matrix(tx, ty)
    rot_matrix = inverse_rotation_matrix(angle=robot_pos.theta)
    refl_matrix = reflection_matrix()

    local_xy_path_ = np.append(
        local_xy_path, np.ones((local_xy_path.shape[0], 1)), axis=1
    )

    tranform_path = rot_matrix.dot(trans_matrix).dot(local_xy_path_.T)
    # tranform_path = refl_matrix.dot(rot_matrix).dot(trans_matrix).dot(local_xy_path_.T)
    norm_path = np.asarray(tranform_path.T)

    # Calculate angle between 2 points.
    diff_points = norm_path[1:] - norm_path[:-1]
    norm_path[:-1, 2] = np.arctan2(diff_points[:, 1], diff_points[:, 0])
    if len(norm_path) > 1:
        norm_path[-1, 2] = norm_path[-2, 2]

    # raw_normalized_path = copy_module.deepcopy(norm_path)

    #######
    # 1. Moving average filter.
    # if len(norm_path) > len_window:
    #     ma_angle = np.convolve(norm_path[:-1, 2], np.ones(len_window), "same") / len_window
    #     norm_path[: len(ma_angle), 2] = ma_angle

    # 2. Moving Average filter with weight.
    # for idx in range(len(norm_path) - 1):
    #     norm_path[idx, 2] = (
    #         weight_filter * norm_path[idx, 2]
    #         + (1 - weight_filter) * norm_path[idx + 1, 2]
    #     )

    # return raw_normalized_path[:-1], norm_path[:-1]
    return norm_path


####################
## General.
def translation_matrix(tx, ty):
    return np.matrix(
        [
            [1, 0, tx],
            [0, 1, ty],
            [0, 0, 1],
        ]
    )


def inverse_rotation_matrix(angle):
    cos, sin = np.cos(-angle), np.sin(-angle)
    return np.matrix(
        [
            [cos, -sin, 0],
            [sin, cos, 0],
            [0, 0, 1],
        ]
    )


def reflection_matrix():
    return np.matrix(
        [
            [1, 0, 0],
            [0, -1, 0],
            [0, 0, 1],
        ]
    )


def filter_trajectory_by_dist(points: np.ndarray, dist: float) -> np.ndarray:
    """TODO:"""
    # Calculate dist of each point distance in order to get
    # the summation distance of a trajectory from a distance.
    diff_points = points[1:, :2] - points[:-1, :2]
    diff_l2_norms = np.linalg.norm(diff_points, axis=1)

    cur_dist = 0
    for idx, val in enumerate(diff_l2_norms):
        cur_dist += val
        if cur_dist > dist:
            break

    return points[:idx]


def ma_filter_trajectory(
    points: np.ndarray, len_window: int, weight_filter: float
) -> np.ndarray:
    """TODO:"""
    # points_x = np.convolve(points[:, 0], np.ones(len_window), "valid") / len_window
    # points_y = np.convolve(points[:, 1], np.ones(len_window), "valid") / len_window

    points_x = np.convolve(points[:, 0], np.ones(len_window), "valid") / len_window
    points_y = np.convolve(points[:, 1], np.ones(len_window), "valid") / len_window

    for idx in range(len(points) - 1):
        points[idx, 0] = (
            weight_filter * points[idx, 0] + (1 - weight_filter) * points[idx + 1, 0]
        )

        points[idx, 1] = (
            weight_filter * points[idx, 1] + (1 - weight_filter) * points[idx + 1, 1]
        )

    # points_x = np.expand_dims(points_x, axis=1)
    # points_y = np.expand_dims(points_y, axis=1)

    # return np.append(points_x, points_y, axis=1)
    return points


def window_to_image(
    points: np.ndarray,
    dist: float,
    robot_w: float,
    robot_l: float,
    frame_w: int,
    frame_h: int,
) -> Tuple[np.ndarray]:
    """TODO:"""
    # Scaling data.
    x_pixels = np.interp(points[:, 0], [-dist, dist], [0, frame_w - 1]).astype(np.uint)
    y_pixels = np.interp(points[:, 1], [-dist, dist], [0, frame_h - 1]).astype(np.uint)
    y_pixels = np.clip(frame_h - y_pixels, 0, frame_h - 1)

    robot_wpixel = np.interp(robot_w, [0, dist * 2], [0, frame_w - 1]).astype(np.uint)
    robot_lpixel = np.interp(robot_l, [0, dist * 2], [0, frame_h - 1]).astype(np.uint)
    theta_pos = points[:, 2]

    # Masking pixel trajectory.
    image = np.zeros((frame_h, frame_w, 3), dtype=np.uint8)
    # image[y_pixels, x_pixels] = 255
    pose_pixels = np.stack((x_pixels, y_pixels, theta_pos), axis=1)

    # Robot centre.
    robot_start_pt = (
        int(frame_w // 2 - robot_lpixel // 2),
        int(frame_h // 2 - robot_wpixel // 2),
    )
    robot_end_pt = (
        int(frame_w // 2 + robot_lpixel // 2),
        int(frame_h // 2 + robot_wpixel // 2),
    )
    cv2.rectangle(image, robot_start_pt, robot_end_pt, (255, 255, 255), -1)

    # Draw robot footprint.
    for (x, y, theta) in pose_pixels:
        theta_deg = 90 - np.rad2deg(theta)
        rect = ((int(x), int(y)), (int(robot_wpixel), int(robot_lpixel)), theta_deg)
        points = cv2.boxPoints(rect)
        box = np.int0(points)
        cv2.drawContours(image, [box], 0, (255, 255, 255), -1)

    return (image, pose_pixels)


def ray_casting(
    img: np.ndarray,
    centre_point: Point,
    num_of_point: int,
    dist: float,
    frame_w: int,
    frame_h: int,
) -> Tuple[np.ndarray]:
    """TODO"""
    x1_pixel = centre_point.x
    y1_pixel = centre_point.y
    angles_rad = np.linspace(0, np.pi * 2, num_of_point)
    eucl_dists_pixel = []
    eucl_dists_m = []

    for angle in angles_rad:
        pixel, length = 255, 1

        # Iterating over pixel until hit black pixel.
        while pixel:
            x2_pixel = int(x1_pixel + length * np.sin(angle))
            y2_pixel = int(y1_pixel + length * np.cos(angle))

            # Check pixel index still in frame size.
            if x2_pixel < frame_w and y2_pixel < frame_h:
                pixel = img[y2_pixel, x2_pixel][0]
                length += 1
            else:
                break

        # Euclidean distance in pixel for plotting purpose.
        l2_norm_pixel = distance.euclidean((x1_pixel, y1_pixel), (x2_pixel, y2_pixel))
        eucl_dists_pixel.append(l2_norm_pixel)

        # Euclidean distance in meter.
        x1_m = np.interp(x1_pixel, [0, frame_w], [0.0, dist * 2])
        y1_m = np.interp(y1_pixel, [0, frame_h], [0.0, dist * 2])
        x2_m = np.interp(x2_pixel, [0, frame_w], [0.0, dist * 2])
        y2_m = np.interp(y2_pixel, [0, frame_h], [0.0, dist * 2])

        # l2_norm_m = np.linalg.norm(np.array([x1_m, y1_m]) - np.array([x2_m, y2_m]))
        l2_norm_m = distance.euclidean((x1_m, y1_m), (x2_m, y2_m))
        eucl_dists_m.append(l2_norm_m)

    return (np.array(eucl_dists_pixel), np.array(eucl_dists_m))


def plot_ray_cast(img: np.ndarray, centre_point: Point, ray_dist_pixel: np.ndarray):
    """TODO:"""
    x1_pixel = centre_point.x
    y1_pixel = centre_point.y
    angles_rad = np.linspace(0, np.pi * 2, len(ray_dist_pixel))

    max_dist = img.shape[0] // 2  # In pixels, about 1 meter.
    for angle, length in zip(angles_rad, ray_dist_pixel):
        x1_pixel, y1_pixel = int(x1_pixel), int(y1_pixel)
        x2_pixel = int(x1_pixel + length * np.sin(angle))
        y2_pixel = int(y1_pixel + length * np.cos(angle))
        color = int((length / max_dist) * 255)
        cv2.line(img, (x1_pixel, y1_pixel), (x2_pixel, y2_pixel), (0, 0, color), 1)

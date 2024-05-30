#!/usr/bin/python3
import cv2
import os
import rospy
import rospkg
import numpy as np
import matplotlib.pyplot as plt

from geometry_msgs.msg import Point
from lfh_utils import utils_data as utils


class LfHViz:
    def __init__(self) -> None:

        self._rate = rospy.Rate(50)
        self._rospack = rospkg.RosPack()
        self._data_dir = os.path.join(self._rospack.get_path("lfh_data"), "data")
        self._rosbag_dir = os.path.join(self._rospack.get_path("lfh_data"), "rosbag")

        self.config()
        self._centre_ray_cast = Point()

        np.set_printoptions(suppress=True)
        rospy.loginfo("LfH visualize data node ready.")

    def config(self) -> None:
        """TODO: Make a global config"""
        self._odometry_topic = "/odometry/filtered"
        self._size_window = 500
        self._local_dist = 1.0 # meter

        self._robot_w = 0.43  # 430mm = 0.43m
        self._robot_l = 0.508  # 508mm = 0.508m
        self._frame_w = 1000
        self._frame_h = 1000
        self._num_of_raycast = 100
        self._num_of_vel = 2
        self._scale_max_v = rospy.get_param("/LfH/scale_max_v")

        self._max_lin_x = (
            rospy.get_param("/jackal_velocity_controller/linear/x/max_velocity")
            / self._scale_max_v
        )

        self._x_train = os.path.join(self._data_dir, "x_train.npy")
        self._y_train = os.path.join(self._data_dir, "y_train.npy")

        self._all_ray_cast = np.empty((0, self._num_of_raycast))
        self._all_vel = np.empty((0, self._num_of_vel))

        self._plot_data = True

    def run(self) -> None:
        """TODO:"""

        _, rosbag_folders, _ = next(os.walk(self._rosbag_dir))

        for rosbag_name in rosbag_folders:
            # 1. Load Odometry.
            rosbag_ = os.path.join(self._rosbag_dir, rosbag_name)
            rospy.loginfo(f"Rosbag: {rosbag_}")
            poses_with_quarternion, cmd_vel = utils.rosbag_load_odometry(
                file_name=rosbag_, topic=self._odometry_topic
            )

            # 2. Convert odometry pose to pose 2D.
            global_poses_2d = utils.convert_poses_quarternion_to_poses_2d(
                poses=poses_with_quarternion
            )

            # Plotting global trajectory, X,Y axes need to be swap.
            # plt.plot(global_poses_2d[:,1], global_poses_2d[:,0])

            # Ray Cast Data
            ray_cast_dists = np.empty((len(global_poses_2d), self._num_of_raycast))
            
            # 3. Iterating over global trajectory.
            for idx in range(len(global_poses_2d)):
                # 4. Local trajectory.
                local_window = global_poses_2d[idx : idx + self._size_window]
                # plt.plot(local_window[:,1], local_window[:,0])
                plt.scatter(local_window[:,1], local_window[:,0], s=1)

                # 5. Filter by distance.
                filtered_local_window = utils.filter_trajectory_by_dist(
                    points=local_window, dist=self._local_dist
                )
                # plt.plot(filtered_local_window[:,1], filtered_local_window[:,0])
                # plt.xlim([-1, 14])
                # plt.ylim([-1, 10])

                plt.show()
                break

                if not len(filtered_local_window):
                    break

                # 6. Translate and Rotation Normalization.
                normalized_points = utils.window_normalization(
                    points=filtered_local_window
                )
                # plt.plot(normalized_points[:,0], normalized_points[:,1])

                # 7. Generating image.
                hallc_img, pose_pixels = utils.window_to_image(
                    points=normalized_points,
                    dist=self._local_dist,
                    robot_w=self._robot_w,
                    robot_l=self._robot_l,
                    frame_w=self._frame_w,
                    frame_h=self._frame_h,
                )

                # 8. Ray Casting.
                self._centre_ray_cast.x = pose_pixels[0, 0]
                self._centre_ray_cast.y = pose_pixels[0, 1]

                ray_cast_dists[idx] = utils.ray_casting(
                    img=hallc_img,
                    centre_point=self._centre_ray_cast,
                    num_of_point=self._num_of_raycast,
                    frame_w=self._frame_w,
                    frame_h=self._frame_h,
                )

                # 9. Plot ray cast.
                utils.plot_ray_cast(
                    img=hallc_img,
                    centre_point=self._centre_ray_cast,
                    ray_dist=ray_cast_dists[idx],
                )

                # cv2.imshow("Hallc Image", hallc_img)
                # key = cv2.waitKey(1) & 0xFF
                # if key == 27:
                #     break

                # plt.show(block=False)
                # plt.pause(0.00001) # Pause for interval seconds.
                # plt.clf()
            # plt.show()


if __name__ == "__main__":
    rospy.init_node("lfh_visualize_data", anonymous=True)
    node = LfHViz()
    node.run()

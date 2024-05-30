#!/usr/bin/python3
import cv2
import os
import rospy
import rospkg
import numpy as np
import matplotlib.pyplot as plt

from geometry_msgs.msg import Point

from lfh_utils import utils
from tqdm import tqdm


class LfHConvert:
    def __init__(self) -> None:

        self._rate = rospy.Rate(50)
        self._rospack = rospkg.RosPack()
        self._data_dir = os.path.join(self._rospack.get_path("lfh_data"), "data")
        self._rosbag_dir = os.path.join(self._rospack.get_path("lfh_data"), "rosbag")
        self._video_dir = os.path.join(self._rospack.get_path("lfh_data"), "video")

        self.config()
        self._centre_ray_cast = Point()

        np.set_printoptions(suppress=True)
        rospy.loginfo("LfH convert data node ready.")

    def config(self) -> None:
        """TODO: Make a global config"""
        self._window_size = rospy.get_param("/LfH/window_size")
        self._local_dist = rospy.get_param("/LfH/local_dist")

        self._robot_w = rospy.get_param("/LfH/robot_width")
        self._robot_l = rospy.get_param("/LfH/robot_length")
        self._frame_w = rospy.get_param("/LfH/frame_width")
        self._frame_h = rospy.get_param("/LfH/frame_height")
        self._num_of_raycast = rospy.get_param("/LfH/num_of_raycast")
        self._num_of_vel = rospy.get_param("/LfH/num_of_vel")
        self._max_lin_x = rospy.get_param("/LfH/max_v")

        x_train_file = f"x_train_{self._max_lin_x:.1f}_m.npy"
        y_train_file = f"y_train_{self._max_lin_x:.1f}_m.npy"
        self._x_train_fname = os.path.join(self._data_dir, x_train_file)
        self._y_train_fname = os.path.join(self._data_dir, y_train_file)

        self._all_ray_cast = np.empty((0, self._num_of_raycast))
        self._all_vel = np.empty((0, self._num_of_vel))

        self._plot_data = rospy.get_param("/LfH/plot_data")

    def run(self) -> None:
        """TODO:"""

        self._rosbag_dir = os.path.join(self._rosbag_dir, f"{self._max_lin_x:.1f}_m")
        rospy.loginfo(self._rosbag_dir)
        _, rosbag_folders, _ = next(os.walk(self._rosbag_dir))

        for rosbag_name in rosbag_folders:
            video_name = os.path.join(self._video_dir, f"{rosbag_name}.avi")
            video_writer = cv2.VideoWriter(
                video_name,
                cv2.VideoWriter_fourcc(*"MJPG"),
                60,
                (self._frame_w, self._frame_h),
            )

            # 1. Load Odometry.
            rosbag_ = os.path.join(self._rosbag_dir, rosbag_name)
            rospy.loginfo(f"Rosbag: {rosbag_}")
            poses_with_quarternion, cmd_vel = utils.rosbag_load_odometry(
                file_name=rosbag_, topic="/odometry/filtered"
            )

            # 2. Convert odometry pose to pose 2D.
            global_poses_2d = utils.convert_poses_quarternion_to_poses_2d(
                poses=poses_with_quarternion
            )

            if self._plot_data:
                fig, ax = plt.subplots(3, 1)
                # Plotting global trajectory, X,Y axes need to be swap.
                ax[0].set_title("Global Trajectory")
                ax[0].plot(global_poses_2d[:, 1], global_poses_2d[:, 0])

            rospy.loginfo(
                f"Len global pose: {len(global_poses_2d)}, cmd vel: {len(cmd_vel)}"
            )
            if len(global_poses_2d) != len(cmd_vel):
                rospy.logerr(f"Length data is not equal!")
                break

            # Ray Cast Data
            ray_cast_pixels = np.empty((len(global_poses_2d), self._num_of_raycast))
            ray_cast_meters = np.empty((len(global_poses_2d), self._num_of_raycast))

            # 3. Iterating over global trajectory.
            # for idx in range(len(global_poses_2d)):
            for idx in tqdm(range(len(global_poses_2d)), desc="Loading..."):
                # 4. Local trajectory.
                local_window = global_poses_2d[idx : idx + self._window_size]

                # 5. Filter by distance.
                filtered_local_window = utils.filter_trajectory_by_dist(
                    points=local_window, dist=self._local_dist
                )

                if not len(filtered_local_window):
                    break

                # 6. Translate and Rotation Normalization.
                normalized_points = utils.window_normalization(
                    points=filtered_local_window
                )

                if not len(normalized_points):
                    break

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

                ray_cast_pixels[idx], ray_cast_meters[idx] = utils.ray_casting(
                    img=hallc_img,
                    centre_point=self._centre_ray_cast,
                    num_of_point=self._num_of_raycast,
                    dist=self._local_dist,
                    frame_w=self._frame_w,
                    frame_h=self._frame_h,
                )

                # 9. Plot ray cast.
                utils.plot_ray_cast(
                    img=hallc_img,
                    centre_point=self._centre_ray_cast,
                    ray_dist_pixel=ray_cast_pixels[idx],
                )

                if self._plot_data:
                    ax[1].set_title("Local Window Trajectory")
                    ax[1].plot(local_window[:, 1], local_window[:, 0])
                    ax[1].plot(filtered_local_window[:, 1], filtered_local_window[:, 0])

                    ax[1].set_xlim(ax[0].get_xlim())
                    ax[1].set_ylim(ax[0].get_ylim())

                    ax[2].set_title("Normalized Local Trajectory")
                    ax[2].plot(normalized_points[:, 0], normalized_points[:, 1])
                    ax[2].set_xlim([-1.0, 1.0])
                    ax[2].set_ylim([-1.0, 1.0])

                    cv2.imshow("Ray Image", hallc_img)
                    video_writer.write(hallc_img)
                    key = cv2.waitKey(1) & 0xFF
                    if key == 27:
                        break

                    plt.show(block=False)
                    plt.tight_layout()
                    plt.pause(0.0001)
                    ax[1].cla()
                    ax[2].cla()

                # rospy.loginfo(f"Idx: {idx}")
                # rospy.loginfo(f"Ray Cast: {ray_cast_pixels[idx]}")
                # rospy.loginfo(f"Cmd Vel: {cmd_vel[idx]}")
                # print()

            video_writer.release()
            cv2.destroyAllWindows()
            plt.close("all")

            # 10. Concatenate all rosbag data.
            self._all_ray_cast = np.concatenate(
                (self._all_ray_cast, ray_cast_meters[:idx]), axis=0
            )
            self._all_vel = np.concatenate((self._all_vel, cmd_vel[:idx]), axis=0)

            rospy.loginfo(f"All ray cast: {self._all_ray_cast.shape}")
            rospy.loginfo(f"All vel: {self._all_vel.shape}")

            # break

        # 11. Save as npy data.
        np.save(self._x_train_fname, self._all_ray_cast)
        np.save(self._y_train_fname, self._all_vel)
        rospy.loginfo(f"Saved data succesfully...")
        rospy.loginfo(f"{self._x_train_fname}")
        rospy.loginfo(f"{self._y_train_fname}")


if __name__ == "__main__":
    rospy.init_node("lfh_convert_data", anonymous=True)
    node = LfHConvert()
    node.run()

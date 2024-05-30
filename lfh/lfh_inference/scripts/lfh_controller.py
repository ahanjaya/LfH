#!/usr/bin/python3
import os
import sys
import enum
import rospy
import rospkg
import torch
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PoseStamped, Pose2D, Point

from scipy.spatial.transform import Rotation as R

from lfh_utils import utils
from lfh_utils.model import LfHModel

from copy import deepcopy
import copy as copy_module


class States(enum.Enum):
    INIT = enum.auto()
    ORIENTATION_START = enum.auto()
    GO_TO_GOAL = enum.auto()
    ORIENTATION_FINISH = enum.auto()
    FINISH = enum.auto()


class HallucinationControllerNode:
    def __init__(self) -> None:

        self.bridge = CvBridge()
        self._rospack = rospkg.RosPack()
        self._rate = rospy.Rate(60)

        self._data_dir = os.path.join(self._rospack.get_path("lfh_inference"), "data")
        self._weight_dir = os.path.join(self._rospack.get_path("lfh_config"), "weights")
        self._read_config()

        # Load model
        self._model = LfHModel(self._num_of_raycast, self._num_of_vel)
        self._model.to(self._device)

        self._model.load_state_dict(torch.load(self._ckpt_path))
        self._model.eval()

        # Publisher.
        self._vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self._vis_pub = rospy.Publisher("lfh_vis", Image, queue_size=1)
        self._raw_vis_pub = rospy.Publisher("raw_lfh_vis", Image, queue_size=1)

        self._msg_robot_pos = Pose2D()
        self._msg_goal_pos = Pose2D()
        self._msg_prev_goal_pos = Pose2D()
        self._msg_path = Path()
        self._msg_vel = Twist()
        self._ray_cast_centre = Point()
        self._err_dist = 0.0
        self._err_angle = 0.0

        # Subsriber.
        self._path_global_sub = rospy.Subscriber(
            "/move_base/NavfnROS/plan", Path, self._cb_global_path
            # "/move_base/TrajectoryPlannerROS/global_plan", Path, self._cb_global_path
        )
        self._odom_sub = rospy.Subscriber(
            "/odometry/filtered", Odometry, self._cb_robot_odom
        )
        self._goal_sub = rospy.Subscriber(
            "/move_base/current_goal", PoseStamped, self._cb_goal
        )

        rospy.loginfo("Hallucination Controller node ready.")

    def _read_config(self) -> None:
        self._window_size = rospy.get_param("/LfH/window_size")
        self._local_dist = rospy.get_param("/LfH/local_dist")

        self._robot_w = rospy.get_param("/LfH/robot_width")
        self._robot_l = rospy.get_param("/LfH/robot_length")
        self._frame_w = rospy.get_param("/LfH/frame_width")
        self._frame_h = rospy.get_param("/LfH/frame_height")

        self._len_window = rospy.get_param("/LfH/len_window")
        self._weight_filter = rospy.get_param("/LfH/weight_filter")

        self._num_of_raycast = rospy.get_param("/LfH/num_of_raycast")
        self._num_of_vel = rospy.get_param("/LfH/num_of_vel")
        self._max_lin_x = rospy.get_param("/LfH/max_v")

        self._kp_theta = rospy.get_param("/LfH/kp_theta")
        self._dist_accuracy = rospy.get_param("/LfH/dist_accuracy")
        self._angle_accuracy = rospy.get_param("/LfH/angle_accuracy")

        self._save_data = rospy.get_param("/LfH/save_data")
        self._plot_data = rospy.get_param("/LfH/plot_data")

        # LfH weights file.
        self._device = rospy.get_param("/LfH/device")
        self._ckpt_file = f"lfh_weights_{self._max_lin_x}_m.pt"
        self._ckpt_path = os.path.join(self._weight_dir, self._ckpt_file)

        if not os.path.exists(self._ckpt_path):
            rospy.logerr(f"No such file or directory: `{self._ckpt_path}`...")
            sys.exit(1)

    def _cb_goal(self, msg: PoseStamped) -> None:
        """TODO:"""
        rotation = R.from_quat(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        )
        _, _, goal_yaw = rotation.as_euler("xyz")

        self._msg_goal_pos.x = msg.pose.position.x
        self._msg_goal_pos.y = msg.pose.position.y
        self._msg_goal_pos.theta = goal_yaw
        rospy.loginfo(
            f"Goal pose: [X:{self._msg_goal_pos.x:.2f}, Y:{self._msg_goal_pos.y:.2f}, Theta:{self._msg_goal_pos.theta:.2f}]..."
        )

    def _cb_robot_odom(self, msg: Odometry) -> None:
        # roll, pitch, yaw
        rotation = R.from_quat(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ]
        )
        _, _, robot_yaw = rotation.as_euler("xyz")

        self._msg_robot_pos.x = msg.pose.pose.position.x
        self._msg_robot_pos.y = msg.pose.pose.position.y
        self._msg_robot_pos.theta = robot_yaw

    def _cb_global_path(self, msg: Path) -> None:
        """TODO:"""
        self._msg_path = msg

    def _lfh_controller(self) -> None:
        """TODO:"""
        # Extract global path msg to array poses.
        np_xy_global_path = utils.convert_pose_stamped_to_array(
            poses=self._msg_path.poses
        )

        # Consuming global path based on robot curent distance.
        np_filtered_global_path = utils.filter_global_path_based_robot_dist(
            xy_global_path=np_xy_global_path, robot_pos=self._msg_robot_pos
        )

        # Filter local window by distance.
        np_local_window = np_filtered_global_path[: self._window_size]
        np_filtered_local_window = utils.filter_trajectory_by_dist(
            points=np_local_window, dist=self._local_dist
        )

        np_ma_local_window = utils.ma_filter_trajectory(
            points=np_filtered_local_window,
            len_window=self._len_window,
            weight_filter=self._weight_filter,
        )

        if len(np_filtered_local_window) < 1:
            return

        # Normalization local window trajectory.
        normalized_path = utils.robot_window_norm(
            # local_xy_path=np_filtered_local_window,
            local_xy_path=np_ma_local_window,
            robot_pos=self._msg_robot_pos,
            len_window=self._len_window,
            weight_filter=self._weight_filter,
        )

        # Generating image.
        hallc_img, pose_pixels = utils.window_to_image(
            points=normalized_path,
            dist=self._local_dist,
            robot_w=self._robot_w,
            robot_l=self._robot_l,
            frame_w=self._frame_w,
            frame_h=self._frame_h,
        )

        # 2D ray cast LIDAR.
        self._ray_cast_centre.x, self._ray_cast_centre.y = pose_pixels[0][:2]

        lidar_pixel, lidar_m = utils.ray_casting(
            img=hallc_img,
            centre_point=self._ray_cast_centre,
            num_of_point=self._num_of_raycast,
            dist=self._local_dist,
            frame_w=self._frame_w,
            frame_h=self._frame_h,
        )

        if self._plot_data:
            utils.plot_ray_cast(
                img=hallc_img,
                centre_point=self._ray_cast_centre,
                ray_dist_pixel=lidar_pixel,
            )
            _msg_img = self.bridge.cv2_to_imgmsg(hallc_img)
            self._vis_pub.publish(_msg_img)

        # Predict the velocity.
        converted_laser_scan = (
            torch.tensor(lidar_m).type(torch.float32).to(self._device)
        )
        pred_vel = self._model(converted_laser_scan)
        self._msg_vel.linear.x, self._msg_vel.angular.z = pred_vel
        self._vel_pub.publish(self._msg_vel)

    def _correct_orientation(self, target_angle: float) -> None:
        """TODO:"""
        self._err_angle = utils.norm_error_theta(
            target_angle - self._msg_robot_pos.theta
        )
        vel_theta = self._err_angle * self._kp_theta

        self._msg_vel.linear.x = 0.0
        self._msg_vel.angular.z = vel_theta
        self._vel_pub.publish(self._msg_vel)

        return np.absolute(self._err_angle) < self._angle_accuracy

    def _robot_reach_goal(self) -> None:
        """TODO:"""
        robot_pos = np.array([self._msg_robot_pos.x, self._msg_robot_pos.y])
        goal_pos = np.array([self._msg_goal_pos.x, self._msg_goal_pos.y])
        self._err_dist = np.linalg.norm(robot_pos - goal_pos)
        # rospy.loginfo(f"Err dist: {self._err_dist:.2f}")

        return self._err_dist < self._dist_accuracy

    def run(self) -> None:
        """TODO:"""
        state = States.INIT

        while not rospy.is_shutdown():
            #### Interupt Handling ####
            if self._msg_goal_pos != self._msg_prev_goal_pos:
                state = States.ORIENTATION_START
            self._msg_prev_goal_pos = copy_module.deepcopy(self._msg_goal_pos)

            #### State-Machine ####
            # rospy.loginfo(f"State: {state}...")
            if state == States.INIT:
                # Clear image.
                dummy_img = np.zeros((self._frame_h, self._frame_w), dtype=np.uint8)
                _msg_img = self.bridge.cv2_to_imgmsg(dummy_img)
                self._vis_pub.publish(_msg_img)

            elif state == States.ORIENTATION_START:
                target_ori = np.arctan2(
                    self._msg_goal_pos.y - self._msg_robot_pos.y,
                    self._msg_goal_pos.x - self._msg_robot_pos.x,
                )

                # target_ori = np.arctan2(
                #     self._msg_path.poses[1].pose.position.y
                #     - self._msg_path.poses[0].pose.position.y,
                #     self._msg_path.poses[1].pose.position.x
                #     - self._msg_path.poses[0].pose.position.x,
                # )

                if self._correct_orientation(target_angle=target_ori):
                    state = States.GO_TO_GOAL

            elif state == States.GO_TO_GOAL:
                # target_ori = np.arctan2(
                #     self._msg_path.poses[1].pose.position.y
                #     - self._msg_path.poses[0].pose.position.y,
                #     self._msg_path.poses[1].pose.position.x
                #     - self._msg_path.poses[0].pose.position.x,
                # )

                # if (
                #     np.absolute(
                #         utils.norm_error_theta(target_ori - self._msg_robot_pos.theta)
                #     )
                #     > 30.0 / 180.0 * np.pi
                # ):
                #     self._correct_orientation(target_angle=target_ori)
                # else:
                #     self._lfh_controller()

                if self._robot_reach_goal():
                    state = States.ORIENTATION_FINISH
                self._lfh_controller()

            elif state == States.ORIENTATION_FINISH:
                if self._correct_orientation(target_angle=self._msg_goal_pos.theta):
                    state = States.FINISH

            elif state == States.FINISH:
                rospy.loginfo(
                    f"Goal err dist: `{self._err_dist:.2f}`, angle: `{self._err_angle:.2f}`..."
                )
                state = States.INIT

            self._rate.sleep()


if __name__ == "__main__":

    rospy.init_node("hallucination_controller", anonymous=True)
    node = HallucinationControllerNode()
    node.run()

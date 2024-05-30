#!/usr/bin/python3
import os
import sys
import enum
import rospy
import rospkg
import torch
import actionlib
import numpy as np

from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from nav_msgs.msg import Path, Odometry
from geometry_msgs.msg import Twist, PoseStamped, Pose2D, Point
from tf.transformations import euler_from_quaternion

from lfh_utils import utils_inference as utils
from lfh_utils.model import HallucinationModel
from lfh_interfaces.msg import *


class States(enum.Enum):
    INIT = enum.auto()
    ORIENTATION_START = enum.auto()
    GO_TO_GOAL = enum.auto()
    ORIENTATION_FINISH = enum.auto()
    FINISH = enum.auto()


class HallucinationControllerServer:
    def __init__(self) -> None:

        self.bridge = CvBridge()
        self.rospack = rospkg.RosPack()
        self._rate = rospy.Rate(60)
        self._ray_cast_centre = Point()
        self._err_dist = 0.0
        self._err_angle = 0.0

        self._read_config()

        # Load model
        self._model = HallucinationModel(self._raycast_points, 3)
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(self._device)

        self._model.load_state_dict(torch.load(self._ckpt_path)["model_state_dict"])
        self._model.eval()

        # Publisher.
        self._msg_vel = Twist()
        self._vel_pub = rospy.Publisher("cmd_vel", Twist, queue_size=1)
        self._vis_pub = rospy.Publisher("lfh_vis", Image, queue_size=1)
        self._goal_pub = rospy.Publisher(
            "/move_base_simple/goal", PoseStamped, queue_size=1
        )

        # Subsriber.
        self._msg_path = Path()
        self._path_global_sub = rospy.Subscriber(
            "/move_base/NavfnROS/plan", Path, self._cb_global_path
        )

        self._msg_robot_pos = Pose2D()
        self._odom_sub = rospy.Subscriber(
            "/odometry/filtered", Odometry, self._cb_robot_odom
        )

        self._msg_goal_pos = Pose2D()
        self._goal_sub = rospy.Subscriber(
            "/move_base_simple/goal", PoseStamped, self._cb_goal
        )

        # Action server.
        self._action_feedback = GotoGoalFeedback()
        self._action_result = GotoGoalResult()

        self._action_server = actionlib.SimpleActionServer(
            "go_to_goal", GotoGoalAction, execute_cb=self._cb_execute, auto_start=False
        )
        self._action_server.start()

        rospy.loginfo("Hallucination Controller Server node ready.")

    def _read_config(self) -> None:
        self._window_size = rospy.get_param("/LfH/window_size")
        self._robot_width = rospy.get_param("/LfH/robot_width")
        self._robot_length = rospy.get_param("/LfH/robot_length")
        self._frame_width = rospy.get_param("/LfH/frame_width")
        self._frame_height = rospy.get_param("/LfH/frame_height")
        self._raycast_points = rospy.get_param("/LfH/raycast_points")
        self._kp_theta = rospy.get_param("/LfH/kp_theta")
        self._dist_accuracy = rospy.get_param("/LfH/dist_accuracy")
        self._angle_accuracy = rospy.get_param("/LfH/angle_accuracy")
        self._ckpt_file = rospy.get_param("/LfH/ckpt_file")

        # LfH weights file.
        self._ckpt_path = os.path.join(
            self.rospack.get_path("lfh_inference"), "ckpts", self._ckpt_file
        )
        if not os.path.exists(self._ckpt_path):
            rospy.logerr(f"No such file or directory: `{self._ckpt_path}`...")
            sys.exit(1)

    def _cb_goal(self, msg: PoseStamped) -> None:
        """TODO:"""
        # roll, pitch, yaw
        _, _, goal_yaw = euler_from_quaternion(
            [
                msg.pose.orientation.x,
                msg.pose.orientation.y,
                msg.pose.orientation.z,
                msg.pose.orientation.w,
            ]
        )
        self._msg_goal_pos.x = msg.pose.position.x
        self._msg_goal_pos.y = msg.pose.position.y
        self._msg_goal_pos.theta = goal_yaw
        rospy.loginfo(
            f"Goal pose: [X:{self._msg_goal_pos.x:.2f}, Y:{self._msg_goal_pos.y:.2f}, Theta:{self._msg_goal_pos.theta:.2f}]..."
        )

    def _cb_robot_odom(self, msg: Odometry) -> None:
        # roll, pitch, yaw
        _, _, robot_yaw = euler_from_quaternion(
            [
                msg.pose.pose.orientation.x,
                msg.pose.pose.orientation.y,
                msg.pose.pose.orientation.z,
                msg.pose.pose.orientation.w,
            ]
        )
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
        np_filtered_local_window = utils.filter_by_dist(xy_path=np_local_window, dist=1.0)

        # Normalization local window trajectory.
        normalized_path = utils.robot_window_norm(
            local_xy_path=np_filtered_local_window, robot_pos=self._msg_robot_pos
        )

        # Generating image.
        hallc_img, pose_pixels = utils.window_to_image(
            path=normalized_path,
            robot_w=self._robot_width,
            robot_h=self._robot_length,
            frame_w=self._frame_width,
            frame_h=self._frame_height,
        )

        # 2D ray cast LIDAR.
        self._ray_cast_centre.x, self._ray_cast_centre.y = pose_pixels[0][:2]

        lidar_pixel, lidar_m = utils.ray_cast_dist(
            img=hallc_img,
            centre_point=self._ray_cast_centre,
            num_of_point=self._raycast_points,
            frame_w=self._frame_width,
            frame_h=self._frame_height,
        )

        ray_img = utils.plot_ray_cast(
            img=hallc_img,
            centre_point=self._ray_cast_centre,
            ray_dist_pixel=lidar_pixel,
        )
        _msg_img = self.bridge.cv2_to_imgmsg(ray_img)
        self._vis_pub.publish(_msg_img)

        # Predict the velocity.
        converted_laser_scan = (
            torch.tensor(lidar_m).type(torch.float32).to(self._device)
        )
        pred_vel = self._model(converted_laser_scan)
        self._msg_vel.linear.x = pred_vel[0]
        self._msg_vel.angular.z = pred_vel[2]

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

        return self._err_dist < self._dist_accuracy

    def _cb_execute(self, goal: GotoGoalGoal) -> None:
        """TODO:"""
        _goal_pos = goal.goal_pose.pose.position
        _goal_ori = goal.goal_pose.pose.orientation
        _, _, goal_yaw = euler_from_quaternion(
            [
                _goal_ori.x,
                _goal_ori.y,
                _goal_ori.z,
                _goal_ori.w,
            ]
        )
        rospy.loginfo(
            f"Request new goal pose `X:{_goal_pos.x:.2f}, Y:{_goal_pos.y:.2f} Theta:{goal_yaw:.2f}`..."
        )
        self._goal_pub.publish(goal.goal_pose)
        state = States.INIT

        while not rospy.is_shutdown():
            #### Interupt Handling ####
            if self._action_server.is_preempt_requested():
                rospy.loginfo("Previous goal is preempted...")
                self._action_server.set_preempted()
                break

            #### State-Machine ####
            # rospy.loginfo(f"State: {state}...")
            if state == States.INIT:
                # Clear image.
                dummy_img = np.zeros(
                    (self._frame_height, self._frame_width), dtype=np.uint8
                )
                _msg_img = self.bridge.cv2_to_imgmsg(dummy_img)
                self._vis_pub.publish(_msg_img)
                state = States.ORIENTATION_START

            elif state == States.ORIENTATION_START:
                target_ori = np.arctan2(
                    self._msg_goal_pos.y - self._msg_robot_pos.y,
                    self._msg_goal_pos.x - self._msg_robot_pos.x,
                )
                if self._correct_orientation(target_angle=target_ori):
                    state = States.GO_TO_GOAL

            elif state == States.GO_TO_GOAL:
                if self._robot_reach_goal():
                    state = States.ORIENTATION_FINISH
                self._lfh_controller()

            elif state == States.ORIENTATION_FINISH:
                if self._correct_orientation(target_angle=self._msg_goal_pos.theta):
                    state = States.FINISH

            elif state == States.FINISH:
                rospy.loginfo(
                    f"Finish goal: `dist:{self._err_dist:.2f}, angle:{self._err_angle:.2f}`..."
                )
                self._action_result.finish = True
                self._action_server.set_succeeded(self._action_result)
                break

            # Feedback
            self._action_feedback.state.error_distance = self._err_dist
            self._action_feedback.state.error_angle = self._err_angle
            self._action_server.publish_feedback(self._action_feedback)

            self._rate.sleep()


if __name__ == "__main__":

    rospy.init_node("hallucination_controller_server", anonymous=True)
    node = HallucinationControllerServer()
    rospy.spin()

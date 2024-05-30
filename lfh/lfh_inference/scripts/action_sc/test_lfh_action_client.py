#!/usr/bin/python3
import random
import math
import enum
import unittest
import rostest
import rospy
import actionlib

from geometry_msgs.msg import PoseStamped
from actionlib_msgs.msg import GoalStatus
from tf.transformations import quaternion_from_euler
from typing import Tuple

from lfh_utils import utils
from lfh_interfaces.msg import *


@enum.unique
class GoalStatusState(enum.IntEnum):
    """http://docs.ros.org/en/api/actionlib_msgs/html/msg/GoalStatus.html"""

    PENDING = 0
    ACTIVE = 1
    PREEMPTED = 2
    SUCCEEDED = 3
    ABORTED = 4
    REJECTED = 5
    PREEMPTING = 6
    RECALLING = 7
    RECALLED = 8
    LOST = 9


class HallucinationControllerClient:
    def __init__(self) -> None:
        """TODO:"""
        self._rate = rospy.Rate(60)
        self._err_dist = 0.0
        self._err_angle = 0.0
        self.goal_status = None
        self.goal_result = None

        # Config
        self._timeout_sec = 500

        # Action client.
        self._action_goal = GotoGoalGoal()
        self._action_client = actionlib.SimpleActionClient("go_to_goal", GotoGoalAction)

        rospy.loginfo("Wait for hallucaination controller server...")
        self._action_client.wait_for_server()
        rospy.loginfo("Hallucination Controller Client node ready.")

    def _cb_feedback(self, msg: GotoGoalFeedback) -> None:
        """TODO:"""
        self._err_dist = msg.state.error_distance
        self._err_angle = msg.state.error_angle
        # rospy.loginfo(
        #     f"Error Dist:{self._err_dist:.2f}, Angle:{self._err_angle:.2f}..."
        # )

    def _cb_done(self, status: GoalStatus, result: GotoGoalResult) -> None:
        """TODO:"""
        rospy.loginfo(f"Goal Status: {status}, Result: {result}...")
        self.goal_status = status
        self.goal_result = result

    def lfh_client(self, goal_x: float, goal_y: float, goal_theta: float) -> None:
        """TODO:"""
        self._action_goal.goal_pose.header.frame_id = "odom"
        self._action_goal.goal_pose.pose.position.x = goal_x
        self._action_goal.goal_pose.pose.position.y = goal_y

        quarternion = quaternion_from_euler(0.0, 0.0, goal_theta)
        self._action_goal.goal_pose.pose.orientation.x = quarternion[0]
        self._action_goal.goal_pose.pose.orientation.y = quarternion[1]
        self._action_goal.goal_pose.pose.orientation.z = quarternion[2]
        self._action_goal.goal_pose.pose.orientation.w = quarternion[3]

        self._action_client.send_goal(
            self._action_goal, done_cb=self._cb_done, feedback_cb=self._cb_feedback
        )

        finish = self._action_client.wait_for_result(
            timeout=rospy.Duration(self._timeout_sec)
        )
        rospy.loginfo(f"Finish before timeout: `{finish}`...")
        if not finish:
            self._action_client.cancel_goal()


class LfhClient(unittest.TestCase):
    def _random_goal(self) -> Tuple[float, float, float]:
        # Config
        self._max_x = 5.0
        self._max_y = 5.0
        self._max_theta = math.pi

        return (
            random.uniform(-self._max_x, self._max_x),
            random.uniform(-self._max_y, self._max_y),
            random.uniform(-self._max_theta, self._max_theta),
        )

    def test_run(self) -> None:
        rospy.init_node("hallucination_controller_client", anonymous=True)
        node = HallucinationControllerClient()

        for _ in range(100):
            x, y, theta = self._random_goal()
            node.lfh_client(goal_x=x, goal_y=y, goal_theta=theta)
            self.assertEqual(node.goal_status, GoalStatusState.SUCCEEDED)


if __name__ == "__main__":
    # rospy.init_node("hallucination_controller_client", anonymous=True)
    # node = HallucinationControllerClient()
    # rospy.spin()

    rostest.rosrun(
        package="lfh_inference",
        test_name="test_lfh_action_client",
        test=LfhClient,
    )

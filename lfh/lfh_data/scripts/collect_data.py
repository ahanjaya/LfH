#!/usr/bin/python3
import os
import time
import subprocess
import rospy
import rospkg
import numpy as np
from geometry_msgs.msg import Twist


class LfHData:
    def __init__(self) -> None:
        self._rate = rospy.Rate(50)
        self._rospack = rospkg.RosPack()
        self._rosbag_dir = os.path.join(self._rospack.get_path("lfh_data"), "rosbag")

        # Publisher.
        self._vel_pub = rospy.Publisher(
            "/bluetooth_teleop/cmd_vel", Twist, queue_size=1
        )

        # Messages.
        self._msg_cmd_vel = Twist()

        # Params.
        self._collect_time = rospy.get_param("/LfH/collect_time")
        self._scale_max_w = rospy.get_param("/LfH/scale_max_w")
        self._max_lin_x = rospy.get_param("/LfH/max_v")
        self._max_ang_z = (
            rospy.get_param("/jackal_velocity_controller/angular/z/max_velocity")
            / self._scale_max_w
        )

        rospy.loginfo(f"Max linear x: {self._max_lin_x}, angular z: {self._max_ang_z}")
        rospy.loginfo("LfH collect data node ready.")

    def _pub_cmd_vel(self, x: float, theta: float, run_time: float) -> None:
        """TODO:"""
        self._msg_cmd_vel.linear.x = x
        self._msg_cmd_vel.angular.z = theta
        start_time = time.time()

        while time.time() - start_time <= run_time:
            self._vel_pub.publish(self._msg_cmd_vel)
            self._rate.sleep()

    def run(self) -> None:
        """TODO:"""

        # Start collect rosbag.
        rosbag_name = f"data_{time.strftime('%m%d%H%M%S')}_{self._max_lin_x}_m"
        rosbag_file = os.path.join(self._rosbag_dir, rosbag_name)

        rosbag_process = subprocess.Popen(
            [
                "rosbag",
                "record",
                "-O" + rosbag_file,
                "/odometry/filtered",
                "/bluetooth_teleop/cmd_vel",
            ]
        )

        # Robot Walk
        vels_ang_z = np.linspace(-self._max_ang_z, self._max_ang_z, num=10)
        np.random.seed(24)
        np.random.shuffle(vels_ang_z)

        for ang_z in vels_ang_z:
            self._pub_cmd_vel(x=self._max_lin_x, theta=0.0, run_time=self._collect_time)
            self._pub_cmd_vel(
                x=self._max_lin_x, theta=ang_z, run_time=self._collect_time
            )

        # Stop collect rosbag.
        rosbag_process.terminate()


if __name__ == "__main__":
    rospy.init_node("lfh_collect_data", anonymous=True)
    node = LfHData()
    node.run()

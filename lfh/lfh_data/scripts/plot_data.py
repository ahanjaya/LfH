#!/usr/bin/python3
import os
import rospy
import rospkg
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt


class LfHPlot:
    def __init__(self) -> None:
        plt.rcParams.update({'font.size': 16})
        self._rospack = rospkg.RosPack()
        self._rosbag_dir = os.path.join(self._rospack.get_path("lfh_data"), "metric")
        self._file_name = "lfh_ma_point_w20_12_23.txt"
        self._ax_title = "LfH MA Points"

        self._file_name = "lfh_theta_ma_w20_12_22.txt"
        self._ax_title = "LfH MA Theta"
        
        self._file_name = "out_2m_eevee.txt"
        self._ax_title = "LfLH 2m/s"

        # self._file_name = "lflh_1m_per_sec.txt"
        # self._ax_title = "LfLH 1m/s"

        # self._file_name = "out_savgol_05.txt"
        # self._ax_title = "LfLH 0.5m/s SavGol"

        # self._file_name = "out_model_100.txt"
        # self._ax_title = "LfLH 1m/s Model 100"

        self._file_name = "out_1m_eevee_1.txt"
        self._ax_title = "LfLH 1m/s"

        rospy.loginfo("LfH plot data node ready.")

    def run(self) -> None:
        """TODO:"""
        txt_file = os.path.join(self._rosbag_dir, self._file_name)
        df = pd.read_csv(
            txt_file,
            delim_whitespace=True,
            names=["world_idx", "success", "collided", "timeout", "time", "nav_metic"],
        )
        world_idx = df["world_idx"].unique()
        result_rate = []

        for idx in world_idx:
            df_world = df.loc[(df["world_idx"] == idx)]

            success = df_world["success"].sum()
            fail = df_world["success"].size - success

            result_rate.append([idx, success, fail])
            # rospy.loginfo(f"World idx: {idx}, Success:{success}, Fail:{fail}")

        np_result_rate = np.array(result_rate)

        total_sucess = np.sum(np_result_rate[:,1]) / (len(world_idx)*10)
        total_fail = 1 - total_sucess

        bar_width = 4
        fig, ax = plt.subplots()
        fig.tight_layout()

        ax.bar(
            np_result_rate[:, 0],
            np_result_rate[:, 1],
            width=bar_width,
            edgecolor="gray",
            label=f"Success: {total_sucess:.2f}",
            align="center",
        )
        # ax.bar(
        #     bar_width + np_result_rate[:, 0],
        #     np_result_rate[:, 2],
        #     width=bar_width,
        #     edgecolor="gray",
        #     label=f"Fail: {total_fail:.2f}",
        #     align="center",
        # )

        ax.bar(
            np_result_rate[:, 0],
            np_result_rate[:, 2],
            bottom=np_result_rate[:,1],
            width=bar_width,
            edgecolor="gray",
            label=f"Fail: {total_fail:.2f}",
            align="center",
        )

        ax.set_xticks(
            [r for r in world_idx],
        )
        ax.set_xticklabels([str(r) for r in world_idx], rotation=90)

        # ax.set_ylim([0, 20])
        ax.set_xlabel("World Index")
        ax.set_ylabel("Num of Trial")
        ax.set_title(self._ax_title)
        ax.legend()
        plt.show()


if __name__ == "__main__":
    rospy.init_node("lfh_plot_data", anonymous=True)
    node = LfHPlot()
    node.run()

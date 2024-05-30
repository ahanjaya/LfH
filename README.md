# Learning from Hallucination (LfH) - Proof of Concept (PoC)

## Description
* This repository contains packages for simulating [Learning from Hallucination](https://www.cs.utexas.edu/~xiao/Research/LfH/LfH.html) paper. Using the same robot as [BARN Challenge](https://www.cs.utexas.edu/~xiao/BARN_Challenge/BARN_Challenge.html).

## Getting started
### ClearPath Jackal for [noetic](https://docs.ros.org/en/noetic/api/jackal_tutorials/html/index.html)
* Install ROS Noetic, make sure you also installed Gazebo and RViz.
* Add Clearpath custom metapackages [here](http://wiki.ros.org/ClearpathRobotics/Packages).
* Follow this [instruction](https://www.clearpathrobotics.com/assets/guides/noetic/jackal/simulation.html).
* Install all jackal packages.
    ```
    sudo apt install ros-noetic-jackal-*
    ```
* Clone this repo and run `catkin_make` in the workspace. Remember to checkout the correct branch.
    ```
    mkdir ~/catkin_ws
    cd ~/catkin_ws 
    git clone https://github.com/ahanjaya/LfH.git src/
    cd ~/catkin_ws
    catkin_make -j16
    ```

### 1. Collect training data.
- Terminal
    ```
    roslaunch lfh_data collect_data.launch
    ```

### 2. Convert training data.
- Convert `.bag` to ros2 format. According to [rosbags repository](https://gitlab.com/ternaris/rosbags)
    ```bash
    rosbags-convert <output_name>.bag
    ```

- Terminal
    ```bash
    roslaunch lfh_data convert_data.launch
    ```

### 3. Train the model.
- Terminal
    ```bash
    roslaunch lfh_data train_data.launch
    ```


#### **Setting RVIZ**
- Copy rviz setting `lfh_navigation.rviz` into `/opt/ros/noetic/share/jackal_viz/rviz`.
    ```bash
    sudo cp ~/catkin_ws/src/lfh_utils/rviz/lfh_navigation.rviz /opt/ros/noetic/share/jackal_viz/rviz
    ```

## Run the simulation and navigation
- Terminal
    ```bash
    roslaunch lfh_inference bringup_lfh.launch
    ```
# Cuboid_detector
ROS 2 repository for 6D pose estimation of cuboid-like objects

## 1. Setup
### Installation
- **The following setting is for ROS 2 Humble in ubuntu 22.04**.

- Build the projects:
```shell
sudo apt install ros-humble-vision-msgs

pip3 install -r requirements.txt
cd Cuboid-detector
colcon build
source install/local_setup.bash
```

## 2. Commands for Demo
Try 'topic':
```shell
ros2 run cuboid_detector cuboid_detector

```

Save rosbag to data:
```shell
python3 script/convert_topic_to_file.py
ros2 bag play YOUR_ROS_BAG_FILE_PATH
```
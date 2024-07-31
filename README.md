# Cuboid_detector
ROS 2 repository for 6D pose estimation of cuboid like objects

## 1. Setup
### Installation
- **The following setting is for ROS 2 Humble in ubuntu 20.04**.

- Build the projects:
```shell
pip3 install -r requirements.txt
cd Cuboid-detector
colcon build
source install/local_setup.bash
```

## 2. Commands for Demo
Try 'topic':
```shell

```

Save rosbag to data:
```shell
python3 script/convert_topic_to_file.py
ros2 bag play YOUR_ROS_BAG_FILE_PATH
```
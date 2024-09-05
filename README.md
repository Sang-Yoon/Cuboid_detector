# Cuboid_detector
ROS 2 repository for 6D pose estimation of cuboid-like objects

## 1. Setup
### Installation
- The following setting is tested for Intel® RealSense™ LiDAR Camera <span style="color: red;">**L515**</span> with <span style="color: red;">**ROS 2 Humble**</span> in <span style="color: red;">**ubuntu 22.04**</span> only.

- Build the projects:
```shell
# librealsense를 먼저 설치

# https://github.com/IntelRealSense/realsense-ros?tab=readme-ov-file#installation-on-ubuntu

cd YOUR_ROS_WORKSPACE/src # Cuboid_detector 폴더 위치시키기

ros2_ws (YOUR_ROS_WORKSPACE)
├── build
├── install
├── log
└── src
    ├── Cuboid_detector
        ├── cuboid_detector
        │   ├── dataset
        │   │   ├── camera_info
        │   │   ├── data
        │   │   │   └── color_000000.png
        │   │   │   └── ...
        │   │   │   └── color_000009.png
        │   │   │   └── pose.json
        │   │   └── mesh
        │   │       └── danpla_box.obj
        │   │   └── ...
        │   ├── script
        │   │   ├── cuboid_detector.py      # Main detection script
        │   │   ├── gt_annotator.py         # Data annotation
        │   │   └── ...
        │   └── ...
        ├── README.md
        └── requirements.txt

sudo apt install ros-humble-vision-msgs ros-humble-realsense2-camera-msgs
cd Cuboid-detector && pip3 install -r requirements.txt
cd ../.. && colcon build
source install/local_setup.bash
```

## 2. Commands for Demo
<span style="color: red;"> **How to evaluate:** </span>
```shell
cd YOUR_ROS_WORKSPACE/src/Cuboid_detector/cuboid_detector/script

python3 cuboid_detector.py --evaluate --visualize
```

<span style="color: red;"> **Try 'topic':** </span>
```shell
# L515 카메라 실행 또는 ros2 bag play YOUR_ROS_BAG_FILE_PATH
ros2 run cuboid_detector cuboid_detector
```

Save ros2 bag to data (Not important):
```shell
cd YOUR_ROS_WORKSPACE/src/Cuboid_detector/cuboid_detector/script

python3 convert_topic_to_file.py
ros2 bag play YOUR_ROS_BAG_FILE_PATH
```
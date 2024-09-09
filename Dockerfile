FROM ros:humble

# ARG DEBIAN_FRONTEND=noninteractive

# Change apt source to Kakao mirror
RUN sed -i".bak.original" -re "s/([a-z]{2}.)?archive.ubuntu.com|security.ubuntu.com/mirror.kakao.com/g" /etc/apt/sources.list

ENV WORKSPACE=/ros2_home
WORKDIR /ros2_home
RUN mkdir -p /ros2_home/src
COPY ./requirements.txt /ros2_home/src/requirements.txt
COPY ./cuboid_detector /ros2_home/src/cuboid_detector

# install ros package
RUN apt-get update && apt-get upgrade -y \
    && apt-get install -y -qq --no-install-recommends \
        ros-${ROS_DISTRO}-cv-bridge \
        ros-${ROS_DISTRO}-vision-msgs \
        ros-${ROS_DISTRO}-realsense2-camera-msgs \
        python3-pip \
        # for cv2 (libGL.so.1)
        ffmpeg \
        libsm6 \
        libxext6 \
    && pip3 install -r /ros2_home/src/requirements.txt \
    # Clean up
    && apt-get autoremove -y \
    && apt-get clean -y \
    && rm -rf /var/lib/apt/lists/*

# RUN apt-get update && apt-get install -y -qq --no-install-recommends \
#     ros-${ROS_DISTRO}-rmw-cyclonedds-cpp \
#     ros-${ROS_DISTRO}-demo-nodes-cpp \
#     iputils-ping \
#     net-tools

# build
COPY ./ros_entrypoint.sh /ros_entrypoint.sh
RUN chmod +x /ros_entrypoint.sh \
    && cd ${WORKSPACE} \
    && . /opt/ros/${ROS_DISTRO}/setup.sh \
    && colcon build --symlink-install

ENV RMW_IMPLEMENTATION=rmw_fastrtps_cpp
# ENV RMW_IMPLEMENTATION=rmw_cyclonedds_cpp

# launch ros package
CMD ["ros2", "run", "cuboid_detector", "cuboid_detector"]
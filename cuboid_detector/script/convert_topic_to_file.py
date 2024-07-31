import os
import numpy as np
import yaml

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge


class imgSaver(Node):
    def __init__(self):
        super().__init__("img_saver")

        self.bridge = CvBridge()
        self.depth_img_sub = self.create_subscription(
            Image, "/depth_to_rgb/image_raw", self.depth_img_callback, 10
        )
        self.rgb_img_sub = self.create_subscription(
            Image, "/rgb/image_raw", self.rgb_img_callback, 10
        )
        self.camera_info_sub = self.create_subscription(
            CameraInfo, "/rgb/camera_info", self.camera_info_callback, 10
        )
        self.img_save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "dataset", "data"
        )
        self.cam_info_save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "dataset", "camera_info"
        )
        if os.path.exists(self.img_save_dir):
            os.system(f"rm -r {self.img_save_dir}")
            os.makedirs(self.img_save_dir)
        else:
            os.makedirs(self.img_save_dir)
        if os.path.exists(self.cam_info_save_dir):
            os.system(f"rm -r {self.cam_info_save_dir}")
            os.makedirs(self.cam_info_save_dir)
        else:
            os.makedirs(self.cam_info_save_dir)

        self.rgb_img_count = 0
        self.depth_img_count = 0

    def depth_img_callback(self, msg):
        self.save_img(msg, "depth", self.depth_img_count)
        self.depth_img_count += 1

    def rgb_img_callback(self, msg):
        self.save_img(msg, "color", self.rgb_img_count)
        self.rgb_img_count += 1

    def camera_info_callback(self, msg):
        self.save_camera_info(msg)

    def save_img(self, msg, prefix, count):
        cv_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv2.imwrite(os.path.join(self.img_save_dir, f"{prefix}_{count:06d}.png"), cv_img)
        self.get_logger().info(f"Saved img: {prefix}_{count}.png")

    def save_camera_info(self, msg):
        camera_info_filename = "camera_info.yaml"

        def convert_array(array):
            if isinstance(array, np.ndarray):
                return array.tolist()
            return array

        camera_info_dict = {
            "width": msg.width,
            "height": msg.height,
            "distortion_model": msg.distortion_model,
            "D": convert_array(msg.d),
            "K": convert_array(msg.k),
            "R": convert_array(msg.r),
            "P": convert_array(msg.p),
        }
        with open(os.path.join(self.cam_info_save_dir, camera_info_filename), "w") as f:
            yaml.dump(camera_info_dict, f)
        self.get_logger().info(f"Saved camera info: {camera_info_filename}")


def main(args=None):
    rclpy.init(args=args)
    img_saver = imgSaver()
    rclpy.spin(img_saver)
    img_saver.destroy_node()
    rclpy.shutdown()


if __name__ == "__main__":
    main()

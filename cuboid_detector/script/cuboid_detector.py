import os
import yaml
import json
import cv2
import numpy as np
import open3d as o3d
from yaml import Loader
from tqdm import tqdm
from scipy.spatial.transform import Rotation as R

import rclpy
from rclpy.node import Node
from cv_bridge import CvBridge
from ament_index_python.packages import (
    get_package_share_directory,
    PackageNotFoundError,
)
from collections import deque
from sensor_msgs.msg import Image, CameraInfo
from geometry_msgs.msg import Point, Quaternion, PoseWithCovariance
from vision_msgs.msg import Detection3D, Detection3DArray, ObjectHypothesisWithPose
from realsense2_camera_msgs.msg import Extrinsics


def get_package_path(package_name):
    try:
        package_path = get_package_share_directory(package_name)
        return package_path
    except PackageNotFoundError:
        print(f"Package '{package_name}' not found.")
        return None


def get_dirpath_iteratively(dirpath, num_iter):
    for _ in range(num_iter):
        dirpath = os.path.dirname(dirpath)
    return dirpath


class CuboidDetector(Node):
    def __init__(self):
        super().__init__("cuboid_detector")
        self.bridge = CvBridge()

        self.color_img_sub = self.create_subscription(
            Image, "/L515/color/image_raw", self.color_img_callback, 10
        )
        self.depth_img_sub = self.create_subscription(
            Image, "/L515/depth/image_rect_raw", self.depth_img_callback, 10
        )
        self.color_camera_info_sub = self.create_subscription(
            CameraInfo, "/L515/color/camera_info", self.camera_info_callback, 10
        )
        self.depth_camera_info_sub = self.create_subscription(
            CameraInfo, "/L515/depth/camera_info", self.camera_info_callback, 10
        )
        self.depth_to_color_extrinsics_sub = self.create_subscription(
            Extrinsics,
            "/L515/extrinsics/depth_to_color",
            self.depth_to_color_extrinsics_callback,
            10,
        )
        self.detection_pub = self.create_publisher(
            Detection3DArray, "/cuboid_detections", 10
        )

        self.img_save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "dataset", "data"
        )
        self.cam_info_save_dir = os.path.join(
            os.path.dirname(os.path.dirname(__file__)), "dataset", "camera_info"
        )
        self.pkg_name = "cuboid_detector"
        self.pkg_path = get_dirpath_iteratively(get_package_path(self.pkg_name), 4)
        self.mesh_path = os.path.join(
            self.pkg_path,
            "src/Cuboid_detector",
            self.pkg_name,
            "dataset/mesh/danpla_box.obj",
        )
        self.mesh = self.load_mesh(self.mesh_path)

        self.color_image_buffer = deque()
        self.depth_image_buffer = deque()

        self.color_intrinsic_matrix = None
        self.depth_intrinsic_matrix = None
        self.depth_to_color_extrinsics = None

        self.detection_in_progress = False

    def color_img_callback(self, msg):
        if self.detection_in_progress:
            return
        color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        color_timestamp = msg.header.stamp
        self.color_image_buffer.append((color_timestamp, color_img))

        self.match_and_process_images()

    def depth_img_callback(self, msg):
        if self.detection_in_progress:
            return
        depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        depth_timestamp = msg.header.stamp
        self.depth_image_buffer.append((depth_timestamp, depth_img))

        self.match_and_process_images()

    def camera_info_callback(self, msg):
        intrinsic_matrix = np.array(msg.k).reshape(3, 3)

        if msg.header.frame_id == "L515_color_optical_frame":
            self.color_intrinsic_matrix = intrinsic_matrix
        elif msg.header.frame_id == "L515_depth_optical_frame":
            self.depth_intrinsic_matrix = intrinsic_matrix

    def depth_to_color_extrinsics_callback(self, msg):
        rotation = np.array(msg.rotation).reshape(3, 3)
        translation = np.array(msg.translation)
        extrinsics = np.eye(4)
        extrinsics[:3, :3] = rotation
        extrinsics[:3, 3] = translation
        self.depth_to_color_extrinsics = extrinsics

    def match_and_process_images(self):
        while self.color_image_buffer and self.depth_image_buffer:
            color_timestamp, color_img = self.color_image_buffer[0]
            depth_timestamp, depth_img = self.depth_image_buffer[0]

            time_diff = self.calculate_time_difference(color_timestamp, depth_timestamp)

            if abs(time_diff) < self.time_tolerance():
                self.color_image_buffer.popleft()
                self.depth_image_buffer.popleft()
                self.color_img = color_img
                self.depth_img = depth_img
                self.color_img_timestamp = color_timestamp
                self.depth_img_timestamp = depth_timestamp
                self.perform_detection()
            elif time_diff < 0:
                self.color_image_buffer.popleft()
            else:
                self.depth_image_buffer.popleft()

    def calculate_time_difference(self, color_timestamp, depth_timestamp):
        color_time_in_ns = color_timestamp.sec * 1e9 + color_timestamp.nanosec
        depth_time_in_ns = depth_timestamp.sec * 1e9 + depth_timestamp.nanosec
        return color_time_in_ns - depth_time_in_ns

    def time_tolerance(self):
        # Set a tolerance for time difference in nanoseconds (e.g., 50 milliseconds)
        return 50 * 1e6

    def align_color_depth(self):
        depth_height, depth_width = self.depth_img.shape
        color_height, color_width, _ = self.color_img.shape
        aligned_depth_img = np.zeros((color_height, color_width), dtype=np.uint16)

        R = self.depth_to_color_extrinsics[:3, :3]
        t = self.depth_to_color_extrinsics[:3, 3]

        for v in range(depth_height):
            for u in range(depth_width):
                depth_value = self.depth_img[v, u]
                if depth_value == 0:
                    continue

                z = depth_value / 1000.0
                x = (
                    (u - self.depth_intrinsic_matrix[0, 2])
                    * z
                    / self.depth_intrinsic_matrix[0, 0]
                )
                y = (
                    (v - self.depth_intrinsic_matrix[1, 2])
                    * z
                    / self.depth_intrinsic_matrix[1, 1]
                )

                point_3d = np.array([x, y, z])
                point_3d_color = R @ point_3d + t

                x_c = point_3d_color[0] / point_3d_color[2]
                y_c = point_3d_color[1] / point_3d_color[2]

                u_c = int(
                    self.color_intrinsic_matrix[0, 0] * x_c
                    + self.color_intrinsic_matrix[0, 2]
                )
                v_c = int(
                    self.color_intrinsic_matrix[1, 1] * y_c
                    + self.color_intrinsic_matrix[1, 2]
                    + 60
                )

                if 0 <= u_c < color_width and 0 <= v_c < color_height:
                    aligned_depth_img[v_c, u_c] = depth_value

        mask = (aligned_depth_img == 0).astype(np.uint8)
        aligned_depth_img = cv2.inpaint(
            aligned_depth_img.astype(np.uint16),
            mask,
            inpaintRadius=3,
            flags=cv2.INPAINT_TELEA,
        )

        return aligned_depth_img

    def load_mesh(self, mesh_file):
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh.compute_vertex_normals()
        return mesh

    def check_and_process_images(self):
        if (
            self.color_img is not None
            and self.depth_img is not None
            and self.color_intrinsic_matrix is not None
            and self.depth_intrinsic_matrix is not None
            and self.depth_to_color_extrinsics is not None
            and self.color_img_timestamp == self.depth_img_timestamp
        ):
            self.get_logger().info("Performing detection...")
            self.perform_detection()
        elif self.color_img_timestamp != self.depth_img_timestamp:
            self.get_logger().info("Color and depth images are not synchronized.")
            self.get_logger().info(f"Color timestamp: {self.color_img_timestamp}")
            self.get_logger().info(f"Depth timestamp: {self.depth_img_timestamp}")
        elif self.color_img is None:
            self.get_logger().info("Color image is not received.")
        elif self.depth_img is None:
            self.get_logger().info("Depth image is not received.")
        elif self.color_intrinsic_matrix is None:
            self.get_logger().info("Color intrinsic matrix is not received.")
        elif self.depth_intrinsic_matrix is None:
            self.get_logger().info("Depth intrinsic matrix is not received.")
        elif self.depth_to_color_extrinsics is None:
            self.get_logger().info("Depth to color extrinsics is not received.")

    def perform_detection(self):
        rgb_img = self.color_img

        blue_regions = self.extract_blue_regions(rgb_img)
        blue_regions = cv2.erode(blue_regions, np.ones((5, 5), np.uint8), iterations=1)
        # cv2.imshow("blue_regions", blue_regions)

        small_bbox, large_bbox = self.get_bbox(
            blue_regions, rgb_img.shape[0], rgb_img.shape[1]
        )
        rgb_img_large_bbox = np.zeros_like(rgb_img)
        large_x, large_y, large_w, large_h = large_bbox
        rgb_img_large_bbox[large_y : large_y + large_h, large_x : large_x + large_w] = (
            rgb_img[large_y : large_y + large_h, large_x : large_x + large_w]
        )
        # cv2.imshow("rgb_img_large_bbox", rgb_img_large_bbox)

        small_x, small_y, small_w, small_h = small_bbox
        black_regions = np.zeros_like(blue_regions)
        black_regions[small_y : small_y + small_h, small_x : small_x + small_w] = (
            self.extract_black_regions(rgb_img)[
                small_y : small_y + small_h, small_x : small_x + small_w
            ]
        )
        black_regions = cv2.erode(
            black_regions, np.ones((5, 5), np.uint8), iterations=1
        )

        contours, _ = cv2.findContours(
            black_regions, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
        )
        contour = max(contours, key=cv2.contourArea)
        epsilon = 0.1 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        cv2.drawContours(rgb_img_large_bbox, [approx], 0, (0, 255, 0), 2)
        # cv2.imshow("rgb_img_large_bbox_contour", rgb_img_large_bbox)

        if len(approx) != 4:
            return
        corners = approx.reshape(4, 2)
        for corner in corners:
            cv2.circle(rgb_img_large_bbox, tuple(corner), 5, (0, 0, 255), -1)
        # cv2.imshow("rgb_img_large_bbox_corners", rgb_img_large_bbox)

        box_points = np.asarray(self.mesh.vertices)
        box_x_min, box_x_max = np.min(box_points[:, 0]), np.max(box_points[:, 0])
        box_y_min, box_y_max = np.min(box_points[:, 1]), np.max(box_points[:, 1])
        box_z_min, box_z_max = np.min(box_points[:, 2]), np.max(box_points[:, 2])
        box_top_corners = np.array(
            [
                [box_x_max, box_y_min, box_z_max],
                [box_x_min, box_y_min, box_z_max],
                [box_x_min, box_y_max, box_z_max],
                [box_x_max, box_y_max, box_z_max],
            ]
        )

        object_points = box_top_corners.astype(np.float32)
        image_points = np.array(corners, dtype=np.float32)

        def sort_points_ccw(points):
            center = np.mean(points, axis=0)
            return sorted(
                points, key=lambda x: np.arctan2(x[1] - center[1], x[0] - center[0])
            )

        image_points = sort_points_ccw(image_points)

        image_points = np.expand_dims(image_points, axis=1)
        dist_coeffs = np.zeros((4, 1))

        success, rotation_vector, translation_vector = cv2.solvePnP(
            object_points, image_points, self.color_intrinsic_matrix, dist_coeffs
        )

        mesh_points_3d = self.mesh.sample_points_poisson_disk(number_of_points=10000)
        mesh_points_2d, _ = cv2.projectPoints(
            np.array(mesh_points_3d.points),
            rotation_vector,
            translation_vector,
            self.color_intrinsic_matrix,
            dist_coeffs,
        )
        mesh_points_2d = np.int32(mesh_points_2d).reshape(-1, 2)
        shapes = np.zeros_like(rgb_img, np.uint8)
        for i in range(len(mesh_points_2d)):
            cv2.circle(shapes, tuple(mesh_points_2d[i]), 1, (0, 0, 255), -1)
        out = rgb_img.copy()
        alpha = 0.1
        mask = shapes.astype(bool)
        out[mask] = cv2.addWeighted(rgb_img, alpha, shapes, 1 - alpha, 0)[mask]
        out = cv2.resize(out, (1280, 720))
        cv2.imshow("result", out)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        detection_array_msg = Detection3DArray()
        detection_array_msg.header.stamp = self.get_clock().now().to_msg()
        detection_array_msg.header.frame_id = "L515_color_optical_frame"
        detection_msg = Detection3D()
        hypothesis_with_pose = ObjectHypothesisWithPose()
        hypothesis_with_pose.hypothesis.class_id = "1"
        pose_with_covariance = PoseWithCovariance()
        pose_with_covariance.pose.position = Point(
            x=float(translation_vector[0]),
            y=float(translation_vector[1]),
            z=float(translation_vector[2]),
        )
        quat = R.from_rotvec(rotation_vector.flatten()).as_quat()
        pose_with_covariance.pose.orientation = Quaternion(
            x=float(quat[0]), y=float(quat[1]), z=float(quat[2]), w=float(quat[3])
        )
        hypothesis_with_pose.pose = pose_with_covariance
        detection_msg.results.append(hypothesis_with_pose)
        detection_array_msg.detections.append(detection_msg)
        self.detection_pub.publish(detection_array_msg)
        return detection_array_msg

    def extract_blue_regions(self, rgb_img):
        hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
        lower_blue = np.array([90, 50, 50])
        upper_blue = np.array([130, 255, 255])
        blue_mask = cv2.inRange(hsv_img, lower_blue, upper_blue)
        return blue_mask

    def extract_black_regions(self, rgb_img):
        hsv_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2HSV)
        lower_black = np.array([0, 0, 0])
        upper_black = np.array([180, 255, 80])
        black_mask = cv2.inRange(hsv_img, lower_black, upper_black)
        return black_mask

    def get_bbox(self, mask, height, width):
        x, y, w, h = cv2.boundingRect(mask)
        center_x, center_y = x + w // 2, y + h // 2
        small_w = int(w * 1.15)
        small_h = int(h * 1.15)
        small_x, small_y = center_x - small_w // 2, center_y - small_h // 2
        large_w = int(w * 1.20)
        large_h = int(h * 1.20)
        large_x, large_y = center_x - large_w // 2, center_y - large_h // 2
        if small_x < 0:
            small_x = 0
        if small_y < 0:
            small_y = 0
        if small_x + small_w > width:
            small_w = width - small_x
        if small_y + small_h > height:
            small_h = height - small_y
        if large_x < 0:
            large_x = 0
        if large_y < 0:
            large_y = 0
        if large_x + large_w > width:
            large_w = width - large_x
        if large_y + large_h > height:
            large_h = height - large_y
        return [small_x, small_y, small_w, small_h], [
            large_x,
            large_y,
            large_w,
            large_h,
        ]


def evaluate_detection():
    rclpy.init(args=None)
    cuboid_detector = CuboidDetector()

    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
    img_path = os.path.join(dataset_path, "data")

    intrinsic_path = os.path.join(dataset_path, "camera_info", "camera_info.yaml")

    with open(intrinsic_path, "r") as f:
        # Use the custom loader to handle special types
        camera_info = yaml.load(f, Loader=Loader)

    color_intrinsic_matrix = np.array(camera_info["K"]).reshape(3, 3)

    color_img_list = [img for img in os.listdir(img_path) if "color" in img]
    color_img_list.sort()

    target_color_img_list = [img for i, img in enumerate(color_img_list) if i % 20 == 0]

    pred_pose_list = []
    gt_pose_list = []

    with open(os.path.join(dataset_path, "data", "pose.json"), "r") as f:
        gt_pose = np.array(json.load(f))
    gt_pose_list.append(gt_pose)

    for i, img_name in enumerate(tqdm(target_color_img_list[:-4])):
        img = cv2.imread(os.path.join(img_path, img_name))
        cuboid_detector.color_img = img
        cuboid_detector.color_intrinsic_matrix = color_intrinsic_matrix
        detection_array_msg = cuboid_detector.perform_detection()
        if detection_array_msg is not None:
            pose_data = detection_array_msg.detections[0].results[0].pose.pose
            pose = np.eye(4)
            pose[:3, 3] = [
                pose_data.position.x,
                pose_data.position.y,
                pose_data.position.z,
            ]
            pose[:3, :3] = R.from_quat(
                [
                    pose_data.orientation.x,
                    pose_data.orientation.y,
                    pose_data.orientation.z,
                    pose_data.orientation.w,
                ]
            ).as_matrix()
            pred_pose_list.append(pose)
        else:
            print(f"No pose detected for {img_name}")

    pred_pose_list = np.array(pred_pose_list)
    gt_pose_list = np.array(gt_pose_list)[0]

    translation_errors = []
    rotation_errors = []

    for pred_pose, gt_pose in zip(pred_pose_list, gt_pose_list):
        translation_error = np.linalg.norm(pred_pose[:3, 3] - gt_pose[:3, 3])
        rotation_error = np.arccos(
            (np.trace(pred_pose[:3, :3].T @ gt_pose[:3, :3]) - 1) / 2
        )
        translation_errors.append(translation_error)
        rotation_errors.append(rotation_error)

    mean_translation_error = np.mean(translation_errors)
    mean_rotation_error = np.mean(rotation_errors)
    print(f"Mean translation error: {mean_translation_error * 1000} mm")
    print(f"Mean rotation error: {np.rad2deg(mean_rotation_error)} degrees")


def main(args=None):
    EVALUATE = False
    if EVALUATE:
        evaluate_detection()
        return
    else:
        rclpy.init(args=args)
        detector = CuboidDetector()
        rclpy.spin(detector)
        detector.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

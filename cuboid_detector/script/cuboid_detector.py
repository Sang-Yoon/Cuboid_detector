import os
import cv2
import numpy as np
import yaml
import open3d as o3d
from sklearn.cluster import KMeans

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image, CameraInfo
import cv2
from cv_bridge import CvBridge


class CuboidDetector(Node):
    def __init__(self):
        super().__init__("cuboid_detector")
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

        self.mesh = self.load_mesh(mesh_file)
        self.keypoints, self.descriptors = self.extract_keypoints_and_descriptors(
            rgb_image
        )
        self.mesh_keypoints, self.mesh_descriptors = (
            self.extract_keypoints_and_descriptors_from_mesh(self.mesh)
        )

    def depth_img_callback(self, msg):
        self.depth_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def rgb_img_callback(self, msg):
        self.color_img = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")

    def camera_info_callback(self, msg):
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
        self.intrinsic_matrix = np.array(camera_info_dict["K"]).reshape(3, 3)

    def load_mesh(self, mesh_file):
        mesh = o3d.io.read_triangle_mesh(mesh_file)
        mesh.compute_vertex_normals()
        return mesh

    def detect_cuboid_2d(self):
        pass

    def get_edges(self, rgb_img):
        img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(img, 100, 200)
        return edges

    def get_corners(self, edges):
        corners = cv2.goodFeaturesToTrack(edges, 4, 0.01, 10)
        return corners

    def visualize_features(self, rgb_img, edges, corners):
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(rgb_img, (x, y), 3, 255, -1)
        return rgb_img


def extract_blue_regions(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    lower_blue = np.array([100, 150, 0])
    upper_blue = np.array([140, 255, 255])

    mask = cv2.inRange(hsv, lower_blue, upper_blue)
    return mask


def get_bbox(mask):
    x, y, w, h = cv2.boundingRect(mask)
    center_x, center_y = x + w // 2, y + h // 2
    small_w = int(w * 1.15)
    small_h = int(h * 1.15)
    small_x, small_y = center_x - small_w // 2, center_y - small_h // 2
    large_w = int(w * 1.2)
    large_h = int(h * 1.2)
    large_x, large_y = center_x - large_w // 2, center_y - large_h // 2
    return [small_x, small_y, small_w, small_h], [large_x, large_y, large_w, large_h]


def find_intersections(lines):
    intersections = []
    if lines is not None:
        for i in range(len(lines)):
            for j in range(i + 1, len(lines)):
                x1, y1, x2, y2 = lines[i][0]
                x3, y3, x4, y4 = lines[j][0]

                # 두 직선의 교차점 계산
                A1 = y2 - y1
                B1 = x1 - x2
                C1 = A1 * x1 + B1 * y1

                A2 = y4 - y3
                B2 = x3 - x4
                C2 = A2 * x3 + B2 * y3

                determinant = A1 * B2 - A2 * B1

                if determinant != 0:
                    x = (B2 * C1 - B1 * C2) / determinant
                    y = (A1 * C2 - A2 * C1) / determinant
                    intersections.append((int(x), int(y)))
    return intersections


def test():
    print("Test")
    img_save_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "dataset", "data"
    )
    cam_info_save_dir = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "dataset", "camera_info"
    )

    rgb_imgs, depth_imgs, K = [], [], []
    for dir_name, _, files in os.walk(img_save_dir):
        for file in files:
            if "color" in file:
                rgb_imgs.append(cv2.imread(os.path.join(dir_name, file)))
            elif "depth" in file:
                depth_imgs.append(
                    cv2.imread(os.path.join(dir_name, file), cv2.IMREAD_ANYDEPTH)
                )

    for dir_name, _, files in os.walk(cam_info_save_dir):
        for file in files:
            with open(os.path.join(dir_name, file), "r") as f:
                camera_info_dict = yaml.safe_load(f)
                K.append(camera_info_dict["K"])

    img_idx = 10
    rgb_img, depth_img, cam_K = rgb_imgs[img_idx], depth_imgs[img_idx], K[0]
    rgb_img = cv2.resize(rgb_img, (640, 480))
    depth_img = cv2.resize(depth_img, (640, 480))
    cam_K = np.array(cam_K) * 640 / 2048

    roi = (640 * 1 / 4, 480 * 1 / 4, 640 * 3 / 4, 480 * 3 / 4)
    rgb_img_roi = np.zeros_like(rgb_img)
    rgb_img_roi[int(roi[1]) : int(roi[3]), int(roi[0]) : int(roi[2])] = rgb_img[
        int(roi[1]) : int(roi[3]), int(roi[0]) : int(roi[2])
    ]

    ##############################
    rgb_img_roi = cv2.GaussianBlur(rgb_img_roi, (5, 5), 3)
    cv2.imshow("rgb_img_roi", rgb_img_roi)
    blue_regions = extract_blue_regions(rgb_img_roi)
    cv2.imshow("blue_regions", blue_regions)

    small_bbox, large_bbox = get_bbox(blue_regions)

    bbox_img = rgb_img.copy()
    small_x, small_y, small_w, small_h = small_bbox
    cv2.rectangle(
        bbox_img,
        (small_x, small_y),
        (small_x + small_w, small_y + small_h),
        (0, 255, 0),
        2,
    )
    cv2.imshow("bbox", bbox_img)

    bbox_img = np.zeros_like(rgb_img)
    large_x, large_y, large_w, large_h = large_bbox
    bbox_img[large_y : large_y + large_h, large_x : large_x + large_w] = rgb_img[
        large_y : large_y + large_h, large_x : large_x + large_w
    ]
    gray_img = cv2.cvtColor(bbox_img, cv2.COLOR_BGR2GRAY)
    gray_img = cv2.GaussianBlur(gray_img, (5, 5), 1)

    canny_edges = cv2.Canny(gray_img, 100, 200)
    edges = np.zeros_like(canny_edges)
    edges[small_y : small_y + small_h, small_x : small_x + small_w] = canny_edges[
        small_y : small_y + small_h, small_x : small_x + small_w
    ]
    cv2.imshow("edges", edges)

    lines = cv2.HoughLinesP(
        edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=50
    )

    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            cv2.line(bbox_img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.imshow("lines", bbox_img)

    corner_on_lines = cv2.goodFeaturesToTrack(edges, 4, 0.01, 10)
    for corner in corner_on_lines:
        x, y = corner.ravel()
        cv2.circle(bbox_img, (x, y), 3, 255, -1)
    cv2.imshow("corner_on_lines", bbox_img)

    # Define the object points of the box (in 3D space)
    box_model_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "dataset", "mesh", "danpla_box.obj"
    )
    box_model = o3d.io.read_triangle_mesh(box_model_path)
    box_points = np.asarray(box_model.vertices)
    box_width = np.max(box_points[:, 0]) - np.min(box_points[:, 0])
    box_height = np.max(box_points[:, 1]) - np.min(box_points[:, 1])
    box_depth = np.max(box_points[:, 2]) - np.min(box_points[:, 2])

    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(args=None):
    test()


if __name__ == "__main__":
    main()

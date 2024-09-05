import os
import numpy as np
import cv2
import json
import yaml
import open3d as o3d
from yaml import Loader

points = []

def array_constructor(loader, node):
    seq = loader.construct_sequence(node)
    try:
        return np.array(seq)
    except ValueError:
        return np.array([np.array(i) for i in seq], dtype=object)

yaml.add_constructor('tag:yaml.org,2002:python/object/apply:array.array', array_constructor, Loader=Loader)

def mouse_callback(event, x, y, flags, param):
    global points

    if event == cv2.EVENT_LBUTTONDOWN:
        points.append((x, y))
        cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("image", img)

        if len(points) == 4:
            save_ground_truth(points)
            points.clear()

def save_ground_truth(points):
    corners = np.array(points).reshape(4, 2)
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
    intrinsic_path = os.path.join(dataset_path, "camera_info", "camera_info.yaml")
    
    with open(intrinsic_path, "r") as f:
        camera_info = yaml.load(f, Loader=Loader)

    color_intrinsic_matrix = np.array(camera_info["K"]).reshape(3, 3)
    mesh_path = os.path.join(dataset_path, "mesh", "danpla_box.obj")
    mesh = o3d.io.read_triangle_mesh(mesh_path)
    mesh.compute_vertex_normals()

    box_points = np.asarray(mesh.vertices)
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

    _, rotation_vector, translation_vector = cv2.solvePnP(
        object_points, image_points, color_intrinsic_matrix, dist_coeffs
    )
    pose = np.eye(4)
    pose[:3, :3] = cv2.Rodrigues(rotation_vector)[0]
    pose[:3, 3] = translation_vector.flatten()

    def append_to_pose_file(pose):
        dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
        pose_file_path = os.path.join(dataset_path, "data", "pose.json")

        if os.path.exists(pose_file_path):
            with open(pose_file_path, "r") as f:
                try:
                    data = json.load(f)
                except json.JSONDecodeError:
                    data = []
        else:
            data = []

        data.append(pose.tolist())

        with open(pose_file_path, "w") as f:
            json.dump(data, f, indent=4)

    append_to_pose_file(pose)

    print("Ground truth saved!")

def main():
    global img
    dataset_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "dataset")
    img_path = os.path.join(dataset_path, "data")

    color_img_list = [img for img in os.listdir(img_path) if "color" in img]
    color_img_list.sort()

    target_color_img_list = [img for i, img in enumerate(color_img_list)]

    for img_name in target_color_img_list:
        img = cv2.imread(os.path.join(img_path, img_name))
        cv2.imshow("image", img)
        cv2.setMouseCallback("image", mouse_callback)
        cv2.waitKey(0)

if __name__ == "__main__":
    main()

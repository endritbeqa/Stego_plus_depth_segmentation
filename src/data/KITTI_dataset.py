import os
from torch.utils.data import Dataset
from typing import List, Union, Optional, Callable, Any, Tuple, Dict
from PIL import Image
import open3d as o3d
import numpy as np
import re


class KITTI(Dataset):

    def __init__(self,
                 root: str,
                 image_transform: Optional[Callable],
                 target_transform: Optional[Callable],
                 point_cloud_transform: Optional[Callable]
                 ):
        super(KITTI, self).__init__()
        self.root = root
        self.image_dir = os.path.join(root, "data_2d_raw")
        self.target_dir = os.path.join(root, "data_2d_semantics")
        self.point_clouds_dir = os.path.join(root, "data_3d_semantics")
        self.data_poses_dir = os.path.join(root, "data_poses")
        self.calibration_dir = os.path.join(root, "calibration")
        self.image_transform = image_transform
        self.target_transform = target_transform
        self.point_cloud_transform = point_cloud_transform
        self.calibration_files = []
        self.calibration_matrixes= {}
        self.images = []
        self.targets = []
        self.point_clouds = []
        self.cam0_to_world = []
        self.poses = []

        for calibration_file in os.listdir(self.calibration_dir):
            if calibration_file[-3:] == "txt":
                self.calibration_files.append(os.path.join(self.calibration_dir, calibration_file))

        self.calibration_matrixes = self.getCalibration_matrixes()

        for sequence in os.listdir(self.image_dir):
            image_sequence_dir = os.path.join(self.image_dir, sequence, "image_00", "data_rect")
            target_sequence_semantic_dir = os.path.join(self.target_dir, "train", sequence, "image_00", "semantic")
            target_sequence_semantic_rgb_dir = os.path.join(self.target_dir, "train", sequence, "image_00", "semantic_rgb")
            target_sequence_semantic_instance_dir = os.path.join(self.target_dir, "train", sequence, "image_00", "instance")
            point_cloud_sequence_dir = os.path.join(self.point_clouds_dir, "train", sequence, "static")

            data_pose_sequence_dir = os.path.join(self.data_poses_dir, sequence)

            for frame in os.listdir(image_sequence_dir):
                self.images.append(os.path.join(image_sequence_dir, frame))
                frame_targets = []
                frame_targets.append(os.path.join(target_sequence_semantic_dir, frame))
                frame_targets.append(os.path.join(target_sequence_semantic_rgb_dir, frame))
                frame_targets.append(os.path.join(target_sequence_semantic_instance_dir, frame))
                frame_targets = tuple(frame_targets)
                self.targets.append(frame_targets)
            for frame in os.listdir(point_cloud_sequence_dir):
                start_frame = int(frame.split(".")[0].split("_")[0])
                end_frame = int(frame.split(".")[0].split("_")[1])
                for i in range(end_frame - start_frame):
                    self.point_clouds.append(os.path.join(point_cloud_sequence_dir, frame))

            cam0_to_world_file = open(os.path.join(data_pose_sequence_dir, "cam0_to_world.txt"), "r")
            poses_file = open(os.path.join(data_pose_sequence_dir, "poses.txt"), "r")

            # this appends the cam to world matrix to
            previous_frame = 0

            for row in cam0_to_world_file.read().strip("\n").split("\n"):
                values = row.strip(" ").split(" ")
                values = [float(x) for x in values]
                frame = int(values[0])
                matrix = np.array(values)[1:]
                matrix = matrix.reshape((4, 4))
                for i in range(frame - previous_frame):
                    self.cam0_to_world.append(matrix)
                previous_frame = frame
            self.cam0_to_world.insert(0,self.cam0_to_world[0])#takes care of the edge case for frame 0

        print("Breakpoint")

    def __getitem__(self, index) -> Dict:

        image = Image.open(self.images[index]).convert("RGB")
        target_semantic = Image.open(self.targets[index][0])
        target_semantic_rgb = Image.open(self.targets[index][1]).convert("RGB")
        target_semantic_instance = Image.open(self.targets[index][2])
        point_cloud = o3d.io.read_point_cloud(self.point_clouds[index])
        point_cloud = np.asarray(point_cloud.points)
        cam0_to_world = self.cam0_to_world[index]

        image = self.image_transform(image)
        target_semantic = self.target_transform(target_semantic)
        target_semantic_rgb = self.target_transform(target_semantic_rgb)
        target_semantic_instance = self.target_transform(target_semantic_instance)
        depth_map = self.point_cloud_transform(point_cloud,cam0_to_world, self.calibration_matrixes["P_rect_00"])


        pack = {
            "ind": index,
            "img": image,
            "semantic": target_semantic,
            "semantic_rgb": target_semantic_rgb,
            "instance": target_semantic_instance,
            "depth": depth_map,
        }

        return pack

    def __len__(self) -> int:
        return len(self.images)

    def getCalibration_matrixes(self):

        calibration_matrixes = {}

        for file_path in self.calibration_files:
            file_name = file_path.split("/")[-1].split(".")[0]
            if file_name == "perspective":
                file = open(file_path, "r")
                for row in file.read().strip("\n").split("\n"):
                    if row.startswith("S_rect_00"):
                        ox, oy = row.split(":")[1].strip("\n").strip(" ").split(" ")
                        calibration_matrixes["Oxy"] = (ox, oy)
                    if row.startswith("R_rect_00"):
                        R_rect_00 = row.split(":")[1].strip("\n").strip(" ").split(" ")
                        R_rect_00 = [float(x) for x in R_rect_00]
                        R_rect_00 = np.array(R_rect_00)
                        R_rect_00 = R_rect_00.reshape((3, 3))
                        calibration_matrixes["R_rect_00"] = R_rect_00
                    if row.startswith("P_rect_00"):
                        P_rect_00 = row.split(":")[1].strip("\n").strip(" ").split(" ")
                        P_rect_00 = [float(x) for x in P_rect_00]
                        P_rect_00 = np.array(P_rect_00)
                        P_rect_00 = P_rect_00.reshape((3, 4))
                        calibration_matrixes["P_rect_00"] = P_rect_00
            elif file_name == "calib_cam_to_pose":
                file = open(file_path, "r")
                camera_values = file.read().strip("\n").split("\n")
                for row in camera_values:
                    cam_name, parameters = row.split(":")
                    parameters = parameters.strip().split(" ")
                    parameters = [float(x) for x in parameters]
                    parameters = np.array(parameters)
                    parameters = parameters.reshape((3, 4))
                    calibration_matrixes[file_name + "_" + cam_name] = parameters
            else:
                file = open(file_path, "r")
                parameters = file.read().strip().split(" ")
                parameters = [float(x) for x in parameters]
                parameters = np.array(parameters)
                parameters = parameters.reshape((3, 4))
                calibration_matrixes[file_name] = parameters

        return calibration_matrixes




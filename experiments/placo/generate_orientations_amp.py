import argparse
import json
import os

import FramesViewer.utils as fv_utils
import numpy as np
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, required=True)
parser.add_argument("-o", "--output_dir", type=str, required=True)
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)

episode = json.load(open(args.file))

frame_duration = episode["FrameDuration"]

frames = episode["Frames"]

step = 5
yaw_orientations = np.arange(360, step=step) - (180 - step)
# print(yaw_orientations)
# yaw_orientations = [180]

for yaw_orientation in yaw_orientations:
    new_frames = []
    for i, frame in enumerate(frames):
        root_position = frame[:3]  # x, y, z
        root_orientation_quat = frame[3:7]  # quat
        root_orientation_mat = R.from_quat(root_orientation_quat).as_matrix()
        # rotate around z axis
        root_orientation_mat = (
            R.from_euler("z", yaw_orientation, degrees=True).as_matrix()
            @ root_orientation_mat
        )
        root_orientation_quat = R.from_matrix(root_orientation_mat).as_quat()

        # rotate root position too around z at origin
        root_position = (
            R.from_euler("z", yaw_orientation, degrees=True).as_matrix() @ root_position
        )

        new_frames.append(list(root_position) + list(root_orientation_quat) + frame[7:])

    new_episode = episode.copy()
    new_episode["Frames"] = new_frames

    output_file = os.path.join(
        args.output_dir,
        os.path.basename(args.file).replace(".txt", f"_{yaw_orientation}.txt"),
    )

    with open(output_file, "w") as f:
        json.dump(new_episode, f)

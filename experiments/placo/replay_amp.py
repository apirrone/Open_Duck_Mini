import argparse
import json
import time

import FramesViewer.utils as fv_utils
import numpy as np
from FramesViewer.viewer import Viewer
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, required=True)
args = parser.parse_args()

fv = Viewer()
fv.start()

episode = json.load(open(args.file))

frame_duration = episode["FrameDuration"]

frames = episode["Frames"]
if "Debug_info" in episode:
    debug = episode["Debug_info"]
else:
    debug = None
pose = np.eye(4)
for i, frame in enumerate(frames):
    root_position = frame[:3]
    root_orientation_quat = frame[3:7]
    root_orientation_mat = R.from_quat(root_orientation_quat).as_matrix()

    if debug is not None:
        left_foot_pose = np.array(debug[i]["left_foot_pose"]).reshape(4, 4)
        right_foot_pose = np.array(debug[i]["right_foot_pose"]).reshape(4, 4)

    pose[:3, 3] = root_position
    pose[:3, :3] = root_orientation_mat

    fv.pushFrame(pose, "aze")

    if debug is not None:
        fv.pushFrame(left_foot_pose, "left")
        fv.pushFrame(right_foot_pose, "right")

    time.sleep(frame_duration)

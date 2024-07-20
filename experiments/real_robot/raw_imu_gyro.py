import argparse
import pickle
import time

import FramesViewer.utils as fv_utils
import numpy as np
from FramesViewer.viewer import Viewer
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--data", type=str, required=True)
args = parser.parse_args()

FPS = 30
gyro_data = pickle.load(open(args.data, "rb"))
fv = Viewer()
fv.start()

pose_quat = fv_utils.make_pose([0.1, 0.1, 0.1], [0, 0, 0])

quat = gyro_data[0][0]
# quat = [quat[3], quat[0], quat[1], quat[2]]
rot = R.from_quat(quat).as_matrix()
pose_quat[:3, :3] = rot
initial_pose = pose_quat.copy()
initial_pose = fv_utils.translateAbsolute(initial_pose, [0.1, 0.1, 0])

pose_ang_vel = initial_pose.copy()
i = 1
while True:
    print(i)
    quat = gyro_data[i][0]
    ang_vel = gyro_data[i][1]
    rot = R.from_quat(quat).as_matrix()
    pose_quat[:3, :3] = rot

    rot_ang_vel = R.from_matrix(pose_ang_vel[:3, :3]).as_euler("xyz", degrees=False)
    new_rot_ang_vel = np.array(rot_ang_vel) + (np.array(ang_vel) * (1 / FPS))
    rot_ang_vel = R.from_euler("xyz", new_rot_ang_vel, degrees=False).as_matrix()
    pose_ang_vel[:3, :3] = rot

    fv.pushFrame(pose_quat, "quat", color=(255, 0, 0))
    fv.pushFrame(pose_ang_vel, "ang_vel")
    time.sleep(1 / FPS)
    i += 1
    if i >= len(gyro_data) - 1:
        print("end, looping ")
        time.sleep(2)
        pose_ang_vel = initial_pose.copy()
        i = 0

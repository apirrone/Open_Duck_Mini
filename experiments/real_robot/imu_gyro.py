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
# from utils import ImuFilter

FPS = 30
gyro_data = pickle.load(open(args.data, "rb"))
fv = Viewer()
fv.start()

pose_euler = fv_utils.make_pose([0.1, 0.1, 0.1], [0, 0, 0])

quat = gyro_data[0][0]
quat = [quat[3], quat[0], quat[1], quat[2]]
rot = R.from_quat(quat).as_matrix()
rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ rot
pose_euler[:3, :3] = rot
pose_euler = fv_utils.rotateInSelf(pose_euler, [0, 0, 90])
initial_pose = pose_euler.copy()
initial_pose = fv_utils.translateAbsolute(initial_pose, [0.1, 0.1, 0])

pose_ang_vel = initial_pose.copy()
i = 1
while True:
    quat = gyro_data[i][0]
    ang_vel = gyro_data[i][1]
    ang_vel = [-ang_vel[1], ang_vel[0], ang_vel[2]]

    quat = [quat[3], quat[0], quat[1], quat[2]]
    rot = R.from_quat(quat).as_matrix()

    rot = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]]) @ rot

    pose_euler[:3, :3] = rot

    pose_euler = fv_utils.rotateInSelf(pose_euler, [0, 0, 90])

    rot_euler = R.from_matrix(pose_ang_vel[:3, :3]).as_euler("xyz", degrees=False)
    new_rot_euler = np.array(rot_euler) + (np.array(ang_vel) * (1 / FPS))
    rot = R.from_euler("xyz", new_rot_euler, degrees=False).as_matrix()
    pose_ang_vel[:3, :3] = rot

    fv.pushFrame(pose_euler, "euler", color=(255, 0, 0))
    fv.pushFrame(pose_ang_vel, "ang_vel")
    time.sleep(1 / FPS)
    i += 1
    if i >= len(gyro_data) - 1:
        print("end, looping ")
        time.sleep(2)
        pose_ang_vel = initial_pose.copy()
        i = 0


# imu_filter = ImuFilter(window_size=100)

linear_velocity = np.array([0.0, 0.0, 0.0])
position = np.array([0.0, 0.0, 0.0])
for linear_acceleration in linear_accelerations:
    linear_acceleration = np.array(linear_acceleration)
    imu_filter.push_data(linear_acceleration)
    linear_acceleration = imu_filter.get_filtered_data()
    # print(linear_acceleration)
    linear_velocity += linear_acceleration * (1 / FPS)
    position += linear_velocity * (1 / FPS)
    # pose[:3, 3] = linear_acceleration * 0.1
    pose[:3, 3] = position
    print(position)
    fv.pushFrame(pose, "imu")
    time.sleep(1 / FPS)

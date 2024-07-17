import pickle
import numpy as np
import time
from FramesViewer.viewer import Viewer
import FramesViewer.utils as fv_utils
from scipy.spatial.transform import Rotation as R

# from utils import ImuFilter

FPS = 30
euler_ang_vels = pickle.load(open("euler_ang_vel.pkl", "rb"))
fv = Viewer()
fv.start()

pose_euler = fv_utils.make_pose([0.1, 0.1, 0.1], [0, 0, 0])
pose_ang_vel = fv_utils.make_pose([0.2, 0.1, 0.1], [0, 0, 0])


i = 0
while True:
    quat = euler_ang_vels[i][0]
    ang_vel = euler_ang_vels[i][1]

    quat = [quat[3], quat[0], quat[1], quat[2]]
    rot = R.from_quat(quat).as_matrix()
    # rot = R.from_euler("xyz", euler, degrees=True).as_matrix()
    # rot = R.from_euler("zyx", euler, degrees=True).as_matrix()
    pose_euler[:3, :3] = np.linalg.inv(rot)

    rot_euler = R.from_matrix(pose_ang_vel[:3, :3]).as_euler("xyz", degrees=False)
    new_rot_euler = np.array(rot_euler) + (np.array(ang_vel) * (1 / FPS))
    rot = R.from_euler("xyz", new_rot_euler, degrees=False).as_matrix()
    pose_ang_vel[:3, :3] = rot

    fv.pushFrame(pose_euler, "euler", color=(255, 0, 0))
    fv.pushFrame(pose_ang_vel, "ang_vel")
    time.sleep(1 / FPS)
    i += 1
    if i >= len(euler_ang_vels) - 1:
        # pose_ang_vel = fv_utils.make_pose([0.2, 0.1, 0.1], [0, 0, 0])
        i -= 1


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

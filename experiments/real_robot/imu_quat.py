import pickle
import numpy as np
import time
from FramesViewer.viewer import Viewer
import FramesViewer.utils as fv_utils
from scipy.spatial.transform import Rotation as R


FPS = 60
# quats = pickle.load(open("gyro_data_turn_x.pkl", "rb"))
eulers = pickle.load(open("gyro_data.pkl", "rb"))
fv = Viewer()
fv.start()
pose = fv_utils.make_pose([0.1, 0.1, 0.1], [0, 0, 0])
i = 0
while True:
    # quat = quats[i]
    # print(quat)
    # mat = R.from_quat(quat).as_matrix()
    euler = eulers[i]
    print(euler)
    mat = R.from_euler("xyz", euler, degrees=True).as_matrix()

    pose[:3, :3] = mat
    pose = fv_utils.rotateInSelf(pose, [0, 0, -90], degrees=True)
    fv.pushFrame(pose, "imu")
    time.sleep(1 / FPS)
    i += 1
    if i >= len(eulers) - 1:
        i = 0

# for linear_acceleration in linear_accelerations:
#     linear_acceleration = np.array(linear_acceleration)
#     imu_filter.push_data(linear_acceleration)
#     linear_acceleration = imu_filter.get_filtered_data()
#     # print(linear_acceleration)
#     linear_velocity += linear_acceleration * (1 / FPS)
#     position += linear_velocity * (1 / FPS)
#     # pose[:3, 3] = linear_acceleration * 0.1
#     pose[:3, 3] = position
#     print(position)
#     fv.pushFrame(pose, "imu")
#     time.sleep(1 / FPS)

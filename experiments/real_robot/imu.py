import pickle
import numpy as np
import time
from FramesViewer.viewer import Viewer
import FramesViewer.utils as fv_utils

# from utils import ImuFilter

FPS = 60
quats = pickle.load(open("gyro_data.pkl", "rb"))
fv = Viewer()
fv.start()
pose = fv_utils.make_pose([0, 0, 0], [0, 0, 0])

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

"""
Replays imu data from a .pkl which is a list of quaternions
Shows the orientation in a 3D viewer
"""

from FramesViewer.viewer import Viewer
import pickle
from scipy.spatial.transform import Rotation as R
import numpy as np
import time

data = pickle.load(open("imu_data.pkl", "rb"))

fv = Viewer()
fv.start()


def reorient(quat):
    """
    Reorients because the IMU is mounted upside down
    """

    euler = R.from_quat(quat).as_euler("xyz")
    euler = [euler[1], euler[2], euler[0]]
    reoriented_quat = R.from_euler("xyz", euler).as_quat()
    return reoriented_quat


imu = np.eye(4)
imu[:3, 3] = [0.1, 0.1, 0.1]  # just easier to see

for raw_quat in data:
    quat = reorient(raw_quat)

    rot_mat = R.from_quat(quat).as_matrix()
    imu[:3, :3] = rot_mat
    fv.pushFrame(imu, "imu")
    time.sleep(1/30)




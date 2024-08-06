import argparse
import json
import time

import FramesViewer.utils as fv_utils
import numpy as np
from FramesViewer.viewer import Viewer
from scipy.spatial.transform import Rotation as R

parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", type=str, required=True)
parser.add_argument(
    "--hardware",
    action="store_true",
    help="use AMP_for_hardware format. If false, use IsaacGymEnvs format",
)
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
T_world_body = np.eye(4)
if args.hardware:
    vels = {}
    vels["linear_vel"] = []
    vels["angular_vel"] = []
    vels["joint_vels"] = []
for i, frame in enumerate(frames):
    root_position = frame[:3]
    root_orientation_quat = frame[3:7]
    root_orientation_mat = R.from_quat(root_orientation_quat).as_matrix()

    T_world_body[:3, 3] = root_position
    T_world_body[:3, :3] = root_orientation_mat

    fv.pushFrame(T_world_body, "aze")

    if debug is not None:
        left_foot_pose = np.array(debug[i]["left_foot_pose"]).reshape(4, 4)
        right_foot_pose = np.array(debug[i]["right_foot_pose"]).reshape(4, 4)
        fv.pushFrame(left_foot_pose, "left")
        fv.pushFrame(right_foot_pose, "right")

    if args.hardware:
        # vels["linear_vel"].append(frame[28:31])
        # vels["angular_vel"].append(frame[31:34])
        # vels["joint_vels"].append(frame[34:49])

        # FR, FL, RR, RL

        # [root_pos, root_rot, joint_pos, foot_pos, lin_vel, ang_vel,
        #     joint_vel, foot_vel])
        FR = frame[19 : 19 + 3]
        FL = frame[22 : 22 + 3]
        RR = frame[25 : 25 + 3]
        RL = frame[28 : 28 + 3]

        T_body_FR = fv_utils.make_pose(FR, [0, 0, 0])
        T_body_FL = fv_utils.make_pose(FL, [0, 0, 0])
        T_body_RR = fv_utils.make_pose(RR, [0, 0, 0])
        T_body_RL = fv_utils.make_pose(RL, [0, 0, 0])

        T_world_FR = T_world_body @ T_body_FR
        T_world_FL = T_world_body @ T_body_FL
        T_world_RR = T_world_body @ T_body_RR
        T_world_RL = T_world_body @ T_body_RL

        fv.pushFrame(T_world_FR, "FR")
        fv.pushFrame(T_world_FL, "FL")
        fv.pushFrame(T_world_RR, "RR")
        fv.pushFrame(T_world_RL, "RL")

        # left_toe_pos = frame[22:25]
        # right_toe_pos = frame[25:28]
        # fv.pushFrame(fv_utils.make_pose(left_toe_pos, [0, 0, 0]), "left_toe")
        # fv.pushFrame(fv_utils.make_pose(right_toe_pos, [0, 0, 0]), "right_toe")

    time.sleep(frame_duration * 2)


# if args.hardware:
#     # plot vels
#     from matplotlib import pyplot as plt

#     # TODO
#     x_lin_vel = [vels["linear_vel"][i][0] for i in range(len(frames))]
#     y_lin_vel = [vels["linear_vel"][i][1] for i in range(len(frames))]
#     z_lin_vel = [vels["linear_vel"][i][2] for i in range(len(frames))]

#     joints_vel = [vels["joint_vels"][i] for i in range(len(frames))]
#     angular_vel_x = [vels["angular_vel"][i][0] for i in range(len(frames))]
#     angular_vel_y = [vels["angular_vel"][i][1] for i in range(len(frames))]
#     angular_vel_z = [vels["angular_vel"][i][2] for i in range(len(frames))]

#     plt.plot(angular_vel_x, label="angular_vel_x")
#     plt.plot(angular_vel_y, label="angular_vel_y")
#     plt.plot(angular_vel_z, label="angular_vel_z")

#     plt.legend()
#     plt.show()

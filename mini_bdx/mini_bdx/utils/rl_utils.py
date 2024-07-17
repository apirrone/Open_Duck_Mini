# This is the joints order when loading using IsaacGymEnvs
# ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle', 'neck_pitch', 'head_pitch', 'head_yaw', 'left_antenna', 'right_antenna', 'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle']
# This is the "standard" order (from mujoco)
# ['right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle', 'left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle', 'neck_pitch', 'head_pitch', 'head_yaw', 'left_antenna', 'right_antenna']
#
# We need to reorder the joints to match the IsaacGymEnvs order
#

mujoco_joints_order = [
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "left_antenna",
    "right_antenna",
]

isaac_joints_order = [
    "left_hip_yaw",
    "left_hip_roll",
    "left_hip_pitch",
    "left_knee",
    "left_ankle",
    "neck_pitch",
    "head_pitch",
    "head_yaw",
    "left_antenna",
    "right_antenna",
    "right_hip_yaw",
    "right_hip_roll",
    "right_hip_pitch",
    "right_knee",
    "right_ankle",
]


def isaac_to_mujoco(joints):
    new_joints = [
        # right leg
        joints[10],
        joints[11],
        joints[12],
        joints[13],
        joints[14],
        # left leg
        joints[0],
        joints[1],
        joints[2],
        joints[3],
        joints[4],
        # head
        joints[5],
        joints[6],
        joints[7],
        joints[8],
        joints[9],
    ]

    return new_joints


def mujoco_to_isaac(joints):
    new_joints = [
        # left leg
        joints[5],
        joints[6],
        joints[7],
        joints[8],
        joints[9],
        # head
        joints[10],
        joints[11],
        joints[12],
        joints[13],
        joints[14],
        # right leg
        joints[0],
        joints[1],
        joints[2],
        joints[3],
        joints[4],
    ]
    return new_joints


def action_to_pd_targets(action, pd_action_offset, pd_action_scale):
    return pd_action_offset + pd_action_scale * action

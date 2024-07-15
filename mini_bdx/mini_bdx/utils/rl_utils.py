# This is the joints order when loading using IsaacGymEnvs
# ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle', 'neck_pitch', 'head_pitch', 'head_yaw', 'left_antenna', 'right_antenna', 'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle']
# This is the "standard" order (from mujoco)
# ['left_hip_yaw', 'left_hip_roll', 'left_hip_pitch', 'left_knee', 'left_ankle', 'right_hip_yaw', 'right_hip_roll', 'right_hip_pitch', 'right_knee', 'right_ankle', 'neck_pitch', 'head_pitch', 'head_yaw', 'left_antenna', 'right_antenna']
#
# We need to reorder the joints to match the IsaacGymEnvs order
def isaac_to_mujoco(joints):
    new_joints = [
        joints[0],
        joints[1],
        joints[2],
        joints[3],
        joints[4],
        joints[10],
        joints[11],
        joints[12],
        joints[13],
        joints[14],
        joints[5],
        joints[6],
        joints[7],
        joints[8],
        joints[9],
    ]

    return new_joints


def mujoco_to_isaac(joints):
    new_joints = [
        joints[0],
        joints[1],
        joints[2],
        joints[3],
        joints[4],
        joints[10],
        joints[11],
        joints[12],
        joints[13],
        joints[14],
        joints[5],
        joints[6],
        joints[7],
        joints[8],
        joints[9],
    ]
    return new_joints


if __name__ == "__main__":
    mujoco_joints = [
        "left_hip_yaw",
        "left_hip_roll",
        "left_hip_pitch",
        "left_knee",
        "left_ankle",
        "right_hip_yaw",
        "right_hip_roll",
        "right_hip_pitch",
        "right_knee",
        "right_ankle",
        "neck_pitch",
        "head_pitch",
        "head_yaw",
        "left_antenna",
        "right_antenna",
    ]
    isaac_joints = [
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

    print(mujoco_to_isaac(mujoco_joints))
    print(isaac_to_mujoco(isaac_joints))

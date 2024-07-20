import numpy as np

mujoco_init_pos = np.array(
    [
        # right_leg
        0.013627156377842975,
        0.07738878096596595,
        0.5933527914082196,
        -1.630548419252953,
        0.8621333440557593,
        # left leg
        -0.013946457213457239,
        0.07918837709879874,
        0.5325073962634973,
        -1.6225192902713386,
        0.9149246381274986,
        # head
        -0.17453292519943295,
        -0.17453292519943295,
        8.65556854322817e-27,
        0,
        0,
    ]
)
id_to_dof = {
    0: "right_hip_yaw",
    1: "right_hip_roll",
    2: "right_hip_pitch",
    3: "right_knee",
    4: "right_ankle",
    5: "left_hip_yaw",
    6: "left_hip_roll",
    7: "left_hip_pitch",
    8: "left_knee",
    9: "left_ankle",
    10: "neck_pitch",
    11: "head_pitch",
    12: "head_yaw",
    13: "left_antenna",
    14: "right_antenna",
}
dof_to_id = {v: k for k, v in id_to_dof.items()}

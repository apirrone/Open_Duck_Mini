import placo
from placo_utils.visualization import robot_viz

import numpy as np
import time

from mini_bdx_runtime.io_330 import Dxl330IO

dxl_io = Dxl330IO("/dev/ttyUSB0", baudrate=3000000, use_sync_read=True)

# Based on performance graph here https://emanual.robotis.com/docs/en/dxl/x/xc330-m288/
A = 2.69
B = 0.19
current_limit = 2.3
torque_limit = 1.0


def current_to_torque(current: float) -> float:
    """
    Input current in A
    Output torque in Nm
    """
    torque = (current - B) / A
    return min(torque_limit, max(-torque_limit, torque))


def torque_to_current(torque: float) -> float:
    """
    Input torque in Nm
    Output current in A
    """
    current = A * torque + B
    return min(current_limit, max(-current_limit, current))


def torque_to_current2(torque: float) -> float:
    """
    Input torque in Nm
    Output current in A
    """
    kt = 2.73
    return torque / kt


joints = {
    "right_hip_yaw": 10,
    "right_hip_roll": 11,
    "right_hip_pitch": 12,
    "right_knee": 13,
    "right_ankle": 14,
}

dxl_io.set_operating_mode({id: 0x0 for id in joints.values()})  # set in current mode


def get_right_leg_position():
    present_positions_list = dxl_io.get_present_position(joints.values())
    present_positions = {
        joint: -np.deg2rad(position)
        for joint, position in zip(joints, present_positions_list)
    }
    return present_positions


robot = placo.HumanoidRobot("../../mini_bdx/robots/bdx/robot.urdf")

input("press any key to record position")
right_leg_position = get_right_leg_position()
for joint, position in right_leg_position.items():
    robot.set_joint(joint, position)


# viz = robot_viz(robot)
# while True:
#     viz.display(robot.state.q)
#     time.sleep(1 / 20)
# exit()
def get_target_current():
    target_torques = robot.static_gravity_compensation_torques_dict("trunk")
    right_leg_target_torques = {}
    for joint, torque in target_torques.items():
        if joint in list(joints.keys()):
            right_leg_target_torques[joint] = torque

    print("target torque", right_leg_target_torques)

    right_leg_target_current = {}
    for joint, torque in right_leg_target_torques.items():
        right_leg_target_current[joint] = -torque_to_current2(torque) * 1000

    print("target current", right_leg_target_current)
    right_leg_target_current_id = {
        joints[joint]: round(current)
        for joint, current in right_leg_target_current.items()
    }
    print("target current id", right_leg_target_current_id)
    return right_leg_target_current_id


time.sleep(1)
input("press enter to set torques")
# exit()
dxl_io.enable_torque(joints.values())
dxl_io.set_goal_current(get_target_current())
try:
    while True:
        right_leg_position = get_right_leg_position()
        for joint, position in right_leg_position.items():
            robot.set_joint(joint, position)
        dxl_io.set_goal_current(get_target_current())
        print("running")
        time.sleep(1.0)
except KeyboardInterrupt:
    print("STOP")
    dxl_io.disable_torque(joints.values())
    pass

import argparse
import time

import cv2
import numpy as np
import placo

from mini_bdx.hwi import HWI
from mini_bdx.utils.xbox_controller import XboxController
from mini_bdx.walk_engine import WalkEngine

parser = argparse.ArgumentParser()
parser.add_argument("-x", action="store_true", default=False)
args = parser.parse_args()

if args.x:
    xbox = XboxController()


hwi = HWI(usb_port="/dev/ttyUSB0")

max_target_step_size_x = 0.03
max_target_step_size_y = 0.03
max_target_yaw = np.deg2rad(15)
target_step_size_x = 0
target_step_size_y = 0
target_yaw = 0
target_head_pitch = 0
target_head_yaw = 0
target_head_z_offset = 0
time_since_last_left_contact = 0
time_since_last_right_contact = 0
walking = False
start_button_timeout = time.time()

robot = placo.RobotWrapper(
    "../../mini_bdx/robots/bdx/robot.urdf", placo.Flags.ignore_collisions
)

walk_engine = WalkEngine(
    robot,
    frequency=1.5,
    swing_gain=0.0,
    default_trunk_x_offset=-0.013,
    default_trunk_z_offset=-0.023,
    target_trunk_pitch=-11.0,
    max_rise_gain=0.01,
)


def xbox_input():
    global target_step_size_x, target_step_size_y, target_yaw, walking, t, walk_engine, target_head_pitch, target_head_yaw, target_head_z_offset, start_button_timeout, max_target_step_size_x, max_target_step_size_y, max_target_yaw
    inputs = xbox.read()
    # print(inputs)
    target_step_size_x = -inputs["l_y"] * max_target_step_size_x
    target_step_size_y = inputs["l_x"] * max_target_step_size_y
    if inputs["l_trigger"] > 0.2:
        target_head_pitch = inputs["r_y"] / 2 * np.deg2rad(70)
        print("=== target head pitch", target_head_pitch)
        target_head_yaw = -inputs["r_x"] / 2 * np.deg2rad(150)
        target_head_z_offset = inputs["r_trigger"] * 4 * 0.2
        print(target_head_z_offset)
        # print("======", target_head_z_offset)
    else:
        target_yaw = -inputs["r_x"] * max_target_yaw

    if inputs["start"] and time.time() - start_button_timeout > 0.5:
        walking = not walking
        start_button_timeout = time.time()


im = np.zeros((80, 80, 3), dtype=np.uint8)


# TODO
def get_imu():
    return [0, 0, 0], [0, 0, 0]


# hwi.turn_off()
# exit()
hwi.turn_on()
time.sleep(1)
# hwi.goto_init()

# exit()
gyro = [0, 0.0, 0]
accelerometer = [0, 0, 0]

skip = 10
prev = time.time()
while True:
    dt = time.time() - prev
    t = time.time()
    if args.x:
        xbox_input()

    # TODO use current to find out when the foot is on the ground
    print("present CURRENT right ankle", hwi.get_present_current("right_ankle"))
    print("goal CURRENT right ankle", hwi.get_goal_current("right_ankle"))
    print(
        "sub",
        hwi.get_goal_current("right_ankle") - hwi.get_present_current("right_ankle"),
    )
    print("===")

    # Get sensor data
    # gyro, accelerometer = get_imu()

    walk_engine.update(
        walking,
        gyro,
        accelerometer,
        False,
        False,
        target_step_size_x,
        target_step_size_y,
        target_yaw,
        target_head_pitch,
        target_head_yaw,
        target_head_z_offset,
        dt,
        ignore_feet_contact=True,
    )
    angles = walk_engine.get_angles()

    if skip > 0:
        skip -= 1
        continue
    hwi.set_position_all(angles)

    # print("-")
    cv2.imshow("image", im)
    key = cv2.waitKey(1)
    if key == ord("p"):
        # gyro[1] += 0.001
        walk_engine.target_trunk_pitch += 0.1
    if key == ord("o"):
        walk_engine.target_trunk_pitch -= 0.1
        # gyro[1] -= 0.001
    if key == ord("m"):
        walk_engine.max_rise_gain += 0.001
    if key == ord("l"):
        walk_engine.max_rise_gain -= 0.001
    if key == ord("b"):
        walk_engine.default_trunk_x_offset += 0.001
    if key == ord("n"):
        walk_engine.default_trunk_x_offset -= 0.001
    if key == ord("i"):
        walk_engine.default_trunk_z_offset += 0.001
    if key == ord("u"):
        walk_engine.default_trunk_z_offset -= 0.001
    if key == ord("f"):
        walk_engine.frequency -= 0.1
    if key == ord("g"):
        walk_engine.frequency += 0.1
    if key == ord("c"):
        walk_engine.swing_gain -= 0.001
    if key == ord("v"):
        walk_engine.swing_gain += 0.001
    if key == ord("w"):
        walking = not walking

    # print("gyro : ", gyro)
    # print("target_trunk pitch", walk_engine.trunk_pitch)
    # print("trunk x offset", walk_engine.default_trunk_x_offset)
    # print("trunk z offset", walk_engine.default_trunk_z_offset)
    # print("max rise gain", walk_engine.max_rise_gain)
    # print("frequency", walk_engine.frequency)
    # print("swing gain", walk_engine.swing_gain)
    # print("===")

    prev = t

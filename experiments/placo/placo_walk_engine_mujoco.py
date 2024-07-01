import argparse

import mujoco
import mujoco.viewer

from mini_bdx.placo_walk_engine import PlacoWalkEngine
from mini_bdx.utils.mujoco_utils import check_contact
from mini_bdx.utils.xbox_controller import XboxController

parser = argparse.ArgumentParser()
parser.add_argument("-x", action="store_true", default=False)
args = parser.parse_args()

if args.x:
    xbox = XboxController()

d_x = 0
d_y = 0
d_theta = 0


pwe = PlacoWalkEngine("../../mini_bdx/robots/bdx/robot.urdf")


def xbox_input():
    global d_x, d_y, d_theta
    inputs = xbox.read()
    # print(inputs)
    d_x = -inputs["l_y"] * pwe.parameters.walk_max_dx_forward / 5
    d_y = inputs["l_x"] * pwe.parameters.walk_max_dy / 5
    d_theta = -inputs["r_x"] * pwe.parameters.walk_max_dtheta / 10
    # if inputs["l_trigger"] > 0.2:
    #     target_head_pitch = inputs["r_y"] / 2 * np.deg2rad(70)
    #     print("=== target head pitch", target_head_pitch)
    #     target_head_yaw = -inputs["r_x"] / 2 * np.deg2rad(150)
    #     target_head_z_offset = inputs["r_trigger"] * 4 * 0.2
    #     print(target_head_z_offset)
    #     # print("======", target_head_z_offset)
    # else:
    #     target_yaw = -inputs["r_x"] * max_target_yaw

    # if inputs["start"] and time.time() - start_button_timeout > 0.5:
    #     walking = not walking
    #     start_button_timeout = time.time()


model = mujoco.MjModel.from_xml_path("../../mini_bdx/robots/bdx/scene.xml")
data = mujoco.MjData(model)
viewer = mujoco.viewer.launch_passive(model, data)


def get_feet_contact():
    right_contact = check_contact(data, model, "foot_module", "floor")
    left_contact = check_contact(data, model, "foot_module_2", "floor")
    return right_contact, left_contact


speed = 3  # 1 is slowest, 3 looks real time on my machine
prev = data.time
while True:
    t = data.time
    dt = t - prev

    if args.x:
        xbox_input()

    pwe.d_x = d_x
    pwe.d_y = d_y
    pwe.d_theta = d_theta
    right_contact, left_contact = get_feet_contact()
    pwe.tick(dt, left_contact, right_contact)

    angles = pwe.get_angles()
    data.ctrl[:] = list(angles.values())

    mujoco.mj_step(model, data, speed)  # 4 seems good
    viewer.sync()
    prev = t

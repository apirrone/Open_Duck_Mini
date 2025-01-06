from mini_bdx_runtime.hwi_feetech_pypot import HWI
from mini_bdx.placo_walk_engine.placo_walk_engine import PlacoWalkEngine
import json
import time

hwi = HWI()
hwi.turn_on()

DT = 0.01
pwe = PlacoWalkEngine(
    "/home/antoine/MISC/mini_BDX/mini_bdx/robots/open_duck_mini_v2",
    model_filename="robot.urdf",
    init_params=json.load(open("placo_defaults.json")),
    ignore_feet_contact=True
)
pwe.set_traj(0.0, 0, 0.0)
while True:
    pwe.tick(DT)
    all_angles = list(pwe.get_angles().values())

    angles = {}
    for i, motor_name in enumerate(hwi.joints.keys()):
        angles[motor_name] = all_angles[i]

    hwi.set_position_all(angles)
    time.sleep(DT)
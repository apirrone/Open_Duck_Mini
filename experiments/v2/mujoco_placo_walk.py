from mini_bdx.placo_walk_engine.placo_walk_engine import PlacoWalkEngine
import time
import json
import mujoco
import mujoco.viewer
import pickle
from mini_bdx.utils.mujoco_utils import check_contact

# DT = 0.01
DT = 0.001
pwe = PlacoWalkEngine(
    "/home/antoine/MISC/mini_BDX/mini_bdx/robots/open_duck_mini_v2",
    model_filename="robot.urdf",
    init_params=json.load(open("placo_defaults.json")),
)
model = mujoco.MjModel.from_xml_path(
    "/home/antoine/MISC/mini_BDX/mini_bdx/robots/open_duck_mini_v2/scene.xml"
)
model.opt.timestep = DT
data = mujoco.MjData(model)

angles = pickle.load(open("init_angles.pkl", "rb"))

data.ctrl[:] = angles
data.qpos[3+4:] = angles
data.qpos[3 : 3 + 4] = [1, 0, 0.06, 0]


def get_feet_contact():
    left_contact = check_contact(data, model, "foot_assembly", "floor")
    right_contact = check_contact(data, model, "foot_assembly_2", "floor")
    return right_contact, left_contact


pwe.set_traj(0.05, 0, 0.0)
with mujoco.viewer.launch_passive(
    model, data, show_left_ui=False, show_right_ui=False
) as viewer:
    while True:
        right_contact, left_contact = get_feet_contact()
        pwe.tick(DT * 10, left_contact=left_contact, right_contact=right_contact)
        angles = list(pwe.get_angles().values())

        # pickle.dump(angles, open("init_angles.pkl", "wb"))
        # exit()

        data.ctrl[:] = angles

        mujoco.mj_step(model, data, 10)
        viewer.sync()
        time.sleep(DT)

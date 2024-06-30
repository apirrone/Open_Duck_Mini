import mujoco
import mujoco.viewer

from mini_bdx.placo_walk_engine import PlacoWalkEngine
from mini_bdx.utils.mujoco_utils import check_contact

pwe = PlacoWalkEngine("../../mini_bdx/robots/bdx/robot.urdf")

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

    right_contact, left_contact = get_feet_contact()
    pwe.tick(dt, left_contact, right_contact)

    angles = pwe.get_angles()
    data.ctrl[:] = list(angles.values())

    mujoco.mj_step(model, data, speed)  # 4 seems good
    viewer.sync()
    prev = t

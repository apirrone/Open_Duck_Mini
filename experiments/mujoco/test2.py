import mujoco
import time
import mujoco_viewer

model = mujoco.MjModel.from_xml_path("scene_test.xml")
model.opt.timestep = 1 / 60
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
viewer = mujoco_viewer.MujocoViewer(model, data)

while True:
    mujoco.mj_step(model, data)
    viewer.render()
    # time.sleep(1 / 60)

import FramesViewer.utils as fv_utils
import mujoco
import mujoco.viewer
import mujoco_viewer
import numpy as np
from FramesViewer.viewer import Viewer
from scipy.spatial.transform import Rotation as R

from mini_bdx.placo_walk_engine import PlacoWalkEngine

# fv = Viewer()
# fv.start()
model = mujoco.MjModel.from_xml_path("../../mini_bdx/robots/bdx/scene.xml")
data = mujoco.MjData(model)
# viewer = mujoco.viewer.launch_passive(model, data)
viewer = mujoco_viewer.MujocoViewer(model, data, width=1280, height=720)
pwe = PlacoWalkEngine(
    "/home/antoine/MISC/mini_BDX/mini_bdx/robots/bdx/robot.urdf",
    ignore_feet_contact=True,
)

pwe.set_traj(0.02, 0, 0.001)


def orient_reward():
    euler = R.from_matrix(pwe.robot.get_T_world_fbase()[:3, :3]).as_euler("xyz")
    desired_yaw = euler[2]
    current_yaw = R.from_matrix(
        np.array(data.body("base").xmat).reshape(3, 3)
    ).as_euler("xyz")[2]
    return -((desired_yaw - current_yaw) ** 2)


def draw_frame(pose, i):
    pose = fv_utils.rotateInSelf(pose, [0, 90, 0])
    viewer.add_marker(
        pos=pose[:3, 3],
        mat=pose[:3, :3],
        size=[0.005, 0.005, 0.1],
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        rgba=[1, 0, 0, 1],
        label=str(i),
    )


prev = data.time
while True:
    t = data.time
    dt = t - prev

    pwe.tick(dt)

    next_footsteps = pwe.get_footsteps_in_world()
    for i in range(len(next_footsteps)):
        next_footsteps[i][:3, 3][2] = 0
    for i, footstep in enumerate(next_footsteps):
        draw_frame(footstep, i)
    if t > 1.0:
        pos = data.body("base").xpos
        mat = data.body("base").xmat
        T_world_body = np.eye(4)
        T_world_body[:3, :3] = mat.reshape(3, 3)
        T_world_body[:3, 3] = pos
        print(orient_reward())
        draw_frame(T_world_body, 100)

        T_world_rightFoot = np.eye(4)
        pos = data.body("right_foot").xpos
        mat = data.body("right_foot").xmat
        T_world_rightFoot[:3, 3] = pos
        T_world_rightFoot[:3, :3] = mat.reshape(3, 3)

        T_body_rightFoot = np.linalg.inv(T_world_body) @ T_world_rightFoot
        T_world_rightFoot2 = T_world_body @ T_body_rightFoot
        draw_frame(T_world_rightFoot, 101)

    mujoco.mj_step(model, data, 4)  # 4 seems good

    viewer.render()

    prev = t

import argparse

import mujoco_viewer

import mujoco

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p-", "--path", type=str, required=True, help="Path to the xml file"
)
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(args.path)
data = mujoco.MjData(model)

# create the viewer object
viewer = mujoco_viewer.MujocoViewer(model, data)

# simulate and render
for _ in range(10000):
    if viewer.is_alive:
        mujoco.mj_step(model, data)
        viewer.render()
    else:
        break

# close
viewer.close()
# close
viewer.close()

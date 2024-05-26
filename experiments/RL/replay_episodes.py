import argparse
import pickle
import time

import mujoco
import mujoco.viewer

parser = argparse.ArgumentParser()
parser.add_argument("-d", "--dataset", type=str, required=True)
args = parser.parse_args()

episodes = pickle.load(open(args.dataset, "rb"))

model = mujoco.MjModel.from_xml_path("../../mini_bdx/robots/bdx/scene.xml")
data = mujoco.MjData(model)


def key_callback(keycode):
    pass


viewer = mujoco.viewer.launch_passive(model, data, key_callback=key_callback)

current_episode_id = 0
current_episode = episodes[current_episode_id]

prev = data.time
try:
    while True:
        for i in range(len(current_episode.acts)):
            angles = current_episode.acts[i]
            dt = data.time - prev

            data.ctrl[:] = angles

            prev = data.time
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep / 2.5)
        print("new episode")
        current_episode_id += 1
        if current_episode_id >= len(episodes):
            current_episode_id = 0

        current_episode = episodes[current_episode_id]


except KeyboardInterrupt:
    viewer.close()

viewer.close()

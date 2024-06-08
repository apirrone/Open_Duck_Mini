import os
import time
from queue import Queue
from threading import Thread

import mujoco
import mujoco.viewer


class BDXMujocoServer:
    def __init__(self, model_path="../../mini_bdx/robots/bdx/"):
        self.model = mujoco.MjModel.from_xml_path(os.path.join(model_path, "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.actions_queue = Queue()

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)
        Thread(target=self.run, daemon=True).start()

    def run(self):
        prev = self.data.time
        while True:
            dt = self.data.time - prev

            try:
                actions = self.actions_queue.get(block=False)
                self.data.ctrl[:] = actions
            except:
                pass

            prev = self.data.time
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.001)

    def send_action(self, action):
        self.actions_queue.put(action)

    def get_state(self):
        return self.data.qpos.flat.copy(), self.data.qvel.flat.copy()


if __name__ == "__main__":
    server = BDXMujocoServer()
    while True:
        server.send_action([0.5] * 13)
        time.sleep(0.1)

import os
import time
from queue import Queue
from threading import Thread

import mujoco
import mujoco.viewer
import numpy as np
from scipy.spatial.transform import Rotation as R


class BDXMujocoServer:
    def __init__(self, model_path="../../mini_bdx/robots/bdx/"):
        self.model = mujoco.MjModel.from_xml_path(os.path.join(model_path, "scene.xml"))
        self.data = mujoco.MjData(self.model)
        self.actions_queue = Queue()

        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

        self.dt = 0

    def start(self):
        Thread(target=self.run, daemon=True).start()

    def run(self):
        prev = self.data.time
        while True:
            self.dt = self.data.time - prev
            try:
                actions = self.actions_queue.get(block=False)
                self.data.ctrl[:] = actions
            except Exception:
                pass

            prev = self.data.time
            mujoco.mj_step(self.model, self.data)
            self.viewer.sync()
            time.sleep(0.001)

    def send_action(self, action):
        self.actions_queue.put(action)

    def get_state(self):
        return self.data.qpos.flat.copy(), self.data.qvel.flat.copy()

    def get_imu(self):

        rot_mat = np.array(self.data.body("base").xmat).reshape(3, 3)
        gyro = R.from_matrix(rot_mat).as_euler("xyz")

        accelerometer = np.array(self.data.body("base").cvel)[3:]

        return gyro, accelerometer

    def get_feet_contact(self):
        right_contact = self.check_contact("foot_module", "floor")
        left_contact = self.check_contact("foot_module_2", "floor")
        return right_contact, left_contact

    def check_contact(self, body1_name, body2_name):
        body1_id = self.data.body(body1_name).id
        body2_id = self.data.body(body2_name).id

        for i in range(self.data.ncon):
            contact = self.data.contact[i]

            if (
                self.model.geom_bodyid[contact.geom1] == body1_id
                and self.model.geom_bodyid[contact.geom2] == body2_id
            ) or (
                self.model.geom_bodyid[contact.geom1] == body2_id
                and self.model.geom_bodyid[contact.geom2] == body1_id
            ):
                return True

        return False


if __name__ == "__main__":
    server = BDXMujocoServer()
    while True:
        server.send_action([0.5] * 13)
        time.sleep(0.1)

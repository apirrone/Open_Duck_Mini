import time

import numpy as np

from mini_bdx.bdx_mujoco_server import BDXMujocoServer

bdx_mujoco_server = BDXMujocoServer(
    model_path="../../mini_bdx/robots/bdx", gravity_on=False
)
bdx_mujoco_server.start()
action = np.zeros(13)
while True:
    act = action.copy()
    act[6] = np.sin(time.time())
    bdx_mujoco_server.send_action(act)
    time.sleep(1 / 30)

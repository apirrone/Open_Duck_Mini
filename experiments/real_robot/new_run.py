from mini_bdx_runtime.hwi import HWI
import time
import numpy as np
from mini_bdx.placo_walk_engine import PlacoWalkEngine

PLACO_DT = 0.001

pwe = PlacoWalkEngine("../../mini_bdx/robots/bdx/robot.urdf", ignore_feet_contact=True)
pwe.set_traj(0.00, 0, 0 + 0.001)

hwi = HWI("/dev/ttyUSB0")

pid = (1000, 0, 100)

hwi.turn_on()
hwi.set_pid_all(pid)

# exit()

control_freq = 60
prev = time.time()
while True:
    t = time.time()
    dt = t - prev
    pwe.tick(dt)
    print(pwe.t)

    joints_positions = pwe.get_angles()
    del joints_positions["left_antenna"]
    del joints_positions["right_antenna"]
    print(joints_positions)
    if pwe.t >= 0:
        exit()
    hwi.set_position_all(joints_positions)

    took = time.time() - t
    time.sleep(max(0, (1 / control_freq) - took))
    prev = t

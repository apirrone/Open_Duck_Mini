import time

import numpy as np
from mini_bdx_runtime.hwi import HWI

hwi = HWI(usb_port="/dev/ttyUSB0")

hwi.set_pid_all([1000, 0, 500])
hwi.turn_on()
time.sleep(1)

while True:

    ankle_pos = 0.3 * np.sin(2 * np.pi * 0.5 * time.time())
    hwi.set_position("right_ankle", ankle_pos)
    hwi.set_position("left_ankle", ankle_pos)

    time.sleep(1 / 60)

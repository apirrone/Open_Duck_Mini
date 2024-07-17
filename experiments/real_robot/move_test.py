from mini_bdx.hwi import HWI
import pickle
import time
import numpy as np

move = pickle.load(open("move.pkl", "rb"))
hwi = HWI(usb_port="/dev/ttyUSB0")

move_without_antennas = []
# remove antennas keys
for m in move:
    m = {k: v for k, v in m.items() if "antenna" not in k}
    move_without_antennas.append(m)


command_value = []

hwi.turn_on()
time.sleep(1)
ctrl_freq = 60  # hz
start = time.time()
i = 0
while True:
    # pos = hwi.init_pos.copy()
    # pos["left_hip_pitch"] += np.sin(5 * time.time()) / 3
    # pos["left_knee"] -= np.sin(5 * time.time()) / 3
    # pos["left_ankle"] -= np.sin(5 * time.time()) / 3

    # pos["right_hip_pitch"] += np.sin(5 * time.time()) / 3
    # pos["right_knee"] -= np.sin(5 * time.time()) / 3
    # pos["right_ankle"] -= np.sin(5 * time.time()) / 3
    # print(pos["right_ankle"])

    pos = move_without_antennas[i]
    hwi.set_position_all(pos)

    command_value.append((list(pos.values()), hwi.get_present_positions()))

    time.sleep(1 / ctrl_freq)

    i += 1
    if i >= len(move) - 1:
        i = 0

pickle.dump(command_value, open("command_value.pkl", "wb"))

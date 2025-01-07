from mini_bdx_runtime.hwi_feetech_pypot import HWI
import time
import numpy as np

hwi = HWI()
hwi.turn_on()

times = []
for i in range(2000):
    s = time.time()
    all = hwi.get_present_positions()
    took = time.time() - s
    print("took", np.around(took, 3))
    times.append(took)

    time.sleep(0.01)

print(np.mean(times))





from pypot.feetech import FeetechSTS3215IO
import time
import numpy as np

io = FeetechSTS3215IO(
    "/dev/ttyACM0",
    baudrate=1000000,
    use_sync_read=True,
)

# id = 24
ids = [10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31, 32, 33]

io.enable_torque(ids)
io.set_mode({id: 0 for id in ids})
times = []
for i in range(1000):
    s = time.time()
    io.get_present_position(ids)
    io.set_goal_position({id: 0 for id in ids})
    times.append(time.time() - s)

    time.sleep(1 / 100)

print("avg :", np.mean(times))
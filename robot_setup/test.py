import time

import numpy as np
from io_330 import Dxl330IO

dxl_io = Dxl330IO("/dev/ttyUSB0", baudrate=1000000)
# ids = dxl_io.scan([32])
# ids = [30, 31, 32, 10, 11, 12, 13, 14, 20, 21, 22, 23, 24]
# dxl_io.set_pid_gain({id: [1800, 0, 0] for id in ids})
# exit()
# dxl_io.disable_torque(ids)
dxl_io.disable_torque([12])
time.sleep(1)
exit()
# # dxl_io.set_return_delay_time(
# #     {
# #         30: 0,
# #         31: 0,
# #         32: 0,
# #         10: 0,
# #         11: 0,
# #         12: 0,
# #         13: 0,
# #         14: 0,
# #         20: 0,
# #         21: 0,
# #         22: 0,
# #         23: 0,
# #         24: 0,
# #     }
# # )
# # exit()

prev = time.time()
while True:
    t = time.time()
    dt = time.time() - prev
    present_positions = dxl_io.get_present_position(ids)
    for i in range(len(ids)):
        print("ID:", ids[i], "Position:", present_positions[i])

    print("---")
    # print(dxl_io.get_present_position(ids))
    prev = t
    # print("fps", 1 / dt)
    # time.sleep(0.01)


# F = 1.0
# dxl_io.enable_torque([31, 32])
# while True:
#     goal = 15 * np.sin(2 * np.pi * F * time.time())
#     print(goal)
#     dxl_io.set_goal_position({31: goal})
#     dxl_io.set_goal_position({32: goal})

# ids = [1]

# try:
# except Exception as e:
#     print(e)
#     print("Failed to change ID")
#     exit(1)
# time.sleep(1)
# print("done")
# dxl_io.close()

# dxl_io = Dxl320IO("/dev/ttyUSB0", baudrate=57600)
# ids = dxl_io.scan([1, 32])
# print(ids)

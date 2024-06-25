import time

from mini_bdx.io_330 import Dxl330IO

# from pypot.dynamixel import Dxl320IO

dxl_io = Dxl330IO("/dev/ttyUSB0", baudrate=57600)

new_id = 24
ids = dxl_io.scan([1])
print(ids)
# dxl_io.change_id({1: new_id})
# ids = dxl_io.scan([1, new_id])
# print(ids)
# exit()
# print(ids)

# dxl_io.enable_torque([32])
# dxl_io.disable_torque([32])
# while True:
#     print(dxl_io.get_present_position([32]))

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

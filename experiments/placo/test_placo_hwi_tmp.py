import time

from mini_bdx_runtime.hwi import HWI

hwi = HWI(usb_port="/dev/ttyUSB0")

hwi.set_pid_all([1000, 0, 500])
hwi.turn_on()
time.sleep(1)

import time

from mini_bdx.hwi import HWI

hwi = HWI(usb_port="/dev/ttyUSB0")

while True:
    present_current_right_ankle = hwi.get_present_current("right_ankle")
    present_current_left_ankle = hwi.get_present_current("left_ankle")

    goal_current_right_ankle = hwi.get_goal_current("right_ankle")
    goal_current_left_ankle = hwi.get_goal_current("left_ankle")

    print("present", present_current_right_ankle, "goal", goal_current_right_ankle)
    time.sleep(0.1)

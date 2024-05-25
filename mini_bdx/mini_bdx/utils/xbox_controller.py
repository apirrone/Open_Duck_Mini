import math
import threading

from inputs import get_gamepad


class XboxController(object):
    MAX_TRIG_VAL = math.pow(2, 8)
    MAX_JOY_VAL = math.pow(2, 15)

    def __init__(self):

        self.LeftJoystickY = 0
        self.LeftJoystickX = 0
        self.RightJoystickY = 0
        self.RightJoystickX = 0
        self.LeftTrigger = 0
        self.RightTrigger = 0
        self.LeftBumper = 0
        self.RightBumper = 0
        self.A = 0
        self.X = 0
        self.Y = 0
        self.B = 0
        self.LeftThumb = 0
        self.RightThumb = 0
        self.Back = 0
        self.Start = 0
        self.LeftDPad = 0
        self.RightDPad = 0
        self.UpDPad = 0
        self.DownDPad = 0

        self._monitor_thread = threading.Thread(
            target=self._monitor_controller, args=()
        )
        self._monitor_thread.daemon = True
        self._monitor_thread.start()

    def deadzone(self, val, range):
        if abs(val) < range:
            return 0
        return val

    def read(self):  # return the buttons/triggers that you care about in this method
        ret = {
            "l_x": round(self.deadzone(self.LeftJoystickX, 0.15), 2),
            "l_y": round(self.deadzone(self.LeftJoystickY, 0.15), 2),
            "r_x": round(self.deadzone(self.RightJoystickX, 0.15), 2),
            "r_y": round(self.deadzone(self.RightJoystickY, 0.15), 2),
            "dpad_left": self.LeftDPad,
            "dpad_right": self.RightDPad,
            "dpad_up": self.UpDPad,
            "dpad_down": self.DownDPad,
            "l_trigger": round(self.LeftTrigger / 4.0, 2),
            "r_trigger": round(self.RightTrigger / 4.0, 2),
            "l_bumper": self.LeftBumper,
            "r_bumper": self.RightBumper,
            "a": self.A,
            "x": self.X,
            "y": self.Y,
            "b": self.B,
            "start": self.Start,
            "back": self.Back,
        }

        return ret

    def _monitor_controller(self):
        while True:
            events = get_gamepad()
            for event in events:
                if event.code == "ABS_Y":
                    self.LeftJoystickY = (
                        event.state / XboxController.MAX_JOY_VAL
                    )  # normalize between -1 and 1
                elif event.code == "ABS_X":
                    self.LeftJoystickX = (
                        event.state / XboxController.MAX_JOY_VAL
                    )  # normalize between -1 and 1
                elif event.code == "ABS_RY":
                    self.RightJoystickY = (
                        event.state / XboxController.MAX_JOY_VAL
                    )  # normalize between -1 and 1
                elif event.code == "ABS_RX":
                    self.RightJoystickX = (
                        event.state / XboxController.MAX_JOY_VAL
                    )  # normalize between -1 and 1
                elif event.code == "ABS_Z":
                    self.LeftTrigger = (
                        event.state / XboxController.MAX_TRIG_VAL
                    )  # normalize between 0 and 1
                elif event.code == "ABS_RZ":
                    self.RightTrigger = (
                        event.state / XboxController.MAX_TRIG_VAL
                    )  # normalize between 0 and 1
                elif event.code == "BTN_TL":
                    self.LeftBumper = event.state
                elif event.code == "BTN_TR":
                    self.RightBumper = event.state
                elif event.code == "BTN_SOUTH":
                    self.A = event.state
                elif event.code == "BTN_NORTH":
                    self.X = event.state  # previously switched with Y
                elif event.code == "BTN_WEST":
                    self.Y = event.state  # previously switched with X
                elif event.code == "BTN_EAST":
                    self.B = event.state
                elif event.code == "BTN_THUMBL":
                    self.LeftThumb = event.state
                elif event.code == "BTN_THUMBR":
                    self.RightThumb = event.state
                elif event.code == "BTN_SELECT":
                    self.Back = event.state
                elif event.code == "BTN_START":
                    self.Start = event.state
                elif event.code == "BTN_TRIGGER_HAPPY1":
                    self.LeftDPad = event.state
                elif event.code == "BTN_TRIGGER_HAPPY2":
                    self.RightDPad = event.state
                elif event.code == "BTN_TRIGGER_HAPPY3":
                    self.UpDPad = event.state
                elif event.code == "BTN_TRIGGER_HAPPY4":
                    self.DownDPad = event.state


if __name__ == "__main__":
    joy = XboxController()
    while True:
        print(joy.read())

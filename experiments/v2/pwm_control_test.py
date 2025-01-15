from pypot.feetech import FeetechSTS3215IO
import time
import numpy as np
from threading import Thread
import pickle


io = FeetechSTS3215IO(
    "/dev/ttyACM0",
    baudrate=1000000,
    use_sync_read=True,
)
# io.disable_torque([1])
# input()
# io.enable_torque([1])
# input()
# io.disable_torque([1])
# exit()

class FeetechPWMControl:
    def __init__(self):
        self.io = FeetechSTS3215IO(
            "/dev/ttyACM0",
            baudrate=1000000,
            use_sync_read=True,
        )
        self.id = 1

        # TODO zero first
        self.io.enable_torque([self.id])
        self.io.set_mode({self.id: 0})
        self.io.set_goal_position({self.id: 0})
        time.sleep(1)
        # exit()

        self.io.set_mode({self.id: 2})
        self.kp = self.io.get_P_coefficient([self.id])[0]

        self.control_freq = 100  # Hz
        self.goal_position = 0
        self.present_position = 0
        Thread(target=self.update, daemon=True).start()

    def disable_torque(self):
        self.io.set_mode({self.id: 0})
        self.io.disable_torque([self.id])

    def enable_torque(self):
        self.io.enable_torque([self.id])
        self.io.set_mode({self.id: 2})


    def update(self):
        while True:
            self.present_position = self.io.get_present_position([self.id])[0]
            error = self.goal_position - self.present_position

            pwm = self.kp * error
            # pwm *= 10
            pwm = np.int16(pwm)

            pwm_magnitude = abs(pwm)
            if pwm_magnitude >= 2**10:
                pwm_magnitude = (2**10) - 1

            direction_bit = 1 if pwm >= 0 else 0

            goal_time = (direction_bit << 10) | pwm_magnitude

            self.io.set_goal_time({self.id: goal_time})

            time.sleep(1 / self.control_freq)


motor = FeetechPWMControl()

s = time.time()
while True:

    target = 20 * np.sin(2 * np.pi * 0.5 * time.time())
    motor.goal_position = target

    if time.time() - s > 3 and time.time() - s < 6:
        motor.disable_torque()
        print("disbale")

    if time.time() - s > 6:
        motor.enable_torque()
        print("enable")
    time.sleep(1 / 60)


present_positions = []
goal_positions = []
present_loads = []
present_currents = []
present_speeds = []
times = []

motor.goal_position = 90

log_start = time.time()
s = time.time()
while time.time() - s < 2:
    present_positions.append(motor.io.get_present_position([motor.id])[0])
    goal_positions.append(motor.goal_position)
    present_loads.append(motor.io.get_present_load([motor.id])[0])
    present_currents.append(motor.io.get_present_current([motor.id])[0])
    present_speeds.append(motor.io.get_present_speed([motor.id])[0])
    times.append(time.time() - log_start)

motor.goal_position = -90
s = time.time()
while time.time() - s < 2:
    present_positions.append(motor.io.get_present_position([motor.id])[0])
    goal_positions.append(motor.goal_position)
    present_loads.append(motor.io.get_present_load([motor.id])[0])
    present_currents.append(motor.io.get_present_current([motor.id])[0])
    present_speeds.append(motor.io.get_present_speed([motor.id])[0])
    times.append(time.time() - log_start)
motor.goal_position = 90

s = time.time()
while time.time() - s < 2:
    present_positions.append(motor.io.get_present_position([motor.id])[0])
    goal_positions.append(motor.goal_position)
    present_loads.append(motor.io.get_present_load([motor.id])[0])
    present_currents.append(motor.io.get_present_current([motor.id])[0])
    present_speeds.append(motor.io.get_present_speed([motor.id])[0])
    times.append(time.time() - log_start)


data = {
    "present_positions": present_positions,
    "goal_positions": goal_positions,
    "present_loads": present_loads,
    "present_currents": present_currents,
    "present_speeds": present_speeds,
    "times": times,
}

pickle.dump(data, open("data_pwm_control.pkl", "wb"))

from pypot.feetech import FeetechSTS3215IO
import time
import pickle

io = FeetechSTS3215IO(
    "/dev/ttyACM0",
    baudrate=1000000,
    use_sync_read=True,
)

id = 24
max_acceleration = 254
io.set_D_coefficient({id: 0})
io.set_acceleration({id: max_acceleration})

io.set_goal_position({id: 0})

# present position
# goal position
# present load
# present current
# present speed
# 0 deg pendant 2s, 90° pendant 2s etc


present_positions = []
goal_positions = []
present_loads = []
present_currents = []
present_speeds = []
times = []

input("press enter to start")

io.set_goal_position({id: 90})
s = time.time()
while time.time() - s < 2:
    present_positions.append(io.get_present_position([id])[0])
    goal_positions.append(io.get_goal_position([id])[0])
    present_loads.append(io.get_present_load([id])[0])
    present_currents.append(io.get_present_current([id])[0])
    present_speeds.append(io.get_present_speed([id])[0])
    times.append(time.time())


io.set_goal_position({id: 0})
s = time.time()
while time.time() - s < 2:
    present_positions.append(io.get_present_position([id])[0])
    goal_positions.append(io.get_goal_position([id])[0])
    present_loads.append(io.get_present_load([id])[0])
    present_currents.append(io.get_present_current([id])[0])
    present_speeds.append(io.get_present_speed([id])[0])
    times.append(time.time())

io.set_goal_position({id: 90})
s = time.time()
while time.time() - s < 2:
    present_positions.append(io.get_present_position([id])[0])
    goal_positions.append(io.get_goal_position([id])[0])
    present_loads.append(io.get_present_load([id])[0])
    present_currents.append(io.get_present_current([id])[0])
    present_speeds.append(io.get_present_speed([id])[0])
    times.append(time.time())


data = {
    "present_positions": present_positions,
    "goal_positions": goal_positions,
    "present_loads": present_loads,
    "present_currents": present_currents,
    "present_speeds": present_speeds,
    "times": times,
}

pickle.dump(data, open(f"data_max_acc_{max_acceleration}.pkl", "wb"))





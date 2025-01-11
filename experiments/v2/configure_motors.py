from pypot.feetech import FeetechSTS3215IO
import argparse
import time

parser = argparse.ArgumentParser()
parser.add_argument("--usb_port", type=str, default="/dev/ttyACM0")
args = parser.parse_args()


joints = {
    "right_hip_yaw": 10,
    "right_hip_roll": 11,
    "right_hip_pitch": 12,
    "right_knee": 13,
    "right_ankle": 14,
    "left_hip_yaw": 20,
    "left_hip_roll": 21,
    "left_hip_pitch": 22,
    "left_knee": 23,
    "left_ankle": 24,
    "neck_pitch": 30,
    "head_pitch": 31,
    "head_yaw": 32,
    "head_roll": 33,
}

# TODO scan baudrates
io = FeetechSTS3215IO(
    args.usb_port,
    baudrate=1000000,
    use_sync_read=True,
)


id = 200
SKIP_SCAN = False
if not SKIP_SCAN:
    try:
        io.get_present_position([id])
    except Exception:
        print(
            "Didn't find motor with id 1, motor has probably been configured before. scanning for other motors"
        )
        print("Scanning... ")
        found_ids = io.scan()
        print("Found ids: ", found_ids)
        if len(found_ids) > 1:
            print("More than one motor found, please connect only one motor")
            exit()
        elif len(found_ids) == 0:
            print("No motor found")
            exit()

        id = found_ids[0]

exit()
print("Select the dof you want to configure : ")
for i, key in enumerate(joints.keys()):
    print(f"{i}: {key}")

dof_index = int(input("> "))
if dof_index not in range(len(joints)):
    print("Invalid choice")
    exit()

dof_name = list(joints.keys())[dof_index]
dof_id = joints[dof_name]

print("")
print("===")
print("Configuring motor ", dof_name, " with id ", dof_id)
print("===")

io.set_lock({id: 1})

io.set_offset({id: 0})
print("- setting new id ")
io.change_id({id: dof_id})
id = dof_id
print("- setting new baudrate")
io.change_baudrate({id: 0})  # 1 000 000

exit()
# WARNING offset management is not understood yet.

print("")
print("The motor will now move to the zero position.")
print("Press enter to continue")
input()

io.enable_torque([id])
io.set_goal_position({id: 0})
time.sleep(1)
zero_pos = io.get_present_position([id])[0]

print("")
print(
    "Now, place the contraption (???) on the motor, and adjust the position to fit (???) (TODO: clarify)"
)
print("Press enter to continue")

io.disable_torque([id])
input()
new_pos = io.get_present_position([id])[0]
offset = zero_pos - new_pos
print("Offset: ", offset)
io.set_offset({id: offset})
time.sleep(1)


io.set_lock({id: 0})

print("")
print(
    "To confirm the offset, please move the motor to a random position, then press enter to go back to zero."
)
input()
io.enable_torque([id])
io.set_goal_position({id: 0})
time.sleep(1)
print("position: ", io.get_present_position([id])[0])

io.disable_torque([id])

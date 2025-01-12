from mini_bdx_runtime.hwi_feetech_pypot import HWI
from mini_bdx.placo_walk_engine.placo_walk_engine import PlacoWalkEngine
from placo_utils.visualization import robot_viz
import json
import time
import argparse
from queue import Queue

parser = argparse.ArgumentParser()
parser.add_argument("--xbox", action="store_true")
parser.add_argument("--viz", action="store_true")
args = parser.parse_args()


MAX_X = 0.02
MAX_Y = 0.02
MAX_THETA = 0.1


if args.xbox:
    from threading import Thread
    import pygame

    pygame.init()
    _p1 = pygame.joystick.Joystick(0)
    _p1.init()
    print(f"Loaded joystick with {_p1.get_numaxes()} axes.")

    cmd_queue = Queue(maxsize=1)

    def commands_worker():
        global cmd_queue

        while True:
            for event in pygame.event.get():
                l_x = round(_p1.get_axis(0), 3)
                l_y = round(_p1.get_axis(1), 3)
                r_x = round(_p1.get_axis(3), 3)
                r_y = round(_p1.get_axis(4), 3)
                l_x = 0.0 if abs(l_x) < 0.1 else l_x
                l_y = 0.0 if abs(l_y) < 0.1 else l_y
                r_x = 0.0 if abs(r_x) < 0.1 else r_x
                r_y = 0.0 if abs(r_y) < 0.1 else r_y

            pygame.event.pump()  # process event queue
            cmd = {
                "l_x": l_x,
                "l_y": l_y,
                "r_x": r_x,
                "r_y": r_y,
            }

            cmd_queue.put(cmd)
            time.sleep(1 / 30)

    Thread(target=commands_worker, daemon=True).start()
    last_commands = {
        "l_x": 0.0,
        "l_y": 0.0,
        "r_x": 0.0,
        "r_y": 0.0,
    }

    def get_last_command():
        global cmd_queue, last_commands
        try:
            last_commands = cmd_queue.get(False)  # non blocking
        except Exception:
            pass

        return last_commands


if not args.viz:
    hwi = HWI()
    hwi.turn_on()


DT = 0.01
pwe = PlacoWalkEngine(
    "/home/antoine/MISC/mini_BDX/mini_bdx/robots/open_duck_mini_v2",
    model_filename="robot.urdf",
    init_params=json.load(open("placo_defaults.json")),
    ignore_feet_contact=True,
)
if args.viz:
    viz = robot_viz(pwe.robot)

pwe.set_traj(0.0, 0, 0.0)
while True:
    pwe.tick(DT)

    if args.xbox:
        commands = get_last_command()
        placo_commands = [
            -commands["l_y"] * MAX_X,
            -commands["l_x"] * MAX_Y,
            -commands["r_x"] * MAX_THETA,
        ]
        print(placo_commands)
        print("====")
        if commands is not None:
            pwe.set_traj(*placo_commands)

    if not args.viz:
        all_angles = list(pwe.get_angles().values())
        angles = {}
        for i, motor_name in enumerate(hwi.joints.keys()):
            angles[motor_name] = all_angles[i]
        hwi.set_position_all(angles)
    else:
        viz.display(pwe.robot.state.q)

    time.sleep(DT)

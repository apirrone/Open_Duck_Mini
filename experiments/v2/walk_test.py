from mini_bdx.placo_walk_engine.placo_walk_engine import PlacoWalkEngine
import time
from placo_utils.visualization import footsteps_viz, robot_frame_viz, robot_viz
import json



DT = 0.01
pwe = PlacoWalkEngine("/home/antoine/MISC/mini_BDX/mini_bdx/robots/open_duck_mini_v2", model_filename="robot.urdf", init_params=json.load(open("placo_defaults.json")))
viz = robot_viz(pwe.robot)

pwe.set_traj(0.1, 0, 0.1)
while True:
    pwe.tick(DT)
    viz.display(pwe.robot.state.q)
    time.sleep(DT)

import os
import argparse
from placo_record_amp import record
from mini_bdx.placo_walk_engine import PlacoWalkEngine
import numpy as np

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("-o", "--output_dir", type=str, default="recordings")
parser.add_argument(
    "-n", "--num_samples", type=int, default=10, help="Number of samples"
)
args = parser.parse_args()

pwe = PlacoWalkEngine("../../mini_bdx/robots/bdx/robot.urdf", ignore_feet_contact=True)

dx_range = [-0.04, 0.04]
dy_range = [-0.05, 0.05]
dtheta_range = [-0.15, 0.15]
length = 8
num_samples = args.num_samples

args_dict = {}
args_dict["name"] = "test"
args_dict["dx"] = 0
args_dict["dy"] = 0
args_dict["dtheta"] = 0
args_dict["length"] = length
args_dict["meshcat_viz"] = False
args_dict["skip_warmup"] = False
args_dict["stand"] = False
args_dict["hardware"] = True
args_dict["output_dir"] = args.output_dir

for i in range(num_samples):
    args_dict["dx"] = round(np.random.uniform(dx_range[0], dx_range[1]), 2)
    args_dict["dy"] = round(np.random.uniform(dy_range[0], dy_range[1]), 2)
    args_dict["dtheta"] = round(np.random.uniform(dtheta_range[0], dtheta_range[1]), 2)
    args_dict["name"] = str(i)
    print("RECORDING ", args_dict["name"])
    print("dx", args_dict["dx"], "dy", args_dict["dy"], "dtheta", args_dict["dtheta"])
    print("==")
    pwe.reset()
    record(pwe, args_dict)

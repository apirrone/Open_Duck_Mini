import pickle
import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--latent", type=str, required=False, default="mujoco_saved_latent.pkl"
)
args = parser.parse_args()

latent = pickle.load(open(args.latent, "rb"))

import matplotlib.pyplot as plt

plt.plot(latent)
plt.show()

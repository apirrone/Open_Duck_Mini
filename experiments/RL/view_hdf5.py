import h5py

data = h5py.File(
    "/home/antoine/MISC/mini_BDX/experiments/RL/data/test_raw/episode_0.hdf5", "r"
)
print(len(data["/action"]))
exit()
for i in range(len(data["/action"])):
    print(data["/action"][i])
    print(data["/observations/qpos"][i])
    print(data["/observations/qvel"][i])

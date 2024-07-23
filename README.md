# Mini BDX Droid

![Capture d’écran du 2024-07-22 18-15-29](https://github.com/user-attachments/assets/41876cb7-b4f2-4c68-8ef9-f92f2eb45044)

I'm making a miniature version of the BDX Droid by Disney. It will be about 35 centimeters tall with its legs extended.

https://github.com/apirrone/mini_BDX/assets/6552564/56cb1919-2f20-4a5c-a4fe-0855db1080ac

This is with a custom walk engine based an [old Rhoban walk](https://github.com/Rhoban/walk_engine/tree/master) and the IK is based on [placo](https://github.com/Rhoban/placo)

If you have a xbox controller, you can control the robot with it in MuJoCo (after installing the package, see Install section):

```bash
$ cd experiments/mujoco
$ python3 mujoco_walk_engine.py -x
```

Start button to start the walk, left joystick to translate in x/y, right joystick to rotate. By pressing the left trigger, you control the head pitch and yaw with the right joystick, and the head height with the right trigger.

Right now I use the keyboard inputs to tune the walk engine's hyperparameters.

## CAD

https://cad.onshape.com/documents/a18ff8cc622a533762a3a6f5/w/27ef6089ad5fe9ba396b6036/e/9ce9b71d7a21eb04415b067f

See [this document](docs/prepare_robot.md) for getting from a onshape design to a simulated robot in MuJoCo

## Install

### Install everything (simulation, rl etc)
```bash
$ pip install -e .[all]
```

## RL stuff

I switched to Isaac Gym for reinforcement learning. My fork of IsaacGymEnvs is here https://github.com/apirrone/IsaacGymEnvs

This is the best walk I got so far using IsaacGymEnv's implementation of AMP (https://xbpeng.github.io/projects/AMP/index.html)

https://github.com/user-attachments/assets/833c86b8-c889-4985-8325-34d4058953bc


# BOM

https://docs.google.com/spreadsheets/d/18hrYgjaE9uL2pnrnq5pNUzFLZcI4Rg0AvSc9sqwE680/edit?usp=sharing

# Assembly Guide 

TODO

The current version (alpha) is not very easy to build, has some mechanical problems (too much play at some joints). After everything works on this version of the robot, I will redesign it from the ground up, trying to make it more accessible and better mechanically designed !

# Interesting papers and resources
- https://www.nature.com/articles/s41598-023-38259-7.pdf
- https://arxiv.org/pdf/2304.13653
- https://arxiv.org/pdf/2401.16889
- https://arxiv.org/pdf/1801.01290
- First comment of this video explains the reward https://www.youtube.com/watch?v=L_4BPjLBF4E

# Directly BDX related (from disney)
- https://www.nvidia.com/en-us/on-demand/session/gtc24-s63374/
- https://la.disneyresearch.com/publication/design-and-control-of-a-bipedal-robotic-character/


## Bootstraping by behavior cloning
- https://stable-baselines.readthedocs.io/en/master/guide/pretrain.html 

# References
- https://cults3d.com/en/3d-model/gadget/robot-disney-star-wars
- https://www.disneytouristblog.com/wp-content/uploads/2023/10/bd1-droid-testing-disney-imagineering-2.jpg
- nice close up video https://www.youtube.com/watch?v=QuWaaNN-1hs

## Imitation learning
- https://github.com/rgalljamov/DRLoco
- https://xbpeng.github.io/projects/DeepMimic/index.html
-
## MPC
- https://github.com/google-deepmind/mujoco_mpc

# Mini BDX Droid

This project is still a work in progress !

![Capture d’écran du 2024-07-22 18-15-29](https://github.com/user-attachments/assets/41876cb7-b4f2-4c68-8ef9-f92f2eb45044)

I'm making a miniature version of the BDX Droid by Disney. It will be about 35 centimeters tall with its legs extended.

https://github.com/user-attachments/assets/f2b3efac-9300-4959-95ae-06420a2d1517

This is a policy trained in Isaac Gym with [Amp_for_hardware](https://github.com/apirrone/AMP_for_hardware/tree/bdx) running in Mujoco. 

https://github.com/user-attachments/assets/6a742583-223a-43f7-8489-4a4fffeaaef6

This is an attempt at running the policy on the real robot. Still some work to do :)


# This repo

This is kind of a hub where I centralize all resources related to this project. This is a working repo, so there are a lot of undocumented scripts :) I'll try to clean things up at some point.

# CAD

https://cad.onshape.com/documents/a18ff8cc622a533762a3a6f5/w/27ef6089ad5fe9ba396b6036/e/9ce9b71d7a21eb04415b067f

See [this document](docs/prepare_robot.md) for getting from a onshape design to a simulated robot in MuJoCo

# RL stuff

The RL method used to learn to walk is [Adversarial Motion Priors Make Good Substitutes for Complex Reward Functions](https://sites.google.com/berkeley.edu/amp-in-real/home). 

My fork of the original repo is [there](https://github.com/apirrone/AMP_for_hardware/tree/bdx)

This is the current policy running in Isaac Gym. It can be controlled to walk forward and to turn left or right.

https://github.com/user-attachments/assets/c37d9bfc-67e1-4b21-a9dc-a1310bdf3016

# BOM

Note : I switched to using `xc330-M288-T` servomotors instead of `xl330-M288-T` for the legs. They are more expensive, but way more powerful. Maybe we'll try to go back to xl330 servos once everything is working properly with the xc330.

https://docs.google.com/spreadsheets/d/18hrYgjaE9uL2pnrnq5pNUzFLZcI4Rg0AvSc9sqwE680/edit?usp=sharing

# Assembly Guide 

TODO

The current version (alpha) is not very easy to build, has some mechanical problems (too much play at some joints). After everything works on this version of the robot, I will redesign it from the ground up, trying to make it more accessible and better mechanically designed !


# Resources

## Interesting papers and resources
- https://www.nature.com/articles/s41598-023-38259-7.pdf
- https://arxiv.org/pdf/2304.13653
- https://arxiv.org/pdf/2401.16889
- https://arxiv.org/pdf/1801.01290
- First comment of this video explains the reward https://www.youtube.com/watch?v=L_4BPjLBF4E

## Directly BDX related (from disney)
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

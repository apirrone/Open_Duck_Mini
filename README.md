# Open Duck Mini

This project is still a work in progress !

![Capture d’écran du 2024-07-22 18-15-29](https://github.com/user-attachments/assets/41876cb7-b4f2-4c68-8ef9-f92f2eb45044)

I'm making a miniature version of the BDX Droid by Disney. It will be about 35 centimeters tall with its legs extended.

https://github.com/user-attachments/assets/9072d20d-6a6c-49be-8bfc-e947a4c3eb76

This is a policy trained in Isaac Gym with [Amp_for_hardware](https://github.com/apirrone/AMP_for_hardware/tree/bdx).

https://github.com/user-attachments/assets/50c76252-7b52-4111-ba29-e0c4bffc3f62

This is the same policy, running in Mujoco

https://github.com/user-attachments/assets/4e03fe2b-371c-4bd5-a462-8fa0ee84efa2

This is a policy trained in Isaac Gym for standing up while being robust to perturbations.

https://github.com/user-attachments/assets/6a742583-223a-43f7-8489-4a4fffeaaef6

This is an attempt at running the walking policy on the real robot. Still some work to do :)


# This repo

This is kind of a hub where I centralize all resources related to this project. This is a working repo, so there are a lot of undocumented scripts :) I'll try to clean things up at some point.

# CAD

https://cad.onshape.com/documents/a18ff8cc622a533762a3a6f5/w/27ef6089ad5fe9ba396b6036/e/9ce9b71d7a21eb04415b067f

See [this document](docs/prepare_robot.md) for getting from a onshape design to a simulated robot in MuJoCo

Idler cap https://cad.onshape.com/documents/be101dee8170780f05a04bfa/w/c49eb6dbb859e081d7ac252c/e/935d77385a5196902c262118

# RL stuff

We now use [AWD](https://github.com/rimim/AWD)

# BOM

Note : I switched to using `xc330-M288-T` servomotors instead of `xl330-M288-T` for the legs. They are more expensive, but way more powerful. Maybe we'll try to go back to xl330 servos once everything is working properly with the xc330.

https://docs.google.com/spreadsheets/d/18hrYgjaE9uL2pnrnq5pNUzFLZcI4Rg0AvSc9sqwE680/edit?usp=sharing

# Assembly Guide

TODO

The current version (alpha) is not very easy to build, has some mechanical problems (too much play at some joints). After everything works on this version of the robot, I will redesign it from the ground up, trying to make it more accessible and better mechanically designed !


# Embedded runtime

This repo contains the code to run the policies on the onboard computer (Raspberry pi zero 2w) https://github.com/apirrone/Open_Duck_Mini_Runtime

# Resources

## Interesting papers and resources
- https://www.nature.com/articles/s41598-023-38259-7.pdf
- https://arxiv.org/pdf/2304.13653
- https://arxiv.org/pdf/2401.16889
- https://arxiv.org/pdf/1801.01290
- First comment of this video explains the reward https://www.youtube.com/watch?v=L_4BPjLBF4E
- https://www.haonanyu.blog/post/sim2real/

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


Thanks a lot to HuggingFace for sponsoring this project !

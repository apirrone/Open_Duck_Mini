# Open Duck Mini v2

<table>
  <tr>
    <td> <img src="https://github.com/user-attachments/assets/2a407765-70ad-48dd-8a5d-488f82503716" alt="1" width="300px" ></td>
    <td> <img src="https://github.com/user-attachments/assets/3b8fe350-73a9-4c9f-ad29-efc781be7aee" alt="2" width="300px" ></td>
    <td> <img src="https://github.com/user-attachments/assets/fd7e5949-1492-4d31-851f-feaa9b695557" alt="3" width="300px" ></td>
   </tr> 
</table>

We are making a miniature version of the BDX Droid by Disney. It is about 42 centimeters tall with its legs extended.
The full BOM cost should be under $400 !

This repo is kind of a hub where we centralize all resources related to this project. This is a working repo, so there are a lot of undocumented scripts :) We'll try to clean things up at some point.


# State of sim2real

https://github.com/user-attachments/assets/58721d0f-2f95-4088-8900-a5d02f41bba7

https://github.com/user-attachments/assets/4129974a-9d97-4651-9474-c078043bb182

https://github.com/user-attachments/assets/a0afcd38-15d8-40c6-8171-a619107406b8


# Updates

> Update 02/04/2024: You can try two policies we trained : [this one](BEST_WALK_ONNX.onnx) and [this one](BEST_WALK_ONNX_2.onnx)
> Run with the following arguments :
> python v2_rl_walk_mujoco.py --onnx_model_path ~/BEST_WALK_ONNX_2.onnx

> Update 15/03/2025: join our discord server to get help or show us your duck :) https://discord.gg/UtJZsgfQGe

> Update 07/02/2025: Big progress on sim2real, see videos above :)

> Update 24/02/2025: Working hard on sim2real ! 

> Update 07/02/2025 : We are writing documentation on the go, but the design and BOM should not change drastically. Still missing the "expression" features, but they can be added after building the robot!

> Update 22/01/2025 : The mechanical design is pretty much finalized (fixing some mistakes here and there). The current version does not include all the "expression" features we want to include in the final robot (LEDs for the eyes, a camera, a speaker and a microphone). We are now working on making it walk with reinforcement learning !

# Community 

![duck_collage](https://github.com/user-attachments/assets/e240c06e-769f-4c87-b65f-189a442cf1e9)

Join our discord community ! https://discord.gg/UtJZsgfQGe

# CAD

https://cad.onshape.com/documents/64074dfcfa379b37d8a47762/w/3650ab4221e215a4f65eb7fe/e/0505c262d882183a25049d05

See [this document](docs/prepare_robot.md) for getting from a onshape design to a simulated robot in MuJoCo (Warning, outdated. Has not been updated in a while)

# RL stuff

We are switching to Mujoco Playground, see this [repo](https://github.com/apirrone/Open_Duck_Playground)

https://github.com/user-attachments/assets/037a1790-7ac1-4140-b154-2c901d20d5f5


## Reference motion generation for imitation learning 

https://github.com/user-attachments/assets/4cb52e17-99a5-47a8-b841-4141596b7afb

See [this repo](https://github.com/apirrone/Open_Duck_reference_motion_generator)

## Actuator identification 

We used Rhoban's [BAM](https://github.com/Rhoban/bam)

# BOM

https://docs.google.com/spreadsheets/d/1gq4iWWHEJVgAA_eemkTEsshXqrYlFxXAPwO515KpCJc/edit?usp=sharing

Chinese: https://zihao-ai.feishu.cn/wiki/AfAtw69vRigXaRk5UkbcrAiLnJw?from=from_copylink

# Build Guide

> New : you can now use the Tnkr guide ! https://tnkr.ai/explore/docs/open-duck-mini/open-duck-mini-v2#home

Chinese: https://zihao-ai.feishu.cn/wiki/space/7488517034406625281
Chinese: https://www.ncnynl.com/category/OpenDuckMini/  (Step-by-step guide)

## Print Guide

See [print_guide](docs/print_guide.md).

## Assembly Guide

See [assembly guide (incomplete)](docs/assembly_guide.md).
Chinese：https://www.ncnynl.com/archives/202506/6739.html

# Embedded runtime

This repo contains the code to run the policies on the onboard computer (Raspberry pi zero 2w) https://github.com/apirrone/Open_Duck_Mini_Runtime

# Training your own policies

If you want to train your own policies, and contribute to making the ducks walk nicely, see [this document](docs/sim2real.md)

> Thanks a lot to HuggingFace and Pollen Robotics for sponsoring this project !

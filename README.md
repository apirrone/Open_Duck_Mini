# Open Duck Mini v2

<table>
  <tr>
    <td> <img src="https://github.com/user-attachments/assets/1cec3e46-de46-4abb-9c9e-20f936f15121" alt="1" width="300px" ></td>
    <td> <img src="https://github.com/user-attachments/assets/d2588204-32db-47c1-9ac5-2d2f71dbb98a" alt="2" width="300px" ></td>
    <td> <img src="https://github.com/user-attachments/assets/94fbd245-655e-4465-a727-950a89ff02c2" alt="3" width="300px" ></td>
   </tr> 
</table>

We are making a miniature version of the BDX Droid by Disney. It is about 42 centimeters tall with its legs extended.
The full BOM cost should be under $400 !

This repo is kind of a hub where we centralize all resources related to this project. This is a working repo, so there are a lot of undocumented scripts :) We'll try to clean things up at some point.


# State of sim2real

The gait is getting better ! 

https://github.com/user-attachments/assets/5ea58549-78dc-4448-8f86-d3e0499df830

https://github.com/user-attachments/assets/a0afcd38-15d8-40c6-8171-a619107406b8


# Updates

> Update 15/03/2025: join our discord server to get help or show us your duck :) https://discord.gg/UtJZsgfQGe

> Update 07/02/2025: Big progress on sim2real, see videos above :)

> Update 24/02/2025: Working hard on sim2real ! 

> Update 07/02/2025 : We are writing documentation on the go, but the design and BOM should not change drastically. Still missing the "expression" features, but they can be added after building the robot!

> Update 22/01/2025 : The mechanical design is pretty much finalized (fixing some mistakes here and there). The current version does not include all the "expression" features we want to include in the final robot (LEDs for the eyes, a camera, a speaker and a microphone). We are now working on making it walk with reinforcement learning !

# CAD

https://cad.onshape.com/documents/64074dfcfa379b37d8a47762/w/3650ab4221e215a4f65eb7fe/e/0505c262d882183a25049d05

See [this document](docs/prepare_robot.md) for getting from a onshape design to a simulated robot in MuJoCo

# RL stuff

We are switching to Mujoco Playground, see this [repo](https://github.com/apirrone/Open_Duck_Playground)

https://github.com/user-attachments/assets/293ed5f2-bd24-472e-b44d-c2e4153b8cba

We are still working with [AWD](https://github.com/rimim/AWD) too

https://github.com/user-attachments/assets/07059dbe-2ebb-4718-afe7-16ce80691073

## Reference motion generation for imitation learning 

https://github.com/user-attachments/assets/4cb52e17-99a5-47a8-b841-4141596b7afb

See [this repo](https://github.com/apirrone/Open_Duck_reference_motion_generator)

## Actuator identification 

We used Rhoban's [BAM](https://github.com/Rhoban/bam)

# BOM

https://docs.google.com/spreadsheets/d/1gq4iWWHEJVgAA_eemkTEsshXqrYlFxXAPwO515KpCJc/edit?usp=sharing

## BOM for Chinese Taobao

https://docs.google.com/spreadsheets/d/13O1BSdV3yDxIHN2xSvWhlGm0d4rU0Y4x1Pkh8teHhc0

# Build Guide

## Print Guide

See [print_guide](docs/print_guide.md).

## Assembly Guide

See [assembly guide (incomplete)](docs/assembly_guide.md).

# Embedded runtime

This repo contains the code to run the policies on the onboard computer (Raspberry pi zero 2w) https://github.com/apirrone/Open_Duck_Mini_Runtime


> Thanks a lot to HuggingFace and Pollen Robotics for sponsoring this project !


# Global project TODO

- [ ] Centralize the URDF/MJCF. Separate repo ? Menagerie ? 

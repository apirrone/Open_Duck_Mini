# Open Duck Mini v2

![Capture d’écran du 2025-01-14 15-44-05](https://github.com/user-attachments/assets/d8e6c804-8b2a-4a32-8d05-eab38d4ba475)

I'm making a miniature version of the BDX Droid by Disney. It is about 42 centimeters tall with its legs extended.
The full BOM cost should be under $400 !

# This repo

This is kind of a hub where I centralize all resources related to this project. This is a working repo, so there are a lot of undocumented scripts :) I'll try to clean things up at some point.

# CAD

https://cad.onshape.com/documents/64074dfcfa379b37d8a47762/w/3650ab4221e215a4f65eb7fe/e/0505c262d882183a25049d05

See [this document](docs/prepare_robot.md) for getting from a onshape design to a simulated robot in MuJoCo

# RL stuff

We now use [AWD](https://github.com/rimim/AWD)

## Actuator identification 

We used Rhoban's [BAM](https://github.com/Rhoban/bam)

# BOM

> The BOM is not yet fully finalized, wait a bit before ordering stuff

https://docs.google.com/spreadsheets/d/1gq4iWWHEJVgAA_eemkTEsshXqrYlFxXAPwO515KpCJc/edit?usp=sharing

# Build Guide

## Print Guide

TODO

## Assembly Guide
TODO

TODO

# Embedded runtime

This repo contains the code to run the policies on the onboard computer (Raspberry pi zero 2w) https://github.com/apirrone/Open_Duck_Mini_Runtime


> Thanks a lot to HuggingFace and Pollen Robotics for sponsoring this project !

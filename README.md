# sEMG-Controlled Robotic Arm

*Individual graduation project: a 6-DOF arm driven by surface-EMG hand-gesture recognition.*

## Overview

A 3D-printed (ABS) 6-DOF robotic arm (~640 mm reach) with a rack-and-pinion parallel gripper, controlled from 2-channel surface-EMG (sEMG) signals. EMG is acquired, conditioned and classified into 10 hand/wrist gestures, which drive the arm through a ROS 2 + MoveIt motion-planning stack. The control chain runs ESP32-S3 -> PCA9685 PWM driver -> MG995 servos over a micro-ROS link, with a Raspberry Pi running inference.

**Tech & tools:** ROS 2 (Humble), MoveIt, Gazebo, micro-ROS, ESP32-S3, PCA9685, Raspberry Pi, Python, C/C++, scikit-learn/PyTorch, SolidWorks (CAD).

## What's inside

- `emg_ws/src/` - ROS 2 packages: `emg_control`, `rotatedauto` (robot description/bringup), `rotatedauto_moveit_config` (MoveIt config)
- `emg_ws/raspberry_pi/` - on-Pi `arm_control` and `inference` code
- `emg_ws/emg_ml/` - EMG machine-learning: `reference/` training & preprocessing scripts, `reference/firmware/` ESP32 collection sketches, `wifi/` TCP streaming, `docs/`
- `docs/Research_Project_Report.pdf` - full research report
- `docs/experimental_results.md` - logged experiments

## How to use

1. Install ROS 2 Humble and `colcon`; place `emg_ws/src` in a workspace and run `colcon build`.
2. Source the workspace and launch the description/MoveIt config (see package `launch/` files).
3. Flash the EMG collection/streaming sketch from `emg_ml/reference/firmware` to the ESP32.
4. Train or load a model with the scripts in `emg_ml/reference/`, then run inference on the Pi (`raspberry_pi/inference`).
5. Run the arm controller node to map predicted gestures to joint/gripper commands.

## Status & notes

Trained model binaries (`.joblib`/`.pt`) and large EMG datasets are **not** included to keep the repo lightweight (they were excluded on purpose) - retrain using the scripts, or I'll add them via Git LFS / a release later. Hardware was hand-built, so pin numbers and gains in firmware may need tuning for your setup.

---

*By Salma Farid — Mechatronics & Robotics Engineering, BUE / LSBU. This is coursework/graduation work shared as a portfolio; I'll keep refining it.*

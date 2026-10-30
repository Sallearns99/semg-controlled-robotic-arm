# sEMG-Controlled Robotic Arm

*Individual graduation project: a 6-DOF arm driven by surface-EMG hand-gesture recognition.*

## Overview

A 3D-printed (ABS) 6-DOF robotic arm with a two-finger gripper, controlled from 2-channel surface-EMG (sEMG) signals. EMG is acquired, conditioned and classified into hand/wrist gestures, which drive the arm in Gazebo (ROS 2 + MoveIt 2) and on the real hardware.

This was my BEng graduation thesis at the British University in Egypt (BUE): *Design and Implementation of an sEMG-Based Gesture-Controlled Robotic Arm Using Deep Learning* (`docs/Research_Project_Report.pdf`).

**Tech & tools:** ROS 2 Jazzy, MoveIt 2, Gazebo Harmonic, ros2_control, ESP32-S3 (Arduino), Raspberry Pi 4, PCA9685, Python, PyTorch, scikit-learn, librosa, SolidWorks (CAD).

## System

```
 2x sEMG sensors        ESP32-S3              Raspberry Pi 4 / PC                      Arm
 (Muscle Sensor v3,  -> samples, streams  ->  50 Hz notch -> 200 ms windows ->      -> PCA9685 -> MG995 servos
  CHEEZ.semg)           USB or Wi-Fi/TCP      76-D features -> classifier ->           (real arm)
                                              5-vote + REST gate -> gesture         -> ros2_control -> Gazebo (sim)
```

- **Acquisition:** ESP32-S3 reading two envelope channels. The ESP32 only samples and streams; all inference and control run on the host.
- **Features:** each 200 ms window (100 ms hop) gives a 76-dimensional hybrid vector per window. For each of the 2 channels that is 8 MFCCs × {mean, std, max, min} plus 6 time-domain descriptors, so (8×4 + 6) × 2 = 76 (`raspberry_pi/inference/preprocess_v0_windowed.py`).
- **Classifiers:** a classical pipeline (random forest / SVM / gradient boosting on the 76-D features) and deep models: a CNN-BiLSTM on feature sequences (`emg_ml/reference/train_emg_model.py`) and a recurrent CNN on the raw window (`raspberry_pi/inference/rcnn_model.py`).
- **Gestures:** data was collected for 12 gestures (`emg_config.py`). The arm is driven by a 7-class control subset: REST, EXTENSION, FLEXION, DOWNWARD_ROT_CW/CCW and EXTEND/RETRACT_ELBOW.
- **Arm:** 6-DOF SolidWorks design (j1–j5 plus a gear-linked two-finger gripper) driven by MG995 servos on a PCA9685, with two mirrored servos on the shoulder (j2). In the thesis the base (j1) is a stepper on a DRV8825 driver. See the note under *Status* about the code.
- **Control:** gestures are debounced by a 3 s majority vote, and each decision moves one joint by a fixed step (`emg_bridge_node.py` in ROS, `emg_arm_control.py` on the Pi).

## Repository structure

```
docs/Research_Project_Report.pdf     Thesis (85 pages)
emg_ws/
├── experimental_results.md          Experiment log and results record
├── emg_ros_env.sh                   Shell setup for the ROS 2 workspace
├── src/                             ROS 2 packages (colcon workspace)
│   ├── emg_control/                 emg_inference_node, emg_bridge_node, emg_monitor + launch/config
│   ├── rotatedauto/                 Arm URDF/xacro, meshes, ros2_control, Gazebo + RViz launch
│   └── rotatedauto_moveit_config/   MoveIt 2 config (SRDF, kinematics, controllers)
├── raspberry_pi/
│   ├── inference/                   Stand-alone real-time classifier (USB or Wi-Fi)
│   ├── arm_control/                 PCA9685 servo driver, calibration, keyboard + EMG control (no ROS)
│   ├── PI_SETUP.md                  Raspberry Pi setup
│   └── CONTROL_REPORT.md            Control-system notes
└── emg_ml/
    ├── reference/                   Earlier training/inference scripts (2000 Hz, 4-ch CNN-BiLSTM, V4)
    │   └── firmware/                ESP32 sketches: streaming, collection, ADC diagnostic
    ├── wifi/                        Wi-Fi streaming sketch, TCP receiver, Pi notes
    └── docs/                        Label map, windowing config, saved results (JSON, confusion matrix)
```

## How to run

### Real-time classifier (no ROS)

```bash
cd emg_ws/raspberry_pi/inference
pip install numpy pandas joblib scipy scikit-learn librosa pyserial torch
# USB
python3 inference_rt.py --port /dev/ttyACM0 --model <model>.joblib --mains 50
# Wi-Fi (ESP32 TCP server on port 5005)
python3 inference_rt.py --host esp32.local --model <model>.joblib --mains 50
```

Pipe the result into the arm controller on the Pi. Use `--dry-run` to test without hardware:

```bash
python3 inference_rt.py --host esp32.local --model <model>.joblib \
  | python3 ../arm_control/emg_arm_control.py --stdin --vote-window 3.0
python3 ../arm_control/emg_arm_control.py --demo --dry-run   # scripted gestures, mock servos
```

`raspberry_pi/arm_control/README.md` covers wiring, `servo_calibrate.py` and keyboard teleop.

### Preprocess a recording

```bash
python3 preprocess_v0_windowed.py --input data/raw --output data/processed_v0_windowed \
  --mains-filter 50 --mains-causal
```

The input is the 11-column, 500 Hz collection CSV defined in `emg_config.py`.

### Simulation (ROS 2 Jazzy)

```bash
mkdir -p ~/emg_ws && cp -r emg_ws/src ~/emg_ws/
cd ~/emg_ws && rosdep install --from-paths src --ignore-src -r -y && colcon build
source install/setup.bash
ros2 launch rotatedauto gazebo.launch.py                 # arm in Gazebo with controllers
ros2 launch rotatedauto_moveit_config moveit.launch.py   # MoveIt 2 + RViz
ros2 launch emg_control emg.launch.py port:=/dev/ttyACM0 retrained_model_path:=<model>.joblib
```

The model paths in `emg_params.yaml` and `emg.launch.py` default to `~/emg_ws/emg_ml/models/`.

## Results

These results come from the thesis and `emg_ws/experimental_results.md`. They are not re-run here.

- **Deployed system (single participant, 7 classes, cross-cycle test, chance 14.3%):** the random forest with the causal 50 Hz notch reached **38.4%** test accuracy (macro-F1 0.29), up from 25.0% without the notch. That is about 2.7× chance and is bounded by cross-session drift and one faulty sensor channel.
- **Offline, laboratory dataset (thesis ch. 3):** the classical and deep models reached roughly 69–75% accuracy (random forest 74.3%, macro-F1 0.744).
- Saved artefacts in `emg_ml/docs/results/` record earlier 10-class runs (`results.json`, `v4_results.json`, confusion matrix), which were near chance.

## Status & notes

- **Not included:** the trained models (`.joblib`/`.pt`) and the EMG datasets, which are large. The 500 Hz firmware (`semg_stream_live*.ino`, `semg_data_collection.ino`) and the 500 Hz training scripts (`train_classical.py`, `train_rcnn.py`, `make_subset.py`) are also missing. The sketches in `emg_ml/reference/firmware/` and `emg_ml/wifi/` are the earlier **2000 Hz** versions and do not match the 500 Hz pipeline in `raspberry_pi/inference/`.
- **Base stepper:** the code still drives j1 as a PCA9685 servo (channel 0). DRV8825 step/dir control for the base is not implemented in this repo yet.
- Hardware was hand-built, so pin numbers, servo centres and gains need calibrating for any other build.

---

*By Salma Farid — Mechatronics & Robotics Engineering, BUE / LSBU. This is coursework/graduation work shared as a portfolio; I'll keep refining it.*

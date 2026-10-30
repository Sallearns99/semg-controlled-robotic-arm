# Changelog

## v1.1.0 — 2026-10-30

### Fixed
- `emg_ml/reference/train_emg_v4.py`: `--help` crashed with `TypeError` because the `--swa_start` help text contained an unescaped `%` ("25%"). It is now escaped as `25%%`.
- `src/emg_control/config/emg_params.yaml`, `src/emg_control/launch/emg.launch.py`: the default model paths were absolute (`/home/salma/...`), so they only worked on the original machine. They now use `~/emg_ws/...`, which `emg_inference_node.py` already expands with `os.path.expanduser`.
- `README.md`, which had several statements that don't match the code and thesis:
  - It said 10 gestures; the deployed control set has 7, out of 12 collected.
  - It said ROS 2 Humble; the code targets Jazzy.
  - It said the ESP32 drives the PCA9685 over micro-ROS. In fact the Raspberry Pi drives the PCA9685 over I²C, and the ESP32 only streams.
  - It said "~640 mm reach", which no file in the repo supports, so I removed it.
  - It gave the wrong path for `experimental_results.md`.

### Removed (cleanup policy)
- Personal absolute paths (`/home/salma/...`) in `emg_ros_env.sh` (now `$HOME`), `prior_inference_node*.py` (now `os.path.expanduser`), and the usage examples in `train_emg_model.py`, `train_emg_v4.py`, `PI_SETUP.md` and `INVENTORY.md`.
- Student ID from `experimental_results.md`.
- Wi-Fi hotspot name and password from `emg_ml/wifi/emg_stream_wifi.ino`, replaced with `your_ssid` / `your_password` placeholders.
- SolidWorks-to-URDF export leftovers from `src/rotatedauto/`:
  - `export.log` (3 MB exporter log containing local Windows paths)
  - the ROS 1 `launch/display.launch` and `launch/gazebo.launch`, which reference `gazebo_ros`, `rviz` and a missing `urdf.rviz`; they had already been replaced by the `.launch.py` files
  - `config/joint_names_rotatedauto.yaml`, which is unused and lists duplicate `j6` names

Note: these files and strings are still in the git history (commit `8d46974`).

### Added
- `README.md`: rewritten with the system pipeline, repo structure, real run commands, results with their source and protocol stated, and a list of what is missing from the repo.
- `CHANGELOG.md` (this file).

### Verified (no change needed)
- All Python files pass `py_compile`. `flake8 --select=F,E9` found only unused imports and variables.
- End-to-end test on **synthetic** 500 Hz data (not real EMG; checks the plumbing only):
  - `preprocess_v0_windowed.py --mains-filter 50 --mains-causal` produces 76-D features.
  - A random-forest pipeline and an `RCNNInference` wrapper both save and load with joblib.
  - `inference_rt.py --host` reads a fake ESP32 TCP stream and prints voted gestures, with both model types.
- The feature extraction in the ROS `emg_inference_node.py` is bit-identical (max diff 0) to `preprocess_v0_windowed.py` over 50 random windows. `mains_notch.py` and `rcnn_model.py` are identical in both places.
- `emg_arm_control.py --demo --dry-run` and `--stdin --dry-run` run and drive the mock PCA9685 correctly, including the mirrored j2 pair.
- The CNN-BiLSTM in `train_emg_model.py` builds (372k parameters) and forward-passes a `(batch, 76, T)` input.
- `rotatedauto.xacro` expands with `xacro`. It gives a valid tree of 10 links, with the `j6r` mimic of `j6l` (−2.0885) and every mesh present. The SRDF joints match the URDF.
- All four ESP32 sketches pass a `g++ -fsyntax-only` check against stub Arduino/WiFi headers (not a real ESP32 build).
- Not run: ROS 2 / Gazebo / MoveIt launch files, `colcon build`, a real ESP32 build, and training (no dataset in the repo).

## v1.0.0

- Initial upload: ROS 2 workspace, Raspberry Pi inference and arm control, EMG ML reference scripts, firmware, and the thesis.

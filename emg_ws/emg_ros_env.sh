# Source this in any ROS terminal before launching the EMG sim:
#     source ~/emg_ws/emg_ros_env.sh
#
# Fixes the two gotchas:
#  1. Uses the CURRENT workspace (~/emg_ws), not the old ~/ros2_emg_ws.
#  2. Forces a consistent numpy<2 stack (from ~/emg_venv) for the inference node,
#     and ignores ~/.local (which has numpy 2.x that clashes with apt scipy).
source /opt/ros/jazzy/setup.bash
source ~/emg_ws/install/setup.bash
export PYTHONNOUSERSITE=1
export PYTHONPATH=/home/salma/emg_venv/lib/python3.12/site-packages:$PYTHONPATH

# Gazebo GUI fix (2026-06-07): after installing the NVIDIA driver, gz's EGL renderer
# tried the NVIDIA GPU (10_nvidia.json) and crashed ("failed to create dri2 screen").
# The desktop runs on the AMD iGPU, so force gz to use Mesa/AMD EGL only.
export __EGL_VENDOR_LIBRARY_FILENAMES=/usr/share/glvnd/egl_vendor.d/50_mesa.json

echo "[emg_ros_env] workspace=~/emg_ws  python-stack=~/emg_venv (numpy<2)  gz-render=Mesa/AMD"

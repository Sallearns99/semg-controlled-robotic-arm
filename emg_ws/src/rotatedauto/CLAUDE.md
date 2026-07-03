# rotatedauto — ROS 2 Jazzy + Gazebo Harmonic handoff

This package was prepared on Windows (no ROS available there), so everything
below is **ready to build and debug on-target**, not yet proven to run. Your job
on the Ubuntu/Jazzy box: build it, launch it, and walk the **Verification
checklist** — each item maps to a known failure mode that was designed around but
couldn't be tested here.

## Target stack
- ROS 2 **Jazzy**
- Gazebo **Harmonic** (`gz sim`, via `ros_gz`) — NOT Gazebo Classic (EOL on Jazzy)
- `ros2_control` + **`gz_ros2_control`**

## What this robot is
6-axis arm (`j1`–`j5` + a 2-finger gripper) exported from SolidWorks. History worth knowing:
- The SolidWorks export modeled the arm **lying along −Y** (SW is Y-up; ROS is Z-up).
  It's stood upright by a fixed `world → base_footprint → base_link` chain with
  `rpy="-1.5708 0 0"` and a `+0.09 m` lift so the base rests on the floor.
- The gripper originally had **two joints both named `j6`** (invalid URDF) — only one
  finger moved. Now split into `j6l` (driver) and `j6r` (follower).

## Files
- `urdf/rotatedauto.xacro` — **source of truth for Gazebo** (geometry + `ros2_control` + gz plugin). Auto-generated from the urdf; if you change geometry, edit the urdf and regenerate, or just edit the xacro directly from now on.
- `urdf/rotatedauto.urdf` — plain description (kept for reference / quick RViz).
- `config/controllers.yaml` — `joint_state_broadcaster`, `arm_controller` (JTC: j1–j5), `gripper_controller` (JTC: j6l only).
- `launch/gazebo.launch.py` — gz sim + spawn + robot_state_publisher + `/clock` bridge + controller spawners.
- `launch/display.launch.py` — RViz2 + joint_state_publisher_gui (no Gazebo).
- `meshes/` — STL meshes (already correctly assembled; do not re-bake).
- The old ROS1 `launch/*.launch`, catkin `CMakeLists`/`package.xml` were replaced with ament equivalents. Ignore/delete any leftover `*.launch` (ROS1 XML).

## Gripper mimic (important)
- **`j6l` is the driver.** `j6r` mimics it via the URDF tag:
  `<mimic joint="j6l" multiplier="-2.0885"/>` (the 1.13 : 2.36 reverse gear ratio).
- `j6r` has **no command interface** and is in **no** controller — that's deliberate
  (it's what makes j6r non-commandable and lets `gz_ros2_control` enforce the mimic).
- If the fingers move the wrong way relative to each other, flip the sign of the
  multiplier in `urdf/rotatedauto.xacro` (and `.urdf`): `-2.0885` ↔ `+2.0885`.
  If the *magnitude* looks off, the ratio is `2.36/1.13 = 2.0885`.

## Build & launch
```bash
cd ~/ros2_ws/src
cp -r /path/to/rotatedauto .          # copy this folder in
cd ~/ros2_ws
rosdep install --from-paths src -y --ignore-src
colcon build --packages-select rotatedauto
source install/setup.bash

# Gazebo + control:
ros2 launch rotatedauto gazebo.launch.py

# (separate terminal, sourced) sanity:
ros2 control list_controllers
ros2 control list_hardware_interfaces
```
RViz-only (no physics), good first smoke test:
```bash
ros2 launch rotatedauto display.launch.py
```

## Move it (quick tests)
Arm (JTC) — send one point:
```bash
ros2 topic pub --once /arm_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
'{joint_names: [j1,j2,j3,j4,j5], points: [{positions: [0.5,0.3,-0.3,0.4,0.2], time_from_start: {sec: 2}}]}'
```
Gripper (drives j6l; j6r should follow automatically):
```bash
ros2 topic pub --once /gripper_controller/joint_trajectory trajectory_msgs/msg/JointTrajectory \
'{joint_names: [j6l], points: [{positions: [0.4], time_from_start: {sec: 1}}]}'
```

## Verification checklist (do these in order)
1. **`colcon build` succeeds.** If `package.xml` deps are missing on the system:
   `sudo apt install ros-jazzy-ros-gz ros-jazzy-gz-ros2-control ros-jazzy-ros2-control ros-jazzy-ros2-controllers`.
2. **Model spawns and is visible** in Gazebo. If invisible / no collision geometry →
   `package://` mesh resolution. The launch sets `GZ_SIM_RESOURCE_PATH` to the
   install `share` parent; if meshes still don't load, export it manually to the dir
   that *contains* the `rotatedauto/` share folder.
3. **Base stays put and upright** (doesn't fall/tip). It's anchored by the fixed
   joint to the `world` link. If it falls, gz didn't treat `world` as the anchor —
   tell me and we'll add an explicit `<gazebo><static>` or a world-fixed joint in SDF.
4. **Controllers go `active`:** `ros2 control list_controllers` shows
   `joint_state_broadcaster`, `arm_controller`, `gripper_controller` all `active`.
   If a controller is stuck `unconfigured`/`inactive`, check `controllers.yaml`
   joint names match the URDF.
5. **The arm actually moves** when you publish a trajectory. If it loads but won't
   move → re-check joint `effort`/`velocity` limits are non-zero (they were set to
   100 / 3.0; the SW export had them at 0, which freezes motion).
6. **Gripper mimic works:** publish to `j6l`; confirm `j6r` moves too, **opposite
   direction**, **~2.09×** the angle. Check via `ros2 topic echo /joint_states`.
   - Wrong direction → flip multiplier sign (see Gripper mimic section).
   - `j6r` doesn't move at all → `gz_ros2_control` mimic support: confirm the
     `<mimic>` tag survived into the spawned description
     (`ros2 param get /robot_state_publisher robot_description` and grep), and that
     `j6r` has no command interface.
7. **Sim time:** controllers use `use_sim_time: true` and the `/clock` bridge is
   running. If motion is jerky/never completes, confirm `ros2 topic echo /clock`
   publishes.

## Joint limits & actuator (MG995 servo, set 2026-06-02)
Joints driven by **MG995** metal-gear servos (~69 g, already in the SW CAD).
Per-servo limits at **6 V**: stall torque 15 kg·cm → `1.47` N·m; no-load
0.13 s/60° → `velocity="8.05"` rad/s. (4.8 V alt: 1.27 N·m / 6.16 rad/s.)
- `j1`, `j5`: **continuous** (effort/velocity only).
- `j2`: **revolute** `±1.5708`. **TWO MG995 servos** → `effort="2.94"` (2 × 1.47).
- `j3`, `j4`: **revolute**, `±1.5708`, single servo `effort="1.47"`.
- `j6l` (gripper driver): **revolute**, `lower=-0.51 upper=0.288`.
- `j6r` (follower): **continuous**, mimic of j6l ×−2.0885 (range ≈[−0.60, 1.07]).
- Every joint has `<dynamics damping="0.05" friction="0.05"/>` — **tunable placeholders,
  NOT spec-derived**.

## Gripper direction convention (verified in Gazebo 2026-06-02)
**j6l = +0.288 (upper) → CLOSED; j6l = −0.51 (lower) → OPEN.** The joint sign is
INVERTED vs. intuition. Encoded in `scripts/teleop_keyboard.py` (o=open decreases j6l).
**MoveIt note:** when building the SRDF, set the gripper `open` group state to
j6l ≈ −0.51 and `close` to ≈ +0.288, or the named states will be backwards.

## Effort enforcement — VERIFIED working (2026-06-02)
gz_ros2_control **does** cap actuator torque at the URDF `effort` limit, even with a
position command interface (it runs an internal position PID; the effort limit bounds
the force it can apply). Evidence:
- At `effort=1.47` (one motor) j2 **sagged** when commanded toward horizontal.
- After setting j2 to `2.94` (its real two-motor budget), commanding j2=0.785 rad it
  **settles at 0.785 with ~0 velocity** — holds rock-solid. Sag resolved by correct torque,
  no separate PID tuning needed.
- So "add PID gains" wasn't the fix here — the position interface already PID-controls;
  the missing piece was torque headroom. If you later want soft/compliant *tunable* PID,
  switch the arm to an **effort** command interface + JTC `gains:` (see
  `/opt/ros/jazzy/share/gz_ros2_control_demos/config/cart_controller_effort.yaml`).
- Watch j3 (elbow): if it droops when extended, it may also need 2 motors / higher effort.

## MoveIt expectation (important)
MoveIt is a **planner** — it generates trajectories and sends them to the **same**
JointTrajectoryControllers. It does **NOT** change holding/torque/PID behavior. So
whatever the arm does now under the JTC is what MoveIt execution will do. The good news:
holding is fixed, so MoveIt execution will track correctly.

## Known soft spots / next steps
- Inertias come straight from SolidWorks (servos included) — fine for a first sim; tune if dynamics look wrong.
- The plain `urdf/rotatedauto.urdf` is **out of sync** (still old continuous ±π / effort=100);
  both launches use the `.xacro`, so it doesn't matter — but don't "regenerate from the urdf" or it'll revert the limits.
- No MoveIt yet (out of scope). The JTC interface is MoveIt-ready when you want it.

## If you (Claude on Ubuntu) need to pick up
The geometry and kinematic chain are correct and verified numerically. The open
work is purely ROS2/gz integration: get it to build, spawn, and pass the checklist
above. Most likely first issues are (in order): missing apt deps, `package://`
mesh path resolution, and the `world`-anchor behavior in gz Harmonic.

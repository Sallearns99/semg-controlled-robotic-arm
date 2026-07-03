# Control System — Technical Report (Draft for Expansion)

> **Note to the writing assistant (Claude Sonnet 4.6):** This is the *control
> section* of a larger project report on a 6-DOF EMG-controlled robotic arm.
> Everything below is factual and reflects the implemented system; assume the
> hardware is fully working (adequate power supply fitted, all servos
> calibrated). Expand each section into full prose to match the surrounding
> report's depth, tone, and formatting requirements. Add figures/diagrams where
> placeholders are marked `[FIGURE]`. Cross-reference the project's earlier
> sections (mechanical design, simulation, EMG sensing/ML) where indicated.

---

## 1. Scope and Objectives

The control system is responsible for converting motion commands into physical
joint movement of the robot arm. The project is delivered in two phases, and the
control architecture deliberately spans both:

1. **Simulation control** — commanding the arm inside a ROS 2 + Gazebo physics
   simulation, used to validate kinematics, joint limits, and the operator
   interface before any hardware risk.
2. **Real-hardware control** — driving the physical arm's seven servo motors
   from a Raspberry Pi, using a keyboard teleoperation interface that mirrors
   the simulation's controls.

The design objective for this stage is **deterministic, operator-in-the-loop
keyboard control of all degrees of freedom**, with a software architecture that
(a) keeps simulation and hardware code cleanly separated, (b) preserves identical
control conventions across both, and (c) provides a clear integration point for
the later EMG-driven control mode.

---

## 2. The Robot and Its Actuated Joints

The arm is a 6-DOF design (5 arm joints + a gripper) exported from SolidWorks and
actuated by **MG995 metal-gear servos**. Although the arm has six controllable
degrees of freedom, it uses **seven servo motors**, because the shoulder joint
(`j2`) carries the largest gravitational load and is driven by a **pair** of
servos working together.

| Joint | DOF role        | Servos | PCA9685 channel(s) | Notes |
|-------|-----------------|:------:|:------------------:|-------|
| `j1`  | Base rotation   | 1      | 0                  | Yaw about the vertical axis |
| `j2`  | Shoulder pitch  | 2      | 1 and 2            | **Two servos, mirrored** (load sharing) |
| `j3`  | Elbow pitch     | 1      | 3                  | |
| `j4`  | Wrist pitch     | 1      | 4                  | |
| `j5`  | Wrist roll      | 1      | 5                  | |
| `j6`  | Gripper         | 1      | 6                  | Single servo; opposing finger is a passive linkage |

**Shoulder dual-servo arrangement (`j2`).** Two MG995 servos are mounted on
opposite sides of the shoulder axis to combine their torque (≈2 × 1.47 N·m ≈
2.94 N·m at 6 V) and share the cantilever load of the extended arm. Because they
face opposite directions, the two servos must rotate in **opposite angular
directions** to produce a single coherent joint motion — i.e. they are
*mirrored*. The control software handles this transparently (see §4.2).

**Gripper (`j6`).** Mechanically the gripper has two fingers, but only one is
servo-driven; the second finger follows through a passive mechanical linkage.
Control therefore treats the gripper as a single actuator commanded between an
*open* and a *closed* endpoint.

`[FIGURE: labelled arm with joint axes j1–j6 and the dual j2 servos]`

---

## 3. Control Architecture Overview

The system implements two parallel control paths that share the same operator
conventions but target different back-ends:

```
                       ┌─────────────────────────────┐
   Operator keyboard ──┤  Teleoperation layer        │
                       │  (identical keymap + limits) │
                       └──────────────┬───────────────┘
                                      │
                ┌─────────────────────┴─────────────────────┐
                │                                            │
     ┌──────────▼───────────┐                   ┌────────────▼─────────────┐
     │ SIMULATION PATH       │                   │ HARDWARE PATH             │
     │ ROS 2 (rclpy)         │                   │ Plain Python (no ROS)     │
     │ JointTrajectory msgs  │                   │ Adafruit ServoKit         │
     │ → ros2_control JTCs   │                   │ → PCA9685 (I²C) → servos  │
     │ → Gazebo physics      │                   │ → physical MG995 motors   │
     └───────────────────────┘                   └───────────────────────────┘
```

`[FIGURE: redraw the above as a clean block diagram]`

A core design decision is that the **two paths are kept in physically separate
folders** and share **no runtime dependency**. The hardware path imports neither
ROS nor the simulation package, so it can be copied onto the Raspberry Pi and run
standalone. What the two paths *do* share is the **operator-facing contract**:
the same keys move the same joints in the same directions, with the same joint
limits and the same sign conventions. This means control skills learned in
simulation transfer directly to the hardware.

### 3.1 Simulation control path (summary)

In simulation, joint commands are published as `trajectory_msgs/JointTrajectory`
messages to two `ros2_control` JointTrajectoryControllers:

- `arm_controller` — drives `j1`–`j5`.
- `gripper_controller` — drives the gripper joint.

These controllers run inside Gazebo via the `gz_ros2_control` plugin, which
applies the URDF effort/velocity limits and enforces the gripper's mechanical
coupling. This path validated the kinematics, the joint limits, and — critically
— that the shoulder requires its doubled torque budget to hold the arm
horizontal without sagging. *(Expand from the simulation section of the report.)*

### 3.2 Hardware control path (focus of this report)

On the physical arm, there is no ROS and no trajectory controller. A single
Python program reads the keyboard, maintains a target angle for every joint, and
writes servo commands to a **PCA9685 16-channel PWM driver** over the I²C bus.
The PCA9685 generates the 50 Hz pulse-width-modulated signals the MG995 servos
require, offloading precise pulse timing from the Raspberry Pi's CPU. The
remainder of this report details this path.

---

## 4. Hardware Control: From Joint Angle to Servo Pulse

### 4.1 Signal chain

```
Keyboard ─► Teleop loop ─► joint target (radians)
        ─► angle-mapping (per-joint calibration)
        ─► servo angle (degrees)
        ─► ServoKit / PCA9685 (I²C, 0x40)
        ─► PWM pulse (≈500–2500 µs @ 50 Hz)
        ─► MG995 servo horn position
```

The Raspberry Pi communicates with the PCA9685 over I²C (default address
`0x40`). Each MG995 is positioned by the **width of a PWM pulse** repeated at
50 Hz: a pulse of roughly 500 µs drives the servo toward one travel extreme and
2500 µs toward the other, spanning approximately 180° of horn rotation. These
pulse-width endpoints are configured per channel in software.

### 4.2 Angle mapping (radians → servo degrees)

The control software represents joint state in **radians**, matching the
simulation's URDF convention, then converts to **servo degrees** for output. For
a joint *j* commanded to angle θ (radians), each of its servo channels *c* is
driven to:

```
servo_deg(c) = center_deg(c) + sign(c) · k(c) · θ
```

where:

- **center_deg(c)** is the servo angle that places the joint at its mechanical
  zero pose (nominally 90°, the mid-travel point, refined by calibration);
- **sign(c) ∈ {+1, −1}** sets the rotation direction of that channel;
- **k(c)** is the gain in servo-degrees per radian of joint motion (≈57.3°/rad
  for a direct 1:1 drive; different if the joint is geared).

The result is clamped to the servo's physical travel `[0°, 180°]`.

**Mirrored shoulder.** The dual-servo `j2` is expressed as two channels with
opposite signs and equal gain:

```
servo_deg(ch1) = 90° + (+1)·k·θ
servo_deg(ch2) = 90° + (−1)·k·θ
```

so the two horns are always symmetric about their centres — for any joint angle,
`(ch1 − 90°)` and `(ch2 − 90°)` are equal and opposite. A single operator command
moves the shoulder as one rigid joint while the two motors physically
counter-rotate to share the load.

### 4.3 Joint limits

A physical servo cannot rotate continuously, so every joint is bounded by
software limits (in radians) that mirror the simulation's URDF limits — for
example ±1.5708 rad (±90°) on the shoulder, elbow, and wrist. Commands are
clamped to these bounds before output, preventing the operator from driving a
servo into a mechanical hard stop (which an MG995 will otherwise stall against
and overheat).

### 4.4 Gripper

The gripper is handled directly in servo degrees rather than radians, since it is
a single actuator with two meaningful endpoints. Calibration records an
`open_deg` and a `closed_deg`; the *open* and *close* keys step the servo between
them.

---

## 5. Software Design

The hardware control software is intentionally small, layered, and
self-contained, residing in `raspberry_pi/arm_control/`:

| Module | Responsibility |
|--------|----------------|
| `servo_config.py` | **Single source of calibration truth** — joint→channel map, per-channel `center_deg`/`sign`/gain, joint limits, gripper endpoints, pulse-width settings, motion defaults. |
| `arm_driver.py` | Hardware abstraction (`ArmDriver`). Converts joint radians to per-channel servo degrees, applies the mirror and limit logic, and writes to the PCA9685. Provides smooth (rate-limited) moves for homing/reset and a release (limp) function. |
| `arm_keyboard_teleop.py` | The operator interface. Reads keystrokes, maintains joint targets, and calls `ArmDriver`. |
| `servo_calibrate.py` | Interactive single-channel tool for discovering each servo's centre, travel range, and the gripper endpoints. |

### 5.1 Hardware-abstraction and the simulation back-end

`ArmDriver` selects one of two back-ends at construction:

- **Hardware back-end** — the real PCA9685 via the `adafruit-circuitpython-
  servokit` library.
- **Mock back-end** — a drop-in stand-in that prints the servo commands it would
  send and moves nothing. It engages automatically when the Adafruit libraries
  or the I²C bus are unavailable, or on demand via a `--dry-run` flag.

This abstraction is significant for the report's discussion of *verification and
validation*: it allowed the entire control logic — the radian-to-degree mapping,
the mirrored-shoulder symmetry, joint-limit clamping, and the gripper stepping —
to be **unit-tested on a development PC without any hardware connected**, and it
lets the same code be dry-run on the Raspberry Pi before the motors are wired,
de-risking first power-on.

### 5.2 Operator interface and keybindings

The teleoperation loop uses non-blocking, single-keystroke (cbreak) terminal
input so the operator can "tap" joints incrementally. Each keypress nudges a
joint target by a configurable step (default 0.10 rad) and immediately issues the
command. The keymap is **identical to the simulation's teleop**, preserving
operator muscle memory across phases:

| Key(s) | Action |
|--------|--------|
| `a` / `z` | `j1` + / − |
| `s` / `x` | `j2` + / − (shoulder, both servos) |
| `d` / `c` | `j3` + / − |
| `f` / `v` | `j4` + / − |
| `g` / `b` | `j5` + / − |
| `o` / `p` | gripper open / close |
| `space` | re-send (hold) current pose |
| `h` | smooth move to home pose |
| `[` / `]` | decrease / increase step size |
| `k` | release servos (go limp — arm must be supported) |
| `q` | quit (servos keep holding) |

### 5.3 Smooth motion and homing

Per-keystroke jogs are small, so they are issued directly. For larger automated
moves — driving to the **home** pose on start-up, or resetting — the driver
**interpolates** all joints to the target at a bounded angular rate (~50 Hz
update, configurable slew limit), so the arm transitions smoothly rather than
snapping. This protects the structure and the gears from impulsive loads.

---

## 6. Calibration Procedure

Because servo horns can be installed at arbitrary orientations and the linkages
have mechanical offsets, every channel must be calibrated once before use. The
`servo_calibrate.py` tool drives one channel at a time in raw degrees while the
operator:

1. nudges the servo (±1° / ±5° keys) until the **joint sits at its zero pose** —
   that angle becomes the channel's `center_deg`;
2. notes the angles at which the servo begins to strain — these inform the safe
   software limits;
3. for the gripper, records the fully-open and fully-closed angles
   (`open_deg`, `closed_deg`).

These values are transcribed into `servo_config.py`. If a joint is found to move
in the wrong direction, its `sign` is flipped. The tool surfaces I²C/power
faults explicitly (rather than hanging) and provides a per-channel *relax* key so
that only the channel under test need draw holding current during calibration
(see §7).

`[FIGURE: calibration tool terminal output / per-channel center values table]`

---

## 7. Power System

The single most important electrical constraint is **current**. MG995 servos draw
on the order of an amp each when working, and several drawing simultaneously
exceeds 5 A; transient stall currents are higher still. The control electronics
and the motor power are therefore separated:

- **Logic power:** the PCA9685's logic supply (VCC) is taken from the Raspberry
  Pi's 3.3 V rail; the I²C lines (SDA/SCL) connect to the Pi.
- **Motor power:** the PCA9685's **V+** servo rail is fed from a **dedicated
  5–6 V supply rated for ≥5 A** (with headroom recommended). The servos are
  **never** powered from the Pi's 5 V pin.
- **Common ground:** the motor supply ground, the PCA9685 ground, and the Pi
  ground are tied together so the PWM and I²C signals share a reference.

**Brownout failure mode (lesson learned).** An under-rated source — for example a
handheld battery or a phone charger — sags under the combined servo load: the
rail voltage collapses, servos lose torque, and the PCA9685 can reset, wedging
the I²C bus. The symptom is servos that "hold but barely move", followed by the
control program appearing to freeze. The resolution is twofold: **(1)** size the
supply correctly (≥5 A at 5–6 V, e.g. a switching supply or a 2S LiPo + buck
converter), and **(2)** the software now reports I²C write failures explicitly and
can relax idle channels to reduce simultaneous holding current — converting a
silent hang into a clear, recoverable fault.

`[FIGURE: power-and-signal wiring diagram — Pi, PCA9685, external supply, common ground]`

---

## 8. Safety Considerations

- **First power-on:** the arm is supported by hand on the first run; the start-up
  homing sequence includes a countdown and can be skipped.
- **Hard-stop protection:** software joint limits prevent commanding servos past
  their travel, avoiding sustained stall and overheating.
- **Hold vs. release:** on exit the servos *hold* their last pose (safer for a
  loaded arm than going limp); an explicit operator action is required to relax
  them, and only with the arm supported.
- **Electrical isolation:** separation of logic and motor power, with a common
  ground, prevents motor transients from disturbing the Pi.

---

## 9. Relationship to the Wider System and Future Work

The hardware control layer is deliberately structured so that the **keyboard is
just one source of joint commands**. The same `ArmDriver` interface — "set joint
*j* to angle θ" / "set gripper to open/closed" — is the integration point for the
project's **EMG-driven control mode**: the EMG classifier's gesture outputs can be
mapped onto the identical joint-command API, substituting the operator's
keystrokes with decoded muscle intent, without changing the actuation layer
beneath it. This continuity between the simulation interface, the keyboard
hardware interface, and the future EMG interface is a central design theme of the
control system. *(Link to the EMG sensing/ML section.)*

`[FIGURE: end-to-end pipeline — EMG → classifier → joint-command API → ArmDriver → servos]`

---

## 10. Summary

The control system provides operator-in-the-loop keyboard control of all six
degrees of freedom of the arm, across both a ROS 2/Gazebo simulation and the
physical hardware, under a shared set of conventions. On hardware, joint angles
in radians are mapped through per-joint calibration into servo pulses driven by a
PCA9685, with special handling for the load-sharing mirrored shoulder and the
single-servo gripper. A layered, hardware-abstracted software design enabled
full logic verification without hardware and clean separation from the simulation
codebase, while a correctly-specified, electrically-isolated power system and
explicit fault reporting ensure reliable operation. The architecture leaves a
direct, well-defined seam for the subsequent EMG control mode.

---
---

# ADDENDUM — Expanded Material (Simulation Path, EMG Bridge, Networking, Challenges, Specifics)

> **Note to the writing assistant (Claude Sonnet 4.6):** Sections 1–10 above lean
> on the *hardware* control path. This addendum supplies the depth that was
> missing for the **simulation control path**, the **EMG→arm software bridge**,
> the **distributed (Pi/ESP/PC) networking**, the **verification work**, and a
> consolidated **challenges** section — these represent the bulk of the implemented
> control engineering and are where the project's problem-solving is most
> demonstrable. Everything here is factual to the implemented system. When you
> write the final control section: (a) integrate this material with §1–10 rather
> than appending it verbatim; (b) lead the "simulation control" subsection with
> A11–A12 and the "challenges" discussion with A16 (it is the strongest
> evidence of engineering depth); (c) convert the tables in A17 into the report's
> formatting; (d) expand every `[FIGURE]`/`[SCREENSHOT]` placeholder; (e) keep the
> two-phase framing (simulation validated the design before any hardware risk;
> hardware and EMG reuse the identical command contract). Distinguish clearly
> between what is **verified in simulation** (A15 lists exactly what was tested)
> and what is **designed-and-ready but pending hardware** (real arm, EMG with a
> working sensor). Do not overstate: the ML gesture model is not yet accurate, so
> frame EMG control as a *validated pipeline awaiting a better-quality dataset*,
> not a finished classifier.

## A11. Simulation Control Stack (software environment)

The simulation control path is built on a current, supported robotics stack:

| Component | Choice | Role |
|-----------|--------|------|
| Middleware | **ROS 2 Jazzy Jalisco** (Ubuntu 24.04) | Nodes, topics, actions, parameters, launch |
| Physics simulator | **Gazebo (gz sim 8 "Harmonic")** via `ros_gz` | Rigid-body dynamics, gravity, contacts |
| Control framework | **`ros2_control`** + **`gz_ros2_control`** | Real-time joint command/state interfaces inside Gazebo |
| Motion planning | **MoveIt 2** | Kinematics, planning, collision-aware execution |
| Description | **URDF generated from a xacro** | Single source of robot geometry/kinematics/limits |

Gazebo *Classic* was deliberately avoided — it is end-of-life on Jazzy; the modern
`gz sim` + `gz_ros2_control` path is the supported one. The project is organised
into three ROS packages: **`rotatedauto`** (robot description + Gazebo bring-up +
keyboard teleop), **`rotatedauto_moveit_config`** (MoveIt configuration), and
**`emg_control`** (EMG inference + the gesture→arm bridge).

## A12. Kinematic Model, ros2_control Wiring, and the Mimic Gripper

**Kinematic tree.** `world → base_footprint → base_link → l1 → l2 → l3 → l4 → l5 →
{l6l, l6r}`. The first two joints are fixed (the world anchor); `j1`–`j5` are the
actuated arm joints; the gripper is the `l5→l6l`/`l5→l6r` pair.

**ros2_control / Gazebo interface.** The xacro declares a `ros2_control` system of
type `gz_ros2_control/GazeboSimSystem`. Each arm joint and the gripper driver
expose a **position command interface** plus position/velocity/effort state
interfaces. Three controllers run under the controller manager:

- `joint_state_broadcaster` — publishes `/joint_states`;
- `arm_controller` — a **JointTrajectoryController** over `j1`–`j5`;
- `gripper_controller` — a JointTrajectoryController over the gripper driver only.

Both the keyboard teleop and MoveIt command these through the controllers'
`FollowJointTrajectory` action / `joint_trajectory` topic, so the **command
contract is identical** whether motion comes from a human, a planner, or the EMG
bridge — the same seam emphasised for the hardware path in §9.

**Mimic gripper (a notable modelling problem; see A16-3).** The two-finger gripper
is modelled as a **driver joint `j6l`** and a **passive follower `j6r`** linked by
a URDF `<mimic>` tag with multiplier **−2.0885** (the physical **1.13 : 2.36
reverse gear ratio**, sign for opposing motion). `j6r` has *no* command interface
and is in *no* controller; `gz_ros2_control` enforces the coupling, so commanding
one driver opens/closes both fingers correctly.

**Actuator (torque/speed) modelling.** Joint `effort`/`velocity` limits in the
URDF are derived from the MG995 datasheet: stall torque **1.47 N·m** and no-load
speed **8.05 rad/s** at 6 V. The dual-servo shoulder is modelled with **effort =
2.94 N·m** (2 × 1.47), matching the physical load-sharing arrangement (§2). Joint
ranges: `j1`,`j5` continuous; `j2`,`j3`,`j4` ±1.5708 rad; gripper driver
−0.51 → 0.288 rad. **Gripper sign is inverted vs. intuition:** +0.288 rad =
*closed*, −0.51 rad = *open* — this convention is carried consistently into the
teleop and (future) EMG mapping.

`[FIGURE: RViz/Gazebo screenshot of the arm standing upright with joint frames]`

## A13. Motion Planning Layer (MoveIt 2)

A MoveIt configuration (`rotatedauto_moveit_config`) was generated and integrated
with the running Gazebo controllers, giving collision-aware planning on top of the
raw joint control:

- **Planning groups:** `arm` (kinematic chain `base_link → l5`, i.e. `j1`–`j5`) and
  `gripper` (`j6l`); the end effector is defined on `l5`; `j6r` is declared a
  passive joint.
- **Named states** for quick commanding/testing: arm `home`/`ready`; gripper
  `open` (−0.51) / `closed` (+0.288).
- **Self-collision matrix** generated headlessly with `collisions_updater`
  (no GUI) — 21 link pairs disabled (adjacent / never-colliding).
- **Controller bridge:** a `moveit_simple_controller_manager` maps the `arm` and
  `gripper` groups onto the same `FollowJointTrajectory` controllers used by the
  teleop, so planned trajectories execute on the simulated arm.
- **IK solver:** KDL. **Known limitation (A16-7):** the arm is *5-DOF*, while KDL
  assumes a 6-DOF wrist, so full position+orientation Cartesian goals frequently
  have no IK solution. **Joint-space goals and named states work reliably**;
  Cartesian goals are best-effort. This is an inherent property of a 5-DOF arm,
  not a software defect, and is the natural place to note future work (a
  5-DOF-aware or position-only IK).

`[SCREENSHOT: MoveIt MotionPlanning panel — plan + execute of a joint-space goal]`

## A14. EMG → Arm Software Bridge (`emg_control` package)

This is the concrete realisation of the "EMG is just another command source" seam.
Two ROS 2 nodes:

1. **`emg_inference_node`** — ingests the 2-channel EMG stream (USB serial *or*
   the WiFi/TCP relay, identically), maintains a **200 ms sliding window with a
   100 ms hop**, extracts features (MFCC + time-domain descriptors), classifies
   the current gesture, applies **majority-vote smoothing** over the last 5
   predictions and a **REST confidence gate**, and publishes `/detected_gesture`
   (stable label) and `/inference_result` (JSON: gesture + confidence + class
   probabilities).
2. **`emg_bridge_node`** — converts gestures into arm motion using the *same
   incremental-target logic as the keyboard teleop*. Mapping for the initial
   gesture set: **EXTENSION/FLEXION jog a selected joint ±**, **GRIP toggles the
   gripper** (with a cooldown so chatter cannot double-toggle), and **REST / no
   confident gesture / stale signal → HOLD** (a deadman governed by a
   `signal_timeout`). It publishes to the same `arm_controller`/`gripper_controller`
   topics as everything else.

**Robustness for an imperfect classifier (A16-11).** Because the gesture model is
not yet accurate, the bridge layers three safeguards so misclassifications do not
produce dangerous motion: inference-side majority voting, a bridge-side
**confidence threshold**, and the **deadman timeout** that returns the arm to HOLD
whenever the signal is uncertain or interrupted. Key behaviours (controlled joint,
jog step, confidence threshold, timeouts) are runtime parameters.

`[FIGURE: data-flow — EMG stream → window/features → classifier → vote/gate → bridge → JTC → arm]`

## A15. Distributed / Networked Control Architecture

For the portable, real-arm configuration the control system is **physically
distributed** across three devices on one 2.4 GHz network:

```
[EMG sensors] → [ESP32-S3]  --WiFi/TCP-->  [Raspberry Pi]  -->  [control host]
   analog        2-ch ADC,                   relay /              ROS 2: inference +
   envelope      2000 Hz, TCP                forwarder            bridge + (sim) arm
                 server :5005
```

- The **ESP32-S3** samples both channels at 2000 Hz and serves the
  `ch1,ch2\n` stream over a **TCP server on port 5005** (advertised via mDNS as
  `esp32.local`). It batches ~10 ms of samples per TCP write so network I/O never
  disturbs the 500 µs sample schedule; ADC inputs use **ADC1 pins (GPIO3/GPIO8)**,
  which remain usable while WiFi is active (ADC2 is not).
- The **Raspberry Pi** runs a small TCP client (`tcp_receiver.py`) that relays the
  byte stream into a **pseudo-terminal (PTY)**, so the existing serial-based ROS
  consumer reads it **unchanged** — a zero-modification integration point.
- The wire format is **identical** to the direct-USB stream, so the same inference
  pipeline serves both bench (USB) and deployed (WiFi) configurations.

This distributed layer is where several real bring-up challenges arose (A16-9,
A16-10).

`[FIGURE: network/topology diagram — ESP32 ↔ WiFi ↔ Pi ↔ control host, ports/format labelled]`

## A16. Challenges Encountered and How They Were Resolved

> Strongest material for demonstrating engineering depth — recommend a dedicated
> subsection. Each item is a real problem hit during implementation.

1. **SolidWorks-to-URDF mesh frames.** The first CAD export produced STL meshes
   referenced to the *assembly's global origin* rather than each link's local
   frame, so every link rendered grossly offset in RViz. Attempting to correct
   each link with manual visual-origin transforms proved brittle and unphysical.
   **Root cause:** subassembly export semantics. **Resolution:** re-export with an
   explicit **reference coordinate system per link** (origin at the joint pivot,
   Z along the rotation axis); the corrected model was validated in an online URDF
   viewer before ROS import.
2. **Coordinate-convention mismatch (Y-up vs Z-up).** SolidWorks models Y-up; ROS
   is Z-up, so the arm imported lying on its side. **Resolution:** a fixed
   `world → base_footprint → base_link` chain (rpy = −1.5708, 0, 0; +0.09 m lift)
   stands it upright on the floor plane.
3. **Invalid gripper joints.** The exporter emitted **two joints both named `j6`**
   (illegal URDF; only one finger actuated). **Resolution:** split into `j6l`
   (driver) and `j6r` (passive **mimic**, ratio −2.0885), with `j6r` excluded from
   all controllers so `gz_ros2_control` enforces the linkage.
4. **`robot_description` parsed as YAML.** On Jazzy the xacro-generated description
   failed to load until the `Command(['xacro', …])` substitution was wrapped in
   `ParameterValue(…, value_type=str)`; otherwise the launch system tried to parse
   the XML as YAML.
5. **Shoulder sag / effort enforcement.** With the shoulder limited to one MG995's
   1.47 N·m, `j2` could not hold the arm horizontal (it needs ≈2–2.5 N·m) and
   sagged. This is the design driver for the **dual-servo shoulder**: at 2.94 N·m
   the joint holds. **Verification:** commanded `j2` to 0.785 rad and it settled at
   0.785 with ~zero residual velocity, confirming both that the doubled torque is
   required *and* that `gz_ros2_control`'s position interface genuinely enforces the
   URDF effort limit.
6. **Recovery from extreme poses.** At full extension the gravitational torque
   exceeds the servo torque, so a joint can hold a partially-loaded pose yet be
   unable to lift back from the limit — physically realistic for an under-powered
   servo, and a flag that the open-loop servos give no position feedback on the
   real arm (future closed-loop consideration).
7. **5-DOF inverse kinematics.** The arm has five DOF but KDL assumes a 6-DOF
   wrist, so arbitrary 6-D pose goals often have no IK solution (A13). Mitigated by
   relying on joint-space/named-state goals; noted as future work.
8. **Sim/hardware code coupling.** To guarantee the hardware path can run without
   ROS, the two paths share **no runtime dependency** — only the operator contract
   (keymap, joint signs, limits). A **mock servo back-end** lets the hardware
   control logic be exercised with no hardware attached (§5.1).
9. **Cross-device networking.** Bringing up the distributed configuration exposed
   two classic faults: devices landing on **different subnets/bands** (so they
   could not reach each other despite "same WiFi"), and **client (AP) isolation**
   on some phone hotspots that blocks device-to-device traffic entirely.
   **Resolution:** consolidate all devices on a single 2.4 GHz network and verify
   reachability (ping/ARP) before higher-level debugging.
10. **ESP32 WiFi streaming.** Several embedded issues: per-sample TCP writes
    stalled the 2000 Hz sampler (**fixed by ~10 ms batching**); `ADC2` is unusable
    with WiFi on (**inputs kept on ADC1 / GPIO3,8**); a guard on
    `availableForWrite()` returned 0 on the installed core and **silently dropped
    every batch** — the symptom was "TCP connected but no data" — fixed by writing
    unconditionally; and the USB fallback must run at **921600 baud** because
    2000 Hz × 2 channels needs ≈200 kbit/s (115200 baud cannot keep up).
11. **Control with a low-accuracy classifier.** Until a better EMG dataset is
    collected, gesture predictions are unreliable; the bridge therefore combines
    majority-vote smoothing, a confidence gate, and a deadman timeout so spurious
    classifications default to HOLD rather than motion (A14).

## A17. Specifics / Parameters Appendix (for citation)

**Joints, actuators, limits, and (hardware) PWM channels:**

| Joint | Function | Servos | Sim limit (rad) | Effort (N·m) | PCA9685 ch |
|-------|----------|:------:|-----------------|:------------:|:----------:|
| j1 | base yaw | 1 | continuous | 1.47 | 0 |
| j2 | shoulder pitch | **2 (mirrored)** | ±1.5708 | **2.94** | 1, 2 |
| j3 | elbow pitch | 1 | ±1.5708 | 1.47 | 3 |
| j4 | wrist pitch | 1 | ±1.5708 | 1.47 | 4 |
| j5 | wrist roll | 1 | continuous | 1.47 | 5 |
| j6 (j6l/j6r) | gripper | 1 driven + passive | −0.51→0.288 | 1.47 | 6 |

**Key parameters:** MG995 — 1.47 N·m stall / 8.05 rad/s no-load @ 6 V; gripper
convention +0.288 = closed, −0.51 = open; mimic ratio −2.0885 (1.13:2.36); EMG
window 200 ms / hop 100 ms; sampling 2000 Hz, 2 channels, 12-bit ADC; teleop step
0.10 rad; PCA9685 at I²C 0x40, 50 Hz PWM, ≈500–2500 µs; servo rail 5–6 V ≥ 5 A.

**Controllers:** `joint_state_broadcaster`, `arm_controller` (JTC j1–j5),
`gripper_controller` (JTC, gripper driver). **Command interface:** position.

## A18. Verification and Validation (what was actually tested)

The control software was validated incrementally, much of it **headless** (no GUI):

- **Description integrity:** the URDF parses and `check_urdf` confirms the
  expected kinematic tree.
- **Bring-up:** Gazebo spawns the arm; all three controllers reach the `active`
  state; `/joint_states` publishes; the arm rests stably upright under gravity.
- **Torque/limits:** the shoulder hold/sag test (A16-5) confirmed effort
  enforcement and the need for the dual servo.
- **Motion planning:** a MoveIt joint-space plan **planned and executed** in
  Gazebo (e.g. j2→0.6, j3→−0.6, reached and held).
- **Teleop & bridge:** the keyboard teleop and the EMG bridge were exercised by
  injecting gestures and confirming the simulated arm jogged the correct joints
  and the gripper toggled correctly (including the inverted-sign and mimic
  behaviour).
- **End-to-end EMG pipeline (playback):** with no sensor hardware, a *recorded*
  EMG CSV was replayed through the inference node into the bridge, driving the
  simulated arm — verifying the entire chain (stream → features → classifier →
  gesture → joint command) independent of the live sensor.
- **Hardware logic:** exercised via the mock servo back-end / dry-run (§5.1),
  de-risking first power-on.

**Not yet validated (designed/ready, pending):** motion on the *physical* arm
(awaiting the adequate power supply), and EMG control with a *usable* classifier
(awaiting re-collected, higher-quality sensor data).

## A19. Glossary (for the writer / reader)

- **DOF** — degree of freedom. **JTC** — JointTrajectoryController.
- **URDF/xacro** — robot description format / its macro pre-processor.
- **ros2_control / gz_ros2_control** — ROS 2 control framework / its Gazebo plugin.
- **mimic joint** — a joint whose motion is a fixed function of another's.
- **PCA9685** — 16-channel I²C PWM driver. **PWM** — pulse-width modulation.
- **PTY** — pseudo-terminal (virtual serial device). **mDNS** — `.local` name
  resolution. **deadman** — safety default that halts motion absent a valid command.

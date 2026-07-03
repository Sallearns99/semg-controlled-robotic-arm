# Raspberry Pi — 7-servo arm keyboard control

Self-contained, **no ROS**. Copy this whole `arm_control/` folder onto the Pi
and run it. Drives the real arm's 7 MG995 servos through a **PCA9685** I2C PWM
board. Same keybindings as the Gazebo sim (`teleop_keyboard.py`), so the controls
feel identical.

> This folder is deliberately separate from the simulation. Nothing here imports
> ROS or the `rotatedauto` package — move it to the Pi and it just runs.

## Files
| file | what it is |
|------|------------|
| `arm_keyboard_teleop.py` | **run this** — keyboard control of the arm |
| `servo_config.py` | **calibrate this** — channel map, limits, center angles |
| `arm_driver.py` | hardware layer (PCA9685 ↔ joints); auto-mocks if no board |
| `servo_calibrate.py` | interactive per-servo calibration helper |
| `requirements.txt` | pip deps (Adafruit ServoKit) |

## The 7 servos
| joint | PCA9685 channel | notes |
|-------|-----------------|-------|
| j1 | 0 | base rotation |
| j2 | **1 and 2** | shoulder — TWO servos, **mirrored** (load sharing) |
| j3 | 3 | elbow |
| j4 | 4 | wrist pitch |
| j5 | 5 | wrist roll |
| j6 | 6 | gripper (one servo; the sim's 2nd finger is a passive linkage) |

If you plugged servos into different channels, edit the `channel` numbers in
`servo_config.py`.

## Wiring (do this with power OFF)
```
Raspberry Pi                         PCA9685                 Servos
------------                         -------                 ------
3.3V  (pin 1)  ── VCC (logic)        V+  ── 5–6V EXTERNAL supply (NOT the Pi!)
GND   (pin 6)  ── GND  ──────────────GND ── shares ground with the Pi
SDA   (pin 3)  ── SDA                CH0..CH6 ── j1, j2a, j2b, j3, j4, j5, gripper
SCL   (pin 5)  ── SCL
```
- **Power**: 7× MG995 can pull several amps. Feed the PCA9685 **V+** from a
  separate 5–6 V / ≥5 A supply. **Never** power the servos from the Pi's 5V pin.
- **Common ground**: the external supply GND, PCA9685 GND, and Pi GND must all
  be tied together, or the I2C/PWM signals float.
- The PCA9685 **VCC** (logic) goes to the Pi 3.3V; **V+** (servo rail) goes to
  the external supply. They are different pins.

## One-time Pi setup
```bash
sudo raspi-config        # Interface Options -> I2C -> Enable, then reboot
sudo apt install -y python3-pip i2c-tools
i2cdetect -y 1           # should show a device at 0x40 (the PCA9685)
pip3 install -r requirements.txt
```
If `pip3 install` complains about an "externally managed environment" on a
recent Pi OS, either add `--break-system-packages` or use a venv.

## Calibrate first (arm supported by hand)
The default `center_deg=90` and gripper endpoints in `servo_config.py` are
placeholders. Find the real values:
```bash
python3 servo_calibrate.py
```
For each channel, nudge the servo (`j`/`l` = ±1°, `J`/`L` = ±5°) until:
- the **arm joint sits at its sim "zero" pose** → that angle is its `center_deg`;
- note where the servo **strains** at each end → keep within that range;
- for the gripper (ch 6): note the angle when **open** → `open_deg`, when
  **closed** → `closed_deg`.

Copy those numbers into `servo_config.py`. If a joint later moves the **wrong
way**, flip the `sign` (+1 ↔ -1) for its channel(s).

## Run
```bash
python3 arm_keyboard_teleop.py            # real hardware
python3 arm_keyboard_teleop.py --dry-run  # print commands, move nothing (test logic)
python3 arm_keyboard_teleop.py --no-home  # skip the startup move to home
```

### Controls
```
 j1:  a / z      j2:  s / x      j3:  d / c
 j4:  f / v      j5:  g / b
 gripper:  o (open) / p (close)
 space : re-send current pose (hold)
 h     : smooth move to HOME
 [ / ] : smaller / bigger step
 k     : RELEASE servos (go limp — HOLD THE ARM FIRST)
 q     : quit (servos keep holding)
```

## Safety
- **First power-up**: on start the arm snaps to HOME. Support it by hand and keep
  the servo-power switch / E-stop in reach. There's a 3-2-1 countdown before it
  moves; use `--no-home` to skip it.
- Move in **small steps** near mechanical limits; MG995 servos will happily
  stall and overheat against a hard stop. The software limits in `servo_config.py`
  are your guard — tighten them once you know each joint's real range.
- `q` leaves servos **holding** their last pose (safer for a heavy arm). To make
  the arm limp, press `k` **while supporting it**, or cut the servo supply.

## Relationship to the simulation
The keybindings and the radian sign conventions match the sim's
`src/rotatedauto/scripts/teleop_keyboard.py`. The difference: the sim publishes
`JointTrajectory` ROS topics; this sends servo angles to a PCA9685. Tune in sim,
then drive the real arm here with the same muscle memory.

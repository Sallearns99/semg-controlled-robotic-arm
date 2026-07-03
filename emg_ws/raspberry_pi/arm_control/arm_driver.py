#!/usr/bin/env python3
"""
arm_driver.py  --  thin hardware layer over the PCA9685 + MG995 servos.

Converts joint angles (radians) into per-channel servo angles using the
calibration in servo_config.py, and drives the PCA9685 via Adafruit ServoKit.

Backends:
  * "hardware" : real PCA9685 over I2C (needs adafruit-circuitpython-servokit).
  * "mock"     : prints what it WOULD send. Used automatically when the
                 adafruit libs aren't installed, or forced with dry_run=True.
                 Lets you test the keyboard logic on any PC (or on the Pi
                 before the board is wired) without moving anything.
"""

import time

import servo_config as cfg


# --------------------------------------------------------------------------
# Backends
# --------------------------------------------------------------------------
class _MockServo:
    def __init__(self, channel, log):
        self.channel = channel
        self.actuation_range = cfg.SERVO_ACTUATION_RANGE
        self._angle = None
        self._log = log

    def set_pulse_width_range(self, lo, hi):
        pass

    @property
    def angle(self):
        return self._angle

    @angle.setter
    def angle(self, value):
        self._angle = value
        if self._log:
            shown = "RELEASED" if value is None else f"{value:6.1f} deg"
            print(f"    [mock] ch{self.channel:>2} -> {shown}")


class _MockKit:
    """Stand-in for adafruit_servokit.ServoKit."""

    def __init__(self, channels=16, log=False):
        self.servo = [_MockServo(i, log) for i in range(channels)]


def _make_kit(dry_run, verbose):
    if dry_run:
        return _MockKit(16, log=verbose), "mock (forced dry-run)"
    try:
        from adafruit_servokit import ServoKit
        return ServoKit(channels=16), "hardware (PCA9685 @ I2C)"
    except (ImportError, NotImplementedError, ValueError) as exc:
        # ImportError: libs missing.  Others: Blinka can't find an I2C bus.
        print(f"[arm_driver] adafruit ServoKit unavailable ({exc.__class__.__name__}: {exc}).")
        print("[arm_driver] Falling back to MOCK backend. Install on the Pi with:")
        print("    pip3 install adafruit-circuitpython-servokit")
        print("    and enable I2C via raspi-config. See README.md.")
        return _MockKit(16, log=verbose), "mock (libs/bus unavailable)"


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------
class ArmDriver:
    def __init__(self, dry_run=False, verbose=True):
        self.kit, self.backend = _make_kit(dry_run, verbose)
        self.verbose = verbose

        # electrical setup per used channel
        used = {c["channel"] for j in cfg.JOINTS.values() for c in j["channels"]}
        used.add(cfg.GRIPPER["channel"])
        for ch in used:
            s = self.kit.servo[ch]
            s.actuation_range = cfg.SERVO_ACTUATION_RANGE
            s.set_pulse_width_range(cfg.SERVO_PULSE_MIN_US, cfg.SERVO_PULSE_MAX_US)

        # logical state
        self.joint_rad = {name: 0.0 for name in cfg.JOINTS}
        self.gripper_deg = cfg.GRIPPER["closed_deg"]

    # --- low level --------------------------------------------------------
    def _write_channel(self, channel, deg):
        deg = cfg.clamp(deg, 0.0, float(cfg.SERVO_ACTUATION_RANGE))
        self.kit.servo[channel].angle = deg

    def _joint_channel_degs(self, name, rad):
        out = []
        for c in cfg.JOINTS[name]["channels"]:
            deg = c["center_deg"] + c["sign"] * c["deg_per_rad"] * rad
            out.append((c["channel"], deg))
        return out

    # --- joint API --------------------------------------------------------
    def set_joint(self, name, rad):
        lo, hi = cfg.JOINTS[name]["limit_rad"]
        rad = cfg.clamp(rad, lo, hi)
        self.joint_rad[name] = rad
        for channel, deg in self._joint_channel_degs(name, rad):
            self._write_channel(channel, deg)
        return rad

    def set_gripper(self, deg):
        lo, hi = sorted((cfg.GRIPPER["open_deg"], cfg.GRIPPER["closed_deg"]))
        deg = cfg.clamp(deg, lo, hi)
        self.gripper_deg = deg
        self._write_channel(cfg.GRIPPER["channel"], deg)
        return deg

    def apply_all(self):
        for name, rad in self.joint_rad.items():
            for channel, deg in self._joint_channel_degs(name, rad):
                self._write_channel(channel, deg)
        self._write_channel(cfg.GRIPPER["channel"], self.gripper_deg)

    # --- smooth moves (used for power-on home + reset) --------------------
    def smooth_move(self, target_rad, target_gripper_deg, rate_deg_s=None):
        """Interpolate every joint+gripper to the target at a bounded speed so
        the arm doesn't snap. Blocking; only used for home/reset, not jogging."""
        rate = rate_deg_s or cfg.SLEW_DEG_PER_S
        start_rad = dict(self.joint_rad)
        start_grip = self.gripper_deg

        # biggest servo-degree move decides the duration
        max_deg = abs(target_gripper_deg - start_grip)
        for name in cfg.JOINTS:
            a = self._joint_channel_degs(name, start_rad[name])
            b = self._joint_channel_degs(name, target_rad.get(name, start_rad[name]))
            for (_, da), (_, db) in zip(a, b):
                max_deg = max(max_deg, abs(db - da))
        duration = max(0.2, max_deg / rate)
        steps = max(2, int(duration / 0.02))  # ~50 Hz update

        for i in range(1, steps + 1):
            f = i / steps
            for name in cfg.JOINTS:
                r = start_rad[name] + f * (target_rad.get(name, start_rad[name]) - start_rad[name])
                self.set_joint(name, r)
            self.set_gripper(start_grip + f * (target_gripper_deg - start_grip))
            time.sleep(duration / steps)

    def go_home(self):
        self.smooth_move(dict(cfg.HOME_RAD), cfg.GRIPPER["closed_deg"])

    # --- shutdown ---------------------------------------------------------
    def release(self):
        """Cut PWM so servos go limp. NOTE: a heavy arm will DROP -- only call
        with the arm supported. By default the teleop holds position instead."""
        for ch in range(16):
            try:
                self.kit.servo[ch].angle = None
            except Exception:
                pass

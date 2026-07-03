#!/usr/bin/env python3
"""
arm_keyboard_teleop.py  --  drive the REAL 7-servo arm from the Pi's keyboard.

Plain Python (no ROS). Talks to a PCA9685 over I2C via Adafruit ServoKit.
Same keybindings as the Gazebo sim's teleop_keyboard.py, so the controls feel
identical to the simulation.

    j1:  a / z      j2:  s / x      j3:  d / c
    j4:  f / v      j5:  g / b
    gripper:  o (open) / p (close)
    space : re-send current pose (hold)
    h     : smooth move to HOME pose
    [ / ] : decrease / increase step size
    k     : RELEASE servos (go limp -- HOLD THE ARM FIRST!)
    q     : quit (servos keep holding their last pose)

Run:
    python3 arm_keyboard_teleop.py            # real hardware
    python3 arm_keyboard_teleop.py --dry-run  # print commands, move nothing
    python3 arm_keyboard_teleop.py --no-home  # don't drive to home on startup

SAFETY: on the first power-up the servos snap to the HOME pose. Support the arm
by hand the first time, and keep the E-stop / servo power switch in reach.
"""

import sys
import time
import select
import termios
import tty
import argparse

import servo_config as cfg
from arm_driver import ArmDriver

# key -> (joint, direction). Mirrors teleop_keyboard.py from the sim.
KEYMAP = {
    "a": ("j1", +1), "z": ("j1", -1),
    "s": ("j2", +1), "x": ("j2", -1),
    "d": ("j3", +1), "c": ("j3", -1),
    "f": ("j4", +1), "v": ("j4", -1),
    "g": ("j5", +1), "b": ("j5", -1),
}


def parse_args():
    p = argparse.ArgumentParser(description="Keyboard teleop for the real arm.")
    p.add_argument("--dry-run", action="store_true",
                   help="use the mock backend; print servo commands, move nothing")
    p.add_argument("--no-home", action="store_true",
                   help="do NOT drive to the home pose on startup")
    p.add_argument("--quiet", action="store_true",
                   help="suppress per-channel mock prints")
    return p.parse_args()


class Teleop:
    def __init__(self, arm):
        self.arm = arm
        self.step = cfg.DEFAULT_STEP_RAD

    def jog(self, joint, direction):
        new = self.arm.joint_rad[joint] + direction * self.step
        self.arm.set_joint(joint, new)
        self._status(joint)

    def gripper(self, direction):
        # direction +1 == toward open_deg, -1 == toward closed_deg
        target = cfg.GRIPPER["open_deg"] if direction > 0 else cfg.GRIPPER["closed_deg"]
        cur = self.arm.gripper_deg
        sign = 1 if target > cur else -1
        self.arm.set_gripper(cur + sign * cfg.GRIPPER_STEP_DEG)
        self._status("grip")

    def handle(self, key):
        if key in KEYMAP:
            self.jog(*KEYMAP[key])
        elif key == "o":
            self.gripper(+1)
        elif key == "p":
            self.gripper(-1)
        elif key == " ":
            self.arm.apply_all()
            self._msg("hold pose")
        elif key == "h":
            self._msg("homing...")
            self.arm.go_home()
            self._status("home")
        elif key == "]":
            self.step = min(0.5, self.step + 0.02)
            self._msg(f"step = {self.step:.2f} rad")
        elif key == "[":
            self.step = max(0.01, self.step - 0.02)
            self._msg(f"step = {self.step:.2f} rad")
        elif key == "k":
            self.arm.release()
            self._msg("servos RELEASED (limp)")

    def _status(self, who):
        vals = "  ".join(f"{j}={self.arm.joint_rad[j]:+.2f}" for j in cfg.JOINTS)
        sys.stdout.write(
            f"\r[{who}]  {vals}  grip={self.arm.gripper_deg:5.1f}deg  (step {self.step:.2f})   ")
        sys.stdout.flush()

    def _msg(self, text):
        sys.stdout.write(f"\r{text:<70}\n")
        sys.stdout.flush()


def main():
    args = parse_args()
    arm = ArmDriver(dry_run=args.dry_run, verbose=not args.quiet)

    print(__doc__)
    print(f"\nbackend: {arm.backend}")

    if not args.no_home:
        print("\n>>> Driving to HOME pose. SUPPORT THE ARM. <<<")
        for n in (3, 2, 1):
            print(f"    {n}...", flush=True)
            time.sleep(1)
        arm.go_home()
    print("\nReady. Press keys to move (q to quit).\n")

    teleop = Teleop(arm)
    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while True:
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not r:
                continue
            key = sys.stdin.read(1)
            if key in ("q", "\x03"):  # q or Ctrl-C
                break
            teleop.handle(key)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print("\nbye (servos still holding last pose; cut servo power to relax)")


if __name__ == "__main__":
    main()

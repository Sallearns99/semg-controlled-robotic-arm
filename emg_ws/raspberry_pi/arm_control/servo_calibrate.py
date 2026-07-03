#!/usr/bin/env python3
"""
servo_calibrate.py  --  find the numbers that go into servo_config.py.

Drives ONE PCA9685 channel at a time in raw servo degrees so you can discover:
  * the safe travel range (where the servo stops without buzzing/straining),
  * the `center_deg` that puts a joint at its sim "zero" pose,
  * the gripper's open_deg / closed_deg.

Controls:
    n         : next channel        b : previous channel
    j / l     : -1 deg / +1 deg     J / L : -5 deg / +5 deg
    c         : print current angle as a center_deg snippet
    0         : go to 90 deg (mid)
    r         : RELAX this channel (limp) -- frees its hold-current; hold the arm
    q         : quit

NOTE: every channel you visit stays powered/holding. With many MG995 holding at
once a weak supply BROWNS OUT and the tool will report a failed I2C write (it no
longer freezes). Press `r` on channels you're done with to shed current, and use
a servo supply rated >=5A.

SAFETY: move in SMALL steps. Support the arm. If a servo strains at an end,
stop and record that as the limit. Start each joint near 90 deg.
"""

import sys
import select
import termios
import tty

import servo_config as cfg
from arm_driver import _make_kit, ArmDriver  # reuse backend selection


def getch(timeout=0.1):
    r, _, _ = select.select([sys.stdin], [], [], timeout)
    return sys.stdin.read(1) if r else None


def safe_set(servo, value):
    """Write a servo angle but never let an I2C/power glitch silently freeze the
    tool. A browned-out PCA9685 (too many MG995 holding at once) often NACKs the
    I2C write -- surface that instead of appearing to 'stop responding'."""
    try:
        servo.angle = value
        return True
    except Exception as exc:  # OSError (I2C NACK), etc.
        sys.stdout.write(
            f"\n[!] servo write FAILED: {exc.__class__.__name__}: {exc}\n"
            "    Most likely a POWER BROWNOUT (several MG995 holding at once) or a\n"
            "    loose V+/ground. Check: separate 5-6V supply rated >=5A, common\n"
            "    ground with the Pi, and that `i2cdetect -y 1` still shows 0x40.\n")
        sys.stdout.flush()
        return False


def main():
    dry = "--dry-run" in sys.argv
    kit, backend = _make_kit(dry, verbose=True)
    channels = sorted({c["channel"] for j in cfg.JOINTS.values() for c in j["channels"]}
                      | {cfg.GRIPPER["channel"]})
    for ch in channels:
        kit.servo[ch].actuation_range = cfg.SERVO_ACTUATION_RANGE
        kit.servo[ch].set_pulse_width_range(cfg.SERVO_PULSE_MIN_US, cfg.SERVO_PULSE_MAX_US)

    print(__doc__)
    print(f"backend: {backend}\nchannels to calibrate: {channels}\n")

    idx = 0
    angle = {ch: 90.0 for ch in channels}
    safe_set(kit.servo[channels[idx]], angle[channels[idx]])

    def show():
        ch = channels[idx]
        sys.stdout.write(f"\rchannel {ch:>2}   angle = {angle[ch]:6.1f} deg     ")
        sys.stdout.flush()

    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        show()
        while True:
            k = getch()
            if k is None:
                continue
            if k in ("q", "\x03"):
                break
            ch = channels[idx]
            if k == "n":
                idx = (idx + 1) % len(channels)
            elif k == "b":
                idx = (idx - 1) % len(channels)
            elif k == "j":
                angle[ch] = cfg.clamp(angle[ch] - 1, 0, cfg.SERVO_ACTUATION_RANGE)
            elif k == "l":
                angle[ch] = cfg.clamp(angle[ch] + 1, 0, cfg.SERVO_ACTUATION_RANGE)
            elif k == "J":
                angle[ch] = cfg.clamp(angle[ch] - 5, 0, cfg.SERVO_ACTUATION_RANGE)
            elif k == "L":
                angle[ch] = cfg.clamp(angle[ch] + 5, 0, cfg.SERVO_ACTUATION_RANGE)
            elif k == "0":
                angle[ch] = 90.0
            elif k == "c":
                sys.stdout.write(f"\r  -> use center_deg={angle[ch]:.1f} for channel {ch}\n")
            elif k == "r":
                # relax THIS channel (stop its PWM) to drop its hold-current --
                # helps avoid browning out the supply while calibrating others.
                # Support the arm: a relaxed joint goes limp.
                safe_set(kit.servo[ch], None)
                sys.stdout.write(f"\r  -> channel {ch} relaxed (limp)\n")
                show()
                continue
            safe_set(kit.servo[channels[idx]], angle[channels[idx]])
            show()
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print("\nDone. Copy your center_deg / open_deg / closed_deg values into servo_config.py")


if __name__ == "__main__":
    main()

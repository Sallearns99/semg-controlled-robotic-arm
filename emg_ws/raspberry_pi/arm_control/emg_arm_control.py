#!/usr/bin/env python3
"""
emg_arm_control.py -- drive the REAL arm from EMG gestures (no ROS, no keyboard).

This is the hardware twin of the Gazebo `emg_bridge_node.py`: it maps the SAME
12-gesture set to the SAME joints, but instead of publishing ROS trajectories it
drives `ArmDriver` (PCA9685 servos), with a dry-run mock backend so you can test
the logic on the dev PC.

Gesture -> joint map (identical to the sim's emg_params.yaml default):
    EXTENSION / FLEXION          -> j3 +/-
    EXTEND_ELBOW / RETRACT_ELBOW -> j2 +/-
    DOWNWARD_ROT_CW / CCW        -> j1 -/+   (base turntable)
    FRONT_ROT_CW / CCW           -> j5 +/-   (wrist roll)
    ULNAR_DEV / RADIAL_DEV       -> j4 +/-   (wrist)
    GRIP                         -> toggle gripper open<->closed (rising edge)
    REST / unlisted / stale      -> HOLD (deadman)

GESTURE SOURCE (decoupled so we don't duplicate the inference pipeline):
  --demo            run a scripted gesture sequence (proves the mapping; great
                    with --dry-run). No hardware or model needed.
  --stdin           read one gesture name per line from stdin. This lets you reuse
                    the EXISTING, proven classifier without re-implementing it:

      # standalone on the Pi (ESP on /dev/ttyACM0):
      python3 inference_rt.py --port /dev/ttyACM0 \
          --model best_model.joblib \
        | python3 emg_arm_control.py --stdin

    (inference_rt.py lives in the ML repo: ".../moving to ubuntu (2)/moving to
     ubuntu/code/". It prints the predicted gesture each hop; this script picks
     the gesture token out of each line.) You can equally pipe ROS:
      ros2 topic echo --field data /detected_gesture | python3 emg_arm_control.py --stdin

Run examples:
    python3 emg_arm_control.py --demo --dry-run      # see the mapping, move nothing
    python3 emg_arm_control.py --demo                # on the Pi, real servos
    python3 inference_rt.py --port /dev/ttyACM0 --model m.joblib | python3 emg_arm_control.py --stdin
"""
import argparse
import re
import sys
import time
from collections import Counter

import servo_config as cfg
from arm_driver import ArmDriver

# Gesture -> (joint, sign). Mirrors src/emg_control/config/emg_params.yaml.
GESTURE_MAP = {
    "EXTENSION":        ("j3", +1),
    "FLEXION":          ("j3", -1),
    "EXTEND_ELBOW":     ("j2", +1),
    "RETRACT_ELBOW":    ("j2", -1),
    "DOWNWARD_ROT_CW":  ("j1", -1),
    "DOWNWARD_ROT_CCW": ("j1", +1),
    "FRONT_ROT_CW":     ("j5", +1),
    "FRONT_ROT_CCW":    ("j5", -1),
    "ULNAR_DEV":        ("j4", +1),
    "RADIAL_DEV":       ("j4", -1),
}
GRIP_GESTURE = "GRIP"
ALL_GESTURES = set(GESTURE_MAP) | {GRIP_GESTURE, "REST"}
# longest-first so DOWNWARD_ROT_CCW matches before DOWNWARD_ROT_CW, etc.
_GESTURE_RE = re.compile("|".join(sorted((re.escape(g) for g in ALL_GESTURES),
                                         key=len, reverse=True)))


class EmgArmBridge:
    """Turns a stream of gesture names into arm motion. Pure + testable."""

    def __init__(self, arm, step_rad=cfg.DEFAULT_STEP_RAD, signal_timeout=0.5,
                 grip_cooldown=0.8, verbose=True,
                 vote_window=0.0, fire_step=0.20, min_vote_frac=0.34):
        self.arm = arm
        self.step = step_rad
        self.timeout = signal_timeout
        self.grip_cooldown = grip_cooldown
        self.verbose = verbose
        # --- vote mode (debounce a noisy classifier) ---
        self.vote_window = vote_window          # s; 0 = continuous jog (no voting)
        self.fire_step = fire_step              # rad per FIRE in vote mode
        self.min_vote_frac = min_vote_frac      # winner must be >= this share of the window
        self.votes = []
        self.window_start = time.time()
        # --- continuous-mode state ---
        self.last_update = time.time()
        self.active = "REST"
        self.prev = "REST"
        self.grip_closed_state = False           # start OPEN -> first GRIP closes
        self.last_grip_toggle = 0.0

    def set_gesture(self, name):
        g = name.strip().upper()
        self.active = g
        self.last_update = time.time()
        if self.vote_window > 0:
            self.votes.append(g)

    def _effective(self):
        if time.time() - self.last_update > self.timeout:
            return "REST"                         # deadman
        return self.active

    def tick(self):
        """Call at a steady rate. Dispatches to vote or continuous mode."""
        if self.vote_window > 0:
            self._vote_tick(time.time())
        else:
            self._continuous_tick(time.time())

    def _vote_tick(self, now):
        """Every `vote_window` seconds, take the majority gesture and fire ONCE."""
        if now - self.window_start < self.vote_window:
            return
        votes, self.votes = self.votes, []
        self.window_start = now
        if not votes:
            return
        winner, count = Counter(votes).most_common(1)[0]
        frac = count / len(votes)
        if winner == "REST" or frac < self.min_vote_frac:
            self._log(f"window: {winner} {count}/{len(votes)} ({frac:.0%}) -> hold")
            return
        self._fire(winner)
        self._log(f"window: {winner} {count}/{len(votes)} ({frac:.0%}) -> FIRE")

    def _fire(self, gesture):
        """One discrete action for a decided gesture (vote mode)."""
        if gesture == GRIP_GESTURE:
            self.grip_closed_state = not self.grip_closed_state
            deg = cfg.GRIPPER["closed_deg"] if self.grip_closed_state else cfg.GRIPPER["open_deg"]
            self.arm.set_gripper(deg)
            self._log(f"gripper -> {'CLOSED' if self.grip_closed_state else 'OPEN'} ({deg:.0f} deg)")
        elif gesture in GESTURE_MAP:
            joint, sign = GESTURE_MAP[gesture]
            self.arm.set_joint(joint, self.arm.joint_rad[joint] + sign * self.fire_step)
            self._log(f"move {gesture}: {joint} -> {self.arm.joint_rad[joint]:+.3f} rad")

    def _continuous_tick(self, now):
        """Jog the mapped joint every tick while a gesture is held (original mode)."""
        g = self._effective()
        if g == GRIP_GESTURE:
            if self.prev != GRIP_GESTURE and (now - self.last_grip_toggle) > self.grip_cooldown:
                self.last_grip_toggle = now
                self.grip_closed_state = not self.grip_closed_state
                deg = cfg.GRIPPER["closed_deg"] if self.grip_closed_state else cfg.GRIPPER["open_deg"]
                self.arm.set_gripper(deg)
                self._log(f"gripper -> {'CLOSED' if self.grip_closed_state else 'OPEN'} ({deg:.0f} deg)")
        elif g in GESTURE_MAP:
            joint, sign = GESTURE_MAP[g]
            self.arm.set_joint(joint, self.arm.joint_rad[joint] + sign * self.step)
            self._log(f"jog {g}: {joint} -> {self.arm.joint_rad[joint]:+.3f} rad")
        # REST / unlisted -> HOLD
        self.prev = g

    def _log(self, msg):
        if self.verbose:
            print(f"[emg_arm] {msg}")


def extract_gesture(line):
    """Pull a known gesture token out of a line of text (inference_rt / ros echo)."""
    m = _GESTURE_RE.search(line.upper())
    return m.group(0) if m else None


def run_stdin(bridge, rate_hz):
    """Read gesture names from stdin; tick the bridge at a fixed rate.

    stdin is drained non-blocking-ish: we read whatever lines are available since
    the last tick and take the most recent recognised gesture as 'active'."""
    import select
    period = 1.0 / rate_hz
    print(f"[emg_arm] reading gestures from stdin at {rate_hz:.0f} Hz tick. Ctrl-C to stop.")
    while True:
        t0 = time.time()
        # drain available lines
        while select.select([sys.stdin], [], [], 0)[0]:
            line = sys.stdin.readline()
            if line == "":                        # EOF
                print("[emg_arm] stdin closed; holding then exiting.")
                return
            g = extract_gesture(line)
            if g:
                bridge.set_gesture(g)
        bridge.tick()
        dt = period - (time.time() - t0)
        if dt > 0:
            time.sleep(dt)


def run_demo(bridge, rate_hz):
    """Scripted sequence so you can SEE each gesture hit the right joint."""
    period = 1.0 / rate_hz
    script = [
        ("EXTEND_ELBOW", 15), ("REST", 5), ("RETRACT_ELBOW", 15), ("REST", 5),
        ("EXTENSION", 15), ("FLEXION", 15), ("REST", 5),
        ("DOWNWARD_ROT_CCW", 15), ("DOWNWARD_ROT_CW", 15), ("REST", 5),
        ("FRONT_ROT_CW", 12), ("ULNAR_DEV", 12), ("RADIAL_DEV", 12), ("REST", 5),
        ("GRIP", 3), ("REST", 10), ("GRIP", 3), ("REST", 10),
    ]
    print("[emg_arm] DEMO: scripted gesture sequence")
    for gesture, ticks in script:
        print(f"\n--- {gesture} x{ticks} ---")
        for _ in range(ticks):
            bridge.set_gesture(gesture)
            bridge.tick()
            time.sleep(period)
    print("\n[emg_arm] demo done.")


def main():
    p = argparse.ArgumentParser(description="Drive the real arm from EMG gestures.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--stdin", action="store_true",
                     help="read gesture names from stdin (pipe inference_rt.py or ros2 topic echo)")
    src.add_argument("--demo", action="store_true",
                     help="run a scripted gesture sequence (use with --dry-run to test the map)")
    p.add_argument("--dry-run", action="store_true",
                   help="mock backend: print servo commands, move nothing")
    p.add_argument("--no-home", action="store_true", help="don't drive to home on startup")
    p.add_argument("--rate", type=float, default=20.0, help="control tick rate Hz (default 20)")
    p.add_argument("--step", type=float, default=cfg.DEFAULT_STEP_RAD,
                   help=f"rad per tick while a jog gesture is held, continuous mode (default {cfg.DEFAULT_STEP_RAD})")
    p.add_argument("--vote-window", type=float, default=0.0,
                   help="VOTE MODE: seconds to collect predictions, then fire the majority "
                        "gesture once (e.g. 3.0). 0 = continuous jog (default). Debounces a noisy classifier.")
    p.add_argument("--fire-step", type=float, default=0.20,
                   help="rad to move per FIRE in vote mode (default 0.20)")
    p.add_argument("--min-vote", type=float, default=0.34,
                   help="winner must be >= this fraction of the window to fire (default 0.34)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    arm = ArmDriver(dry_run=args.dry_run, verbose=not args.quiet)
    if not args.no_home:
        arm.go_home()
    bridge = EmgArmBridge(arm, step_rad=args.step, verbose=not args.quiet,
                          vote_window=args.vote_window, fire_step=args.fire_step,
                          min_vote_frac=args.min_vote)
    if args.vote_window > 0:
        print(f"[emg_arm] VOTE MODE: decide every {args.vote_window:.1f}s, "
              f"fire {args.fire_step:+.2f} rad if winner >= {args.min_vote:.0%}")

    try:
        if args.demo:
            run_demo(bridge, args.rate)
        else:
            run_stdin(bridge, args.rate)
    except KeyboardInterrupt:
        pass
    finally:
        arm.release()
        print("\n[emg_arm] released. bye.")


if __name__ == "__main__":
    main()

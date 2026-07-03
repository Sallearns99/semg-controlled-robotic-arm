#!/usr/bin/env python3
"""
servo_config.py  --  THE ONE FILE YOU CALIBRATE.

Maps each robot joint to PCA9685 channel(s) and converts joint angles
(radians, same convention as the Gazebo sim) into servo angles (degrees).

Hardware: 7x MG995 servos on a PCA9685 16-channel I2C PWM board, powered
from a SEPARATE 5-6 V supply (NOT the Pi's 5V rail).

Wiring map (edit `channel` numbers to match how you actually plugged them in):

    joint   PCA9685 ch   notes
    -----   ----------   -----
    j1          0        base rotation
    j2         1, 2      shoulder -- TWO servos, MIRRORED (load sharing)
    j3          3        elbow
    j4          4        wrist pitch
    j5          5        wrist roll
    j6          6        gripper (single servo; the sim's j6r finger is a
                         passive linkage, NOT a motor)

CALIBRATION WORKFLOW (do this on the Pi, arm supported by hand, see README):
    1. Run:  python3 servo_calibrate.py
    2. For each channel find: the pulse range, the `center_deg` that puts the
       joint at its sim "zero" pose, and whether `sign` needs flipping.
    3. Copy those numbers into JOINTS below.
"""

import math

RAD2DEG = 180.0 / math.pi

# --- Per-channel electrical calibration (MG995). -------------------------
# actuation_range / pulse widths are passed to the servo driver. MG995 reaches
# its full ~180-200 deg sweep around 500-2500 us; the conservative ServoKit
# default (1000-2000) only gives ~90 deg, so we widen it here. If a servo
# buzzes/strains at an endpoint, narrow the min/max for THAT channel.
SERVO_ACTUATION_RANGE = 180     # degrees of travel mapped across the pulse range
SERVO_PULSE_MIN_US = 500        # microseconds at 0 deg
SERVO_PULSE_MAX_US = 2500       # microseconds at SERVO_ACTUATION_RANGE deg

# --- Joint definitions ----------------------------------------------------
# Each joint maps to one or more channels. Per channel:
#   center_deg : servo angle (deg, 0..SERVO_ACTUATION_RANGE) when the JOINT is
#                at 0 rad. Start at 90 (mid travel) and trim during calibration.
#   sign       : +1 or -1. Flips the direction a channel turns. For the
#                MIRRORED j2 pair the two channels carry OPPOSITE signs so the
#                joint moves as one while the servos physically counter-rotate.
#   deg_per_rad: servo degrees per radian of joint motion. 57.2958 == direct
#                1:1 drive. Change if the joint is geared/belted.
#
# limit_rad : software joint limits (radians), mirroring the sim's URDF limits.
#             A real servo can't be "continuous", so j1/j5 are clamped here.
#
# Tip: after wiring, if a joint moves the WRONG way, flip every `sign` for that
# joint's channels (or flip the single sign for a 1-servo joint).

JOINTS = {
    "j1": {
        "channels": [{"channel": 0, "center_deg": 90.0, "sign": +1, "deg_per_rad": RAD2DEG}],
        "limit_rad": (-1.5708, 1.5708),
    },
    "j2": {  # TWO servos, MIRRORED: opposite signs, same magnitude.
        "channels": [
            {"channel": 1, "center_deg": 90.0, "sign": +1, "deg_per_rad": RAD2DEG},
            {"channel": 2, "center_deg": 90.0, "sign": -1, "deg_per_rad": RAD2DEG},
        ],
        "limit_rad": (-1.5708, 1.5708),
    },
    "j3": {
        "channels": [{"channel": 3, "center_deg": 90.0, "sign": +1, "deg_per_rad": RAD2DEG}],
        "limit_rad": (-1.5708, 1.5708),
    },
    "j4": {
        "channels": [{"channel": 4, "center_deg": 90.0, "sign": +1, "deg_per_rad": RAD2DEG}],
        "limit_rad": (-1.5708, 1.5708),
    },
    "j5": {
        "channels": [{"channel": 5, "center_deg": 90.0, "sign": +1, "deg_per_rad": RAD2DEG}],
        "limit_rad": (-1.5708, 1.5708),
    },
}

# --- Gripper (single servo, handled in servo DEGREES, not radians) --------
# The sim drives j6l in radians with a mimic'd second finger; on the real arm
# it's ONE servo. We just define the two physical endpoints and step between.
# Find OPEN_DEG / CLOSED_DEG with servo_calibrate.py.
GRIPPER = {
    "channel": 6,
    "open_deg": 60.0,     # servo angle for a fully OPEN gripper  (CALIBRATE)
    "closed_deg": 120.0,  # servo angle for a fully CLOSED gripper (CALIBRATE)
}

# --- Motion / safety defaults --------------------------------------------
HOME_RAD = {"j1": 0.0, "j2": 0.0, "j3": 0.0, "j4": 0.0, "j5": 0.0}
DEFAULT_STEP_RAD = 0.10       # per keypress for arm joints
GRIPPER_STEP_DEG = 8.0        # per keypress for the gripper
SLEW_DEG_PER_S = 120.0        # max servo speed when moving to a new target (rate limiting)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))

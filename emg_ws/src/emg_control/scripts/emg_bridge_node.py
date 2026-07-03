#!/usr/bin/env python3
"""emg_bridge_node.py — map EMG gestures to rotatedauto arm/gripper jogs.

Subscribes:
  /inference_result  std_msgs/String  JSON {gesture, confidence, probs}  (every hop, continuous)
  /detected_gesture  std_msgs/String  gesture name  (also accepted, for hand-testing)
  /joint_states      sensor_msgs/JointState  (seed targets from the real arm)
Publishes:
  /arm_controller/joint_trajectory      trajectory_msgs/JointTrajectory  (j1..j5)
  /gripper_controller/joint_trajectory  trajectory_msgs/JointTrajectory  (j6l)

Mapping is CONFIG-DRIVEN via the `gesture_map` parameter (see config/emg_params.yaml).
Each entry is "GESTURE:joint:sign"; while that gesture is held the joint jogs by
sign*step every tick. `grip_gesture` toggles the gripper (rising edge + cooldown).
Any gesture not in the map (incl. REST), low confidence, or signal loss -> HOLD.

Default map (12-gesture scheme):
  EXTENSION/FLEXION         -> j3 +/-      (wrist/forearm)
  EXTEND_ELBOW/RETRACT_ELBOW-> j2 +/-      (shoulder/elbow lift)
  DOWNWARD_ROT_CW/CCW       -> j1 -/+      (base turntable)
  FRONT_ROT_CW/CCW          -> j5 +/-      (wrist roll)
  ULNAR_DEV/RADIAL_DEV      -> j4 +/-      (wrist)
  GRIP                      -> toggle gripper

Vote mode (param vote_window > 0): instead of jogging continuously while a gesture is
held, collect predictions for `vote_window` seconds, then fire the MAJORITY gesture ONCE
(move `fire_step` rad / toggle gripper), provided the winner is >= `min_vote` of the window
and not REST. Debounces a noisy classifier. vote_window = 0 keeps continuous jog.

Test without EMG:
  ros2 topic pub -1 /detected_gesture std_msgs/msg/String "{data: DOWNWARD_ROT_CCW}"
  ros2 topic pub -1 /detected_gesture std_msgs/msg/String "{data: GRIP}"
"""
import json
import time
from collections import Counter

import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

ARM_JOINTS = ["j1", "j2", "j3", "j4", "j5"]
GRIPPER_JOINT = "j6l"
# joint limits (rad); None = continuous (no clamp). Matches rotatedauto.xacro.
LIMITS = {"j1": None, "j2": (-1.5708, 1.5708), "j3": (-1.5708, 1.5708),
          "j4": (-1.5708, 1.5708), "j5": None, "j6l": (-0.51, 0.288)}

# Default gesture -> "joint:sign" map (overridable via the gesture_map param).
DEFAULT_GESTURE_MAP = [
    "EXTENSION:j3:+1",
    "FLEXION:j3:-1",
    "EXTEND_ELBOW:j2:+1",
    "RETRACT_ELBOW:j2:-1",
    "DOWNWARD_ROT_CW:j1:-1",
    "DOWNWARD_ROT_CCW:j1:+1",
    "FRONT_ROT_CW:j5:+1",
    "FRONT_ROT_CCW:j5:-1",
    "ULNAR_DEV:j4:+1",
    "RADIAL_DEV:j4:-1",
]


def clamp(joint, val):
    lim = LIMITS[joint]
    return val if lim is None else max(lim[0], min(lim[1], val))


class EmgBridgeNode(Node):
    def __init__(self):
        super().__init__("emg_bridge_node")
        self.declare_parameter("gesture_map", DEFAULT_GESTURE_MAP)
        self.declare_parameter("grip_gesture", "GRIP")
        self.declare_parameter("step", 0.03)              # rad per tick
        self.declare_parameter("rate_hz", 20.0)
        self.declare_parameter("min_confidence", 0.0)     # 0 = act on any gesture (raise as model improves)
        self.declare_parameter("signal_timeout", 0.5)     # s with no gesture -> HOLD (deadman)
        self.declare_parameter("gripper_open", -0.51)
        self.declare_parameter("gripper_closed", 0.288)
        self.declare_parameter("grip_cooldown", 0.8)      # s; min time between gripper toggles
        self.declare_parameter("vote_window", 0.0)        # s; >0 = majority-vote over window then fire ONCE
        self.declare_parameter("fire_step", 0.20)         # rad per FIRE in vote mode
        self.declare_parameter("min_vote", 0.34)          # winner must be >= this fraction of the window
        self.declare_parameter("fire_time", 0.6)          # s to execute ONE vote-mode move (smooth, deliberate)

        self.grip_gesture = str(self.get_parameter("grip_gesture").value).strip().upper()
        self.step = float(self.get_parameter("step").value)
        rate = float(self.get_parameter("rate_hz").value)
        self.min_conf = float(self.get_parameter("min_confidence").value)
        self.timeout = float(self.get_parameter("signal_timeout").value)
        self.grip_open = float(self.get_parameter("gripper_open").value)
        self.grip_closed = float(self.get_parameter("gripper_closed").value)
        self.grip_cooldown = float(self.get_parameter("grip_cooldown").value)
        self.vote_window = float(self.get_parameter("vote_window").value)
        self.fire_step = float(self.get_parameter("fire_step").value)
        self.min_vote = float(self.get_parameter("min_vote").value)
        self.fire_time = float(self.get_parameter("fire_time").value)

        self.jog_map = self._parse_gesture_map(
            self.get_parameter("gesture_map").value)

        self.arm_pub = self.create_publisher(JointTrajectory, "/arm_controller/joint_trajectory", 10)
        self.grip_pub = self.create_publisher(JointTrajectory, "/gripper_controller/joint_trajectory", 10)
        self.create_subscription(String, "/inference_result", self._on_result, 10)
        self.create_subscription(String, "/detected_gesture", self._on_gesture, 10)
        self.create_subscription(JointState, "/joint_states", self._on_states, 10)

        self.targets = {j: 0.0 for j in ARM_JOINTS}
        self.targets[GRIPPER_JOINT] = self.grip_closed
        # `current` = the arm's ACTUAL live positions (from /joint_states). Used to
        # freeze every non-selected joint exactly where it is on each fire, so only
        # one joint can move at a time.
        self.current = dict(self.targets)
        self._seeded = False

        self.active = "REST"
        self.confidence = 1.0
        self.last_update = 0.0
        self.prev_effective = "REST"
        self.grip_closed_state = False   # gripper starts OPEN -> first GRIP CLOSES
        self.last_grip_toggle = 0.0
        self.votes = []
        self.window_start = time.time()

        self.create_timer(1.0 / rate, self._tick)
        pretty = ", ".join(f"{g}->{j}{'+' if s > 0 else '-'}"
                           for g, (j, s) in self.jog_map.items())
        self.get_logger().info(
            f"emg_bridge up. step={self.step} min_confidence={self.min_conf} "
            f"timeout={self.timeout}s grip_gesture={self.grip_gesture}.")
        self.get_logger().info(f"gesture map: {pretty}  (others HOLD)")
        if self.vote_window > 0:
            self.get_logger().info(
                f"VOTE MODE: decide every {self.vote_window:.1f}s, fire {self.fire_step:+.2f} rad "
                f"if winner >= {self.min_vote:.0%} (else hold).")

    def _parse_gesture_map(self, entries):
        """Parse ['GESTURE:joint:sign', ...] -> {GESTURE: (joint, sign_float)}.
        Bad entries are skipped with a warning so one typo can't kill the node."""
        out = {}
        for raw in (entries or []):
            try:
                gesture, joint, sign = str(raw).split(":")
                gesture = gesture.strip().upper()
                joint = joint.strip()
                sign = float(sign)
                if joint not in ARM_JOINTS:
                    raise ValueError(f"joint '{joint}' not in {ARM_JOINTS}")
                if sign == 0:
                    raise ValueError("sign must be non-zero")
                out[gesture] = (joint, 1.0 if sign > 0 else -1.0)
            except (ValueError, AttributeError) as e:
                self.get_logger().warn(f"gesture_map: skipping bad entry '{raw}' ({e})")
        return out

    def _on_states(self, msg):
        # Always track the live actual positions.
        for name, pos in zip(msg.name, msg.position):
            if name in self.current:
                self.current[name] = pos
        if not self._seeded:                     # seed targets from the real arm once
            for name, pos in zip(msg.name, msg.position):
                if name in self.targets:
                    self.targets[name] = pos
            self._seeded = True

    def _on_gesture(self, msg):
        self.active = msg.data.strip().upper()
        self.confidence = 1.0
        self.last_update = time.time()
        self._record_vote(self.active, 1.0)

    def _on_result(self, msg):
        try:
            d = json.loads(msg.data)
            self.active = str(d.get("gesture", "REST")).upper()
            self.confidence = float(d.get("confidence", 1.0))
            self.last_update = time.time()
            self._record_vote(self.active, self.confidence)
        except (ValueError, TypeError):
            pass

    def _record_vote(self, gesture, conf):
        """In vote mode, buffer each incoming prediction (confidence-gated)."""
        if self.vote_window > 0 and conf >= self.min_conf:
            self.votes.append(gesture)

    def _effective_gesture(self):
        if time.time() - self.last_update > self.timeout:
            return "REST"                       # deadman: stale signal
        if self.confidence < self.min_conf:
            return "REST"                       # confidence gate
        return self.active

    def _tick(self):
        """Timer callback. Dispatches to vote or continuous mode."""
        if self.vote_window > 0:
            self._vote_tick(time.time())
        else:
            self._continuous_tick(time.time())

    def _vote_tick(self, now):
        """Every vote_window seconds, take the majority gesture and fire ONCE."""
        if now - self.window_start < self.vote_window:
            return
        votes, self.votes = self.votes, []
        self.window_start = now
        if not votes:
            return
        winner, count = Counter(votes).most_common(1)[0]
        frac = count / len(votes)
        if winner == "REST" or frac < self.min_vote:
            self.get_logger().info(f"window: {winner} {count}/{len(votes)} ({frac:.0%}) -> hold")
            return
        if winner == self.grip_gesture:
            self.grip_closed_state = not self.grip_closed_state
            self.targets[GRIPPER_JOINT] = (self.grip_closed if self.grip_closed_state
                                           else self.grip_open)
            self._send_gripper()
            self.get_logger().info(
                f"window: {winner} ({frac:.0%}) -> gripper {'CLOSED' if self.grip_closed_state else 'OPEN'}")
        elif winner in self.jog_map:
            joint, sign = self.jog_map[winner]
            # ONE joint at a time: freeze every other arm joint at its ACTUAL
            # current position (stops anything still travelling from a prior fire),
            # and move only the selected joint by fire_step.
            for j in ARM_JOINTS:
                self.targets[j] = self.current.get(j, self.targets[j])
            self.targets[joint] = clamp(joint, self.current.get(joint, self.targets[joint])
                                        + sign * self.fire_step)
            self._send_arm(self.fire_time)
            self.get_logger().info(
                f"window: {winner} {count}/{len(votes)} ({frac:.0%}) -> FIRE {joint}={self.targets[joint]:+.3f} "
                f"(others frozen)")

    def _continuous_tick(self, now):
        g = self._effective_gesture()

        if g == self.grip_gesture:
            if (self.prev_effective != self.grip_gesture
                    and (now - self.last_grip_toggle) > self.grip_cooldown):
                # rising edge + cooldown -> toggle (chatter can't double-fire)
                self.last_grip_toggle = now
                self.grip_closed_state = not self.grip_closed_state
                self.targets[GRIPPER_JOINT] = (self.grip_closed if self.grip_closed_state
                                               else self.grip_open)
                self._send_gripper()
                self.get_logger().info(
                    f"gripper -> {'CLOSED' if self.grip_closed_state else 'OPEN'}")
        elif g in self.jog_map:
            joint, sign = self.jog_map[g]
            self.targets[joint] = clamp(joint, self.targets[joint] + sign * self.step)
            self._send_arm()
            self.get_logger().info(
                f"jog {g}: {joint}={self.targets[joint]:+.3f}", throttle_duration_sec=0.4)

        self.prev_effective = g

    def _send_arm(self, duration=0.15):
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = [self.targets[j] for j in ARM_JOINTS]
        pt.time_from_start.sec = int(duration)
        pt.time_from_start.nanosec = int((duration - int(duration)) * 1e9)
        traj.points.append(pt)
        self.arm_pub.publish(traj)

    def _send_gripper(self):
        traj = JointTrajectory()
        traj.joint_names = [GRIPPER_JOINT]
        pt = JointTrajectoryPoint()
        pt.positions = [self.targets[GRIPPER_JOINT]]
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 300_000_000
        traj.points.append(pt)
        self.grip_pub.publish(traj)


def main():
    rclpy.init()
    node = EmgBridgeNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()

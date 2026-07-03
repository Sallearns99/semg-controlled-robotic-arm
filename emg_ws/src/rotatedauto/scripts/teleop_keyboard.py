#!/usr/bin/env python3
"""Keyboard teleop for the rotatedauto arm + gripper (ROS 2 Jazzy).

Streams JointTrajectory goals to the arm_controller (j1-j5) and
gripper_controller (j6l). j6r follows j6l automatically via the URDF mimic.

Run Gazebo first, then in another sourced terminal:
    ros2 run rotatedauto teleop_keyboard.py

Controls (hold-and-tap):
    j1:  a / z      j2:  s / x      j3:  d / c
    j4:  f / v      j5:  g / b
    gripper:  o (open) / p (close)
    space : hold current pose (re-publish)
    r     : reset all joints to 0
    [ / ] : decrease / increase step size
    q     : quit
"""

import sys
import termios
import tty
import select
import threading

import rclpy
from rclpy.node import Node
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from sensor_msgs.msg import JointState

ARM_JOINTS = ['j1', 'j2', 'j3', 'j4', 'j5']
GRIPPER_JOINT = 'j6l'

# (lower, upper) limits; None = continuous (no clamp)
LIMITS = {
    'j1': None,
    'j2': (-1.5708, 1.5708),
    'j3': (-1.5708, 1.5708),
    'j4': (-1.5708, 1.5708),
    'j5': None,
    'j6l': (-0.51, 0.288),
}

# GRIPPER DIRECTION CONVENTION (verified in Gazebo 2026-06-02):
#   j6l = +0.288 (upper) -> gripper CLOSED
#   j6l = -0.51  (lower) -> gripper OPEN
# i.e. the joint sign is INVERTED vs. intuition. So 'o' (open) must DECREASE j6l.
# NOTE for MoveIt: set the gripper "open" group state to j6l ~ -0.51 and
# "close" to j6l ~ +0.288, or the gripper SRDF states will be backwards.

# key -> (joint, direction)
KEYMAP = {
    'a': ('j1', +1), 'z': ('j1', -1),
    's': ('j2', +1), 'x': ('j2', -1),
    'd': ('j3', +1), 'c': ('j3', -1),
    'f': ('j4', +1), 'v': ('j4', -1),
    'g': ('j5', +1), 'b': ('j5', -1),
    'o': ('j6l', -1), 'p': ('j6l', +1),   # o = OPEN (toward -0.51), p = CLOSE (toward +0.288)
}

HELP = __doc__


def clamp(joint, val):
    lim = LIMITS[joint]
    if lim is None:
        return val
    return max(lim[0], min(lim[1], val))


class TeleopKeyboard(Node):
    def __init__(self):
        super().__init__('teleop_keyboard')
        self.arm_pub = self.create_publisher(
            JointTrajectory, '/arm_controller/joint_trajectory', 10)
        self.grip_pub = self.create_publisher(
            JointTrajectory, '/gripper_controller/joint_trajectory', 10)

        # current target per joint
        self.targets = {j: 0.0 for j in ARM_JOINTS}
        self.targets[GRIPPER_JOINT] = 0.0
        self.step = 0.10  # rad per keypress
        self._got_state = False

        # seed targets from the real robot state if available
        self.create_subscription(JointState, '/joint_states', self._on_state, 10)

    def _on_state(self, msg):
        if self._got_state:
            return
        for name, pos in zip(msg.name, msg.position):
            if name in self.targets:
                self.targets[name] = pos
        self._got_state = True

    def _send_arm(self):
        traj = JointTrajectory()
        traj.joint_names = ARM_JOINTS
        pt = JointTrajectoryPoint()
        pt.positions = [self.targets[j] for j in ARM_JOINTS]
        pt.time_from_start.sec = 0
        pt.time_from_start.nanosec = 300_000_000  # 0.3 s
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

    def send_all(self):
        self._send_arm()
        self._send_gripper()

    def handle_key(self, key):
        if key in KEYMAP:
            joint, direction = KEYMAP[key]
            self.targets[joint] = clamp(joint, self.targets[joint] + direction * self.step)
            if joint == GRIPPER_JOINT:
                self._send_gripper()
            else:
                self._send_arm()
            self._print_state(joint)
        elif key == ' ':
            self.send_all()
            self.get_logger().info('hold pose')
        elif key == 'r':
            for j in self.targets:
                self.targets[j] = 0.0
            self.send_all()
            self.get_logger().info('reset to zero')
        elif key == ']':
            self.step = min(0.5, self.step + 0.02)
            self.get_logger().info(f'step = {self.step:.2f} rad')
        elif key == '[':
            self.step = max(0.01, self.step - 0.02)
            self.get_logger().info(f'step = {self.step:.2f} rad')

    def _print_state(self, joint):
        vals = '  '.join(f'{j}={self.targets[j]:+.2f}' for j in ARM_JOINTS)
        sys.stdout.write(
            f'\r[{joint}]  {vals}  grip={self.targets[GRIPPER_JOINT]:+.2f}  (step {self.step:.2f})   ')
        sys.stdout.flush()


def main():
    rclpy.init()
    node = TeleopKeyboard()

    # spin in background so /joint_states seeding works
    spin_thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
    spin_thread.start()

    print(HELP)
    print('\nReady. Press keys to move (q to quit).\n')

    old = termios.tcgetattr(sys.stdin)
    try:
        tty.setcbreak(sys.stdin.fileno())
        while rclpy.ok():
            r, _, _ = select.select([sys.stdin], [], [], 0.1)
            if not r:
                continue
            key = sys.stdin.read(1)
            if key == 'q' or key == '\x03':  # q or Ctrl-C
                break
            node.handle_key(key)
    finally:
        termios.tcsetattr(sys.stdin, termios.TCSADRAIN, old)
        print('\nbye')
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

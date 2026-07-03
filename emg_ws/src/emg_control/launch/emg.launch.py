#!/usr/bin/env python3
"""Launch EMG inference + bridge for the rotatedauto arm.

Run AFTER Gazebo (which provides the controllers):
    ros2 launch rotatedauto gazebo.launch.py
    ros2 launch emg_control emg.launch.py

Args:
    port:=/dev/ttyACM0      ESP serial port
    bridge_only:=true       skip inference node (hand-publish /detected_gesture to test)
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import UnlessCondition
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    cfg = os.path.join(get_package_share_directory('emg_control'), 'config', 'emg_params.yaml')
    port = LaunchConfiguration('port')
    bridge_only = LaunchConfiguration('bridge_only')
    replay_csv = LaunchConfiguration('replay_csv')
    replay_gesture = LaunchConfiguration('replay_gesture')
    replay_speed = LaunchConfiguration('replay_speed')
    use_retrained = LaunchConfiguration('use_retrained')
    retrained_model_path = LaunchConfiguration('retrained_model_path')
    vote_window = LaunchConfiguration('vote_window')

    return LaunchDescription([
        DeclareLaunchArgument('port', default_value='/dev/ttyACM0'),
        DeclareLaunchArgument('bridge_only', default_value='false',
                              description='Run only the bridge (hand-publish /detected_gesture)'),
        DeclareLaunchArgument('replay_csv', default_value='',
                              description='Path to a recorded CSV to replay instead of reading the ESP'),
        DeclareLaunchArgument('replay_gesture', default_value='-1',
                              description='gesture_id to replay (-1 = all)'),
        DeclareLaunchArgument('replay_speed', default_value='1.0'),
        DeclareLaunchArgument('use_retrained', default_value='true',
                              description='Use the retrained model slot (default: the CRNN). '
                                          'Set false to fall back to the old envelope classifier.'),
        DeclareLaunchArgument('retrained_model_path',
                              default_value='/home/salma/emg_ws/emg_ml/models/classical_6g.joblib',
                              description='Retrained model (default: the classical_6g RF). '
                                          'Swap to .../models/rcnn_6g.joblib for the CRNN.'),
        DeclareLaunchArgument('vote_window', default_value='3.0',
                              description='Bridge vote mode: collect predictions for this many s, then '
                                          'fire the MAJORITY gesture ONCE (one move / 3 s). 0 = continuous jog.'),
        Node(
            package='emg_control', executable='emg_inference_node.py',
            name='emg_inference_node', output='screen',
            parameters=[cfg, {'port': port, 'replay_csv': replay_csv,
                              'replay_gesture': replay_gesture, 'replay_speed': replay_speed,
                              'use_retrained': use_retrained,
                              'retrained_model_path': retrained_model_path}],
            condition=UnlessCondition(bridge_only),
        ),
        Node(
            package='emg_control', executable='emg_bridge_node.py',
            name='emg_bridge_node', output='screen',
            parameters=[cfg, {'vote_window': ParameterValue(vote_window, value_type=float)}],
        ),
    ])

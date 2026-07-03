#!/usr/bin/env python3
"""RViz2 + joint_state_publisher_gui for rotatedauto (no Gazebo).

j6r has a <mimic> tag, so the GUI will only show a slider for j6l;
j6r follows automatically. Use this to eyeball the model before Gazebo.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command


def generate_launch_description():
    pkg_share = get_package_share_directory('rotatedauto')
    xacro_file = os.path.join(pkg_share, 'urdf', 'rotatedauto.xacro')
    robot_description = {
        'robot_description': ParameterValue(
            Command(['xacro ', xacro_file]), value_type=str)
    }

    return LaunchDescription([
        Node(package='robot_state_publisher', executable='robot_state_publisher',
             output='screen', parameters=[robot_description]),
        Node(package='joint_state_publisher_gui', executable='joint_state_publisher_gui',
             output='screen'),
        Node(package='rviz2', executable='rviz2', output='screen',
             arguments=['-d', os.path.join(pkg_share, 'config', 'view.rviz')]
             if os.path.exists(os.path.join(pkg_share, 'config', 'view.rviz')) else []),
    ])

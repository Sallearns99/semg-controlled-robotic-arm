#!/usr/bin/env python3
"""MoveIt move_group + RViz MotionPlanning for rotatedauto.

Run this ALONGSIDE Gazebo (which provides the controllers + joint states):
    Terminal 1:  ros2 launch rotatedauto gazebo.launch.py
    Terminal 2:  ros2 launch rotatedauto_moveit_config moveit.launch.py

use_sim_time is true so MoveIt syncs with Gazebo's /clock.
"""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch.conditions import IfCondition
from launch_ros.actions import Node
from moveit_configs_utils import MoveItConfigsBuilder


def generate_launch_description():
    rotatedauto_xacro = os.path.join(
        get_package_share_directory('rotatedauto'), 'urdf', 'rotatedauto.xacro')

    moveit_config = (
        MoveItConfigsBuilder('rotatedauto', package_name='rotatedauto_moveit_config')
        .robot_description(file_path=rotatedauto_xacro)
        .robot_description_semantic(file_path='config/rotatedauto.srdf')
        .robot_description_kinematics(file_path='config/kinematics.yaml')
        .joint_limits(file_path='config/joint_limits.yaml')
        .trajectory_execution(file_path='config/moveit_controllers.yaml')
        .planning_pipelines(pipelines=['ompl'], default_planning_pipeline='ompl')
        .to_moveit_configs()
    )

    use_sim_time = {'use_sim_time': True}

    declare_rviz = DeclareLaunchArgument(
        'rviz', default_value='true', description='Launch RViz MotionPlanning')

    move_group = Node(
        package='moveit_ros_move_group',
        executable='move_group',
        output='screen',
        parameters=[
            moveit_config.to_dict(),
            use_sim_time,
            {'publish_robot_description_semantic': True},
        ],
    )

    rviz_config = os.path.join(
        get_package_share_directory('rotatedauto_moveit_config'), 'config', 'moveit.rviz')

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        condition=IfCondition(LaunchConfiguration('rviz')),
        parameters=[
            moveit_config.robot_description,
            moveit_config.robot_description_semantic,
            moveit_config.robot_description_kinematics,
            moveit_config.planning_pipelines,
            moveit_config.joint_limits,
            use_sim_time,
        ],
    )

    return LaunchDescription([declare_rviz, move_group, rviz])

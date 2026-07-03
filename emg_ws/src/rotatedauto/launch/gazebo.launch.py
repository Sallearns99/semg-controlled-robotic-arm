#!/usr/bin/env python3
"""Launch rotatedauto in Gazebo Harmonic with ros2_control (ROS 2 Jazzy)."""

import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument, IncludeLaunchDescription, RegisterEventHandler,
    SetEnvironmentVariable, ExecuteProcess,
)
from launch.event_handlers import OnProcessExit
from launch.conditions import IfCondition, UnlessCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution, Command
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    pkg = 'rotatedauto'
    pkg_share = get_package_share_directory(pkg)
    xacro_file = os.path.join(pkg_share, 'urdf', 'rotatedauto.xacro')

    # Help Gazebo Harmonic resolve package:// mesh URIs.
    resource_path = os.path.dirname(pkg_share)  # the .../share dir's parent of this pkg

    # gui:=false runs the gz server headless (no GUI window) — useful for CI / validation.
    gui = LaunchConfiguration('gui')
    declare_gui = DeclareLaunchArgument('gui', default_value='true',
                                        description='Launch the Gazebo GUI client (false = headless server only)')

    robot_description = {
        'robot_description': ParameterValue(
            Command(['xacro ', xacro_file]), value_type=str),
        'use_sim_time': True,
    }

    gz_sim_gui = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                  'launch', 'gz_sim.launch.py'])
        ]),
        launch_arguments={'gz_args': '-r -v 4 empty.sdf'}.items(),
        condition=IfCondition(gui),
    )
    gz_sim_headless = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            PathJoinSubstitution([FindPackageShare('ros_gz_sim'),
                                  'launch', 'gz_sim.launch.py'])
        ]),
        launch_arguments={'gz_args': '-s -r -v 4 empty.sdf'}.items(),
        condition=UnlessCondition(gui),
    )

    robot_state_publisher = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[robot_description],
    )

    spawn = Node(
        package='ros_gz_sim',
        executable='create',
        output='screen',
        arguments=['-topic', 'robot_description', '-name', 'rotatedauto', '-z', '0.0'],
    )

    clock_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[gz.msgs.Clock'],
        output='screen',
    )

    load_jsb = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'joint_state_broadcaster'],
        output='screen',
    )
    load_arm = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'arm_controller'],
        output='screen',
    )
    load_gripper = ExecuteProcess(
        cmd=['ros2', 'control', 'load_controller', '--set-state', 'active',
             'gripper_controller'],
        output='screen',
    )

    return LaunchDescription([
        declare_gui,
        SetEnvironmentVariable('GZ_SIM_RESOURCE_PATH', resource_path),
        gz_sim_gui,
        gz_sim_headless,
        robot_state_publisher,
        clock_bridge,
        spawn,
        # chain the controller spawners after the model is spawned
        RegisterEventHandler(OnProcessExit(target_action=spawn,
                                           on_exit=[load_jsb])),
        RegisterEventHandler(OnProcessExit(target_action=load_jsb,
                                           on_exit=[load_arm])),
        RegisterEventHandler(OnProcessExit(target_action=load_arm,
                                           on_exit=[load_gripper])),
    ])

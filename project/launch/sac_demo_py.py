"""SAC demo launch (physical car).

Paths: ~/f1tenth_ws/src/project/...
Runs the best checkpoint with the safety node. No training.

    ros2 launch project sac_demo_py.py
"""
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

_HOME = os.path.expanduser('~')
_PROJECT = os.path.join(_HOME, 'f1tenth_ws', 'src', 'project')


def generate_launch_description():
    return LaunchDescription([
        # safety node
        Node(
            package='project',
            executable='safety_node',
            output='screen',
            parameters=[PathJoinSubstitution([
                FindPackageShare('project'), 'config', 'safety_params.yaml'])
            ],
        ),
        # SAC demo node (inference only, best checkpoint)
        Node(
            package='project',
            executable='sac_demo_node',
            output='screen',
            parameters=[{
                'checkpoint_path': os.path.join(_PROJECT, 'sac', 'sac_checkpoint_best.pth'),
                'scalers_path': os.path.join(_PROJECT, 'processed', 'scalers.npz'),
                'max_speed': 1.0,
                'min_speed': 0.5,
            }],
        ),
    ])

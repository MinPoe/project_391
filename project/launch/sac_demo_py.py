"""
SAC demo launch (physical car).

Paths: ~/f1tenth_ws/src/project/...
Runs the best checkpoint with the safety node. No training. This file is designed to be used with the physical car.

Launch:
    ros2 launch project sac_demo_py.py
"""
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():
    """ 
    Generate the launch description for the SAC demo node.

    Args:
        None

    Returns:
        The launch description.
    """
    share = get_package_share_directory('project')

    return LaunchDescription([
        # safety node
        Node(
            package='project',
            executable='safety_node',
            output='screen',
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('project'), 'config', 'safety_params.yaml']),
                {'odom_topic': '/odom'},
            ],
        ),
        # SAC demo node (inference only, best checkpoint)
        Node(
            package='project',
            executable='sac_demo_node',
            output='screen',
            parameters=[{
                'checkpoint_path': os.path.join(share, 'sac', 'sac_checkpoint_best.pth'),
                'scalers_path': os.path.join(share, 'processed', 'scalers.npz'),
                'max_speed': 1.0,
                'min_speed': 0.7,
            }],
        ),
    ])

"""Launch file for the Behavioural Cloning model.

Launches the safety node and the BC inference node.
"""

import os
from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

# Get the share directory
_SHARE = get_package_share_directory('project')


def generate_launch_description() -> LaunchDescription:
    """
    Generate the launch description for the Behavioural Cloning model.

	Args:
		None

    Returns:
        The launch description.
    """
    # Return the launch description
    return LaunchDescription([
        # Safety node
        Node(
            package='project',
            executable='safety_node',
            output='screen',
            parameters=[
                PathJoinSubstitution([
                    FindPackageShare('project'), 'config', 'safety_params.yaml']),
                {'odom_topic': '/ego_racecar/odom'}, # Can change this to /odom if using physical car
            ],
        ),
        # BC inference node
        Node(
            package='project',
            executable='bc_inference_node',
            output='screen',
            parameters=[{
                'model_path': os.path.join(_SHARE, 'bc', 'bc_model.pth'), # Load the BC model
                'scalers_path': os.path.join(_SHARE, 'processed', 'scalers.npz'), # Load the scalers
                'max_speed': 2.0,
            }],
        ),
    ])

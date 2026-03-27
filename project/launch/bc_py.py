import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

_SHARE = get_package_share_directory('project')


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
        # BC inference node
        Node(
            package='project',
            executable='bc_inference_node',
            output='screen',
            parameters=[{
                'model_path': os.path.join(_SHARE, 'bc', 'bc_model_sim.pth'),
                'scalers_path': os.path.join(_SHARE, 'processed', 'processed_simulator', 'scalers.npz'),
                'max_speed': 0.5,
            }],
        ),
    ])

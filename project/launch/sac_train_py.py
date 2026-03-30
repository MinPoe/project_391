"""SAC training launch (simulator).

Paths: ~/sim_ws/src/Project_C10/project/...

    ros2 launch project sac_train_py.py
"""
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

_HOME = os.path.expanduser('~')
_PROJECT = os.path.join(_HOME, 'sim_ws', 'src', 'Project_C10', 'project')


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
        # SAC training node
        Node(
            package='project',
            executable='sac_train_node',
            output='screen',
            parameters=[{
                'bc_weights_path': os.path.join(_PROJECT, 'bc', 'bc_model_sim.pth'),
                'scalers_path': os.path.join(_PROJECT, 'processed', 'scalers.npz'),
                'checkpoint_path': os.path.join(_PROJECT, 'sac', 'sac_checkpoint.pth'),
                'log_path': os.path.join(_PROJECT, 'sac', 'training_log.csv'),
                'max_speed': 2.0,
                'min_speed': 0.7,
                'deterministic': False,
            }],
        ),
    ])

import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

# Locate installed package data so paths work regardless of CWD.
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
        # SAC inference + training node
        Node(
            package='project',
            executable='sac_inference_node',
            output='screen',
            parameters=[{
                'bc_weights_path': os.path.join(_SHARE, 'bc', 'bc_model_sim.pth'),
                'scalers_path': os.path.join(_SHARE, 'processed', 'processed_simulator', 'scalers.npz'),
                'checkpoint_path': os.path.join(_SHARE, 'sac', 'sac_checkpoint.pth'),
                'log_path': os.path.join(_SHARE, 'sac', 'training_log.csv'),
                'max_speed': 2.0,
                'min_speed': 0.5,
                'max_steering': 0.4189,
                'training': True,
                'deterministic': False,
                'lr': 3e-4,
                'gamma': 0.99,
                'tau': 0.005,
                'buffer_size': 100000,
                'batch_size': 256,
                'update_every': 4,
                'warmup_steps': 0,
                'save_every': 5000,
            }],
        ),
    ])

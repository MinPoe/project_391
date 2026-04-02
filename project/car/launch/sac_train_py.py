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
        # SAC training node
        Node(
            package='project',
            executable='sac_train_node',
            output='screen',
            parameters=[{
                'bc_weights_path': os.path.join(_PROJECT, 'bc', 'bc_model_sim.pth'),
                'scalers_path': os.path.join(_PROJECT, 'processed', 'scalers.npz'),
                'initial_checkpoint_path': '',
                #'initial_checkpoint_path': os.path.join(
                #    _PROJECT, 'bc', 'bc_model_sim.pth'),
                'checkpoint_path': os.path.join(_PROJECT, 'sac', 'sac_checkpoint.pth'),
                'log_path': os.path.join(_PROJECT, 'sac', 'training_log.csv'),
                'max_speed': 1.0,
                'min_speed': 0.7,
                'deterministic': False,
                'resume_training': False,
                'lr_actor': 1e-4,
                'lr_critic': 3e-4,
                'warmup_steps': 2000,
                'learning_starts': 2000,
                'actor_learning_starts': 10000,
                'bc_reg_weight': 2.0,
                'bc_reg_decay_steps': 50000,
            }],
        ),
    ])

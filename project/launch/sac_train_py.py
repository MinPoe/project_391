"""
SAC training launch (simulator).

Paths: ~/sim_ws/src/Project_C10/project/...
Runs the SAC training with the safety node.

Launch:
    ros2 launch project sac_train_py.py
"""
import os

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from ament_index_python.packages import get_package_share_directory

# Get the share directory and source directory
_SHARE = get_package_share_directory('project')
_SRC = os.path.join(os.path.expanduser('~'), 'f1tenth_ws', 'src', 'project')


def generate_launch_description() -> LaunchDescription:
    """
    Generate the launch description for the SAC training.

    Returns:
        The launch description.
    """
    # Return the launch description
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
        # SAC training node
        Node(
            package='project',
            executable='sac_train_node',
            output='screen',
            parameters=[{
                'bc_weights_path': os.path.join(_SHARE, 'bc', 'bc_model_sim.pth'),
                'scalers_path': os.path.join(_SHARE, 'processed', 'scalers.npz'),
                # Start from the BC policy, then begin SAC updates after warmup.
                'initial_checkpoint_path': os.path.join(_SRC, 'sac', 'sac_checkpoint_best.pth'),
                'checkpoint_path': os.path.join(_SRC, 'sac', 'sac_checkpoint.pth'),
                'log_path': os.path.join(_SRC, 'sac', 'training_log.csv'),
                'max_speed': 1.2,
                'min_speed': 0.7,
                'deterministic': False,
                'resume_training': False,
                'lr_actor': 1e-4,
                'lr_critic': 3e-4,
                'warmup_steps': 4000,
                'learning_starts': 4000,
                'actor_learning_starts': 12000,
                'bc_reg_weight': 5.0,
                'bc_reg_decay_steps': 150000,
            }],
        ),
    ])

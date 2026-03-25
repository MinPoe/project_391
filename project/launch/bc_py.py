from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


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
                'model_path': 'bc/bc_model_sim.pth',
                'scalers_path': 'processed/processed_simulator/scalers.npz',
                'max_speed': 0.5,
            }],
        ),
    ])

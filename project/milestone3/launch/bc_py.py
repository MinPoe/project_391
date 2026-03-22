from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare


def generate_launch_description():
    return LaunchDescription([
        # safety node
        Node(
            package='milestone3',
            executable='safety_node',
            output='screen',
            parameters=[PathJoinSubstitution([
                FindPackageShare('milestone3'), 'config', 'safety_params.yaml'])
            ],
        ),
        # BC inference node
        Node(
            package='milestone3',
            executable='bc_inference_node',
            output='screen',
            parameters=[{
                'model_path': 'bc/bc_model.pth',
                'scaler_lidar_path': 'processed/processed_simulator/scaler_lidar.pkl',
                'scaler_action_path': 'processed/processed_simulator/scaler_action.pkl',
                'max_speed': 1.0,
            }],
        ),
    ])

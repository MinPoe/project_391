from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

"""
This file contains the launch code required for milestone 3.
"""

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
        # wall following node
        Node(
            package='milestone3',
            executable='cam_node',
            output='screen',
            parameters=[PathJoinSubstitution([
                FindPackageShare('milestone3'), 'config', 'cam_params.yaml'])
            ],
        ),
        # lap counter node
        Node(
            package='milestone3',
            executable='lap_counter',
            output='screen',
            parameters=[PathJoinSubstitution([
                FindPackageShare('milestone3'), 'config', 'lap_counter_params.yaml'])
            ],
        ),
    ])

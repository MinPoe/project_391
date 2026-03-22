from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

"""
This file contains the launch code required for milestone 2.
On running: ros2 launch milestone1 milestone2_py.py, it runs both the gap following node as well as the safety node.
Code structure was obtained from the official ros2 docs.
"""

def generate_launch_description():
    return LaunchDescription([
        # safety node
        Node( 
            package='milestone3',
            executable='safety_node_old',
            output='screen',
            parameters=[PathJoinSubstitution([
                FindPackageShare('milestone3'), 'config', 'safety_params_old.yaml'])
            ],
        ),
        # gap following node
        Node(
            package='milestone3',
            executable='gap_node_old',
            output='screen',
            parameters=[PathJoinSubstitution([
                FindPackageShare('milestone3'), 'config', 'gap_follow_params_old.yaml'])
            ],
        ),
    ])
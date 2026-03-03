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
            output='screen'
        ),
        # wall following node
        Node(
            package='milestone3',
            executable='cam_node',
            output='screen'
        ),
        # lap counter node
        Node(
            package='milestone3',
            executable='lap_counter',
            output='screen'
        ),
    ])

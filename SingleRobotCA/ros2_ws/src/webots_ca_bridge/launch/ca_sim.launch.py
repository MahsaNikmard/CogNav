import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg = get_package_share_directory('webots_ca_bridge')
    rviz_config = os.path.join(pkg, 'config', 'ca_rviz.rviz')

    bridge = Node(
        package    = 'webots_ca_bridge',
        executable = 'bridge_node',
        name       = 'webots_ca_bridge',
        output     = 'screen',
        emulate_tty= True,
    )

    rviz = Node(
        package    = 'rviz2',
        executable = 'rviz2',
        name       = 'rviz2',
        arguments  = ['-d', rviz_config],
        output     = 'screen',
    )

    return LaunchDescription([bridge, rviz])

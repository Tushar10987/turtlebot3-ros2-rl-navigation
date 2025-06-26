from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    # Path to the TurtleBot3 Gazebo launch file
    turtlebot3_gazebo = get_package_share_directory('turtlebot3_gazebo')

    # Path to your custom world
    world_file = os.path.join(
        get_package_share_directory('gym_rl_nav'),
        'worlds',
        'nav_world.world'
    )

    return LaunchDescription([
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(
                os.path.join(turtlebot3_gazebo, 'launch', 'turtlebot3_world.launch.py')
            ),
            launch_arguments={'world': world_file}.items()
        )
    ])


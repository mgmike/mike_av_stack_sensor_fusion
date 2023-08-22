import os

from ament_index_python.packages import get_package_share_directory


from launch_ros.substitutions import FindPackageShare

from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution, TextSubstitution

from launch_ros.actions import Node

viz = 'true'

# Carla configs
host = 'localhost'
port = '2000'
timeout = "10"

# Ego vehicle
role_name ='ego_vehicle'
vehicle_filter ='vehicle.audi.etron'
spawn_point ="65.516594,7.808423,0.275307,0.0,0.0,0.0"
# use comma separated format "x,y,z,roll,pitch,yaw"

# Map to load on startup (either a predefined CARLA town (e.g. 'Town01'), or a OpenDRIVE map file)
town = 'Town03'

# Synchronous mode
synchronous_mode = ''
synchronous_mode_wait_for_vehicle_control_command = ''
fixed_delta_seconds = '0.008333333'

# Localization settings
map_name = "map.pcd"
# Options: icp, ndt, icps -
scan_matching_algorithm = "icp"
iters = "100"

def generate_launch_description():
    carla_ros_bridge = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('carla_ros_bridge')),
            '/carla_ros_bridge.launch.py']),
        launch_arguments={
            'host': host,
            'port': port,
            'town': town,
            'timeout': timeout,
            # 'synchronous_mode': synchronous_mode,
            # 'synchronous_mode_wait_for_vehicle_control_command': synchronous_mode_wait_for_vehicle_control_command,
            'fixed_delta_seconds': fixed_delta_seconds
            }.items())
    carla_spawn_objects = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([os.path.join(
            get_package_share_directory('carla_spawn_objects')),
            '/carla_spawn_objects.launch.py']),
        launch_arguments={
            'objects_definition_file': os.path.join(
                get_package_share_directory('mike_av_stack_sensor_fusion'),'configs','objects.json')
            }.items()
        )
    rviz = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', os.path.join(get_package_share_directory('mike_av_stack_sensor_fusion'), 'configs','config.rviz')]
    )
   
    return LaunchDescription([
        carla_ros_bridge,
        carla_spawn_objects,
        rviz
    ])
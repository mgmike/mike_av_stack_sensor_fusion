# mike_av_stack_sensor_fusion

Requrements

easydict
sensor_msgs
vision_msgs
open3d
ros2_numpy added as a module
yolov7
docker

Make sure you have the correct drivers and docker versions for nvidia integration
https://docs.nvidia.com/ai-enterprise/deployment-guide-vmware/0.1.0/docker.html

Add nvidia runtime
https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/user-guide.html


Commands:


cd ../../
source /opt/ros/foxy/setup.bash
colcon build --packages-select mike_av_stack_sensor_fusion
source install/setup.bash
ros2 run mike_av_stack_sensor_fusion sensor_fusion



To run:

Start Carla (Host machine)
$ ./CarlaUE4.sh

Start carla ros bridge (Host machine)
$ export ROS_DOMAIN_ID=1
$ colcon build --packages-select mike_av_stack_sensor_fusion
$ source install/setup.bash
$ ros2 launch mike_av_stack_sensor_fusion av_sf.launch.py

Start mike AV sensor fusion node (Docker container)
$ source install/setup.bash
$ ros2 run mike_av_stack_sensor_fusion sensor_fusion --ros-args --log-level debug
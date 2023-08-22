# mike_av_stack_sensor_fusion

Requrements

easydict
sensor_msgs
vision_msgs
open3d
ros2_numpy added as a module
yolov7

Commands:

pip install -r requirements.txt
pip3 install open3d==0.17.0
<!-- pip3 install carla==0.9.13 -->

cd ../../
source /opt/ros/foxy/setup.bash
colcon build --packages-select mike_av_stack_sensor_fusion
source install/setup.bash
ros2 run mike_av_stack_sensor_fusion sensor_fusion



To run:

Start Carla (Host machine)
$ ./CarlaUE4.sh

Start carla ros bridge (Host machine)
$ colcon build --packages-select mike_av_stack_sensor_fusion
$ ros2 launch mike_av_stack_sensor_fusion av_sf.launch.py

Start mike AV sensor fusion node (Docker container)
$ ros2 run mike_av_stack_sensor_fusion sensor_fusion
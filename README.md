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
pip3 install open3d>=0.17.0

source /opt/ros/foxy/setup.bash
colcon build --packages-select mike_av_stack_sensor_fusion
source install/setup.bash
ros2 run mike_av_stack_sensor_fusion sensor_fusion
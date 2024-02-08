
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
import threading
import os

import sys
print(sys.path)

import json
print(json.__file__)
from easydict import EasyDict as edict

from sensor_msgs.msg import Image, PointCloud2
import vision_msgs
from vision_msgs.msg import Detection3DArray
# from tracking.trackmanagement import Trackmanagement
from .tracking.measurements import Sensor, Lidar, Camera
# from tracking.trackmanagement import Trackmanagement

# from mike_av_stack_sensor_fusion.tracking.measurements import Sensor, Lidar, Camera

from ament_index_python.packages import get_package_share_directory

package_name = 'mike_av_stack_sensor_fusion'

class SFTest(Node):
    def __init__(self):
        super().__init__('sensor_fusion_test')
        sub_cb_group = ReentrantCallbackGroup()
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )
        self.subscriber_pc = self.create_subscription(
            msg_type=PointCloud2,
            topic="/carla/ego_vehicle/lidar1", 
            callback=self.detection_callback,
            qos_profile=qos_profile,
            callback_group=sub_cb_group
            )
        self.subscriber_pc

    def detection_callback(self, point_cloud):
        self.get_logger().info('Got pointcloud')

def get_sensor(sensor, trackmanager, executor):
    name = sensor.type.split('.')[1]
    name = sensor.type.split('.')[2] if 'other' in name else name
    result = None
    if name == 'lidar':
        result = Lidar(name, sensor, trackmanager)
        executor.add_node(result)
    elif name == 'camera':
        result = Camera(name, sensor, trackmanager)
        executor.add_node(result)
    return result

def main(args=None):
    
    curr_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = os.path.abspath(os.path.join(curr_path, os.pardir))
    share_path = get_package_share_directory(package_name=package_name)
    print("current path: ", curr_path)  
    print("share path: ", share_path)

    sensors_j = edict()
    # Create edict json object of all the sensors in sensors.json

    with open(os.path.join(share_path, 'configs', 'sensors.json')) as j_object:
        sensors_j.update(json.load(j_object))

    # print(sensors_j.sensors)

    # trackmanager = Trackmanagement()

    rclpy.init(args=args)
    executor = MultiThreadedExecutor()

    # Create list of Sensors
    sensors = {sensor.id : get_sensor(sensor, None, executor=executor) for sensor in sensors_j.sensors}
    logger = None
    if logger is None and len(sensors) > 0:
        for sensor in sensors.values():
            if sensor is Lidar or sensor is Camera:
                logger = sensor.get_logger()

    # node1 = SFTest()

    # executor.add_node(node1)

    # spin() simply keeps python from exiting until this node is stopped
    # executor_thread = threading.Thread(target=executor.spin, daemon=True)
    # executor_thread.start()
    
    try:
        print('Beginning client, shut down with CTRL-C')
        if logger is not None:
            logger.info('Beginning client, shut down with CTRL-C')
        executor.spin()
    except KeyboardInterrupt:
        print('Keyboard interrupt, shutting down.\n')
        if logger is not None:
            logger.info('Keyboard interrupt, shutting down.\n')
    for sensor in sensors.values():
        print(f'sensor {sensor}')
        if sensor is Lidar or sensor is Camera:
            sensor.destroy_node()
    rclpy.shutdown()
    return

    
if __name__ == '__main__':
    main()
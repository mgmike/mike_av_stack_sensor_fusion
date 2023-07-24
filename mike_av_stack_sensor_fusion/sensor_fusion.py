#!/home/mike/anaconda3/envs/waymo/bin/python3

import rclpy
from rclpy.node import Node
import threading
import os
import json
from easydict import EasyDict as edict

from sensor_msgs.msg import Image, PointCloud2
import vision_msgs
from vision_msgs.msg import Detection3DArray
# from tracking.trackmanagement import Trackmanagement
from tracking.measurements import Sensor, Lidar, Camera
# from tracking.trackmanagement import Trackmanagement
    

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
    
    sensors_j = edict()
    # Create edict json object of all the sensors in sensors.json
    with open(os.path.join(curr_path, 'configs', 'sensors.json')) as j_object:
        sensors_j.update(json.load(j_object))

    # print(sensors_j.sensors)

    # trackmanager = Trackmanagement()

    rclpy.init(args=args)
    executor = rclpy.Executor()

    # Create list of Sensors
    sensors = {sensor.id : get_sensor(sensor, trackmanager, executor=executor) for sensor in sensors_j.sensors}

    # spin() simply keeps python from exiting until this node is stopped
    executor_thread = threading.Thread(target=executor.spin, daemon=True)
    executor_thread.start()
    # try:
    #     while rclpy.ok():
    #         print('Help me body, you are my only hope')
    #         rate.sleep()
    # except KeyboardInterrupt:
    #     pass
    # rclpy.shutdown()
    # executor_thread.join()

    
if __name__ == '__main__':
    main()
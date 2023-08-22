
# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Classes for sensor and measurement 
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy


from std_msgs.msg import Header
from sensor_msgs.msg import Image, PointCloud2
from vision_msgs.msg import BoundingBox3D, ObjectHypothesisWithPose, Detection3D, Detection3DArray, Detection2D, Detection2DArray
from geometry_msgs.msg import Pose, Point, Vector3, Quaternion
# from tf.transformations import quaternion_from_euler, euler_from_quaternion
from ..detection import objdet_pcl as pcl
from ..detection import objdet_detect as odet
# from ..detection.objdet_models.yolov7.yolov7 import Yolov7
import numpy as np
import time
from tracking.trackmanagement import Measurement, LidarMeasurement, CameraMeasurement

import os
import sys
dir_tracking = os.path.dirname(os.path.realpath(__file__))
dir_sf = os.path.dirname(dir_tracking)
dir_scripts = os.path.dirname(dir_sf)
sys.path.append(dir_scripts)
from tools.ros_conversions.transformations import quaternion_from_euler
from ros2_numpy.ros2_numpy.point_cloud2 import pointcloud2_to_array
import cv2
import json


class Sensor(Node):
    '''Sensor class including measurement matrix'''
    def __init__(self, name, configs, trackmanager):
        super().__init__('sensor_fusion')
        self.verbose = True
        self.configs = configs
        self.name = name
        self.trackmanager = trackmanager

        self.frame_id = 0

    def detection_callback(self):
        """ Override this for subscriber """

    def track_manage_callback(self):
        """ Override this for trackmanagement. Update Track list from Trackmanagement"""
    
    def in_fov(self, x):
        """ check if an object x can be seen by this sensor """
        
             
    def get_hx(self, x):    
        """ calculate nonlinear measurement expectation value h(x) """

        
    def get_H(self, x, params):
        """ calculate Jacobian H at current x from h(x) """

class Lidar(Sensor):
    def __init__(self, name, configs, trackmanager):
        super().__init__(name, configs, trackmanager)

        self.configs.update(odet.load_configs())
        self.model = odet.create_model(self.configs)
        self.configs.fov = [-np.pi/2, np.pi/2] # angle of field of view in radians
        self.configs.dim_meas = 3

        # Set up transforms
        self.sens_to_veh = np.matrix(np.identity((4))) # transformation sensor to vehicle coordinates equals identity matrix because lidar detections are already in vehicle coordinates
        print(type(self.sens_to_veh))
        self.veh_to_sens = np.linalg.inv(self.sens_to_veh) # transformation vehicle to sensor coordinates

        self.pub_detection = self.create_publisher(
            Detection3DArray, 
            "/sensor_fusion/detection/lidar/" + self.name,
            10
            )

        # Set up ros subscribers
        sub_cb_group = ReentrantCallbackGroup()
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )
        self.subscriber_pc = self.create_subscription(
            msg_type=PointCloud2,
            topic="/carla/ego_vehicle/lidar1/", 
            callback=self.detection_callback,
            qos_profile=qos_profile,
            callback_group=sub_cb_group
            )
        self.subscriber_det = self.create_subscription(
            msg_type=Detection3DArray,
            topic="/sensor_fusion/detection/lidar/", 
            callback=self.track_manage_callback,
            qos_profile=qos_profile,
            callback_group=sub_cb_group
            )
        self.subscriber_pc
        self.subscriber_det
        
    def detection_callback(self, point_cloud):
        if self.verbose:
            self.get_logger().info('Got pointcloud')

        point_cloud_2d = self.get_point_cloud_2d(point_cloud)

        bev = pcl.bev_from_pcl(point_cloud_2d, self.configs)
        detections = odet.detect_objects(bev, self.model, self.configs)

        if self.verbose:
            print(len(detections))

        dets = []
        for det in detections:
            d3d = Detection3D()
                            # Unpack the list 
            ori = Quaternion(*(quaternion_from_euler(0, 0, det[7])))
            d3d.bbox.center.orientation = ori
            d3d.bbox.center.position.x = det[1]
            d3d.bbox.center.position.y = det[2]
            d3d.bbox.center.position.z = det[3]
            d3d.bbox.size.x = det[4]
            d3d.bbox.size.y = det[5]
            d3d.bbox.size.z = det[6]
            dets.append(d3d)

        self.frame_id += 1
        detection3DArray = Detection3DArray()
        # detection3DArray.header.frame_id = self.frame_id 
        detection3DArray.header.stamp = self.get_clock().now().to_msg()
        detection3DArray.detections = dets
        self.pub_detection.publish(detection3DArray)

    def get_point_cloud_2d(self, pointcloud):
        # Convert the data from a 1d list of uint8s to a 2d list
        field_len = len(pointcloud.data)

        # The result of this is <vector<vector>> where [[]]
        point_cloud_2d = np.array([np.array(x.tolist()) for x in pointcloud2_to_array(pointcloud)])
    
        if self.verbose:
            print("Shape of pc2d: ", point_cloud_2d.shape, " First element: ", type(point_cloud_2d[0]), point_cloud_2d[0])
            print("First og: ", pointcloud.data[0], ", ", pointcloud.data[1], ", ", pointcloud.data[2], ", ", pointcloud.data[3])
            print("height: %d, width: %d, length of data: %d" % (pointcloud.height, pointcloud.width, field_len))
            for field in pointcloud.fields:
                print("\tname: ", field.name, "offset: ", field.offset, "datatype: ", field.datatype, "count: ", field.count)

        # TODO: Will need to transform to vehicle coordinate system

        # perform coordinate conversion
        # xyz_sensor = np.stack([x,y,z,np.ones_like(z)])
        # xyz_vehicle = np.einsum('ij,jkl->ikl', extrinsic, xyz_sensor)
        # xyz_vehicle = xyz_vehicle.transpose(1,2,0)

        # transform 3d points into vehicle coordinate system
        # pcl = xyz_vehicle[ri_range > 0,:3]
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(pcl)
        # o3d.visualization.draw_geometries([pcd])

        return point_cloud_2d


    def track_manage_callback(self, detection3DArray):
        if self.verbose:
            self.get_logger().info('Got lidar detection')

        meas_list = []
        for detection in detection3DArray.detections:
            time = detection.header.stamp
            meas = LidarMeasurement(self, time, detection, self.trackmanager.params)
            meas_list.append(meas)

        self.trackmanager.association.associate_and_update(self.trackmanager, meas_list, self.trackmanager.filter)

    def in_fov(self, x):
        x_s = x[0:4]
        x_s[3] = 1
        x_v = self.veh_to_sens * x_s
        if x_v[0] != 0:
            alpha = np.arctan(x_v[1] / x_v[0])
            if alpha > self.configs.fov[0] and alpha < self.configs.fov[1]:
                return True

        return False

    def get_hx(self, x):
        pos_veh = np.ones((4, 1)) # homogeneous coordinates
        pos_veh[0:3] = x[0:3] 
        pos_sens = self.veh_to_sens*pos_veh # transform from vehicle to lidar coordinates
        return pos_sens[0:3]


    def get_H(self, x, params):
        H = np.matrix(np.zeros((self.configs.dim_meas, params.dim_state)))
        R = self.veh_to_sens[0:3, 0:3] # rotation
        T = self.veh_to_sens[0:3, 3] # translation
        H[0:3, 0:3] = R
        return H
    

class Camera(Sensor):
    def __init__(self, name, configs, trackmanager):
        super().__init__(name, configs, trackmanager)

        # Add yolo configs
        # with open(os.path.join(dir_sf, 'configs', 'yolov7.json')) as j_object:
        #     configs.yolov7 = json.load(j_object)

        # self.init_yolo()

        sub_cb_group = ReentrantCallbackGroup()
        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.RMW_QOS_POLICY_RELIABILITY_BEST_EFFORT,
            history=QoSHistoryPolicy.RMW_QOS_POLICY_HISTORY_KEEP_LAST,
            depth=1
        )
        self.subscription_det = self.create_subscription(
            msg_type=Image,
            topic=self.configs.base_topic, 
            callback=self.detection_callback,
            qos_profile=qos_profile,
            callback_group=sub_cb_group
            )
        self.subscription_det

        self.configs.fov = [-0.35, 0.35] # angle of field of view in radians, inaccurate boundary region was removed

        self.configs.dim_meas = 2
        if 'calib' not in self.configs :
            self.get_logger().info('No calibration settings in the sensors.json file. Consider adding them')
            return

        calib = self.configs.calib

        self.sens_to_veh = np.matrix(calib.extrinsic.transform).reshape(4,4) # transformation sensor to vehicle coordinates
        self.f_i = calib.intrinsic[0] # focal length i-coordinate
        self.f_j = calib.intrinsic[1] # focal length j-coordinate
        self.c_i = calib.intrinsic[2] # principal point i-coordinate
        self.c_j = calib.intrinsic[3] # principal point j-coordinate

        self.veh_to_sens = np.linalg.inv(self.sens_to_veh) # transformation vehicle to sensor coordinates

        self.pub_detection = self.create_publisher(
            Detection2DArray,
            "/sensor_fusion/detection/camera/" + self.configs.id,
            10
            )
        
    # def init_yolo(self):
    #     self.get_logger().info(f'Initializing Yolov7, cv version: {cv2.__version__}')
    #     self.yolo = Yolov7(self.configs.yolov7)


    def detection_callback(self, image):
        # self.yolo.detect(image)
        print("detection")

    def track_manage_callback(self, detection2DArray):        
        meas_list = []
        for detection in detection2DArray.detections:
            time = detection.header.stamp
            frame_id = detection.header.frame_id
            meas = CameraMeasurement(self, time, detection, self.configs)
            meas_list.append(meas)

        self.trackmanager.association.associate_and_update(self.trackmanager, meas_list, self.trackmanager.filter)


    def get_hx(self, x):
        ############
        # Step 4: implement nonlinear camera measurement function h:
        # - transform position estimate from vehicle to camera coordinates
        # - project from camera to image coordinates
        # - make sure to not divide by zero, raise an error if needed
        # - return h(x)
        ############

        pos_veh = np.ones((4,1))
        pos_veh[0:3] = x[0:3]
        pos_sens = self.veh_to_sens * pos_veh

        hx = np.zeros((self.configs.dim_meas,1))
        px, py, pz, _ = pos_sens

        if px == 0:
            raise NameError('Jacobain is not defined for px=0!')
        else:
            hx[0] = self.c_i - (self.f_i * py) / px
            hx[1] = self.c_j - (self.f_j * pz) / px
        
        return hx  

    def get_H(self, x, params):
        H = np.matrix(np.zeros((self.configs.dim_meas, params.dim_state)))
        R = self.veh_to_sens[0:3, 0:3] # rotation
        T = self.veh_to_sens[0:3, 3] # translation
        # check and print error message if dividing by zero
        if R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0] == 0: 
            raise NameError('Jacobian not defined for this x!')
        else:
            H[0,0] = self.f_i * (-R[1,0] / (R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])
                                + R[0,0] * (R[1,0]*x[0] + R[1,1]*x[1] + R[1,2]*x[2] + T[1]) \
                                    / ((R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])**2))
            H[1,0] = self.f_j * (-R[2,0] / (R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])
                                + R[0,0] * (R[2,0]*x[0] + R[2,1]*x[1] + R[2,2]*x[2] + T[2]) \
                                    / ((R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])**2))
            H[0,1] = self.f_i * (-R[1,1] / (R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])
                                + R[0,1] * (R[1,0]*x[0] + R[1,1]*x[1] + R[1,2]*x[2] + T[1]) \
                                    / ((R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])**2))
            H[1,1] = self.f_j * (-R[2,1] / (R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])
                                + R[0,1] * (R[2,0]*x[0] + R[2,1]*x[1] + R[2,2]*x[2] + T[2]) \
                                    / ((R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])**2))
            H[0,2] = self.f_i * (-R[1,2] / (R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])
                                + R[0,2] * (R[1,0]*x[0] + R[1,1]*x[1] + R[1,2]*x[2] + T[1]) \
                                    / ((R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])**2))
            H[1,2] = self.f_j * (-R[2,2] / (R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])
                                + R[0,2] * (R[2,0]*x[0] + R[2,1]*x[1] + R[2,2]*x[2] + T[2]) \
                                    / ((R[0,0]*x[0] + R[0,1]*x[1] + R[0,2]*x[2] + T[0])**2))
        return H   
        
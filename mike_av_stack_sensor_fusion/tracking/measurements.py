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
from tools.ros_conversions.transformations import quaternion_from_euler, pointcloud2_to_array, image_to_numpy
from ros2_numpy.ros2_numpy.point_cloud2 import pointcloud2_to_array
from ros2_numpy.ros2_numpy.image import image_to_numpy
from ament_index_python.packages import get_package_share_directory
import cv2
import json
import torch
import open3d as o3d

from ultralytics import YOLO

package_name = 'mike_av_stack_sensor_fusion'

class Sensor(Node):
    '''Sensor class including measurement matrix'''
    def __init__(self, name, configs, trackmanager):
        super().__init__('sensor_fusion')
        self.verbose = True
        self.configs = configs
        self.name = name
        self.trackmanager = trackmanager
        self.get_logger().debug(f'Starting Sensor: {self.configs.id} of type: {self.configs.type}')

        self.share_path = get_package_share_directory(package_name=package_name)
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
        self.model = odet.create_model(self, self.configs)
        self.configs.fov = [-np.pi/2, np.pi/2] # angle of field of view in radians
        self.configs.dim_meas = 3
        self.configs.verbose = True
        self.configs.viz = True

        # Set up transforms
        self.sens_to_veh = np.matrix(np.identity((4))) # transformation sensor to vehicle coordinates equals identity matrix because lidar detections are already in vehicle coordinates
        self.get_logger().debug(f'Type: {type(self.sens_to_veh)}')
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
        self.get_logger().info(f'Starting subscription of topic {self.configs.base_topic}')
        self.subscriber_pc = self.create_subscription(
            msg_type=PointCloud2,
            topic=self.configs.base_topic, 
            callback=self.detection_callback,
            qos_profile=qos_profile,
            callback_group=sub_cb_group
            )
        self.get_logger().info(f'Starting subscription of topic /sensor_fusion/detection/lidar')
        if trackmanager == None:
            self.subscriber_det = self.create_subscription(
                msg_type=Detection3DArray,
                topic="/sensor_fusion/detection/lidar", 
                callback=self.track_manage_callback,
                qos_profile=qos_profile,
                callback_group=sub_cb_group
                )
            self.subscriber_det
        self.subscriber_pc
        
    def __del__(self):
        cv2.destroyAllWindows
        
    def detection_callback(self, point_cloud):
        if self.verbose:
            self.get_logger().info('Got pointcloud')

        point_cloud_2d = self.get_point_cloud_2d(point_cloud)
        self.get_logger().debug('got point cloud 2d')

        bev = self.bev_from_pcl(point_cloud_2d, self.configs)
        self.get_logger().debug('got bev')
        detections = odet.detect_objects(self, bev, self.model, self.configs, verbose=True)
        self.get_logger().debug('got detections')

        if detections is None:
            self.get_logger().warn(f'No detections')
            return
            
        if self.verbose:
            self.get_logger().debug(f'detection size: {len(detections)}')

        dets = []
        for det in detections:
            d3d = Detection3D()
                            # Unpack the list 
            quaternion = quaternion_from_euler(roll=0, pitch=0, yaw=det[7])
            self.get_logger().debug(f'quaternion_from_euler: {quaternion}')
            self.get_logger().debug(f'type: {type(det[4])}')
            d3d.bbox.center.orientation.x = quaternion[0]
            d3d.bbox.center.orientation.y = quaternion[1]
            d3d.bbox.center.orientation.z = quaternion[2]
            d3d.bbox.center.orientation.w = quaternion[3]
            d3d.bbox.center.position.x = det[1]
            d3d.bbox.center.position.y = det[2]
            d3d.bbox.center.position.z = det[3]
            d3d.bbox.size.x = det[4].astype(np.float64)
            d3d.bbox.size.y = det[5].astype(np.float64)
            d3d.bbox.size.z = det[6].astype(np.float64)
            dets.append(d3d)

        self.frame_id += 1
        detection3DArray = Detection3DArray()
        # detection3DArray.header.frame_id = self.frame_id 
        detection3DArray.header.stamp = self.get_clock().now().to_msg() # rospy.Time.now()
        detection3DArray.detections = dets
        self.pub_detection.publish(detection3DArray)

    def get_point_cloud_2d(self, pointcloud):
        # Convert the data from a 1d list of uint8s to a 2d list
        field_len = len(pointcloud.data)

        # The result of this is <vector<vector>> where [[]]
        point_cloud_2d = np.array([np.array(x.tolist()) for x in pointcloud2_to_array(pointcloud)])
    
        if self.verbose:
            self.get_logger().debug(f'Shape of pc2d: {point_cloud_2d.shape} First element: {type(point_cloud_2d[0])} {point_cloud_2d[0]}')
            self.get_logger().debug(f'First og: {pointcloud.data[0]}, {pointcloud.data[1]}, {pointcloud.data[2]}. {pointcloud.data[3]}')
            self.get_logger().debug('height: %d, width: %d, length of data: %d' % (pointcloud.height, pointcloud.width, field_len))
            for field in pointcloud.fields:
                self.get_logger().debug(f'\tname: {field.name}, offset: {field.offset}, datatype: {field.datatype}, count: {field.count}')

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

    # create birds-eye view of lidar data
    def bev_from_pcl(self, lidar_pcl, configs):

        # remove lidar points outside detection area and with too low reflectivity
        mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                        (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]))# &
                        #(lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
        lidar_pcl = lidar_pcl[mask]
        
        # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
        lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  

        if self.configs.verbose:
            self.get_logger().debug(f'{type(lidar_pcl)}')
            self.get_logger().debug('Min and max height, %f, %f' %(np.min(lidar_pcl[:,2]), np.max(lidar_pcl[:,2])))

        # convert sensor coordinates to bev-map coordinates (center is bottom-middle)

        ## step 1 : compute bev-map discretization by dividing x-range by the bev-image height (see configs)
        delta_x_rw_meters = configs.lim_x[1] - configs.lim_x[0]
        delta_y_rw_meters = configs.lim_y[1] - configs.lim_y[0]
        meters_pixel_x = delta_x_rw_meters / configs.bev_height
        meters_pixel_y = delta_y_rw_meters / configs.bev_width

        ## step 2 : create a copy of the lidar pcl and transform all metrix x-coordinates into bev-image coordinates  
        lidar_pcl_copy = np.copy(lidar_pcl)
        lidar_pcl_copy[:, 0] = np.int_(np.floor(lidar_pcl_copy[:, 0] / meters_pixel_x))  

        # step 3 : perform the same operation as in step 2 for the y-coordinates but make sure that no negative bev-coordinates occur
        lidar_pcl_copy[:, 1] = np.int_(np.floor(lidar_pcl_copy[:, 1] / meters_pixel_y) + (configs.bev_width + 1) / 2)

        # step 4 : visualize point-cloud using the function show_pcl from a previous task
        # if viz:
        #     show_pcl(lidar_pcl_copy)
    
    
        # Compute intensity layer of the BEV map

        ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
        intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))

        # step 2 : re-arrange elements in lidar_pcl_cpy by sorting first by x, then y, then -z (use numpy.lexsort)
        lidar_pcl_copy[lidar_pcl_copy[:, 3] > 1.0, 3] = 1.0
        index_vector_int = np.lexsort((-lidar_pcl_copy[:, 3], lidar_pcl_copy[:, 1], lidar_pcl_copy[:, 0]))
        lidar_pcl_top = lidar_pcl_copy[index_vector_int]

        ## step 3 : extract all points with identical x and y such that only the top-most z-coordinate is kept (use numpy.unique)
        ##          also, store the number of points per x,y-cell in a variable named "counts" for use in the next task
        _, idx_int_unique, counts = np.unique(lidar_pcl_top[:, 0:2], return_index=True, return_inverse=False, return_counts=True, axis=0)
        lidar_pcl_top = lidar_pcl_top[idx_int_unique]

        ## step 4 : assign the intensity value of each unique entry in lidar_pcl_top to the intensity map 
        ##          make sure that the intensity is scaled in such a way that objects of interest (e.g. vehicles) are clearly visible    
        ##          also, make sure that the influence of outliers is mitigated by normalizing intensity on the difference between the max. and min. value within the point cloud
        # intensity_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 3] / (np.amax(lidar_pcl_top[:, 3]) - np.amin(lidar_pcl_top[:, 3]))

        lidar_pcl_top_copy = np.copy(lidar_pcl_top[:,3])

        mean = 0.955248
        std = 0.026137
        if self.configs.viz:
            mean = np.mean(lidar_pcl_top_copy)
            std = np.std(lidar_pcl_top_copy)

        devs = 1
        min = 0 if (mean - devs * std) < 0 else mean - devs * std
        max = 1 if (mean + devs * std) > 1 else mean + devs * std

        if self.configs.viz:
            minv = np.min(lidar_pcl_top[:,3])
            maxv = np.max(lidar_pcl_top[:,3])
            pbot = np.percentile(lidar_pcl_top_copy, 10)
            ptop = np.percentile(lidar_pcl_top_copy, 90)
            span = ptop - pbot
            self.get_logger().debug('minv: {minv}, maxv: {maxv}')
            self.get_logger().debug('span: %f, mean: %f, standard deviation: %f' %(span, mean, std))
            self.get_logger().debug('percentile, 90: %f, 10: %f' %(ptop, pbot))
            self.get_logger().debug('lower std: %f, upper std: %f' %(min, max))

        # scale_log = np.frompyfunc(lambda x: 0 if x == 1 else -1 / np.log10(x))

        scale = np.frompyfunc(lambda x, min, max: 1 if x > max else (x - min) / (max - min), 3, 1)
        # intensity_map = scale(intensity_map_ps, min, max).astype(float)

        lidar_pcl_top_copy_post = scale(lidar_pcl_top_copy, min, max)

        intensity_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
        intensity_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top_copy_post

        # if viz:
        #     analyze({'before': lidar_pcl_top_copy, 'after': lidar_pcl_top_copy_post}, title='Intensity Distribution', nqp=False)

        ## step 5 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
        img_intensity = intensity_map * 255
        img_intensity = img_intensity.astype(np.uint8)
        if self.configs.viz:
            cv2.imshow('img_intensity', img_intensity)
            cv2.waitKey(16)

    

        # Compute height layer of the BEV map

        ## step 1 : create a numpy array filled with zeros which has the same dimensions as the BEV map
        height_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))

        ## step 2 : assign the height value of each unique entry in lidar_top_pcl to the height map
        ##          make sure that each entry is normalized on the difference between the upper and lower height defined in the config file
        ##          use the lidar_pcl_top data structure from the previous task to access the pixels of the height_map
        # _, idx_height_unique, counts = np.unique(lidar_pcl_top[:, 0:2], return_index=True, return_inverse=False, return_counts=True, axis=0)
        # lidar_pcl_hei = lidar_pcl_top[idx_height_unique]
        height_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = lidar_pcl_top[:, 2] / float(np.abs(configs.lim_z[1] - configs.lim_z[0]))

        ## step 3 : temporarily visualize the intensity map using OpenCV to make sure that vehicles separate well from the background
        img_height = height_map * 256
        img_height = img_height.astype(np.uint8)
        if self.configs.viz:
            cv2.imshow('img_height', img_height)
            cv2.waitKey(16)

        #######
        ####### ID_S2_EX3 END #######       

        # TODO remove after implementing all of the above steps
        # lidar_pcl_cpy = []
        # lidar_pcl_top = []
        # height_map = []
        # intensity_map = []

        # Compute density layer of the BEV map
        density_map = np.zeros((configs.bev_height + 1, configs.bev_width + 1))
        _, _, counts = np.unique(lidar_pcl_copy[:, 0:2], axis=0, return_index=True, return_counts=True)
        normalizedCounts = np.minimum(1.0, np.log(counts + 1) / np.log(64)) 
        density_map[np.int_(lidar_pcl_top[:, 0]), np.int_(lidar_pcl_top[:, 1])] = normalizedCounts
            
        # assemble 3-channel bev-map from individual maps
        bev_map = np.zeros((3, configs.bev_height, configs.bev_width))
        bev_map[2, :, :] = density_map[:configs.bev_height, :configs.bev_width]  # r_map
        bev_map[1, :, :] = height_map[:configs.bev_height, :configs.bev_width]  # g_map
        bev_map[0, :, :] = intensity_map[:configs.bev_height, :configs.bev_width]  # b_map

        # expand dimension of bev_map before converting into a tensor
        s1, s2, s3 = bev_map.shape
        bev_maps = np.zeros((1, s1, s2, s3))
        bev_maps[0] = bev_map

        bev_maps = torch.from_numpy(bev_maps)  # create tensor from birds-eye view
        input_bev_maps = bev_maps.to(configs.device, non_blocking=True).float()

        # show_bev(input_bev_maps, configs)

        return input_bev_maps


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
        print('Share path: ', self.share_path)
        with open(os.path.join(self.share_path, 'configs', 'yolov8.json')) as j_object:
            configs.yolov8 = json.load(j_object)
        # self.init_yolo()
        self.model = odet.create_model(self, self.configs, self.share_path)

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
            "/sensor_fusion/detection/camera" + self.configs.id,
            10
            )
        
    # def init_yolo(self):
    #     self.get_logger().info(f'Initializing Yolov7, cv version: {cv2.__version__}')
    #     self.yolo = Yolov7(self.configs.yolov7)


    def detection_callback(self, image):
        # self.yolo.detect(image)
        self.get_logger().debug("detection")
        image_np = image_to_numpy(image)
        outputs = self.model(image_np)
        self.get_logger().debug("Outputs: ", outputs)
        

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
        
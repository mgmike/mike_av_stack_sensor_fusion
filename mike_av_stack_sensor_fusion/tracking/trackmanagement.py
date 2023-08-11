# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Classes for track and track management
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
import threading
import collections
import rclpy
from rclpy.node import Node
from rclpy.duration import Duration
import json
from easydict import EasyDict as edict

# add project directory to python path to enable relative imports
import os
import sys
dir_tracking = os.path.dirname(os.path.realpath(__file__))
dir_sf = os.path.dirname(dir_tracking)
dir_scripts = os.path.dirname(dir_sf)
sys.path.append(dir_scripts)
sys.path.append(dir_sf)

from tracking.filter import Filter
from tracking.association import Association
from tools.ros_conversions.transformations import euler_from_quaternion

# Prediction class is needed, because each tracked object will need multiple predictions
# In the udacity code, we only predicted where the object would be in one time step, but in real life
# we dont know when the next camera image, or lidar scan will come in. Therefore we need to predict a 
# few future positions.
class Prediction():
    def __init__(self, stamp, x, P) -> None:
        self.stamp = stamp
        self.x = x
        self.P = P
class Track:
    '''Track class with state, covariance, id, score'''
    def __init__(self, meas, id, params):
        self.params = params
        self.node = params.node
        print('creating track no.', id)
        M_rot = meas.sensor.sens_to_veh[0:3, 0:3] # rotation matrix from sensor to vehicle coordinates
        
        ############
        # Step 2: initialization:
        # - replace fixed track initialization values by initialization of x and P based on 
        # unassigned measurement transformed from sensor to vehicle coordinates
        # - initialize track state and track score with appropriate values
        ############

        x = np.zeros((6,1))
        P = np.zeros((6,6))
        z = np.zeros((4,1))
        z[3] = 1
        z[0:3] = meas.z
        T = meas.sensor.sens_to_veh
        R = meas.R
        M_rot = T[0:3, 0:3]

        # Compute position estimation and error covariance
        x[0:3] = (T * z)[0:3]
        P[0:3, 0:3] = M_rot * R * M_rot.transpose()

        # Compute velocity estimation error covariance
        P_vel = np.matrix([[params.sigma_p44**2, 0, 0],
                           [0, params.sigma_p55**2, 0],
                           [0, 0, params.sigma_p66**2]])
        P[3:6, 3:6] = P_vel

        # self.x = np.matrix([[49.53980697],
        #                 [ 3.41006279],
        #                 [ 0.91790581],
        #                 [ 0.        ],
        #                 [ 0.        ],
        #                 [ 0.        ]])
        # self.P = np.matrix([[9.0e-02, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
        #                 [0.0e+00, 9.0e-02, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00],
        #                 [0.0e+00, 0.0e+00, 6.4e-03, 0.0e+00, 0.0e+00, 0.0e+00],
        #                 [0.0e+00, 0.0e+00, 0.0e+00, 2.5e+03, 0.0e+00, 0.0e+00],
        #                 [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 2.5e+03, 0.0e+00],
        #                 [0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 0.0e+00, 2.5e+01]])

        self.state = 'initialized'
        self.score = 1.0 / params.window

        # Initialize empty list of predictions
        # The 0th element will be the most recent updated measurement
        self.predictions = [Prediction(self.node.get_clock().now().to_msg(), x, P)]
               
        # other track attributes
        self.id = id
        self.width = meas.width
        self.length = meas.length
        self.height = meas.height
        self.yaw =  np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates

    def set_predictions(self, predictions):
        self.predictions = predictions

    # **********!!!*!*!*!*!*!* Make this take in a time stamp and make most fxns in filter.py take in a time stamp
    def get_nearest_prediction(self, meas):
        stamp = meas.stamp
        if stamp < self.predictions[0].stamp:
            self.node.get_logger().warn('Comparing predictions with old measurement')
            return self.predictions[0]
        elif stamp > self.predictions[self.params.predictions]:
            self.node.get_logger().warn('Predictions are outdated')
            return self.predictions[self.params.predictions]
        else :
            for i, prediction in enumerate(self.predictions[1:]):
                if np.abs(meas.stamp - prediction.stamp) < Duration(0, self.params.dt / 2.0):
                    self.node.get_logger().info('Nearest prediction is ', i, ' in the future at time: ', prediction.stamp)
                    return prediction
                
    def get_new_prediction(self, stamp, x, P):
        return Prediction(stamp, x, P)


    def update_attributes(self, meas):
        # use exponential sliding average to estimate dimensions and orientation
        if meas.sensor.name == 'lidar':
            c = self.params.weight_dim
            self.width = c*meas.width + (1 - c)*self.width
            self.length = c*meas.length + (1 - c)*self.length
            self.height = c*meas.height + (1 - c)*self.height
            M_rot = meas.sensor.sens_to_veh
            self.yaw = np.arccos(M_rot[0,0]*np.cos(meas.yaw) + M_rot[0,1]*np.sin(meas.yaw)) # transform rotation from sensor to vehicle coordinates
        
        
###################        

class Trackmanagement(Node):
    '''Track manager with logic for initializing and deleting objects'''
    def __init__(self):
        super().__init__('trackmanagement')
        self.N = 0 # current number of tracks
        self.last_id = -1
        self.track_list = []
        self.track_list_lock = threading.Lock()

        self.params = edict()
        self.params.node = self
        with open(os.path.join(dir_sf, 'configs', 'tracking.json')) as j_object:
            self.params.update(json.load(j_object))
        
        self.filter = Filter(self.params)
        self.association = Association(self.params)
        
    def manage_tracks(self, unassigned_tracks, unassigned_meas, meas_list):  
        ############
        # Step 2: implement track management:
        # - decrease the track score for unassigned tracks
        # - delete tracks if the score is too low or P is too big (check params.py for parameters that might be helpful, but
        # feel free to define your own parameters)
        ############

        # Seems like the value of each element in the unassigned_tracks list will be an index of the track in track_list
        
        # decrease score for unassigned tracks
        for i in unassigned_tracks:
            self.track_list_lock.acquire()
            u = self.track_list[i]
            self.track_list_lock.release()
            if meas_list:
                if meas_list[0].sensor.in_fov(u.x):
                    u.score -= 1.0 / self.params.window
            # else: 
            #     u.score -= 1.0 / params.window
            if u.score <= 0.0:
                u.score = 0.0

        # delete old tracks   
        self.track_list_lock.acquire()
        for track in self.track_list:
            closest_pred = track.get_nearest_prediction(meas_list[0])
            if ((track.state in ['confirmed'] and track.score < self.params.delete_threshold) or
                    ((closest_pred.P[0, 0] > self.params.max_P or closest_pred.P[1, 1] > self.params.max_P)) or
                    track.score < 0.05):
                print('deleting track no.', track.id)
                self.track_list.remove(track)
        self.track_list_lock.release()

        ############
        # END student code
        ############ 
            
        # initialize new track with unassigned measurement
        for j in unassigned_meas: 
            if meas_list[j].sensor.name == 'lidar': # only initialize with lidar measurements
                self.init_track(meas_list[j])
            
    def addTrackToList(self, track):
        self.track_list_lock.acquire()
        self.track_list.append(track)
        self.track_list_lock.release()
        self.N += 1
        self.last_id = track.id

    def init_track(self, meas):
        track = Track(meas, self.last_id + 1, self.params)
        self.addTrackToList(track)

    def delete_track(self, track):
        if track is Track:
            print('deleting track no.', track.id)
            self.track_list_lock.acquire()
            self.track_list.remove(track)
            self.track_list_lock.release()
        if track is int:
            print('deleting track no.', track)
            self.track_list_lock.acquire()
            del self.track_list[track]
            self.track_list_lock.release()
        
    def handle_updated_track(self, track):      
        ############
        # Step 2: implement track management for updated tracks:
        # - increase track score
        # - set track state to 'tentative' or 'confirmed'
        ############

        track.score += 1 / self.params.window
        track.score = min(1.0, track.score)

        if track.score > self.params.confirmed_threshold:
            track.state = 'confirmed'
        else:
            track.state = 'tentative'


class Measurement:
    '''Measurement class including measurement values, covariance, timestamp, sensor'''
    def __init__(self, sensor, stamp):
        # create measurement object
        self.sensor = sensor
        self.stamp = stamp # time

class LidarMeasurement(Measurement):
    def __init__(self, sensor, stamp, detection, params):
        super().__init__(sensor, stamp)

        # create measurement object
        sigma_lidar_x = params.sigma_lidar_x # load params
        sigma_lidar_y = params.sigma_lidar_y
        sigma_lidar_z = params.sigma_lidar_z
        self.z = np.zeros((self.sensor.configs.dim_meas,1)) # measurement vector
        self.z[0] = detection.bbox.center.position.x
        self.z[1] = detection.bbox.center.position.y
        self.z[2] = detection.bbox.center.position.z
        self.R = np.matrix([[sigma_lidar_x**2, 0, 0], # measurement noise covariance matrix
                            [0, sigma_lidar_y**2, 0], 
                            [0, 0, sigma_lidar_z**2]])
        
        self.width = detection.bbox.size.x
        self.length = detection.bbox.size.y
        self.height = detection.bbox.size.z
        q = detection.bbox.center.orientation
        q_array = [q.x, q.y, q.z, q.w]
        self.yaw = euler_from_quaternion(q_array)

class CameraMeasurement(Measurement):
    def __init__(self, sensor, stamp, detection, params):
        super().__init__(sensor, stamp)

        sigma_cam_i = params.sigma_cam_i # load params
        sigma_cam_j = params.sigma_cam_j
        self.z = np.zeros((self.sensor.configs.dim_meas, 1))
        self.z[0] = detection.bbox.center.x
        self.z[1] = detection.bbox.center.y
        self.length = detection.bbox.size_x
        self.width = detection.bbox.size_y
        self.R = np.matrix([[sigma_cam_i**2, 0], # measurement noise covariance matrix
                            [0, sigma_cam_j**2]])
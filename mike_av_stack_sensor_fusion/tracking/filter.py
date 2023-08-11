# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Kalman filter class
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np

# add project directory to python path to enable relative imports
import os
import sys
import rclpy
from rclpy.duration import Duration 

dir_tracking = os.path.dirname(os.path.realpath(__file__))
dir_sf = os.path.dirname(dir_tracking)
dir_scripts = os.path.dirname(dir_sf)
sys.path.append(dir_scripts)
sys.path.append(dir_sf)

class Filter:
    '''Kalman filter class'''
    def __init__(self, params):
        self.params = params
        self.node = params.node
        pass

    def F(self, dt):
        # Implement and return system matrix F
        return np.matrix([[1,0,0,dt,0,0],
                         [0,1,0,0,dt,0],
                         [0,0,1,0,0,dt],
                         [0,0,0,1,0,0],
                         [0,0,0,0,1,0],
                         [0,0,0,0,0,1]])

    def Q(self, dt):
        # Implement and return process noise covariance Q
        q = self.params.q
        q1 = dt * q
        q2 = (dt**2 * q) / 2
        q3 = (dt**3 * q) / 3
        return np.matrix([[q3,0,0,q2,0,0],
                         [0,q3,0,0,q2,0],
                         [0,0,q3,0,0,q2],
                         [q2,0,0,q1,0,0],
                         [0,q2,0,0,q1,0],
                         [0,0,q2,0,0,q1]])

    def predict(self, track, current_pos):
        # Predict state x and estimation error covariance P to next timestep, save x and P in track
        time = self.node.get_clock().now().to_msg()
        predictions = []
        for i in range(track.params.predictions + 1):
            dt = i * track.params.dt
            F = self.F(dt)
            x = F * current_pos.x
            P = F * current_pos.P * F.transpose() + self.Q(dt)
            predictions.append(track.get_new_prediction(time + Duration(0, dt), x, P))
        track.set_predictions(predictions)

    def update(self, track, meas):
        ############
        # Step 1: update state x and covariance P with associated measurement, save x and P in track
        ############
        
        closest_pred = track.get_nearest_prediction(meas)

        # Get the projection matrix that projects the state space into the measurement space
        H = meas.sensor.get_H(closest_pred.x, meas.params)

        H_t = H.transpose()
        K = closest_pred.P * H_t * np.linalg.inv(self.S(track, meas, H))
        I = np.identity(self.params.dim_state)
        x = (closest_pred.x + K * self.gamma(track, meas)) # gamma is transformation of state estimation to measurement state
        P = ((I - K * H) * closest_pred.P)
        current_pos = track.get_new_prediction(meas.stamp, x, P)

        track.update_attributes(meas)

        self.predict(track, current_pos)
    
    def gamma(self, x, meas):
        ############
        # Step 1: calculate and return residual gamma
        ############

        # x = track.x[0:3] # 6x1
        # if meas.sensor.name == 'lidar':
        #     H = meas.sensor.get_H(x)
        #     return meas.z - H[0:3, 0:3] * x
        # elif meas.sensor.name == 'camera':
        #     return meas.z - meas.sensor.get_hx(x)
        
        # return 0

        H = meas.sensor.get_hx(x)
        gamma = meas.z - H
        return gamma
        
        ############
        # END student code
        ############ 

    def S(self, track, meas, H):
        ############
        # Step 1: calculate and return covariance of residual S
        ############
        closest_pred = track.get_nearest_prediction(meas)
        H_t = H.transpose()
        return H * closest_pred.P * H_t + meas.R
        
        ############
        # END student code
        ############ 
# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Data association class with single nearest neighbor association and gating based on Mahalanobis distance
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# imports
import numpy as np
from scipy.stats.distributions import chi2

# add project directory to python path to enable relative imports
import os
import sys
dir_tracking = os.path.dirname(os.path.realpath(__file__))
dir_sf = os.path.dirname(dir_tracking)
dir_scripts = os.path.dirname(dir_sf)
sys.path.append(dir_scripts)
sys.path.append(dir_sf)


class Association:
    '''Data association class with single nearest neighbor association and gating based on Mahalanobis distance'''
    def __init__(self, params):
        self.params = params
        self.association_matrix = np.matrix([])
        self.unassigned_tracks = []
        self.unassigned_meas = []
        
    def associate(self, track_list, meas_list, KF):
             
        ############
        # Step 3: association:
        # - replace association_matrix with the actual association matrix based on Mahalanobis distance (see below) for all tracks and all measurements
        # - update list of unassigned measurements and unassigned tracks
        ############
        
        N = len(track_list) # N tracks
        M = len(meas_list) # M measurements
        self.unassigned_tracks = [] # reset lists
        self.unassigned_meas = []
        
        if len(meas_list) > 0:
            self.unassigned_meas = [a for a in range(len(meas_list))]
        if len(track_list) > 0:
            self.unassigned_tracks = [b for b in range(len(track_list))]
        if len(meas_list) > 0 and len(track_list) > 0: 
            self.association_matrix = np.inf * np.ones((N, M))

            for n in self.unassigned_tracks:
                for m in self.unassigned_meas:
                    MHD = self.MHD(track_list[n], meas_list[m], KF)
                    if self.gating(MHD, meas_list[m].sensor):
                        self.association_matrix[n, m] = MHD
        
        ############
        # END student code
        ############ 
                
    def get_closest_track_and_meas(self):
        ############
        # Step 3: find closest track and measurement:
        # - find minimum entry in association matrix
        # - delete row and column
        # - remove corresponding track and measurement from unassigned_tracks and unassigned_meas
        # - return this track and measurement
        ############

        min = np.min(self.association_matrix)
        if min == np.inf:
            return np.nan, np.nan

        # find indecies for minimum entry in association matrix
        n, m = np.unravel_index(self.association_matrix.argmin(), self.association_matrix.shape)
        # remove row then column
        self.association_matrix = np.delete(np.delete(self.association_matrix, n, 0), m, 1)
        # Get  track and measurement to delete
        update_track = self.unassigned_tracks[n]
        update_meas = self.unassigned_meas[m]
        
        # remove track and measurement from unassigned lists
        self.unassigned_tracks.remove(update_track) 
        self.unassigned_meas.remove(update_meas)
            
        ############
        # END student code
        ############ 
        return update_track, update_meas     

    def gating(self, MHD, sensor): 
        ############
        # Step 3: return True if measurement lies inside gate, otherwise False
        ############
        
        limit = chi2.ppf(self.params.gating_threshold, sensor.configs.dim_meas)
        if MHD < limit:
            return True
        else:
            return False
        
        ############
        # END student code
        ############ 
        
    def MHD(self, track, meas, KF):
        ############
        # Step 3: calculate and return Mahalanobis distance
        ############
        closest_pred = track.get_nearest_prediction(meas)
        gamma = KF.gamma(closest_pred.x, meas) # In terms of z now
        H = meas.sensor.get_H(closest_pred.x, track.params)
        S = KF.S(track, meas, H) # Estimation error covariance P transformed to measurement space plus measurement covariance R
        MHD = np.sqrt(gamma.transpose() * np.linalg.inv(S) * gamma)
        
        return MHD
        
        ############
        # END student code
        ############ 
    
    def associate_and_update(self, manager, meas_list, KF):
        # associate measurements and tracks
        self.associate(manager.track_list, meas_list, KF)
    
        # update associated tracks with measurements
        while self.association_matrix.shape[0]>0 and self.association_matrix.shape[1]>0:
            
            # search for next association between a track and a measurement
            ind_track, ind_meas = self.get_closest_track_and_meas()
            if np.isnan(ind_track):
                print('---no more associations---')
                break
            track = manager.track_list[ind_track]
            closest_pred = track.get_nearest_prediction(meas_list[0])
            # check visibility, only update tracks in fov    
            if not meas_list[0].sensor.in_fov(track.x):
                continue
            
            # Kalman update
            print('update track', track.id, 'with', meas_list[ind_meas].sensor.name, 'measurement', ind_meas)
            KF.update(track, meas_list[ind_meas])
            
            # update score and track state 
            manager.handle_updated_track(track)
            
            # save updated track
            manager.track_list[ind_track] = track
            
        # run track management 
        manager.manage_tracks(self.unassigned_tracks, self.unassigned_meas, meas_list)
        
        for track in manager.track_list:            
            print('track', track.id, 'score =', track.score)
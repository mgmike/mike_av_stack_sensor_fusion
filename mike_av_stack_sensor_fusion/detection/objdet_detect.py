#!/home/mike/anaconda3/envs/waymo/bin/python3

# ---------------------------------------------------------------------
# Project "Track 3D-Objects Over Time"
# Copyright (C) 2020, Dr. Antje Muntzinger / Dr. Andreas Haja.
#
# Purpose of this file : Detect 3D objects in lidar point clouds using deep learning
#
# You should have received a copy of the Udacity license together with this program.
#
# https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013
# ----------------------------------------------------------------------
#

# general package imports
import numpy as np
import torch
import time
import cv2
import json

from easydict import EasyDict as edict
# add project directory to python path to enable relative imports
import os
import sys
dir_detection = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_detection)

PACKAGE_PARENT = '..'
SCRIPT_DIR = os.path.dirname(os.path.realpath(os.path.join(os.getcwd(), os.path.expanduser(__file__))))
sys.path.append(os.path.normpath(os.path.join(SCRIPT_DIR, PACKAGE_PARENT)))

# model-related
from objdet_models.fpn_resnet.utils.evaluation_utils import decode, post_processing 
from objdet_models.fpn_resnet.utils.torch_utils import _sigmoid
import objdet_models.fpn_resnet.models.fpn_resnet as fpn_resnet


from ament_index_python.packages import get_package_share_directory

# from tools.objdet_models.darknet.models.darknet2pytorch import Darknet as darknet
# from tools.objdet_models.darknet.utils.evaluation_utils import post_processing_v2

package_name = 'mike_av_stack_sensor_fusion'

# load all object-detection parameters into an edict
def load_configs(model_name='fpn_resnet'):
    # get parent directory of this file to enable relative paths
    curr_path = os.path.dirname(os.path.realpath(__file__))
    share_path = get_package_share_directory(package_name=package_name)
    
    configs = edict()

    with open(os.path.join(share_path, "configs", "bev.json")) as bevj_object:
        configs.update(json.load(bevj_object))

    # print(configs)

    with open(os.path.join(share_path, "configs", model_name + ".json")) as mj_object:
        configs.update(json.load(mj_object))
    configs.model_path = os.path.join(curr_path, 'objdet_models', model_name)
    configs.pretrained_path = os.path.join(share_path, 'weights', configs.pretrained_filename)
    if 'cfgfile' in configs.values():
        configs.cfgfile = os.path.join(configs.model_path, 'config', configs.cfgfile)


    with open(os.path.join(share_path, "configs", "tracking.json")) as mj_object:
        configs.update(json.load(mj_object))

    # visualization parameters
    configs.output_width = 608 # width of result image (height may vary)
    configs.obj_colors = [[0, 255, 255], [0, 0, 255], [255, 0, 0]] # 'Pedestrian': 0, 'Car': 1, 'Cyclist': 2
    # GPU vs. CPU
    configs.no_cuda = False # if true, cuda is not used
    configs.gpu_idx = 0  # GPU index to use.
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    
    # Evaluation params
    configs.min_iou = 0.5

    return configs


# create model according to selected model type
def create_model(node, configs):

    # check for availability of model file
    # assert os.path.isfile(configs.pretrained_filename), "No file at {}".format(configs.pretrained_filename)
    #                               pretrained_path?

    # create model depending on architecture name
    # if (configs.arch == 'darknet') and (configs.cfgfile is not None):
    #     print('Darknet not implemented yet')
        # model = darknet(cfgfile=configs.cfgfile, use_giou_loss=configs.use_giou_loss)    
    
    if 'fpn_resnet' in configs.arch:
        node.get_logger().info('using ResNet architecture with feature pyramid')
        model = fpn_resnet.get_pose_net(num_layers=configs.num_layers, heads=configs.heads, head_conv=configs.head_conv,
                                        imagenet_pretrained=configs.imagenet_pretrained)

    else:
        assert False, 'Undefined model backbone'

    # load model weights
    model.load_state_dict(torch.load(configs.pretrained_path, map_location='cpu'))
    node.get_logger().info('Loaded weights from {}\n'.format(configs.pretrained_path))

    # set model to evaluation state
    configs.device = torch.device('cpu' if configs.no_cuda else 'cuda:{}'.format(configs.gpu_idx))
    model = model.to(device=configs.device)  # load model to either cpu or gpu
    out_cap = None
    model.eval()          

    return model

def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()

def extract_3d_bb(det, configs, cls_id = 1):
    ## step 3 : perform the conversion using the limits for x, y and z set in the configs structure
    # (scores-0:1, x-1:2, y-2:3, z-3:4, dim-4:7, yaw-7:8)
    _score, _x, _y, _z, _h, _w, _l, _yaw = det
    _yaw = -_yaw
    bound_size_x = configs.lim_x[1] - configs.lim_x[0]
    bound_size_y = configs.lim_y[1] - configs.lim_y[0]
    x = _y / configs.bev_height * bound_size_x + configs.lim_x[0]
    y = _x / configs.bev_width * bound_size_y + configs.lim_y[0]
    z = _z + configs.lim_z[0]
    w = _w / configs.bev_width * bound_size_y
    l = _l / configs.bev_height * bound_size_x
    if x < configs.lim_x[0]  or x > configs.lim_x[1] or y < configs.lim_y[0] or y > configs.lim_y[1]:
        return []
    else:
        return [cls_id, x, y, z, _h, w, l, _yaw]

# detect trained objects in birds-eye view
def detect_objects(node, input_bev_maps, model, configs, verbose=False):

    
    # Extract 3d bounding boxes from model response
    if verbose:
        node.get_logger().debug("student task ID_S3_EX2")
    objects = [] 

    # deactivate autograd engine during test to reduce memory usage and speed up computations
    with torch.no_grad():  

        # perform inference

        # decode model output into target object format
        # if 'darknet' in configs.arch:

        #     # perform post-processing
        #     outputs = model(input_bev_maps)
        #     output_post = post_processing_v2(outputs, conf_thresh=configs.conf_thresh, nms_thresh=configs.nms_thresh) 
        #     detections = []
        #     for sample_i in range(len(output_post)):
        #         if output_post[sample_i] is None:
        #             continue
        #         detection = output_post[sample_i]
        #         for obj in detection:
        #             x, y, w, l, im, re, _, _, _ = obj
        #             yaw = np.arctan2(im, re)
        #             detections.append([1, x, y, 0.0, 1.50, w, l, yaw])    

        #     for det in detections:
        #         obj = extract_3d_bb(det, configs)
        #         if len(obj) > 0:
        #             objects.append(obj)

        node.get_logger().debug(f'Inside torch, arch: {configs.arch}')    

        if 'fpn_resnet' in configs.arch:
            # decode output and perform post-processing
            
            # perform post-processing
            t1 = time_synchronized()
            outputs = model(input_bev_maps)
            outputs['hm_cen'] = _sigmoid(outputs['hm_cen'])
            outputs['cen_offset'] = _sigmoid(outputs['cen_offset'])
            # detections size (batch_size, K, 10)
            detections = decode(outputs['hm_cen'], outputs['cen_offset'], outputs['direction'], outputs['z_coor'],
                                outputs['dim'], K=configs.K)
            detections = detections.cpu().numpy().astype(np.float32)
            detections = post_processing(detections, configs)
            t2 = time_synchronized() 

            detections = detections[0]
            
            if verbose:
                node.get_logger().debug(f'detections: {detections}')            
            ## detections contains an array of length 3 where 0 pertains to pidestrians, 1 is vehicles and 2 is cyclests. 
            ## Each array can contain a list of detections

            ## step 2 : loop over all detections
            for cls_id in range(configs.num_classes):
                ## step 1 : check whether there are any detections
                if len(detections[cls_id]) > 0:
                    for det in detections[cls_id]:
                        ## step 4 : append the current object to the 'objects' array
                        obj = extract_3d_bb(det, configs)
                        if len(obj) > 0:
                            objects.append(obj)
 
    show_objects_in_bev_labels_in_camera(objects, input_bev_maps, configs)
    return objects    




######### Visualization helper functions

# project detected bounding boxes into birds-eye view
def project_detections_into_bev(bev_map, detections, configs, color=[]):
    for row in detections:
        # extract detection
        _id, _x, _y, _z, _h, _w, _l, _yaw = row

        # convert from metric into pixel coordinates
        x = (_y - configs.lim_y[0]) / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
        y = (_x - configs.lim_x[0]) / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
        z = _z - configs.lim_z[0]
        w = _w / (configs.lim_y[1] - configs.lim_y[0]) * configs.bev_width
        l = _l / (configs.lim_x[1] - configs.lim_x[0]) * configs.bev_height
        yaw = -_yaw

        # draw object bounding box into birds-eye view
        if not color:
            color = configs.obj_colors[int(_id)]
        
        # get object corners within bev image
        bev_corners = np.zeros((4, 2), dtype=np.float32)
        cos_yaw = np.cos(yaw)
        sin_yaw = np.sin(yaw)
        bev_corners[0, 0] = x - w / 2 * cos_yaw - l / 2 * sin_yaw # front left
        bev_corners[0, 1] = y - w / 2 * sin_yaw + l / 2 * cos_yaw 
        bev_corners[1, 0] = x - w / 2 * cos_yaw + l / 2 * sin_yaw # rear left
        bev_corners[1, 1] = y - w / 2 * sin_yaw - l / 2 * cos_yaw
        bev_corners[2, 0] = x + w / 2 * cos_yaw + l / 2 * sin_yaw # rear right
        bev_corners[2, 1] = y + w / 2 * sin_yaw - l / 2 * cos_yaw
        bev_corners[3, 0] = x + w / 2 * cos_yaw - l / 2 * sin_yaw # front right
        bev_corners[3, 1] = y + w / 2 * sin_yaw + l / 2 * cos_yaw
        
        # draw object as box
        corners_int = bev_corners.reshape(-1, 1, 2).astype(int)
        cv2.polylines(bev_map, [corners_int], True, color, 2)

        # draw colored line to identify object front
        corners_int = bev_corners.reshape(-1, 2)
        cv2.line(bev_map, (int(corners_int[0, 0]), int(corners_int[0, 1])), (int(corners_int[3, 0]), int(corners_int[3, 1])), (255, 255, 0), 2)


# visualize detection results as overlay in birds-eye view and ground-truth labels in camera image
def show_objects_in_bev_labels_in_camera(detections, bev_maps, configs):

    # project detections into birds-eye view
    bev_map = (bev_maps.cpu().data.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    bev_map = cv2.resize(bev_map, (configs.bev_width, configs.bev_height))
    project_detections_into_bev(bev_map, detections, configs)
    bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)

    img_bev_h, img_bev_w = bev_map.shape[:2]
    ratio_bev = configs.output_width / img_bev_w
    output_bev_h = int(ratio_bev * img_bev_h)
    ret_img_bev = cv2.resize(bev_map, (configs.output_width, output_bev_h))

    # show combined view
    cv2.imshow('labels vs. detected objects', ret_img_bev)

    cv2.waitKey(16) 
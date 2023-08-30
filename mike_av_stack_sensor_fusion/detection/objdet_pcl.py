#!/home/mike/anaconda3/envs/waymo/bin/python3

import numpy as np
import cv2
import torch
import open3d as o3d

def show_bev(node, bev_maps, configs):

    node.get_logger().info('bev_maps shape: {bev_maps.shape}')
    bev_map = (bev_maps.cpu().data.squeeze().permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    bev_map = cv2.resize(bev_map, (configs.bev_width, configs.bev_height))
    bev_map = cv2.rotate(bev_map, cv2.ROTATE_180)
    cv2.imshow('BEV map', bev_map)
      
    cv2.waitKey(16) 
    

# visualize lidar point-cloud
def show_pcl(pcl):

    pcl = pcl[:,:3]

    # step 1 : initialize open3d with key callback and create window
    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()
    
    # step 2 : create instance of open3d point-cloud class
    pcd = o3d.geometry.PointCloud()

    # step 3 : set points in pcd instance by converting the point-cloud into 3d vectors (using open3d function Vector3dVector)
    pcd.points = o3d.utility.Vector3dVector(pcl)

    # step 4 : for the first frame, add the pcd instance to visualization using add_geometry; for all other frames, use update_geometry instead
    # if(cnt_frame == 0):
    #     vis.add_geometry(pcd)
    # else:
    #     vis.update_geometry(pcd)
    #     vis.update_renderer()
    #     vis.poll_events()

    vis.add_geometry(pcd)

    
    # step 5 : visualize point cloud and keep window open until right-arrow is pressed (key-code 262)
    def callback(vis):
        # vis.clear_geometries()
        vis.destroy_window()

    vis.register_key_callback(262, callback)
    vis.run()
    

# create birds-eye view of lidar data
def bev_from_pcl(node, lidar_pcl, configs, viz=False, verbose=False):

    # remove lidar points outside detection area and with too low reflectivity
    mask = np.where((lidar_pcl[:, 0] >= configs.lim_x[0]) & (lidar_pcl[:, 0] <= configs.lim_x[1]) &
                    (lidar_pcl[:, 1] >= configs.lim_y[0]) & (lidar_pcl[:, 1] <= configs.lim_y[1]))# &
                    #(lidar_pcl[:, 2] >= configs.lim_z[0]) & (lidar_pcl[:, 2] <= configs.lim_z[1]))
    lidar_pcl = lidar_pcl[mask]
    
    # shift level of ground plane to avoid flipping from 0 to 255 for neighboring pixels
    lidar_pcl[:, 2] = lidar_pcl[:, 2] - configs.lim_z[0]  

    if verbose:
        node.get_logger().debug(lidar_pcl[0,:])
        node.get_logger().debug('Min and max height, %f, %f' %(np.min(lidar_pcl[:,2]), np.max(lidar_pcl[:,2])))

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
    if viz:
        mean = np.mean(lidar_pcl_top_copy)
        std = np.std(lidar_pcl_top_copy)

    devs = 1
    min = 0 if (mean - devs * std) < 0 else mean - devs * std
    max = 1 if (mean + devs * std) > 1 else mean + devs * std

    if viz:
        minv = np.min(lidar_pcl_top[:,3])
        maxv = np.max(lidar_pcl_top[:,3])
        pbot = np.percentile(lidar_pcl_top_copy, 10)
        ptop = np.percentile(lidar_pcl_top_copy, 90)
        span = ptop - pbot
        node.get_logger().debug('minv: {minv}, maxv: {maxv}')
        node.get_logger().debug('span: %f, mean: %f, standard deviation: %f' %(span, mean, std))
        node.get_logger().debug('percentile, 90: %f, 10: %f' %(ptop, pbot))
        node.get_logger().debug('lower std: %f, upper std: %f' %(min, max))

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
    if viz:
        while (1):
            cv2.imshow('img_intensity', img_intensity)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        cv2.destroyAllWindows

   

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
    if viz:
        while (1):
            cv2.imshow('img_height', img_height)
            if cv2.waitKey(10) & 0xFF == 27:
                break
        cv2.destroyAllWindows

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

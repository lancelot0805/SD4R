import cv2
import numpy as np
from PIL import Image
import torch
import matplotlib.pyplot as plt
import torch
import os
import math
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
import matplotlib
import mmcv
from mmdet3d.core.bbox import Box3DMode, CameraInstance3DBoxes, Coord3DMode, LiDARInstance3DBoxes, points_cam2img
import copy
from shapely.geometry import Polygon, box, Point
from mmdet3d.core import show_multi_modality_result
from scipy.ndimage import label
  
def get_downsample_depths_numpy(depth, down, processing='min'):
    H, W = depth.shape
    depth = depth.reshape(H//down, down, W//down, down, 1)
    depth = depth.transpose(0, 2, 4, 1, 3)
    depth = depth.reshape(-1, down * down)
    depth_tmp = np.where(depth == 0.0, 1e5, depth)
    if processing == 'min': 
        depth = np.min(depth_tmp, axis=-1)
    if processing == 'max': 
        depth = np.max(depth_tmp, axis=-1)
    if processing == 'mean': 
        depth = np.mean(depth_tmp, axis=-1)
    depth = depth.reshape(H//down, W//down)
    # cv2.imwrite('depth.png', depth)
    return depth
def get_upsample_depths_numpy(depth, up):
    H, W = depth.shape
    depth = np.repeat(depth, up, axis=0)
    depth = np.repeat(depth, up, axis=1)
    # cv2.imwrite('depth.png', depth)
    return depth

def colorize(value, vmin=None, vmax=None, cmap='jet', invalid_val=-99, invalid_mask=None, background_color=(128, 128, 128, 255), gamma_corrected=False, value_transform=None):
    """Converts a depth map to a color image.

    Args:
        value (torch.Tensor, numpy.ndarry): Input depth map. Shape: (H, W) or (1, H, W) or (1, 1, H, W). All singular dimensions are squeezed
        vmin (float, optional): vmin-valued entries are mapped to start color of cmap. If None, value.min() is used. Defaults to None.
        vmax (float, optional):  vmax-valued entries are mapped to end color of cmap. If None, value.max() is used. Defaults to None.
        cmap (str, optional): matplotlib colormap to use. Defaults to 'magma_r'.
        invalid_val (int, optional): Specifies value of invalid pixels that should be colored as 'background_color'. Defaults to -99.
        invalid_mask (numpy.ndarray, optional): Boolean mask for invalid regions. Defaults to None.
        background_color (tuple[int], optional): 4-tuple RGB color to give to invalid pixels. Defaults to (128, 128, 128, 255).
        gamma_corrected (bool, optional): Apply gamma correction to colored image. Defaults to False.
        value_transform (Callable, optional): Apply transform function to valid pixels before coloring. Defaults to None.

    Returns:
        numpy.ndarray, dtype - uint8: Colored depth map. Shape: (H, W, 4)
    """
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()

    value = value.squeeze()
    if invalid_mask is None:
        invalid_mask = value == invalid_val
    mask = np.logical_not(invalid_mask)

    # normalize
    vmin = np.percentile(value[mask],2) if vmin is None else vmin
    vmax = np.percentile(value[mask],85) if vmax is None else vmax
    if vmin != vmax:
        value = (value - vmin) / (vmax - vmin)  # vmin..vmax
    else:
        # Avoid 0-division
        value = value * 0.

    # squeeze last dim if it exists
    # grey out the invalid values

    value[invalid_mask] = np.nan
    cmapper = matplotlib.cm.get_cmap(cmap)
    if value_transform:
        value = value_transform(value)
        # value = value / value.max()
    value = cmapper(value, bytes=True)  # (nxmx4)

    # img = value[:, :, :]
    img = value[...]
    img[invalid_mask] = background_color

    #     return img.transpose((2, 0, 1))
    if gamma_corrected:
        # gamma correction
        img = img / 255
        img = np.power(img, 2.2)
        img = img * 255
        img = img.astype(np.uint8)
    return img

def draw_bev_pts_bboxes(points, gt_bbox_corners=None, pd_bbox_corners=None, save_path=None, xlim=[0, 51.2], ylim=[-25.6, 25.6]):
        
    fig, ax = plt.subplots(figsize=(10, 10))
    # fig.patch.set_facecolor('black')
    # ax.set_facecolor('black')
    
    points = points[(xlim[0]<=points[:,0]).astype(bool) & (points[:,0]<=xlim[1]).astype(bool)]
    points = points[(ylim[0]<=points[:,1]).astype(bool) & (points[:,1]<=ylim[1]).astype(bool)]
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])
    ax.autoscale(False)
    
    # plot points
    x = points[:, 0]
    y = points[:, 1]
    distances = np.sqrt(x**2 + y**2)
    normalized_distances = np.clip(distances / 60, 0, 1)
    colors = plt.cm.gray(normalized_distances)
    # ax.scatter(x, y, c=colors, s=3)
    ax.scatter(x, y, c=np.array([[0.3, 0.3, 0.3]]), s=5)
    
    # plot bboxes
    if gt_bbox_corners is not None:
        for bbox in gt_bbox_corners:
            polygon = patches.Polygon(bbox, closed=True, edgecolor='red', linewidth=2, fill=False)
            ax.add_patch(polygon)
    if pd_bbox_corners is not None:
        for bbox in pd_bbox_corners:
            polygon = patches.Polygon(bbox, closed=True, edgecolor='blue', linewidth=2, fill=False)
            ax.add_patch(polygon)
    
    # plt.xlabel('X (m)')
    # plt.ylabel('Y (m)')
    # plt.title('point cloud and bboxes under BEV')
    # plt.grid(True)
    # plt.savefig(save_path)
    ax.axis('off')  # Remove the axes
    plt.savefig(save_path, facecolor=fig.get_facecolor(), bbox_inches='tight', pad_inches=0)
    plt.close(fig)
def _extend_matrix(mat):
    mat = np.concatenate([mat, np.array([[0., 0., 0., 1.]])], axis=0)
    return mat

def get_velodyne2img(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    P2 = np.array([float(info) for info in lines[2].split(' ')[1:13]]).reshape([3, 4]); P2 = _extend_matrix(P2)
    rect = np.array([float(info) for info in lines[4].split(' ')[1:10]]).reshape([3, 3]); rect_4x4 = np.zeros([4, 4], dtype=rect.dtype); rect_4x4[3, 3] = 1.; rect_4x4[:3, :3] = rect; rect = rect_4x4
    Trv2c = np.array([float(info) for info in lines[5].split(' ')[1:13]]).reshape([3, 4]); Trv2c = _extend_matrix(Trv2c)
    return P2, rect, Trv2c

def get_pointcloud(path, load_dim):
    pointcloud = np.fromfile(path, dtype=np.float32).reshape(-1, load_dim)[:, :3]
    return pointcloud

def get_all_depth(depth_path_1, depth_path_2, depth_path_3):
    depth_dc_1 = np.array(Image.open(depth_path_1), dtype=np.float32) / 255.0
    residual = np.ones((400, depth_dc_1.shape[1]))*0.0 # *80.0
    depth_dc_1 = np.concatenate([residual, depth_dc_1], axis=0) # three channel the same
    depth_dc_2 = np.load(depth_path_2)
    H, W = depth_dc_2.shape
    depth_dc_2 = cv2.resize(depth_dc_2, (W*2, H*2), interpolation=cv2.INTER_CUBIC) # bilinear
    # down = 2
    # depth_dc_2 = np.repeat(depth_dc_2.reshape(-1, 1), down*down, axis=-1)
    # depth_dc_2 = depth_dc_2.reshape(H*down, W*down)
    depth_dc_3 = np.load(depth_path_3)
    return depth_dc_1, depth_dc_2, depth_dc_3
def remove_dontcare(ann_info):
    img_filtered_annotations = {}
    relevant_annotation_indices = [
        i for i, x in enumerate(ann_info['name']) if x != 'DontCare'
    ]
    for key in ann_info.keys():
        img_filtered_annotations[key] = (
            ann_info[key][relevant_annotation_indices])
    return img_filtered_annotations

def find_index(annos, file_index):
    for i in range(len(annos)):
        image_idx = annos[i]['image']['image_idx']
        if image_idx == int(file_index): return i


def drop_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x not in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

def get_gt_bbox(annos, CLASSES, radar2cam):
    annos = remove_dontcare(annos)
    loc = annos['location']
    dims = annos['dimensions']
    rots = annos['rotation_y']
    gt_names = annos['name']
    gt_bboxes_3d = np.concatenate([loc, dims, rots[..., np.newaxis]], axis=1).astype(np.float32)
    gt_bboxes_3d = CameraInstance3DBoxes(gt_bboxes_3d).convert_to(Box3DMode.LIDAR, np.linalg.inv(radar2cam))
    gt_bboxes = annos['bbox']
    selected = drop_arrays_by_name(gt_names, ['DontCare'])
    gt_bboxes = gt_bboxes[selected].astype('float32')
    gt_names = gt_names[selected]
    gt_labels = []  # 0123:car pedestrian cyclist truck
    for cat in gt_names:
        cat = cat.lower()
        if cat in CLASSES: gt_labels.append(CLASSES.index(cat))
        else: gt_labels.append(-1)
    gt_labels = np.array(gt_labels).astype(np.int64)
    gt_labels_3d = copy.deepcopy(gt_labels)
    
    gt_bboxes_3d = gt_bboxes_3d[gt_labels_3d!=-1]
    gt_bboxes_2d = gt_bboxes[gt_labels!=-1]
    return dict(gt_bboxes_3d=gt_bboxes_3d, gt_bboxes_2d=gt_bboxes_2d)

def draw_point_on_image(image, points, projection):
    H, W, _  = image.shape
    pc_in_image = np.ascontiguousarray(copy.copy(image))
    pts_hom = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)  # Convert to homogeneous coordinates
    img_pts = np.matmul(projection, pts_hom.T).T  # Apply transformation
    img_pts[:, :2] = img_pts[:, :2] / img_pts[:, 2:3]  # Convert back to Cartesian coordinates
    # Filter points that are within the image boundaries
    valid_mask = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < W) & (img_pts[:, 1] >= 0) & (img_pts[:, 1] < H)
    valid_mask = valid_mask & (img_pts[:, 2] >= 0) & (img_pts[:, 2] <= 80)
    img_pts = img_pts[valid_mask]
    # Extract the depth values and update the depth image
    depth_values = img_pts[:, 2]
    out_depths = np.zeros((H, W))
    # depth_order = np.argsort(depth_values)[::-1]
    # img_pts = img_pts[depth_order]
    # depth_values = np.log1p(depth_values)
    # normalized_depth = ((depth_values - depth_values.min()) / (depth_values.max() - depth_values.min())) * 255
    normalized_depth = cv2.normalize(depth_values, None, 0, 255, cv2.NORM_MINMAX)
    colored_depth = cv2.applyColorMap(normalized_depth.astype(np.uint8), cv2.COLORMAP_JET)
    x_indices = np.floor(img_pts[:, 0]).astype(int)
    y_indices = np.floor(img_pts[:, 1]).astype(int)
    out_depths[y_indices, x_indices] = depth_values
    # Draw the points on the depth image in red
    for i, (x, y) in enumerate(zip(x_indices, y_indices)):
        color = colored_depth[i][0]  # Access the color; colored_depth[i] is a (1, 3) array
        cv2.circle(pc_in_image, (x, y), 3, tuple(int(c) for c in color), -1)
    return pc_in_image, out_depths

def draw_bev_props_mask(gt_bboxes_3d):
    occ_threshold = 0.3
    point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2.76]
    post_center_range = [x + y for x, y in zip(point_cloud_range, [-10, -10, -5, 10, 10, 5])]
    voxel_size = [0.32, 0.32, 5.76]
    grid_config = {
        'xbound': [point_cloud_range[0], point_cloud_range[3], voxel_size[0]],
        'ybound': [point_cloud_range[1], point_cloud_range[4], voxel_size[1]],
        'zbound': [point_cloud_range[2], point_cloud_range[5], voxel_size[2]],
        'dbound': [1.0, 49, 1.0]}
    bev_h_ = int((point_cloud_range[3] - point_cloud_range[0]) / voxel_size[0])
    bev_w_ = int((point_cloud_range[4] - point_cloud_range[1]) / voxel_size[1])
    bev_grid_shape = [bev_h_, bev_w_]
    bev_cell_size = [(grid_config['xbound'][1]-grid_config['xbound'][0])/bev_h_, (grid_config['ybound'][1]-grid_config['ybound'][0])/bev_w_]
    bev_mask = torch.zeros(bev_grid_shape)
    bbox_corners = gt_bboxes_3d.corners[:, [0,2,4,6],:2] # bev corners
    num_rectangles = bbox_corners.shape[0]
    bbox_corners[:,:,0] = (bbox_corners[:,:,0] - grid_config['xbound'][0])/bev_cell_size[0] # id_num, 4, 2
    bbox_corners[:,:,1] = (bbox_corners[:,:,1] - grid_config['ybound'][0])/bev_cell_size[1] # id_num, 4, 2
    # precise bur slow method
    grid_min = torch.clip(torch.floor(torch.min(bbox_corners, axis=1).values).to(torch.int64), 0, bev_grid_shape[0] - 1)
    grid_max = torch.clip(torch.ceil (torch.max(bbox_corners, axis=1).values).to(torch.int64), 0, bev_grid_shape[1] - 1)
    possible_mask_h_all = torch.cat([grid_min[:, 0:1], grid_max[:, 0:1]], dim=1).tolist()
    possible_mask_w_all = torch.cat([grid_min[:, 1:2], grid_max[:, 1:2]], dim=1).tolist()
    for n in range(num_rectangles):
        clock_corners = bbox_corners[n].cpu().numpy()[(0,1,3,2), :]
        poly = Polygon(clock_corners)
        h_list = possible_mask_h_all[n]; h_list = np.arange(h_list[0] - 1, h_list[1] + 1, 1); h_list = np.clip(h_list, 0, bev_grid_shape[0] - 1)
        w_list = possible_mask_w_all[n]; w_list = np.arange(w_list[0] - 1, w_list[1] + 1, 1); w_list = np.clip(w_list, 0, bev_grid_shape[1] - 1)
        for i in h_list:
            for j in w_list:
                cell_center = np.array([i + 0.5, j + 0.5])
                cell_poly = box(i, j, i + 1, j + 1)
                if poly.contains(Point(cell_center)):
                    bev_mask[i, j] = True
                else:
                    intersection = cell_poly.intersection(poly)
                    if (intersection.area / cell_poly.area) > occ_threshold: bev_mask[i, j] = True
    return 255.0*bev_mask.numpy().astype(np.float32)

def unproject_points(depth_image, img2lidar, segmentation_mask, crop_top_pixels=0):
    """
    Convert a depth image to a point cloud using the given img2lidar matrix.

    Parameters:
    - depth_image: numpy array of shape (H, W)
    - img2lidar: numpy array of shape (4, 4)
    - segmentation_mask: numpy array of shape (H, W), boolean mask indicating which pixels to unproject
    - crop_top_pixels: int, number of top pixels to crop

    Returns:
    - point_cloud: numpy array of shape (N, 3), where N is the number of valid points
    """
    H, W = depth_image.shape
    # Crop the top pixels
    top_mask = np.zeros_like(depth_image, dtype=np.bool_)
    top_mask[crop_top_pixels:, :] = True
    # Apply segmentation mask
    valid_mask = segmentation_mask & (depth_image > 0) & top_mask
    # Get coordinates of valid points
    valid_indices = np.where(valid_mask)
    u = valid_indices[1]  # x-coordinates
    v = valid_indices[0]  # y-coordinates
    z = depth_image[valid_indices]  # depth values
    # Convert pixel coordinates (u, v) to camera coordinates (x, y, z)
    x = u * z
    y = v * z
    # Create homogeneous coordinates
    ones = np.ones_like(z)
    points_homogeneous = np.stack([x, y, z, ones], axis=-1).T
    # Transform to LiDAR coordinate system
    points_lidar = img2lidar @ points_homogeneous
    # Normalize by the last row (if necessary)
    points_lidar = points_lidar[:3, :].T
    return points_lidar

def preprocess_depthwithmask(depth_map, mask): # original resolution
    # Define near masks based on depth ranges
    near_mask_1 = np.logical_and(mask, (0.00 <= depth_map) & (depth_map < 15.0))
    near_mask_2 = np.logical_and(mask, (15.0 <= depth_map) & (depth_map < 30.0))
    near_mask_3 = np.logical_and(mask, (30.0 <= depth_map) & (depth_map < 80.0))
    far_mask = np.logical_and(mask, depth_map >= 80.0)
    # Apply erosion to near masks with varying intensity
    eroded_near_mask_1 = cv2.erode(near_mask_1.astype(np.uint8), np.ones((5, 5), np.uint8), iterations=3).astype(np.bool_)
    eroded_near_mask_2 = cv2.erode(near_mask_2.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1).astype(np.bool_)
    eroded_near_mask_3 = cv2.erode(near_mask_3.astype(np.uint8), np.ones((1, 1), np.uint8), iterations=1).astype(np.bool_)
    # Create a final mask by merging eroded near masks
    combined_near_mask = np.logical_or(eroded_near_mask_1, eroded_near_mask_2)
    combined_near_mask = np.logical_or(combined_near_mask, eroded_near_mask_3)
    # Combine the eroded near masks and the untouched far mask into the final mask
    final_mask = np.where(combined_near_mask, combined_near_mask, far_mask)
    return final_mask

def process_mask_using_eroded(mask, depth_map, max_size=1000, max_size_ratio=None, max_depth_ratio=0.6):
    H, W = mask.shape
    if max_size_ratio is not None: 
        max_size = int(max_size_ratio*H*W)
    # Label connected components in the mask
    labeled_mask, num_features = label(mask)
    # Create an array to store the sizes of each connected component
    component_sizes = np.zeros(num_features + 1, dtype=np.int32)
    # Calculate the size of each connected component
    for i in range(1, num_features + 1):
        component_mask = (labeled_mask == i)
        component_size = np.sum(component_mask)
        component_sizes[i] = component_size
        
    # Erode components that are too large
    eroded_mask = np.zeros_like(mask)
    for i in range(1, num_features + 1):
        component_mask = (labeled_mask == i).astype(np.uint8)  
        iterations = max(component_sizes[i] // max_size, 1)  # Example: scale iterations, limit to 1
        eroded_component_mask = cv2.erode(component_mask, np.ones((3, 3), np.uint8), iterations=iterations).astype(np.bool_)
        eroded_mask = np.logical_or(eroded_mask, eroded_component_mask)

    # Relabel the eroded mask
    eroded_labeled_mask, eroded_num_features = label(eroded_mask)
    
    # Recalculate component sizes for eroded components
    final_component_sizes = np.zeros(eroded_num_features + 1, dtype=np.int32)
    final_mask = np.zeros_like(mask)
    
    # Apply depth map filtering and calculate final_component_sizes
    for i in range(1, eroded_num_features + 1):
        eroded_component_mask = (eroded_labeled_mask == i)
        final_component_sizes[i] = np.sum(eroded_component_mask)
        
        # Apply depth map filtering
        if final_component_sizes[i] > 0:
            depth_values = depth_map[eroded_component_mask]
            num_pixels_to_keep = int(max_depth_ratio * final_component_sizes[i])
            if num_pixels_to_keep > 0:
                threshold_value = np.partition(depth_values, num_pixels_to_keep)[num_pixels_to_keep]
                filtered_component_mask = np.logical_and(eroded_component_mask, depth_map <= threshold_value)
                final_mask = np.logical_or(final_mask, filtered_component_mask)
    
    return final_mask

def distinguish_forground_points(image, points, projection, seg_mask, bbox_corners):
    H, W, _  = image.shape
    forground_points_in_image = np.ascontiguousarray(copy.copy(image))
    pts_hom = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)  # Convert to homogeneous coordinates
    img_pts = np.matmul(projection, pts_hom.T).T  # Apply transformation
    img_pts[:, :2] = img_pts[:, :2] / img_pts[:, 2:3]  # Convert back to Cartesian coordinates
    # Filter points that are within the image boundaries
    valid_mask_1 = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < W) & (img_pts[:, 1] >= 0) & (img_pts[:, 1] < H)
    img_pts = img_pts[valid_mask_1]; points = points[valid_mask_1]
    x_indices = np.floor(img_pts[:, 0]).astype(int)
    y_indices = np.floor(img_pts[:, 1]).astype(int)
    # Filter points that are out depth range and seg_mask
    valid_mask_2 = (img_pts[:, 2] >= 0) & (img_pts[:, 2] <= 80) & (seg_mask[y_indices, x_indices]==True)
    img_pts = img_pts[valid_mask_2]; points = points[valid_mask_2]
    x_indices = np.floor(img_pts[:, 0]).astype(int)
    y_indices = np.floor(img_pts[:, 1]).astype(int)
    # Draw the points on the depth image in red or blue based on bbox_corners
    for _, (x, y, pts3d) in enumerate(zip(x_indices, y_indices, points)):
        point_in_bbox = False
        for bbox in bbox_corners:
            bbox = bbox.astype(np.float32)  # Ensure bbox is of type float32
            if cv2.pointPolygonTest(bbox, (pts3d[0], pts3d[1]), False) >= 0:
                point_in_bbox = True
                break
        color = (0, 255, 0) if point_in_bbox else (255, 0, 0)
        cv2.circle(forground_points_in_image, (x, y), 7, color, -1)
    # draw_bev_pts_bboxes(points, gt_bbox_corners=bbox_corners, save_path='1.png')
    return forground_points_in_image

def apply_3d_effect(image, mask):
    mask = mask.astype(np.uint8) * 255
    
    kernel = np.array([[0, -1, 0],
                       [-1, 4, -1],
                       [0, -1, 0]])
    edges = cv2.filter2D(image, -1, kernel)
    mask_3ch = cv2.merge([mask, mask, mask])
    image_with_edges = cv2.addWeighted(image, 0.7, edges, 0.3, 0)
    result = np.where(mask_3ch == 255, image_with_edges, image)
    shadow = cv2.GaussianBlur(mask, (21, 21), 10)
    highlight = cv2.GaussianBlur(mask, (21, 21), 5)

    shadow = shadow.astype(np.float32) * 0.4
    highlight = highlight.astype(np.float32) * 0.6
    shadow = cv2.merge([shadow * 0.5, shadow * 0.5, shadow]) 
    highlight = cv2.merge([highlight * 0.5, highlight * 0.5, highlight])
    result = result.astype(np.float32)

    result = np.where(mask_3ch == 255, cv2.addWeighted(result, 1, shadow, -1.5, 0), result)
    result = np.where(mask_3ch == 255, cv2.addWeighted(result, 1, highlight, 1.5, 0), result)
    result = np.clip(result, 0, 255).astype(np.uint8)
    # cv2.imwrite('result.png', result)
    return result

def scale_bbox(bbox, factor):
    x1, y1, x2, y2 = np.array(bbox).astype(np.int32)
    width = x2 - x1
    heigt = y2 - y1
    cx = x1 + width / 2
    cy = y1 + heigt / 2
    new_width = width * factor
    new_heigt = heigt * factor
    new_x1 = int(cx - new_width / 2)
    new_y1 = int(cy - new_heigt / 2)
    new_x2 = int(cx + new_width / 2)
    new_y2 = int(cy + new_heigt / 2)
    return new_x1, new_y1, new_x2, new_y2

def MVP_depth(gt_bboxes_2d, unproject_mask, radar_sparse_depths, predict_depth, N=1, threshold=8, depth_offset=3, set_bbox=(75, 1.8)):
    H, W = unproject_mask.shape
    depth_output = np.zeros((H, W))
    depth_inputs = np.zeros((H, W))

    for bbox in gt_bboxes_2d:
        x1, y1, x2, y2 = np.array(bbox).astype(np.int32)
        x_len = x2 - x1
        y_len = y2 - y1
        mask_forshow = np.zeros((H, W))
        mask_forshow[y1:y2,x1:x2] = 1
        
        if x2 - x1 < set_bbox[0] or y2 - y1 < set_bbox[0]:
            new_x1, new_y1, new_x2, new_y2 = scale_bbox(bbox, factor=set_bbox[1])
            # depth = (radar_sparse_depths[new_y1:new_y2, new_x1:new_x2]).mean()
            # if depth > 20:
            unproject_mask = np.ones_like(unproject_mask)
            x1, y1, x2, y2 = new_x1, new_y1, new_x2, new_y2

        depth_region = radar_sparse_depths[y1:y2, x1:x2]
        masks_region = unproject_mask[y1:y2, x1:x2]
        preds_region = predict_depth[y1:y2, x1:x2]
        
        
        valid_depth = depth_region[(depth_region > 0) & (masks_region > 0)]
        valid_coord = np.argwhere ((depth_region > 0) & (masks_region > 0))
        valid_preds = preds_region[(depth_region > 0) & (masks_region > 0)]
 
        if len(valid_depth) > 0:
            distance = np.abs(valid_depth-valid_preds)
            valid_depth = valid_depth[distance<threshold]
            valid_coord = valid_coord[distance<threshold]
            
        if len(valid_depth) > 0:
            mean = np.mean(valid_depth)
            stds = np.std (valid_depth)
            lower_bound = mean - 1 * stds
            upper_bound = mean + 1 * stds
            filter = (valid_depth >= lower_bound) & (valid_depth <= upper_bound)
            valid_depth = valid_depth[filter]
            valid_coord = valid_coord[filter]
        
        if len(valid_depth) > 0:
            for coord, depth in zip(valid_coord, valid_depth):
                root_y = np.clip(coord[0] + y1, 0, H-1)
                root_x = np.clip(coord[1] + x1, 0, W-1)
                depth_inputs[root_y, root_x] = depth
                
                # random_offset = np.random.randint(0, pixel_offset, size=(len(valid_coord), N, 2))  # (M, N, 2)
                radom_x = np.random.randint(0, np.max([x_len*0.05, 1]), size=(len(valid_coord), N, 1))
                radom_y = np.random.randint(0, np.max([y_len*0.05, 1]), size=(len(valid_coord), N, 1))
                random_offset = np.concatenate([radom_x, radom_y], axis=-1)
                random_coords = valid_coord[:, np.newaxis, :] + random_offset  # (M, N, 2)
                new_coords = random_coords + np.array([y1, x1])  # (M*N, 2)
                valid_mask = (new_coords[..., 0] >= 0) & (new_coords[..., 0] < H) & \
                         (new_coords[..., 1] >= 0) & (new_coords[..., 1] < W)
                valid_new_coords = new_coords[valid_mask].reshape(-1, 2)  # (K, 2)
                valid_depth_values = np.repeat(valid_depth[:, np.newaxis], N, axis=1).flatten()[valid_mask.flatten()]
                random_depths = np.random.rand(valid_depth_values.shape[0])*depth_offset
                depth_output[valid_new_coords[:, 0], valid_new_coords[:, 1]] = valid_depth_values + random_depths

        # if len(valid_depth) > 0:
        #     N = len(valid_depth)*2
        #     sampled_x = np.random.randint(x1, x2, N)
        #     sampled_y = np.random.randint(y1, y2, N)
        #     sampled_coords = np.vstack((sampled_y-y1, sampled_x-x1)).T 
        #     distances = np.linalg.norm(valid_coord[:, np.newaxis, :] - sampled_coords[np.newaxis, :, :], axis=2)  # (M, N)
        #     distances[distances == 0] = 1e-3
        #     weights = 1 / distances  # (M, N)
        #     weighted_depths = np.sum(valid_depth[:, np.newaxis] * weights, axis=0) / np.sum(weights, axis=0)  # (N,)
        #     depth_output[sampled_y, sampled_x] = weighted_depths
        
    return depth_output # depth_inputs depth_output

def load_raw_datas():
    # choose which to draw
    file_index = '01477' #'01035' '03506' '03331' '02087' '03893' '04524'
    
    # preliminary settins
    out_dir = './docs/test'
    os.makedirs(out_dir, exist_ok=True)
    root_path = 'data/VoD/radar_5frames'
    CLASSES = ('car', 'pedestrian', 'cyclist')
    ann_file = 'data/VoD/radar_5frames/vod_infos_train.pkl'
    annos_all = mmcv.load(ann_file)
    annos_ind = annos_all[find_index(annos_all, file_index)]
    annos = annos_ind['annos']

    # load depth from both file
    depth_path_1='data/VoD/depth_anything_crop400/' + file_index + '.png'
    depth_path_2='data/VoD/NLSPN/' + file_index + '.npy'
    depth_path_3='data/VoD/depth_anything_upfromds2/' + file_index + '.npy'
    depth_path_4='data/VoD/depth_anything_v2_inverse_depth/' + file_index + '.npy'
    depth_dc_1, depth_dc_2, depth_dc_3 = get_all_depth(depth_path_1, depth_path_2, depth_path_3)
    depth_dc_4 = np.load(depth_path_4)
    # depth_dc_1[:400, :] = depth_dc_2[:400, :]
    depth_dc_1_color = colorize(depth_dc_1, vmin=0.0, vmax=80.0)
    Image.fromarray(depth_dc_1_color).save(os.path.join(out_dir, 'extra_depth_1.png'))
    depth_dc_2_color = colorize(depth_dc_2, vmin=0.0, vmax=80.0)
    Image.fromarray(depth_dc_2_color).save(os.path.join(out_dir, 'extra_depth_2.png'))
    depth_dc_3_color = colorize(depth_dc_3, vmin=0.0, vmax=80.0)
    Image.fromarray(depth_dc_3_color).save(os.path.join(out_dir, 'extra_depth_3.png'))
    depth_dc_4_color = colorize(depth_dc_4, vmin=0.0, vmax=80.0)
    Image.fromarray(depth_dc_4_color).save(os.path.join(out_dir, 'extra_depth_4.png'))
    # cv2.imwrite(os.path.join(out_dir, 'extra_depth_1.png'), depth_dc_1)
    # cv2.imwrite(os.path.join(out_dir, 'extra_depth_2.png'), depth_dc_2)
    
    # load raw image
    image_path = annos_ind['image']['image_path']
    image_path = os.path.join(root_path, image_path)
    image = cv2.imread(image_path)
    H, W, C = image.shape
    cv2.imwrite(os.path.join(out_dir, 'raw_image.png'), image)
    
    # preprcocessing bbox
    radar2cam = annos_ind['calib']['R0_rect'] @ annos_ind['calib']['Tr_velo_to_cam']
    gt_bbox_dict = get_gt_bbox(annos, CLASSES, radar2cam)
    gt_bboxes_3d = gt_bbox_dict['gt_bboxes_3d']
    gt_bboxes_2d = gt_bbox_dict['gt_bboxes_2d']
    bbox_corners = gt_bboxes_3d.corners[:, [0,2,4,6],:2]
    bbox_corners = bbox_corners.numpy()[:, (0,1,3,2), :] # clock_corners
    
    # load segmentation mask
    seg_mask_1 = np.zeros((H, W))
    for bbox in gt_bboxes_2d:
        x1, y1, x2, y2 = np.array(bbox).astype(np.int32)
        seg_mask_1[y1:y2, x1:x2] = 255.0
    seg_path='data/VoD/segmentation/' + file_index + '.png'
    cv2.imwrite(os.path.join(out_dir, 'seg_mask_1.png'), seg_mask_1)
    seg_mask_2 = np.array(Image.open(seg_path), dtype=np.bool_)*255.0
    cv2.imwrite(os.path.join(out_dir, 'seg_mask_2.png'), seg_mask_2)
    
    # draw bbox_Mask
    alpha, gray = 0.80, 0
    bgdark_mask_1 = np.full_like(image, (gray, gray, gray), dtype=np.uint8)
    bgdark_mask_2 = np.full_like(image, (gray, gray, gray), dtype=np.uint8)
    mask_1 = (seg_mask_1 == 255)
    mask_2 = (seg_mask_2 == 255)
    bgdark_mask_1[mask_1] = image[mask_1]
    bgdark_mask_2[mask_2] = image[mask_2]
    mask_with_img_1 = cv2.addWeighted(image, 1 - alpha, bgdark_mask_1, alpha, 0)
    mask_with_img_2 = cv2.addWeighted(image, 1 - alpha, bgdark_mask_2, alpha, 0)
    # Edge detection
    edges_1 = cv2.Canny(seg_mask_1.astype(np.uint8), 100, 200)
    edges_2 = cv2.Canny(seg_mask_2.astype(np.uint8), 100, 200)
    # Connect and thicken edges
    kernel = np.ones((3, 3), np.uint8)  # You can adjust the kernel size to change the thickness
    edges_1 = cv2.dilate(edges_1, kernel, iterations=3)  # Dilate to connect and thicken edges
    edges_2 = cv2.dilate(edges_2, kernel, iterations=3)
    # Create red edge overlay
    mask_with_img_1[edges_1 > 0] = [0, 69, 255]  # Red color for edges in mask 1
    mask_with_img_2[edges_2 > 0] = [0, 69, 255]  # Red color for edges in mask 2
    mask_3d_image = apply_3d_effect(image, seg_mask_2.astype(np.bool_))
    mask_with_img_3 = copy.copy(mask_3d_image)
    mask_with_img_3[edges_2 > 0] = [0, 69, 255]  # Red color for edges in mask 2
    cv2.imwrite(os.path.join(out_dir, 'forground_seg.png'), mask_3d_image)
    cv2.imwrite(os.path.join(out_dir, 'mask_with_img_1.png'), mask_with_img_1)
    cv2.imwrite(os.path.join(out_dir, 'mask_with_img_2.png'), mask_with_img_2)
    cv2.imwrite(os.path.join(out_dir, 'mask_with_img_3.png'), mask_with_img_3)
                        
    # draw radar bev
    P2 = annos_ind['calib']['P2']
    rect = annos_ind['calib']['R0_rect']
    Trv2c = annos_ind['calib']['Tr_velo_to_cam']
    radar2img = P2 @ rect @ Trv2c
    radar_filename = annos_ind['point_cloud']['velodyne_path']
    radar_filename = os.path.join(root_path, radar_filename)
    radar_points = get_pointcloud(radar_filename, load_dim=7)
    draw_bev_pts_bboxes(radar_points, gt_bbox_corners=bbox_corners, save_path=os.path.join(out_dir, 'radar_points_bev_withgt.png'))
    draw_bev_pts_bboxes(radar_points, gt_bbox_corners=None, save_path=os.path.join(out_dir, 'radar_points_bev.png'))
    show_multi_modality_result(img=image, gt_bboxes=gt_bboxes_3d, pred_bboxes=None, proj_mat=radar2img, out_dir=out_dir, filename='proj_bbox_on_img', box_mode='lidar', show=False)
    
    # draw lidar bev
    lidar_calib_path = 'data/VoD/lidar/training/calib/' + file_index + '.txt'
    lidar_filename = lidar_calib_path.replace('calib', 'velodyne').replace('txt','bin')
    P2, rect, Trv2c = get_velodyne2img(lidar_calib_path)
    lidar2img = P2 @ rect @ Trv2c
    lidar_points = get_pointcloud(lidar_filename, load_dim=4)
    lidar2radar =  np.matmul(np.linalg.inv(radar2img), lidar2img)
    lidar_points_under_radar = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
    lidar_points_under_radar = np.matmul(lidar_points_under_radar, lidar2radar.T)[:, :3]
    draw_bev_pts_bboxes(lidar_points_under_radar, gt_bbox_corners=bbox_corners, save_path=os.path.join(out_dir, 'lidar_points_bev_withgt.png'))
    draw_bev_pts_bboxes(lidar_points_under_radar, gt_bbox_corners=None, save_path=os.path.join(out_dir, 'lidar_points_bev.png'))
    
    # draw points in image
    lidar_in_image_1, lidar_sparse_depths = draw_point_on_image(np.ones_like(image), lidar_points, lidar2img)
    cv2.imwrite(os.path.join(out_dir, 'lidar_on_image_1.png'), lidar_in_image_1)
    lidar_in_image_2, lidar_sparse_depths = draw_point_on_image(image, lidar_points, lidar2img)
    cv2.imwrite(os.path.join(out_dir, 'lidar_on_image_2.png'), lidar_in_image_2)
    radar_in_image_1, radar_sparse_depths = draw_point_on_image(np.ones_like(image), radar_points, radar2img)
    cv2.imwrite(os.path.join(out_dir, 'radar_on_image_1.png'), radar_in_image_1)
    radar_in_image_2, radar_sparse_depths = draw_point_on_image(image, radar_points, radar2img)
    cv2.imwrite(os.path.join(out_dir, 'radar_on_image_2.png'), radar_in_image_2)
    
    # bev proposal mask
    bev_mask = np.rot90(draw_bev_props_mask(gt_bboxes_3d), 3)
    cv2.imwrite(os.path.join(out_dir, 'bev_mask_object.png'), bev_mask)
    cv2.imwrite(os.path.join(out_dir, 'bev_mask_background.png'), 255-bev_mask)
    
    # draw depth2lidar complete radar points using bbox_mask
    unproject_mask = seg_mask_2
    # depth_input = lidar_sparse_depths # depth_dc_3 # depth_dc_1 # lidar_sparse_depths
    depth_input = MVP_depth(gt_bboxes_2d, unproject_mask, radar_sparse_depths, predict_depth=depth_dc_4, N=1, depth_offset=1.0)
    # downsample unprojection
    downsample = 8
    matrix = np.eye(4)
    matrix[0, 0] = matrix[0, 0] / downsample
    matrix[1, 1] = matrix[1, 1] / downsample
    img2radar = np.linalg.inv(matrix @ radar2img)
    unproject_mask = cv2.resize(unproject_mask.astype(np.float32), (W//downsample, H//downsample), interpolation=cv2.INTER_CUBIC).astype(np.bool_)            
    depth_input = get_downsample_depths_numpy(depth_input, downsample, processing='min')
    # unproject_mask = preprocess_depthwithmask(depth_input, unproject_mask)
    # unproject_mask = process_mask_using_eroded(unproject_mask, depth_input, max_size=1000, max_depth_ratio=0.6)
    virtual_points = unproject_points(depth_input, img2radar, np.ones_like(unproject_mask), crop_top_pixels=int(unproject_mask.shape[0]*0.4))
    draw_bev_pts_bboxes(virtual_points, gt_bbox_corners=bbox_corners, save_path=os.path.join(out_dir, 'radar_points_bev_withgt_virtuals.png'))
    # draw_bev_pts_bboxes(virtual_points, gt_bbox_corners=None, save_path=os.path.join(out_dir, 'radar_points_bev_virtuals.png'))
    # complete_points = np.concatenate([virtual_points[:,:3], radar_points[:,:3]], axis=0)
    draw_bev_pts_bboxes(radar_points[:,:3], gt_bbox_corners=bbox_corners, save_path=os.path.join(out_dir, 'radar_points_bev_withgt_complete.png'))
    # draw_bev_pts_bboxes(radar_points[:,:3], gt_bbox_corners=None, save_path=os.path.join(out_dir, 'radar_points_bev_complete.png'))
    foreground_radar_on_image = distinguish_forground_points(mask_with_img_3, radar_points, radar2img, seg_mask_2.astype(np.bool_), bbox_corners) # mask_with_img_3 mask_3d_image
    cv2.imwrite(os.path.join(out_dir, 'foreground_radar_on_image.png'), foreground_radar_on_image)
    foreground_lidar_on_image = distinguish_forground_points(mask_with_img_3, lidar_points_under_radar, radar2img, seg_mask_2.astype(np.bool_), bbox_corners)
    cv2.imwrite(os.path.join(out_dir, 'foreground_lidar_on_image.png'), foreground_lidar_on_image)
    
def virtual_points():
    # choose which to draw
    texts_path = 'data/VoD/radar_5frames/ImageSets/train.txt'
    lines_list = []
    with open(texts_path, 'r') as f:
        for line in f:
            line_list = line.strip()
            lines_list.append(line_list)
    # lines_list = ['00677']        
    for file_index in lines_list:
        print('processing %s ing'%(file_index))
        # preliminary settins
        out_dir = './docs/virtual_points'
        os.makedirs(out_dir, exist_ok=True)
        root_path = 'data/VoD/radar_5frames'
        CLASSES = ('car', 'pedestrian', 'cyclist')
        ann_file = 'data/VoD/radar_5frames/vod_infos_train.pkl'
        annos_all = mmcv.load(ann_file)
        annos_ind = annos_all[find_index(annos_all, file_index)]
        annos = annos_ind['annos']

        # load depth from both file
        depth_path_1='data/VoD/depth_anything_crop400/' + file_index + '.png'
        depth_path_2='data/VoD/NLSPN/' + file_index + '.npy'
        depth_path_3='data/VoD/depth_anything_upfromds2/' + file_index + '.npy'
        depth_path_4='data/VoD/depth_anything_v2_inverse_depth/' + file_index + '.npy'
        depth_dc_1, depth_dc_2, depth_dc_3 = get_all_depth(depth_path_1, depth_path_2, depth_path_3)
        depth_dc_4 = np.load(depth_path_4)
        
        # load raw image
        image_path = annos_ind['image']['image_path']
        image_path = os.path.join(root_path, image_path)
        image = cv2.imread(image_path)
        H, W, C = image.shape
        # cv2.imwrite(os.path.join(out_dir, 'raw_image.png'), image)
        
        # preprcocessing bbox
        radar2cam = annos_ind['calib']['R0_rect'] @ annos_ind['calib']['Tr_velo_to_cam']
        gt_bbox_dict = get_gt_bbox(annos, CLASSES, radar2cam)
        gt_bboxes_3d = gt_bbox_dict['gt_bboxes_3d']
        gt_bboxes_2d = gt_bbox_dict['gt_bboxes_2d']
        if len(gt_bboxes_3d) == 0: continue
        bbox_corners = gt_bboxes_3d.corners[:, [0,2,4,6],:2]
        bbox_corners = bbox_corners.numpy()[:, (0,1,3,2), :] # clock_corners
        
        # load segmentation mask
        seg_mask_1 = np.zeros((H, W))
        for bbox in gt_bboxes_2d:
            x1, y1, x2, y2 = np.array(bbox).astype(np.int32)
            seg_mask_1[y1:y2, x1:x2] = 255.0
        seg_path='data/VoD/segmentation/' + file_index + '.png'
        # cv2.imwrite(os.path.join(out_dir, 'seg_mask_1.png'), seg_mask_1)
        seg_mask_2 = np.array(Image.open(seg_path), dtype=np.bool_)*255.0
        # cv2.imwrite(os.path.join(out_dir, 'seg_mask_2.png'), seg_mask_2)
                
        # draw radar bev
        P2 = annos_ind['calib']['P2']
        rect = annos_ind['calib']['R0_rect']
        Trv2c = annos_ind['calib']['Tr_velo_to_cam']
        radar2img = P2 @ rect @ Trv2c
        radar_filename = annos_ind['point_cloud']['velodyne_path']
        radar_filename = os.path.join(root_path, radar_filename)
        radar_points = get_pointcloud(radar_filename, load_dim=7)
        # draw_bev_pts_bboxes(radar_points, gt_bbox_corners=bbox_corners, save_path=os.path.join(out_dir, 'radar_points_bev_withgt.png'))
        # draw_bev_pts_bboxes(radar_points, gt_bbox_corners=None, save_path=os.path.join(out_dir, 'radar_points_bev.png'))
        # show_multi_modality_result(img=image, gt_bboxes=gt_bboxes_3d, pred_bboxes=None, proj_mat=radar2img, out_dir=out_dir, filename='proj_bbox_on_img', box_mode='lidar', show=False)
        
        # draw lidar bev
        lidar_calib_path = 'data/VoD/lidar/training/calib/' + file_index + '.txt'
        lidar_filename = lidar_calib_path.replace('calib', 'velodyne').replace('txt','bin')
        P2, rect, Trv2c = get_velodyne2img(lidar_calib_path)
        lidar2img = P2 @ rect @ Trv2c
        lidar_points = get_pointcloud(lidar_filename, load_dim=4)
        lidar2radar =  np.matmul(np.linalg.inv(radar2img), lidar2img)
        lidar_points_under_radar = np.hstack((lidar_points, np.ones((lidar_points.shape[0], 1))))
        lidar_points_under_radar = np.matmul(lidar_points_under_radar, lidar2radar.T)[:, :3]
        # draw_bev_pts_bboxes(lidar_points_under_radar, gt_bbox_corners=bbox_corners, save_path=os.path.join(out_dir, 'lidar_points_bev_withgt.png'))
        # draw_bev_pts_bboxes(lidar_points_under_radar, gt_bbox_corners=None, save_path=os.path.join(out_dir, 'lidar_points_bev.png'))
        
        # draw points in image
        lidar_in_image_1, lidar_sparse_depths = draw_point_on_image(np.ones_like(image), lidar_points, lidar2img)
        lidar_in_image_2, lidar_sparse_depths = draw_point_on_image(image, lidar_points, lidar2img)
        radar_in_image_1, radar_sparse_depths = draw_point_on_image(np.ones_like(image), radar_points, radar2img)
        radar_in_image_2, radar_sparse_depths = draw_point_on_image(image, radar_points, radar2img)

        
        # draw depth2lidar complete radar points using bbox_mask
        unproject_mask = seg_mask_2
        # depth_input = lidar_sparse_depths # depth_dc_3 # depth_dc_1 # lidar_sparse_depths
        depth_input = MVP_depth(gt_bboxes_2d, unproject_mask, radar_sparse_depths, predict_depth=depth_dc_4, N=1, depth_offset=0.8)
        # downsample unprojection
        downsample = 8
        matrix = np.eye(4)
        matrix[0, 0] = matrix[0, 0] / downsample
        matrix[1, 1] = matrix[1, 1] / downsample
        img2radar = np.linalg.inv(matrix @ radar2img)
        unproject_mask = cv2.resize(unproject_mask.astype(np.float32), (W//downsample, H//downsample), interpolation=cv2.INTER_CUBIC).astype(np.bool_)            
        depth_input = get_downsample_depths_numpy(depth_input, downsample, processing='min')
        # unproject_mask = preprocess_depthwithmask(depth_input, unproject_mask)
        # unproject_mask = process_mask_using_eroded(unproject_mask, depth_input, max_size=1000, max_depth_ratio=0.6)
        virtual_points = unproject_points(depth_input, img2radar, np.ones_like(unproject_mask), crop_top_pixels=int(unproject_mask.shape[0]*0.4))
        draw_bev_pts_bboxes(virtual_points, gt_bbox_corners=bbox_corners, save_path=os.path.join(out_dir, file_index + '_radar_points_bev_withgt_virtuals.png'))
        # draw_bev_pts_bboxes(radar_points[:,:3], gt_bbox_corners=bbox_corners, save_path=os.path.join(out_dir, file_index + '_radar_points_bev_withgt_complete.png'))


if __name__ == '__main__':
    load_raw_datas()
    # virtual_points()

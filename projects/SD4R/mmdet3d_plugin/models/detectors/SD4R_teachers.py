import torch, copy, time, os, mmcv, cv2
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
from shapely.geometry import Polygon, box, Point
from torchvision.utils import save_image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from mmcv.runner.dist_utils import master_only
from mmcv import Config, DictAction
from mmdet.models import DETECTORS, ROI_EXTRACTORS, build_detector
from mmdet.models.backbones.resnet import BasicBlock
from mmdet3d.ops import Voxelization
from mmdet3d.core.bbox import box_np_ops
from mmdet3d.models import build_model
from mmdet3d.models import builder
from mmdet3d.models.builder import FUSION_LAYERS, BACKBONES, LOSSES, MIDDLE_ENCODERS
from mmdet3d.models.detectors import MVXFasterRCNN
from mmdet3d.core import bbox3d2result, show_multi_modality_result
from ...datasets.structures.bbox import HorizontalBoxes
from ...utils.visualization import draw_bev_pts_bboxes, draw_paper_bboxes
from ...utils.visualization import custom_draw_lidar_bbox3d_on_img
from ...utils.depth_tools import draw_sum_depth, draw_true_depth, generate_guassian_depth_target
from ...utils import filter_boxes_by_iou

@DETECTORS.register_module()
class SD4R_teachers(MVXFasterRCNN):
    """Multi-modality BEVFusion using Faster R-CNN."""

    def __init__(self, 
                # hyper parameter
                _dim_ =256, 
                bev_h_=160,
                bev_w_=160,
                img_channels=256, 
                rad_channels=384,
                num_classes=3,
                downsample=8,
                point_cloud_range=None,
                grid_config=None, 
                img_norm_cfg=None,
                SAVE_INTERVALS=10,
                # loss settings
                box3d_supervision=None,
                depth_supervision=None,
                msk2d_supervision=None,
                props_supervision=None,
                focus_supervision=None,
                # architecture 
                aux_bbox_head = dict(point=True, image=None, weight=0.1),
                depth_complet = dict(point_depth=True, extra_depth=False),
                camera_stream = dict(aware=dict(depth=True, pixel=True)),
                distill_setts = dict(use=False, semi=False, teacher_cfg=None, checkpoint=None),
                points_stream = dict(fuses_drop=0.0, radar_drop=0.0),
                architectures = dict(DCN=False, use_QFP=False, point_supervision=None, radar_supervision=None),
                # training details
                use_grid_mask=False,
                freeze_depths=False,
                freeze_radars=False,
                freeze_images=True,
                freeze_ranges=False,
                freeze_propss=False,
                # framework config
                depth_net=None,
                pvseg_net=None,
                fuses_net=None,
                bvseg_net=None,
                focus_net=None,
                warps_net=None,
                qfuse_net=None,
                point_net=None,
                meta_info=None,
                **kwargs):
        kwargs = self.preprocess_kwargs(kwargs)
        super(SD4R_teachers, self).__init__(**kwargs)

        # hyper parameter
        self._dim_  = _dim_
        self.bev_h_ = bev_h_
        self.bev_w_ = bev_w_
        self.img_channels = img_channels
        self.rad_channels = rad_channels
        self.num_classes = num_classes

        self.downsample = downsample
        self.point_cloud_range = point_cloud_range
        self.grid_config = grid_config
        self.img_norm_cfg = img_norm_cfg
        self.SAVE_INTERVALS = SAVE_INTERVALS
        # loss settings
        self.box3d_supervision = box3d_supervision
        self.depth_supervision = depth_supervision
        self.msk2d_supervision = msk2d_supervision
        self.props_supervision = props_supervision
        self.focus_supervision = focus_supervision
        # architecture
        self.aux_bbox_head = aux_bbox_head
        self.camera_stream = camera_stream
        self.points_stream = points_stream
        self.depth_complet = depth_complet
        self.distill_setts = distill_setts
        self.architectures = architectures
        # training details
        self.use_grid_mask = use_grid_mask
        self.freeze_images = freeze_images
        self.freeze_depths = freeze_depths
        self.freeze_radars = freeze_radars
        self.freeze_ranges = freeze_ranges
        self.freeze_propss = freeze_propss
        # meta infos
        self.meta_info = meta_info
        self.figures_path = meta_info['figures_path']
        self.project_name = meta_info['project_name']
        if 'vod' in self.project_name.lower(): self.dataset_type = 'VoD'
        if 'tj4d' in self.project_name.lower(): self.dataset_type = 'TJ4D'
        
        # other parameter for convenience
        self.xbound = self.grid_config['xbound']
        self.ybound = self.grid_config['ybound']
        self.zbound = self.grid_config['zbound']
        self.dbound = self.grid_config['dbound']
        self.D = int((self.dbound[1]-self.dbound[0])//self.dbound[2])
        self.bev_grid_shape = [bev_h_, bev_w_]
        self.bev_cell_size = [(self.xbound[1]-self.xbound[0])/bev_h_, (self.ybound[1]-self.ybound[0])/bev_w_]
        self.voxel_size = [self.grid_config['xbound'][2], self.grid_config['ybound'][2], self.grid_config['zbound'][2]]
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        self.xlim, self.ylim = [x_min, x_max], [y_min, y_max]
        # teacher configs
        self.depth_aware = self.camera_stream['aware']['depth']
        self.pixel_aware = self.camera_stream['aware']['pixel']
        depth_net.update(figures_path=self.figures_path)
        self.downsample = depth_net['downsample']
        self.fuses_net_dict = copy.deepcopy(fuses_net)
        self.overall_dropout=self.points_stream['fuses_drop']
        self.feats_1_dropout=self.points_stream['radar_drop']
        
        # for teacher and student framework design
        self.fuses_net = FUSION_LAYERS.build(fuses_net) if fuses_net else None
        self.focus_net = FUSION_LAYERS.build(focus_net) if (focus_net and self.focus_supervision['use']) else None
        self.bvseg_net = FUSION_LAYERS.build(bvseg_net) if (bvseg_net and self.props_supervision['use']) else None
        self.depth_net = FUSION_LAYERS.build(depth_net) if (depth_net and self.depth_supervision['use']) else None
        self.pvseg_net = FUSION_LAYERS.build(pvseg_net) if (pvseg_net and self.msk2d_supervision['use']) else None
        self.dfa3d_net = FUSION_LAYERS.build(self.dfa3d_net) if self.dfa3d_net is not None else None

        # init weights and freeze if needed
        self.init_default_modules()
        self.init_weights()
        if self.freeze_images: self.freeze_img_model()
        if self.freeze_radars: self.freeze_pts_model()
        if self.freeze_ranges: self.freeze_msk_model()
        self.record_fps = {'num': 0, 'time':0}
        self.init_visulization()
    def preprocess_kwargs(self, kwargs):
        # lidar feature extractor backbone
        self.pts_voxel_layer_extra = kwargs.get('pts_voxel_layer_extra', kwargs.get('pts_voxel_layer'))
        self.pts_voxel_encoder_extra = kwargs.get('pts_voxel_encoder_extra', kwargs.get('pts_voxel_encoder'))
        self.pts_middle_encoder_extra = kwargs.get('pts_middle_encoder_extra', kwargs.get('pts_middle_encoder'))
        self.pts_backbone_extra = kwargs.get('pts_backbone_extra', kwargs.get('pts_backbone'))
        self.pts_neck_extra = kwargs.get('pts_neck_extra', kwargs.get('pts_neck'))
        if kwargs.get('pts_voxel_layer_extra', None) is not None: kwargs.pop('pts_voxel_layer_extra')
        if kwargs.get('pts_voxel_encoder_extra', None) is not None:kwargs.pop('pts_voxel_encoder_extra')
        if kwargs.get('pts_middle_encoder_extra', None) is not None:kwargs.pop('pts_middle_encoder_extra')
        if kwargs.get('pts_backbone_extra', None) is not None:kwargs.pop('pts_backbone_extra')
        if kwargs.get('pts_neck_extra', None) is not None:kwargs.pop('pts_neck_extra')
        # rebuild pts_bbox_head
        self.pts_bbox_head = kwargs.pop('pts_bbox_head')
        # if need, dfa3d net build
        self.dfa3d_net = kwargs.get('dfa3d_net', None)
        if self.dfa3d_net is not None: kwargs.pop('dfa3d_net')
        return kwargs
     
    def init_default_modules(self):
        if self.focus_supervision['use']:
            self.token = nn.Embedding(1, self._dim_)
            self.MHSA_1 = nn.MultiheadAttention(embed_dim=self._dim_, num_heads=8)
            self.MHSA_2 = nn.MultiheadAttention(embed_dim=self._dim_, num_heads=8)
        if self.pts_bbox_head and self.box3d_supervision['use']:
            pts_train_cfg = self.train_cfg.pts if self.train_cfg else None
            self.pts_bbox_head.update(train_cfg=pts_train_cfg)
            pts_test_cfg = self.test_cfg.pts if self.test_cfg else None
            self.pts_bbox_head.update(test_cfg=pts_test_cfg)
            self.pts_bbox_head_dict = self.pts_bbox_head
            self.pts_bbox_head = builder.build_head(self.pts_bbox_head)
        # for teacher framework design
        # 1. camera 3D-to-2D mapping
        self.num_in_height = self.camera_stream['height']
        self.voxelpainting_points, self.voxel_coords = self.generate_pillar_ref_points(self.num_in_height)
        self.mapcollapse = nn.Sequential(
            nn.Conv2d(self.num_in_height * self.img_channels, self.img_channels, kernel_size=1),
            BasicBlock(self.img_channels, self.img_channels))
        # extra feature extractor for lidar
        if self.pts_voxel_encoder_extra['type'] == 'RadarPillarFeatureNet':
            self.pts_voxel_encoder_extra['type'] = 'PillarFeatureNet'
            self.pts_voxel_encoder_extra['in_channels'] = 4
            self.pts_voxel_encoder_extra.pop('with_velocity_snr_center')
            self.pts_voxel_layer_extra = self.pts_voxel_layer
        else: self.pts_voxel_layer_extra = Voxelization(**self.pts_voxel_layer_extra)
        self.pts_voxel_encoder_extra = builder.build_voxel_encoder(self.pts_voxel_encoder_extra)
        self.pts_middle_encoder_extra = builder.build_middle_encoder(self.pts_middle_encoder_extra)
        self.pts_backbone_extra = builder.build_backbone(self.pts_backbone_extra)
        self.pts_neck_extra = builder.build_neck(self.pts_neck_extra)
        # lidar and radar feature fusion module
        self.fuses_net_dict['img_channels'] = self.rad_channels
        self.fuses_net_dict['rad_channels'] = self.rad_channels
        self.fuses_net_dict['out_channels'] = self.rad_channels
        self.lidar_radar_fusion = FUSION_LAYERS.build(self.fuses_net_dict)
        # auxiliary head for modality specific loss
        if self.pts_bbox_head and self.box3d_supervision['use'] and self.aux_bbox_head is not None:   
            self.aux_loss_weight = self.aux_bbox_head['weight']   
            if 'point' in self.aux_bbox_head and self.aux_bbox_head['point']:
                dict_aux_point = copy.deepcopy(self.pts_bbox_head_dict)
                dict_aux_point['in_channels'] = self.rad_channels
                dict_aux_point['feat_channels'] = self.rad_channels
                self.aux_bbox_head_point = builder.build_head(dict_aux_point)
            if 'image' in self.aux_bbox_head and self.aux_bbox_head['image']:
                dict_aux_image = copy.deepcopy(self.pts_bbox_head_dict)
                dict_aux_image['in_channels'] = self.img_channels
                dict_aux_image['feat_channels'] = self.img_channels
                self.aux_bbox_head_image = builder.build_head(dict_aux_image)
        else: self.aux_bbox_head_image, self.aux_bbox_head_point = None, None

    def init_visulization(self):
        # self.SAVE_INTERVALS = 250 # 500
        self.vis_time_box3d = 0
        self.vis_time_bev2d = 0
        self.vis_time_bevnd = 0
        self.vis_time_range = 0
        self.vis_time_depth = 0
        self.vis_time_point = 0
        self.mean=np.array(self.img_norm_cfg['mean'])
        self.std=np.array(self.img_norm_cfg['std'])
        self.figures_path_det3d_test = os.path.join(self.figures_path, 'test', 'det3d')
        self.figures_path_bev2d_test = os.path.join(self.figures_path, 'test', 'bev_mask')
        self.figures_path_bevnd_test = os.path.join(self.figures_path, 'test', 'bev_feats')
        self.figures_path_range_test = os.path.join(self.figures_path, 'test', 'range')
        self.figures_path_depth_test = os.path.join(self.figures_path, 'test', 'depth')
        self.figures_path_point_test = os.path.join(self.figures_path, 'test', 'point')
        self.figures_path_det3d_train = os.path.join(self.figures_path, 'train', 'det3d')
        self.figures_path_bev2d_train = os.path.join(self.figures_path, 'train', 'bev_mask')
        self.figures_path_bevnd_train = os.path.join(self.figures_path, 'train', 'bev_feats')
        self.figures_path_range_train = os.path.join(self.figures_path, 'train', 'range')
        self.figures_path_depth_train = os.path.join(self.figures_path, 'train', 'depth')
        self.figures_path_point_train = os.path.join(self.figures_path, 'train', 'point')
        os.makedirs(self.figures_path_det3d_test, exist_ok=True)
        os.makedirs(self.figures_path_bev2d_test, exist_ok=True)
        os.makedirs(self.figures_path_bevnd_test, exist_ok=True)
        os.makedirs(self.figures_path_range_test, exist_ok=True)
        os.makedirs(self.figures_path_depth_test, exist_ok=True)
        os.makedirs(self.figures_path_point_test, exist_ok=True)
        os.makedirs(self.figures_path_det3d_train, exist_ok=True)
        os.makedirs(self.figures_path_bev2d_train, exist_ok=True)
        os.makedirs(self.figures_path_bevnd_train, exist_ok=True)
        os.makedirs(self.figures_path_range_train, exist_ok=True)
        os.makedirs(self.figures_path_depth_train, exist_ok=True)
        os.makedirs(self.figures_path_point_train, exist_ok=True)
            
    # model parameter freezing or not 
    
    def freeze_img_model(self):
        """freeze image backbone and neck for fusion"""
        if self.with_img_backbone:
            for param in self.img_backbone.parameters():
                param.requires_grad = False
        if self.with_img_neck:
            for param in self.img_neck.parameters():
                param.requires_grad = False
    
    def freeze_msk_model(self):
        if self.pvseg_net is not None:
            for param in self.pvseg_net.parameters():
                param.requires_grad = False

    def freeze_pts_model(self):
        """freeze radar backbone and neck for pretrain"""
        if self.pts_voxel_encoder:
            for param in self.pts_voxel_encoder.parameters():
                param.requires_grad = False
        if self.pts_middle_encoder:
            for param in self.pts_middle_encoder.parameters():
                param.requires_grad = False
        if self.pts_backbone:
            for param in self.pts_backbone.parameters():
                param.requires_grad = False
        if self.pts_neck is not None:
            for param in self.pts_neck.parameters():
                param.requires_grad = False
    
            
    # feature pre-extraction
    
    def generate_pillar_ref_points(self, num_in_height):
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range
        voxel_x, voxel_y, voxel_z = self.voxel_size
        
        # Calculate the center points for the grid
        x_centers = torch.arange(x_min + voxel_x / 2, x_max, voxel_x)
        y_centers = torch.arange(y_min + voxel_y / 2, y_max, voxel_y)
        z_step = (z_max - z_min) / num_in_height
        z_centers = torch.arange(z_min + z_step / 2, z_max, z_step)
        assert x_centers.shape[0] == self.bev_h_
        assert y_centers.shape[0] == self.bev_w_

        # Create a mesh grid for x, y, z
        xv, yv, zv = torch.meshgrid(x_centers, y_centers, z_centers)

        # Stack the grid coordinates
        ref_points = torch.stack((xv, yv, zv), dim=-1) # shape: (H, W, Z, 3)
        
        # indices
        hv, wv, zv = torch.meshgrid(torch.arange(self.bev_h_), torch.arange(self.bev_w_), torch.arange(num_in_height))
        idx = torch.arange(self.bev_h_ * self.bev_w_ * num_in_height)
        voxel_coords = torch.cat([hv.reshape(-1, 1), wv.reshape(-1, 1), zv.reshape(-1, 1), idx.reshape(-1, 1)], dim=-1)
        return ref_points, voxel_coords
    
    def voxelpainting_mask_depth_aware(self, context, pvs_logits, depth_logits, points, lidar2img, temperature=1.0):
        epsilon = 1e-6
        B, _, H, W = pvs_logits.shape
        device = lidar2img.device
        cam_depth_range = self.grid_config['dbound']
        painted_points = []

        bev_x_min, bev_y_min, bev_z_min, bev_x_max, bev_y_max, bev_z_max = self.point_cloud_range
        context = context*temperature # enlarge
        
        for i in range(B):
            # preparation
            pts = points[i][:, :3]
            pts_hom = torch.cat((pts, torch.ones((pts.shape[0], 1), device=device)), dim=1)  # (N, 4)
            img_pts = torch.matmul(lidar2img[i], pts_hom.t()).t()  # (N, 4)
            img_pts[:, :2] = img_pts[:, :2] / (img_pts[:, 2:3] + epsilon)
            depth_values = img_pts[:, 2]

            valid_mask = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < W) & (img_pts[:, 1] >= 0) & (img_pts[:, 1] < H)
            img_pts_int = img_pts[:, :2].long()
            
            # Initialize the feature tensor with zeros, 
            context_features = torch.zeros((pts.shape[0], context.shape[1]+3), device=device, dtype=context.dtype)
            pts_norm = (pts - torch.tensor([bev_x_min, bev_y_min, bev_z_min], device=device)) \
                / torch.tensor([bev_x_max - bev_x_min, bev_y_max - bev_y_min, bev_z_max - bev_z_min], device=device)
            # context_features[:, :3] = pts_norm
            context_features[:, :3] = pts # keep same as radar points
            
            # begin valid point decorated
            valid_img_pts_int = img_pts_int[valid_mask]
            valid_depth_values = depth_values[valid_mask]
            # for grid sample for sub pixel
            valid_img_pts_norm = img_pts[:, :2][valid_mask].clone()
            valid_img_pts_norm[:, 0] = (valid_img_pts_norm[:, 0] / (W - 1)) * 2 - 1
            valid_img_pts_norm[:, 1] = (valid_img_pts_norm[:, 1] / (H - 1)) * 2 - 1
            valid_img_pts_norm = torch.clamp(valid_img_pts_norm, -1, 1)
            
            # Extract context features for valid image points
            valid_context_features = F.grid_sample(context[i].unsqueeze(0), valid_img_pts_norm.unsqueeze(0).unsqueeze(1), align_corners=True)
            valid_context_features = valid_context_features.squeeze(0).squeeze(1).permute(1, 0)  # (num_points, C)
            
            # mask values
            if self.pixel_aware:
                valid_mask_values = F.grid_sample(pvs_logits[i].unsqueeze(0), valid_img_pts_norm.unsqueeze(0).unsqueeze(1), align_corners=True)
                valid_mask_values = valid_mask_values.squeeze(0).squeeze(1).permute(1, 0)  # (num_points, C)
                valid_context_features = valid_context_features * valid_mask_values
                
            # Get corresponding depth_logits & Weight context_features using the log-transformed depth probabilities
            depth_logits_for_sampling = depth_logits[i].unsqueeze(0)  # (1, D, H, W)
            depth_probs = F.grid_sample(depth_logits_for_sampling, valid_img_pts_norm.unsqueeze(0).unsqueeze(1), align_corners=True)
            depth_probs = depth_probs.squeeze(2).squeeze(0)  # (D, num_points)
            # depth_probs = depth_logits[i, :, valid_img_pts_int[:, 1], valid_img_pts_int[:, 0]]
            power_exponent = 1.0
            power_depth_probs = depth_probs ** power_exponent
            power_depth_probs /= power_depth_probs.sum(dim=0, keepdim=True) + epsilon  # Normalize
            depth_probs = power_depth_probs
            depth_indices = ((valid_depth_values - cam_depth_range[0]) / cam_depth_range[2])
            lower_indices = torch.floor(depth_indices).long()
            upper_indices = torch.ceil(depth_indices).long()
            lower_indices = torch.clamp(lower_indices, 0, depth_probs.shape[0] - 1)
            upper_indices = torch.clamp(upper_indices, 0, depth_probs.shape[0] - 1)
            upper_weight = depth_indices - lower_indices.float()
            lower_weight = 1 - upper_weight
            lower_prob_values = depth_probs[lower_indices, range(lower_indices.shape[0])]
            upper_prob_values = depth_probs[upper_indices, range(upper_indices.shape[0])]
            depth_prob_values = lower_weight * lower_prob_values + upper_weight * upper_prob_values
            # re-weight decorated features    
            if self.depth_aware:
                valid_context_features = valid_context_features * depth_prob_values.unsqueeze(1)
                
            # Assign the computed features to the correct positions
            indices = torch.nonzero(valid_mask).squeeze(1).to(device)
            add_feature = torch.cat((torch.zeros((indices.size(0), 3), device=device), valid_context_features), dim=1)
            context_features.index_add_(0, indices, add_feature)
    
            if torch.isnan(context_features).any():
                print("NaN detected in %s, num=%d"%('painted_points', torch.isnan(context_features).sum()))
                context_features = torch.nan_to_num(context_features, nan=0.0)
            painted_points.append(context_features)
            
        return painted_points
    
    def extract_pts_feat(self, pts, img_metas):
        """Extract features of radar points."""

        # voxelization
        voxels, coors, num_points = [], [], []
        for res in pts:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        coors = coors_batch
        
        # extract features
        voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,)
        batch_size = coors[-1, 0].item() + 1
        x = self.pts_middle_encoder(voxel_features, coors, batch_size)
        x = self.pts_backbone(x)
        if self.with_pts_neck:
            x = self.pts_neck(x)
        batch_dict = dict(voxels=voxels, num_points=num_points, voxel_coords=coors, batch_size=batch_size)
        return x[0], batch_dict
    
    def extract_pts_feat_extra(self, pts, img_metas):
        """Extract features of lidar points."""
        
        # voxelization
        voxels, coors, num_points = [], [], []
        for res in pts:
            res_voxels, res_coors, res_num_points = self.pts_voxel_layer_extra(res)
            voxels.append(res_voxels)
            coors.append(res_coors)
            num_points.append(res_num_points)
        voxels = torch.cat(voxels, dim=0)
        num_points = torch.cat(num_points, dim=0)
        coors_batch = []
        for i, coor in enumerate(coors):
            coor_pad = F.pad(coor, (1, 0), mode='constant', value=i)
            coors_batch.append(coor_pad)
        coors_batch = torch.cat(coors_batch, dim=0)
        coors = coors_batch
        
        # extract features
        voxel_features = self.pts_voxel_encoder_extra(voxels, num_points, coors,)
        batch_size = coors[-1, 0].item() + 1
        x = self.pts_middle_encoder_extra(voxel_features, coors, batch_size)
        x = self.pts_backbone_extra(x)
        x = self.pts_neck_extra(x)
        batch_dict = dict(voxels=voxels, num_points=num_points, voxel_coords=coors, batch_size=batch_size)
        return x[0], batch_dict
    
    def extract_img_feat(self, img, img_metas):
        """Extract features of images."""
        if self.with_img_backbone and img is not None:
            input_shape = img.shape[-2:]
            # update real input shape of each single img
            for img_meta in img_metas:
                img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.view(B * N, C, H, W)
                
            if self.use_grid_mask:
                img = self.grid_mask(img)
            img_feats = self.img_backbone(img)
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)
        return img_feats
    
    def multimodaldropout(self, feats_1, feats_2):
        # Generate random probabilities for dropout
        p_overall = torch.rand(1).item()
        p_feats_1 = torch.rand(1).item()
        # Apply modality dropout
        if p_overall < self.overall_dropout:
            if p_feats_1 < self.feats_1_dropout: 
                feats_1 = feats_1 * 0.0
            else: feats_2 = feats_2 * 0.0    
        return feats_1, feats_2
    
    # NOTE: core model here, processing multi-modality feats
       
    def extract_feat(self, points, img, img_metas, processed_info):
        """Extract features from images and points."""

        # preparation of camera-geo-aware input
        if img.dim() == 3 and img.size(0)== 3: img = img.unsqueeze(0)
        B, C, H, W = img.shape
        if not isinstance(points, list): points = [points]
        img_metas, gt_bboxes_3d, gt_labels_3d, gt_bboxes_2d, gt_labels_2d, depth_comple, bbox_Mask, segmentation, radar_depth, cam_aware, \
            img_aug_matrix, lidar_aug_matrix, bda_rot, gt_depths, gt_bev_mask_binary, gt_bev_mask_semant, final_lidar2img, lidar_points = processed_info
        device = final_lidar2img.device
        img_inputs = [img, cam_aware[0], cam_aware[1], cam_aware[2], cam_aware[3], cam_aware[4], bda_rot]
        img, rots, trans, intrins, post_rots, post_trans, bda = img_inputs[0:7]
        cam_depth_range = self.grid_config['dbound']
        raw_depth = torch.arange(cam_depth_range[0], cam_depth_range[1], cam_depth_range[2]).to(device)
        matrix = torch.eye(4).to(device)
        matrix[0, 0] = matrix[0, 0] / self.downsample
        matrix[1, 1] = matrix[1, 1] / self.downsample
        projection = matrix @ final_lidar2img
        radar_points = copy.deepcopy(points)
        lidar_points = copy.deepcopy(lidar_points)
        
        # 1. pre-extract img and radar features of raw data | pts_bev_feats using R
        img_feats = self.extract_img_feat(img, img_metas)
        radar_feats, _ = self.extract_pts_feat(radar_points, img_metas)
        lidar_feats, _ = self.extract_pts_feat_extra(lidar_points, img_metas)
        radar_feats, lidar_feats = self.multimodaldropout(radar_feats, lidar_feats)
        pts_bev_feats = self.lidar_radar_fusion(lidar_feats, radar_feats)
            
        # 2. depth estimation & perspective segmentation
        mlp_input = self.depth_net.get_mlp_input(rots, trans, intrins, post_rots, post_trans, bda)
        geo_inputs = [rots, trans, intrins, post_rots, post_trans, bda, mlp_input]
        view_trans_inputs = [rots, trans, intrins, post_rots, post_trans, bda]
        cam_params_list = [[rots[i:i+1], trans[i:i+1], intrins[i:i+1], post_rots[i:i+1], post_trans[i:i+1], bda[i:i+1]] for i in range(img.shape[0])]
        index = self.downsample // 4 - 1
        h, w = img.shape[2] // self.downsample, img.shape[3] // self.downsample
        align_feats = [F.interpolate(feat, (h, w), mode='bilinear', align_corners=True) for feat in img_feats]
        align_feats = torch.cat(align_feats, dim=1)
        # NOTE here, we input radar_depth
        context, depth = self.depth_net([align_feats] + geo_inputs, radar_depth, depth_comple, img_metas) # radar_depth or gt_depths
        cam_depth_range = self.grid_config['dbound']
        raw_depth = torch.arange(cam_depth_range[0], cam_depth_range[1], cam_depth_range[2]).to(device)
        sums_depths = torch.sum(raw_depth.view(1,-1,1,1)*depth, dim=1).unsqueeze(1)
        rangeview_logit = self.pvseg_net(context.squeeze(1)) 
        rangeview_logit_sigmoid = rangeview_logit.sigmoid()
        focus_weight = 1.0*sums_depths/self.point_cloud_range[3]
        mask_reweighted = (1-focus_weight)*rangeview_logit_sigmoid+focus_weight*torch.ones_like(rangeview_logit_sigmoid).to(context.device)
        min_vals = torch.min(mask_reweighted.reshape(B, -1), dim=1)[0].reshape(B, 1, 1, 1)
        max_vals = torch.max(mask_reweighted.reshape(B, -1), dim=1)[0].reshape(B, 1, 1, 1)
        mask_reweighted = (mask_reweighted-min_vals)/(max_vals-min_vals)
        # mask_reweighted = rangeview_logit_sigmoid
        
        # 3. generate 3D-to-2D 'Depth-OFT' img BEV | img_bev_feats using L+R (C)
        h, w, z = self.voxelpainting_points.shape[:3]
        all_decorated_points = [self.voxelpainting_points.reshape(-1, 3).to(device) for _ in range(B)]
        all_decorated_points = self.voxelpainting_mask_depth_aware(context.squeeze(1), mask_reweighted, depth, all_decorated_points, projection)
        paint_bev = [all_decorated_points[i][:, 3:] for i in range(B)]
        paint_bev = [x.view(1, h, w, z, -1) for x in paint_bev]
        paint_bev = torch.cat(paint_bev, dim=0) # B H W Z C
        paint_bev = paint_bev.view(B, h, w, -1).permute(0, 3, 2, 1).contiguous() # B C*Z H W
        img_bev_feats = self.mapcollapse(paint_bev)
        
        # 4. BEV fusion radar and image feats
        bev_feats = self.fuses_net(img_bev_feats, pts_bev_feats)
        bev_feats = bev_feats.permute(0, 1, 3, 2).contiguous()
        assert bev_feats.shape[2] == self.bev_h_
        assert bev_feats.shape[3] == self.bev_w_
        
        # 5. BEV feature enhancement with depth-aware image feature
        if self.dfa3d_net is not None:
            bev_mask = torch.ones((B, 1, self.bev_h_, self.bev_w_), dtype=torch.bool).to(device)
            bev_feats = self.dfa3d_net(
                imgs = img,
                mlvl_feats = [img_feats[index].unsqueeze(1)],
                proposal = bev_mask,
                cam_params=cam_params_list,
                lss_bev=bev_feats,
                img_metas=img_metas,
                mlvl_dpt_dists=[depth.unsqueeze(1)],
                backward_bev_mask_logit=torch.zeros_like(bev_feats).to(device))
            bev_feats = bev_feats.permute(0, 2, 1).view(B, self.img_channels, self.bev_h_, self.bev_w_).contiguous()
        
        # 6. auxliary supervision in bev space
        if self.props_supervision['use']:
            bev_binary_logit = self.bvseg_net.forward_binary(bev_feats)
            bev_semant_logit = self.bvseg_net.forward_semant(bev_feats)
        else: bev_binary_logit, bev_semant_logit = None, None
        if self.focus_supervision['use']:
            scene_tokens = bev_feats.flatten(2,3).permute(2,0,1)
            learn_tokens = self.token.weight.unsqueeze(1).repeat(1,B,1) # N B C
            learn_tokens, _ = self.MHSA_1(learn_tokens, scene_tokens, scene_tokens)
            learn_tokens, _ = self.MHSA_2(learn_tokens, scene_tokens, scene_tokens)
            focus_dicts = self.focus_net(bev_feats, learn_tokens, self.training)
            bev_feats = focus_dicts['x_objt_feat']
            logits_objects = focus_dicts['pred_mask_objt']
            logits_backgrd = focus_dicts['pred_mask_back']
        else: focus_dicts, logits_objects, logits_backgrd = None, None, None
        bev_feats = bev_feats.permute(0, 1, 3, 2).contiguous()
        
        # record visualization
        # step_all_time = step0_time + step1_time + step2_time + step3_time + step4_time + step5_time + step6_time + step7_time + step8_time + step9_time + step10_time
        # print(' fps0:%.1f|fps1:%.1f|fps2:%.1f|fps3:%.1f|fps4:%.1f|fps5:%.1f|fps6:%.1f|fps7:%.1f|fps8:%.1f|fps9:%.1f|fps10:%.1f'
        #       %(1/step0_time,1/step1_time,1/step2_time,1/step3_time,1/step4_time,1/step5_time,1/step6_time,1/step7_time,1/step8_time,1/step9_time,1/step10_time))
        # self.recording_fps(step_all_time=step_all_time)
        if self.depth_supervision['use']: self.draw_depth_estimation(img_metas, gt_depths, depth, img, depth_comple, radar_depth, gt_depths)
        if self.msk2d_supervision['use']: self.draw_gt_pred_rangeview(img_metas, segmentation, bbox_Mask, rangeview_logit.sigmoid()> self.pvseg_net.mask_thre_train, rangeview_logit.sigmoid(), eroded=False)
        self.draw_bev_feature_map(bev_feats, img_metas, bev_feats_name='bev_feats')
        self.draw_bev_feature_map(img_bev_feats, img_metas, bev_feats_name='img_bev_feats')
        self.draw_bev_feature_map(pts_bev_feats, img_metas, bev_feats_name='pts_bev_feats')
        self.draw_bev_feature_map(radar_feats, img_metas, bev_feats_name='bev_feats_radar')
        self.draw_bev_feature_map(lidar_feats, img_metas, bev_feats_name='bev_feats_lidar')
        if self.props_supervision['use']: self.draw_gt_pred_bev(gt_bev_mask_binary.bool(), bev_binary_logit.sigmoid()>self.bvseg_net.mask_thre, bev_binary_logit.sigmoid(), img_metas, 'ptsbinary')
        if self.props_supervision['use']: self.draw_gt_pred_bev_semantic(gt_bev_mask_semant, bev_semant_logit, img_metas, 'semantics')
        if self.focus_supervision['use']: self.draw_gt_pred_bev( gt_bev_mask_binary.bool(), logits_objects.sigmoid()>self.focus_net.objects_thre, logits_objects.sigmoid(), img_metas, 'foreground')
        if self.focus_supervision['use']: self.draw_gt_pred_bev(~gt_bev_mask_binary.bool(), logits_backgrd.sigmoid()>self.focus_net.backgrd_thre, logits_backgrd.sigmoid(), img_metas, 'background')
        
        return dict(img_feats=img_feats,
                    pts_feats=[bev_feats],
                    img_bev_feats=img_bev_feats,
                    pts_bev_feats=pts_bev_feats,
                    true_depths=gt_depths,
                    pred_depths=depth, 
                    sums_depths=sums_depths,
                    bbox_Mask=bbox_Mask,
                    segm_Mask=segmentation,
                    pred_Mask=rangeview_logit,
                    gt_bev_mask_binary=gt_bev_mask_binary,
                    gt_bev_mask_semant=gt_bev_mask_semant,
                    bev_binary_logit=bev_binary_logit,
                    bev_semant_logit=bev_semant_logit,
                    depth_comple=depth_comple,
                    lidar_points=lidar_points,
                    focus_dicts=focus_dicts)
        
    # train and evaluating process
    
    def simple_test(self, 
                    points, 
                    img_metas, 
                    img=None, 
                    rescale=False, 
                    gt_bboxes_3d=None,
                    gt_labels_3d=None,
                    gt_labels=None,
                    gt_bboxes=None):
        """Test function without augmentaiton."""
        if len(img_metas) !=1: img_metas = [img_metas]
        # preparation for testing
        if gt_bboxes_3d is not None: 
            for i in range(len(img_metas)):
                img_metas[i]['gt_labels'] = gt_labels[i]
                img_metas[i]['gt_bboxes'] = HorizontalBoxes(gt_bboxes[i], in_mode='xyxy')
                img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i].to(gt_labels_3d[i].device)
                img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]
        processed_info = self.preprocessing_information(img_metas, gt_labels_3d[i].device)
        self.overall_dropout = 0.0
        self.feats_1_dropout = 0.0
        feature_dict = self.extract_feat(points, img=img, img_metas=img_metas, processed_info=processed_info)
        lidar_points = feature_dict['lidar_points']
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox: # pts means 3D detection
            bbox_pts, outs_pts = self.simple_test_pts(pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox

        if img_feats and self.with_img_bbox: # img means 2D detection
            bbox_img = self.simple_test_img(img_feats, img_metas, rescale=rescale)
            for result_dict, img_bbox in zip(bbox_list, bbox_img):
                result_dict['img_bbox'] = img_bbox
        
        # visualization for test stage 
        threshold = 0.3
        if gt_bboxes_3d is not None and self.box3d_supervision['use']:
            if img.dim() == 3 and img.size(0)== 3: img = img.unsqueeze(0)
            if not isinstance(points, list): points = [points]
            self.draw_gt_pred_figures_3d(points, lidar_points, img, gt_bboxes_3d, gt_labels_3d, img_metas, False, threshold, outs_pts=outs_pts)
        else: # vanilla testing method
            self.vis_time_box3d += 1
            if self.vis_time_box3d % self.SAVE_INTERVALS == 0:
                figures_path_det3d = self.figures_path_det3d_test 
                input_img = np.array(img.cpu()).transpose(1,2,0)
                input_img = input_img*self.std[None, None, :] + self.mean[None, None, :]
                pred_bboxes_3d = bbox_pts[0]['boxes_3d']
                pred_scores_3d = bbox_pts[0]['scores_3d']
                pred_bboxes_3d = pred_bboxes_3d[pred_scores_3d>threshold].to('cpu')
                proj_mat = img_metas[0]["final_lidar2img"] # update lidar2img
                img_name = img_metas[0]['filename'].split('/')[-1].split('.')[0]
                # project 3D bboxes to image and get show figures
                if len(pred_bboxes_3d) == 0: pred_bboxes_3d = None
                filename = str(self.vis_time_box3d) + '_' + img_name + '_det3d'
                show_multi_modality_result(img=input_img, gt_bboxes=None, pred_bboxes=pred_bboxes_3d, proj_mat=proj_mat, out_dir=figures_path_det3d, filename=filename, box_mode='lidar', show=False)
            
        return bbox_list

    def get_loss(self, feature_dict,
                 gt_bboxes_3d,
                 gt_labels_3d,
                 img_metas,
                 points, img,
                 gt_bboxes_ignore):
        img_feats = feature_dict['img_feats']
        pts_feats = feature_dict['pts_feats']
        aux_feats_image = feature_dict['img_bev_feats']
        aux_feats_point = feature_dict['pts_bev_feats']
        true_depths = feature_dict['true_depths']
        pred_depths = feature_dict['pred_depths']
        sums_depths = feature_dict['sums_depths']
        bbox_Mask = feature_dict['bbox_Mask']
        segm_Mask = feature_dict['segm_Mask']
        pred_Mask = feature_dict['pred_Mask']
        gt_bev_mask_binary = feature_dict['gt_bev_mask_binary']
        gt_bev_mask_semant = feature_dict['gt_bev_mask_semant']
        bev_binary_logit = feature_dict['bev_binary_logit']
        bev_semant_logit = feature_dict['bev_semant_logit']
        lidar_points = feature_dict['lidar_points']
        depth_comple = feature_dict['depth_comple']
        focus_dicts = feature_dict['focus_dicts']

        # compute for all explicit losses
        losses = dict()
        outs_pts = None
        if self.box3d_supervision['use'] and gt_bboxes_3d is not None:
            losses_box3d, outs_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d, gt_labels_3d, img_metas, points, lidar_points, img, gt_bboxes_ignore)
            losses.update(losses_box3d)
        if self.depth_supervision['use'] and self.depth_net is not None:
            losses_depth = self.depth_net.get_depth_loss(true_depths, pred_depths, sums_depths, depth_comple)
            losses.update(losses_depth)
        if self.msk2d_supervision['use'] and self.pvseg_net is not None:
            losses_msk2d = self.pvseg_net.get_range_view_mask_loss(gt_mask_box=bbox_Mask, gt_mask_seg=segm_Mask, predit_mask=pred_Mask)
            losses.update(losses_msk2d)
        if self.props_supervision['use'] and self.bvseg_net is not None:
            losses_props_1 = self.bvseg_net.get_bev_mask_loss(bev_binary_logit, gt_bev_mask_binary)
            losses_props_2 = self.bvseg_net.get_bev_mask_loss_semantic(bev_semant_logit, gt_bev_mask_semant)
            losses_props = {}; losses_props.update(losses_props_1); losses_props.update(losses_props_2); 
            losses.update(losses_props)
        if self.focus_supervision['use'] and self.focus_net is not None:
            losses_focus = self.focus_net.get_feature_focus_loss(focus_dicts, gt_bev_mask_binary)
            losses.update(losses_focus)
        if self.box3d_supervision['use'] and self.aux_bbox_head is not None:
            loss_inputs_aux_point = self.aux_bbox_head_point([aux_feats_point]) + (gt_bboxes_3d, gt_labels_3d, img_metas)
            losses_aux_point = self.aux_bbox_head_point.loss(*loss_inputs_aux_point, gt_bboxes_ignore=gt_bboxes_ignore)
            losses_aux_point = {f"{key}_aux_point": [self.aux_loss_weight*value[0]] for key, value in losses_aux_point.items()}
            losses.update(losses_aux_point)
        if self.box3d_supervision['use'] and self.aux_bbox_head_image is not None:
            loss_inputs_aux_image = self.aux_bbox_head_image([aux_feats_image]) + (gt_bboxes_3d, gt_labels_3d, img_metas)
            losses_aux_image = self.aux_bbox_head_image.loss(*loss_inputs_aux_image, gt_bboxes_ignore=gt_bboxes_ignore)
            losses_aux_image = {f"{key}_aux_image": [self.aux_loss_weight*value[0]] for key, value in losses_aux_image.items()}
            losses.update(losses_aux_image)
        return losses, outs_pts
    
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      img_depth=None,
                      proposals=None,
                      gt_bboxes_ignore=None):
        # preparation for loss caculation
        for i in range(len(img_metas)):
            img_metas[i]['gt_labels'] = gt_labels[i]
            img_metas[i]['gt_bboxes'] = HorizontalBoxes(gt_bboxes[i], in_mode='xyxy')
            img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i].to(gt_labels_3d[i].device)
            img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]
        gt_bboxes_3d = [gt_bboxes_3d[i][gt_labels_3d[i]!=-1] for i in range(len(img_metas))] # filter out the ignored labels
        gt_labels_3d = [gt_labels_3d[i][gt_labels_3d[i]!=-1] for i in range(len(img_metas))] # filter out the ignored labels
        processed_info = self.preprocessing_information(img_metas, img.device)
        feature_dict = self.extract_feat(points, img=img, img_metas=img_metas, processed_info=processed_info)
        # compute for 3D object detection losses
        losses, outs_pts = self.get_loss(feature_dict, gt_bboxes_3d, gt_labels_3d, img_metas, points, img, gt_bboxes_ignore)
        return losses
     
    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          points,
                          lidar_points,
                          img,
                          gt_bboxes_ignore=None):
        outs_pts = self.pts_bbox_head(pts_feats)
        loss_inputs = outs_pts + (gt_bboxes_3d, gt_labels_3d, img_metas)
        losses = self.pts_bbox_head.loss(*loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        weight = self.box3d_supervision['weight']
        losses['loss_cls'] = losses['loss_cls'][0]*weight
        losses['loss_bbox'] = losses['loss_bbox'][0]*weight
        losses['loss_dir'] = losses['loss_dir'][0]*weight
    
        # visualization for train stage  
        if gt_bboxes_3d is not None and self.box3d_supervision['use']:
            self.draw_gt_pred_figures_3d(points, lidar_points, img, gt_bboxes_3d, gt_labels_3d, img_metas, False, 0.3, outs_pts=outs_pts)

        return losses, outs_pts
    
    # preprocessing for data and others
    
    def preprocessing_information(self, batch_img_metas, device):
        if self.training:
            # all important informations
            B = len(batch_img_metas)
            H, W = batch_img_metas[0]['img_shape']
            h_down, w_down = H // self.downsample, W // self.downsample
            # gt lidar instances_3d, list of InstancesData
            gt_bboxes_3d = [img_meta['gt_bboxes_3d'] for img_meta in batch_img_metas]
            gt_labels_3d = [img_meta['gt_labels_3d'] for img_meta in batch_img_metas]
            gt_bboxes_2d = [img_meta['gt_bboxes'] for img_meta in batch_img_metas]
            gt_labels_2d = [img_meta['gt_labels'] for img_meta in batch_img_metas]
            gt_bboxes_3d = [gt_bboxes_3d[i][gt_labels_3d[i]!=-1] for i in range(B)]
            gt_labels_3d = [gt_labels_3d[i][gt_labels_3d[i]!=-1] for i in range(B)]
            gt_bboxes_2d = [gt_bboxes_2d[i][gt_labels_2d[i]!=-1] for i in range(B)]
            gt_labels_2d = [gt_labels_2d[i][gt_labels_2d[i]!=-1] for i in range(B)]
            gt_bev_mask_binary, gt_bev_mask_semant = self.generate_bev_mask(gt_bboxes_3d, gt_labels_3d, B, device, occ_threshold=0.3) # B H W
            gt_bev_mask_binary = gt_bev_mask_binary.to(device)
            gt_bev_mask_semant = gt_bev_mask_semant.to(device)
            
            # create gt_depths from LiDAR data, already processing with IMG_AUG, no need with BEV_AUG
            if 'gt_depths' in batch_img_metas[0].keys():
                gt_depths = [img_meta['gt_depths'] for img_meta in batch_img_metas if 'gt_depths' in img_meta]
                gt_depths = torch.stack(gt_depths).unsqueeze(1) # B, 1, H, W
                gt_depths = gt_depths.to(device)
            else: gt_depths = torch.zeros((B, 1, H, W)).to(device)
            # cam_aware: rot, tran, intrin, post_rot, post_tran, _, cam2lidar, focal_length, baseline
            if 'cam_aware' in batch_img_metas[0].keys():
                cam_aware = [img_meta['cam_aware'] for img_meta in batch_img_metas]
                merged_tensors = [None] * len(cam_aware[0])
                for i in range(len(cam_aware[0])):
                    component = [x[i] for x in cam_aware]
                    merged_tensors[i] = torch.stack(component, dim=0)
                cam_aware = merged_tensors
                cam_aware = [x.to(device) for x in cam_aware]
            else: cam_aware = [None for _ in range(9)]
            # img_aug_matrix: 4x4 martix of combined post_rot&post_tran of IMG_AUG
            if 'img_aug_matrix' in batch_img_metas[0].keys():
                img_aug_matrix = [img_meta['img_aug_matrix'] for img_meta in batch_img_metas]
                img_aug_matrix = torch.tensor(np.stack(img_aug_matrix, axis=0))
                img_aug_matrix = img_aug_matrix.to(device)
            else: img_aug_matrix = torch.eye(4).unsqueeze(0).repeat(B,1,1).to(device)
            # lidar_aug_matrix same as bda_rot: 4x4 martix of combined post_rot&post_tran of BEV_AUG
            if 'lidar_aug_matrix' in batch_img_metas[0].keys():
                lidar_aug_matrix = [img_meta['lidar_aug_matrix'] for img_meta in batch_img_metas]
                lidar_aug_matrix = torch.tensor(np.stack(lidar_aug_matrix, axis=0)).to(torch.float32).to(device)
            else: lidar_aug_matrix = torch.eye(4).unsqueeze(0).repeat(len(batch_img_metas), 1, 1).to(device)
            if 'bda_rot' in batch_img_metas[0].keys():
                bda_rot = [img_meta['bda_rot'] for img_meta in batch_img_metas]
                bda_rot = torch.tensor(np.stack(bda_rot, axis=0)).to(torch.float32).to(device)
            else:  bda_rot = lidar_aug_matrix.to(device)
            # re-organize clearly to create NOW lidar2img for project convenience
            if 'cam_aware' in batch_img_metas[0].keys():
                batch_img_metas = self.reorganize_lidar2img(batch_img_metas)
                calib = []
                for sample_idx in range(B):
                    mat = batch_img_metas[sample_idx]['final_lidar2img']
                    mat = torch.Tensor(mat).to(device)
                    calib.append(mat)
                final_lidar2img = torch.stack(calib)
            else: Warning("No cam_aware in batch_img_metas, can not project points to img") 
            # preprocessed seg_mask, pre_inferenced depth_comple
            if 'depth_comple' in batch_img_metas[0].keys():
                depth_comple = [img_meta['depth_comple'] for img_meta in batch_img_metas]
                depth_comple = torch.tensor(np.stack(depth_comple, axis=0)).to(device).unsqueeze(1)
            else: depth_comple = torch.zeros((B, 1, H, W)).to(device)
            if 'radar_depth' in batch_img_metas[0].keys():
                radar_depth = [img_meta['radar_depth'] for img_meta in batch_img_metas]
                radar_depth = torch.tensor(np.stack(radar_depth, axis=0)).to(device).unsqueeze(1)
                radar_depth = radar_depth.to(torch.float32)
            else: radar_depth =torch.zeros((B, 1, H, W)).to(device)
            if 'segmentation' in batch_img_metas[0].keys():
                segmentation = [img_meta['segmentation'].astype(np.float32) for img_meta in batch_img_metas]
                segmentation = torch.tensor(np.stack(segmentation, axis=0), dtype=torch.float32).to(device).unsqueeze(1)
                segmentation = F.interpolate(segmentation, (h_down, w_down), mode='bilinear', align_corners=True)
            else: segmentation = torch.zeros((B, 1, h_down, w_down)).to(device)
            if 'bbox_Mask' in batch_img_metas[0].keys():
                bbox_Mask = [img_meta['bbox_Mask'] for img_meta in batch_img_metas]
                bbox_Mask = torch.tensor(np.stack(bbox_Mask, axis=0)).to(device).unsqueeze(1)
                bbox_Mask = F.interpolate(bbox_Mask, (h_down, w_down), mode='bilinear', align_corners=True)
            else: bbox_Mask = torch.zeros((B, 1, h_down, w_down)).to(device)
            if "lidar_points" in batch_img_metas[0].keys():
                lidar_points_list = []
                true_lidar2img_list = []
                for i in range(B):
                    lidar_points = batch_img_metas[i]['lidar_points'].to(device)
                    true_lidar2cam = batch_img_metas[i]['true_lidar2cam']
                    radar2img = np.array(final_lidar2img[i].cpu())
                    tmp_final_cam2img = copy.deepcopy(batch_img_metas[i]['cam2img'])
                    rots, trans, intrins, post_rots, post_trans = batch_img_metas[i]['cam_aware'][:5]
                    tmp_final_cam2img[:2, :3] = post_rots[:2, :2] @ tmp_final_cam2img[:2, :3]
                    tmp_final_cam2img[:2, 2] = post_trans[:2] + tmp_final_cam2img[:2, 2]
                    true_lidar2img = tmp_final_cam2img @ np.array(true_lidar2cam)
                    lidar2radar = np.matmul(np.linalg.inv(radar2img), true_lidar2img)
                    lidar2radar = torch.tensor(lidar2radar).to(device).to(torch.float32)
                    lidar_points_under_radar = torch.cat((lidar_points[:,:3], torch.ones((lidar_points.shape[0], 1), device=device)), dim=-1).to(device)
                    lidar_points_under_radar = torch.matmul(lidar_points_under_radar, lidar2radar.T)[:, :3]
                    lidar_points_under_radar = torch.cat([lidar_points_under_radar, lidar_points[:, -2:-1]], dim=-1)
                    lidar_points_list.append(lidar_points_under_radar)
                    true_lidar2img_list.append(true_lidar2img)
                lidar_points = lidar_points_list
                true_lidar2img = torch.tensor(np.stack(true_lidar2img_list, axis=0)).to(device).to(torch.float32)
            else: lidar_points, true_lidar2img = None, None

        else:
            # gt_bboxes_3d, gt_labels_3d, gt_bboxes_2d, gt_labels_2d, gt_depths = [], [], [], [], []
            batch_img_metas = batch_img_metas[0]
            H, W = batch_img_metas['img_shape']
            h_down, w_down = H // self.downsample, W // self.downsample
            
            if 'gt_bboxes_3d' in batch_img_metas:
                gt_bboxes_3d = [batch_img_metas['gt_bboxes_3d']]
            else: gt_bboxes_3d = []
            if 'gt_labels_3d' in batch_img_metas:
                gt_labels_3d = [batch_img_metas['gt_labels_3d']]
            else: gt_labels_3d = []
            if 'gt_bboxes' in batch_img_metas:
                gt_bboxes_2d = [batch_img_metas['gt_bboxes']]
            else: gt_bboxes_2d = []
            if 'gt_labels' in batch_img_metas:
                gt_labels_2d = [batch_img_metas['gt_labels']]
            else: gt_labels_2d = []
            gt_bboxes_3d = [gt_bboxes_3d[i][gt_labels_3d[i]!=-1] for i in [0]] if len(gt_labels_3d)!=0 else []
            gt_labels_3d = [gt_labels_3d[i][gt_labels_3d[i]!=-1] for i in [0]] if len(gt_labels_3d)!=0 else []
            gt_bboxes_2d = [gt_bboxes_2d[i][gt_labels_2d[i]!=-1] for i in [0]] if len(gt_labels_2d)!=0 else []
            gt_labels_2d = [gt_labels_2d[i][gt_labels_2d[i]!=-1] for i in [0]] if len(gt_labels_2d)!=0 else []
            gt_bev_mask_binary, gt_bev_mask_semant = self.generate_bev_mask(gt_bboxes_3d, gt_labels_3d, 0, device, occ_threshold=0.3) # B H W
            gt_bev_mask_binary = gt_bev_mask_binary.to(device)
            gt_bev_mask_semant = gt_bev_mask_semant.to(device)
            
            if 'gt_depths' in batch_img_metas.keys():
                gt_depths = [batch_img_metas['gt_depths']]
                gt_depths = torch.stack(gt_depths).unsqueeze(1) # B, 1, H, W
                gt_depths = gt_depths.to(device)
            else: gt_depths = torch.zeros((1, 1, H, W)).to(device)
            if 'cam_aware' in batch_img_metas.keys():
                cam_aware = batch_img_metas['cam_aware']
                cam_aware = [[x.to(device)] for x in cam_aware]
                cam_aware = [torch.stack(x, dim=0) for x in cam_aware]
            else: cam_aware = [None for _ in range(9)]
            if 'img_aug_matrix' in batch_img_metas.keys():
                img_aug_matrix = [batch_img_metas['img_aug_matrix']]
                img_aug_matrix = torch.tensor(np.stack(img_aug_matrix, axis=0))
                img_aug_matrix = img_aug_matrix.to(device)
            else: img_aug_matrix = torch.eye(4).unsqueeze(0).to(device)
            if 'lidar_aug_matrix' in batch_img_metas.keys():
                lidar_aug_matrix = [batch_img_metas['lidar_aug_matrix']]
                lidar_aug_matrix = torch.tensor(np.stack(lidar_aug_matrix, axis=0)).to(torch.float32).to(device)
            else: lidar_aug_matrix = torch.eye(4).unsqueeze(0).to(device)
            if 'bda_rot' in batch_img_metas.keys():
                bda_rot = [batch_img_metas['bda_rot']]
                bda_rot = torch.tensor(np.stack(bda_rot, axis=0)).to(torch.float32).to(device)
            else: bda_rot = torch.eye(4).unsqueeze(0).to(device)
            if 'cam_aware' in batch_img_metas.keys():
                batch_img_metas = self.reorganize_lidar2img([batch_img_metas])[0]
                mat = batch_img_metas['final_lidar2img']
                mat = torch.Tensor(mat).to(device)
                final_lidar2img = torch.stack([mat])
            else: Warning("No cam_aware in batch_img_metas, can not project points to img")
            if 'depth_comple' in batch_img_metas.keys():
                depth_comple = [batch_img_metas['depth_comple']] if isinstance(batch_img_metas, list) else [batch_img_metas['depth_comple']]
                depth_comple = torch.tensor(np.stack(depth_comple, axis=0)).to(device).unsqueeze(1)
                depth_comple = depth_comple.to(torch.float32)
            else: depth_comple = torch.zeros((1, 1, H, W)).to(device)
            if 'radar_depth' in batch_img_metas.keys():
                radar_depth = [batch_img_metas['radar_depth']] if isinstance(batch_img_metas, list) else [batch_img_metas['radar_depth']]
                radar_depth = torch.tensor(np.stack(radar_depth, axis=0)).to(device).unsqueeze(1)
                radar_depth = radar_depth.to(torch.float32)
            else: radar_depth = torch.zeros((1, 1, H, W)).to(device)
            if 'segmentation' in batch_img_metas.keys():
                segmentation = [batch_img_metas['segmentation'].astype(np.float32)] if isinstance(batch_img_metas, list) else [batch_img_metas['segmentation']]
                segmentation = torch.tensor(np.stack(segmentation, axis=0), dtype=torch.float32).to(device).unsqueeze(1)
                segmentation = F.interpolate(segmentation, (h_down, w_down), mode='bilinear', align_corners=True)
            else: segmentation = torch.zeros((1, 1, h_down, w_down)).to(device)
            if 'bbox_Mask' in batch_img_metas.keys():
                bbox_Mask = [batch_img_metas['bbox_Mask']] if isinstance(batch_img_metas, list) else [batch_img_metas['bbox_Mask']]
                bbox_Mask = torch.tensor(np.stack(bbox_Mask, axis=0)).to(device).unsqueeze(1)
                bbox_Mask = F.interpolate(bbox_Mask, (h_down, w_down), mode='bilinear', align_corners=True)
            else: bbox_Mask = torch.zeros((1, 1, h_down, w_down)).to(device)
            if "lidar_points" in batch_img_metas.keys():
                lidar_points = batch_img_metas['lidar_points'].to(device)
                true_lidar2cam = batch_img_metas['true_lidar2cam']
                radar2img = np.array(final_lidar2img[0].cpu())
                tmp_final_cam2img = copy.deepcopy(batch_img_metas['cam2img'])
                rots, trans, intrins, post_rots, post_trans = batch_img_metas['cam_aware'][:5]
                tmp_final_cam2img[:2, :3] = post_rots[:2, :2] @ tmp_final_cam2img[:2, :3]
                tmp_final_cam2img[:2, 2] = post_trans[:2] + tmp_final_cam2img[:2, 2]
                true_lidar2img = tmp_final_cam2img @ np.array(true_lidar2cam)
                lidar2radar = np.matmul(np.linalg.inv(radar2img), true_lidar2img)
                lidar2radar = torch.tensor(lidar2radar).to(device).to(torch.float32)
                lidar_points_under_radar = torch.cat((lidar_points[:,:3], torch.ones((lidar_points.shape[0], 1), device=device)), dim=-1).to(device)
                lidar_points_under_radar = torch.matmul(lidar_points_under_radar, lidar2radar.T)[:, :3]
                lidar_points_under_radar = torch.cat([lidar_points_under_radar, lidar_points[:, -2:-1]], dim=-1)
                lidar_points = [lidar_points_under_radar]
                true_lidar2img = torch.tensor(np.stack([true_lidar2img], axis=0)).to(device).to(torch.float32)
            else: lidar_points, true_lidar2img = None, None
            batch_img_metas = [batch_img_metas]
            
        return batch_img_metas, gt_bboxes_3d, gt_labels_3d, gt_bboxes_2d, gt_labels_2d, depth_comple, bbox_Mask, segmentation, radar_depth, \
            cam_aware, img_aug_matrix, lidar_aug_matrix, bda_rot, gt_depths, gt_bev_mask_binary, gt_bev_mask_semant, final_lidar2img, lidar_points

    def reorganize_lidar2img(self, batch_input_metas):
        """add 'lidar2img' transformation matrix into batch_input_metas.

        Args:
            batch_input_metas (list[dict]): Meta information of multiple inputs
                in a batch.
        Returns:
            batch_input_metas (list[dict]): Meta info with lidar2img added
        """
        for img_metas in batch_input_metas:
            final_cam2img = copy.deepcopy(img_metas['cam2img'])
            final_lidar2img = copy.deepcopy(img_metas['lidar2img'])
            
            # same as visualization in BEVAug3D 
            rots, trans, intrins, post_rots, post_trans = img_metas['cam_aware'][:5]
            final_cam2img[:2, :3] = post_rots[:2, :2] @ final_cam2img[:2, :3]
            final_cam2img[:2, 2] = post_trans[:2] + final_cam2img[:2, 2]
            final_lidar2img = final_cam2img @ img_metas['lidar2cam']
            final_lidar2img = final_lidar2img @ np.linalg.inv(img_metas['lidar_aug_matrix'])
            img_metas['final_lidar2img'] = final_lidar2img

        return batch_input_metas
    
    def generate_bev_mask(self, gt_bboxes_3d, gt_labels_3d, batch_size, device, occ_threshold):
        # As long as it is occupied, it is 1
        gt_bev_mask_binary = []
        if len(gt_bboxes_3d) != 0:
            bev_cell_size = torch.tensor(self.bev_cell_size).to(device)
            for bsid in range(len(gt_bboxes_3d)):
                bev_mask = torch.zeros(self.bev_grid_shape)
                gt_bboxes_3d[bsid].tensor[:,3:4] = gt_bboxes_3d[bsid].tensor[:,3:4]*1.2
                gt_bboxes_3d[bsid].tensor[:,4:5] = gt_bboxes_3d[bsid].tensor[:,4:5]*1.8
                bbox_corners = gt_bboxes_3d[bsid].corners[:, [0,2,4,6],:2] # bev corners
                num_rectangles = bbox_corners.shape[0]
                bbox_corners[:,:,0] = (bbox_corners[:,:,0] - self.xbound[0])/bev_cell_size[0] # id_num, 4, 2
                bbox_corners[:,:,1] = (bbox_corners[:,:,1] - self.ybound[0])/bev_cell_size[1] # id_num, 4, 2
                
                # precise bur slow method
                grid_min = torch.clip(torch.floor(torch.min(bbox_corners, axis=1).values).to(torch.int64), 0, self.bev_grid_shape[0] - 1)
                grid_max = torch.clip(torch.ceil (torch.max(bbox_corners, axis=1).values).to(torch.int64), 0, self.bev_grid_shape[1] - 1)
                possible_mask_h_all = torch.cat([grid_min[:, 0:1], grid_max[:, 0:1]], dim=1).tolist()
                possible_mask_w_all = torch.cat([grid_min[:, 1:2], grid_max[:, 1:2]], dim=1).tolist()
                for n in range(num_rectangles):
                    clock_corners = bbox_corners[n].cpu().numpy()[(0,1,3,2), :]
                    poly = Polygon(clock_corners)
                    h_list = possible_mask_h_all[n]; h_list = np.arange(h_list[0] - 1, h_list[1] + 1, 1); h_list = np.clip(h_list, 0, self.bev_grid_shape[0] - 1)
                    w_list = possible_mask_w_all[n]; w_list = np.arange(w_list[0] - 1, w_list[1] + 1, 1); w_list = np.clip(w_list, 0, self.bev_grid_shape[1] - 1)
                    for i in h_list:
                        for j in w_list:
                            cell_center = np.array([i + 0.5, j + 0.5])
                            cell_poly = box(i, j, i + 1, j + 1)
                            if poly.contains(Point(cell_center)):
                                bev_mask[i, j] = gt_labels_3d[bsid][n]+1 # bev_mask[i, j] = True
                            else:
                                intersection = cell_poly.intersection(poly)
                                if (intersection.area / cell_poly.area) > occ_threshold: 
                                    bev_mask[i, j] = gt_labels_3d[bsid][n]+1 # bev_mask[i, j] = True
                # coarse but quick method
                # for i in range(num_rectangles):
                #     bev_mask[grid_min[i, 0]:grid_max[i, 0], grid_min[i, 1]:grid_max[i, 1]] = True
                # save_image(bev_mask[None,None,:,:]*0.99, 'gt_bev_mask_binary.png')
                gt_bev_mask_binary.append(bev_mask)
            gt_bev_mask_semant = torch.stack(gt_bev_mask_binary, dim=0).unsqueeze(1) # B 1 H W
            gt_bev_mask_binary = copy.deepcopy(gt_bev_mask_semant)
            gt_bev_mask_binary[gt_bev_mask_binary==0] = 0
            gt_bev_mask_binary[gt_bev_mask_binary!=0] = 1
        else:
            gt_bev_mask_binary = torch.zeros((batch_size, 1, self.bev_grid_shape[0], self.bev_grid_shape[1]))
            gt_bev_mask_semant = torch.zeros_like(gt_bev_mask_binary)
        gt_bev_mask_binary = gt_bev_mask_binary.to(torch.bool)
        gt_bev_mask_semant = gt_bev_mask_semant
        # zero means background! gt_labels_3d+1
        return gt_bev_mask_binary, gt_bev_mask_semant
    
    def recording_fps(self, step_all_time):
        self.record_fps['num'] += 1
        self.record_fps['time'] += step_all_time
        if not self.training and self.record_fps['num'] % 50 == 0: 
            print(' FPS: %.2f'%(1.0/step_all_time))
        if not self.training and self.dataset_type=='VoD' and self.record_fps['num'] == 1296 and not self.training: 
            print(' FINAL VOD FPS: %.2f'%(1296/self.record_fps['time']))
            
    @master_only
    def draw_gt_pred_figures_3d(self, radar_points, lidar_points, imgs, gt_bboxes_3ds, gt_labels_3ds, img_metas, rescale=False, threshold=0.3, **kwargs):
        # if training we should decode the bbox from features 'outs_pts' first
        self.vis_time_box3d += 1
        if not self.vis_time_box3d % self.SAVE_INTERVALS == 0: return
        # filter out the ignored labels
        if self.training: figures_path_det3d = self.figures_path_det3d_train
        else: figures_path_det3d = self.figures_path_det3d_test
        gt_bboxes_3ds = [gt_bboxes_3ds[i][gt_labels_3ds[i]!= -1] for i in range(len(img_metas))]
        outs_pts = kwargs['outs_pts']
        if outs_pts is not None:
            bbox_list = self.pts_bbox_head.get_bboxes(*outs_pts, img_metas, rescale=False)
            bbox_list = [bbox3d2result(bboxes, scores, labels)for bboxes, scores, labels in bbox_list]
        else: bbox_list = None
                
        # starting visualization
        for i in range(imgs.shape[0]): # batch size
            # preparation
            input_img = np.array(imgs[i].cpu()).transpose(1,2,0)
            input_img = input_img*self.std[None, None, :] + self.mean[None, None, :]
            pred_bboxes_3d = bbox_list[i]['boxes_3d'] if bbox_list is not None else None
            pred_scores_3d = bbox_list[i]['scores_3d'] if bbox_list is not None else None
            pred_bboxes_3d = pred_bboxes_3d[pred_scores_3d>threshold].to('cpu') if bbox_list is not None else None
            gt_bboxes_3d = gt_bboxes_3ds[i].to('cpu')
            proj_mat = img_metas[i]["final_lidar2img"] # update lidar2img
            img_name = img_metas[i]['filename'].split('/')[-1].split('.')[0]
            # project 3D bboxes to image and get show figures
            if pred_bboxes_3d is not None:
                if len(pred_bboxes_3d) == 0: pred_bboxes_3d = None
                
            # draw in image view
            filename = str(self.vis_time_box3d) + '_' + img_name + '_det3d'
            result_path = figures_path_det3d; mmcv.mkdir_or_exist(result_path)
            # show_multi_modality_result(img=input_img, gt_bboxes=gt_bboxes_3d, pred_bboxes=pred_bboxes_3d, proj_mat=proj_mat, out_dir=figures_path_det3d, filename=filename, box_mode='lidar', show=False)
            # draw in bev view
            save_path_radar = os.path.join(figures_path_det3d, str(self.vis_time_box3d) + '_' + img_name + '_det3d_bev_radar.png')
            save_path_paper_radar = os.path.join(figures_path_det3d, str(self.vis_time_box3d) + '_' + img_name + '_det3d_bev_paper_radar.png')
            save_path_lidar = os.path.join(figures_path_det3d, str(self.vis_time_box3d) + '_' + img_name + '_det3d_bev_lidar.png')
            save_path_paper_lidar = os.path.join(figures_path_det3d, str(self.vis_time_box3d) + '_' + img_name + '_det3d_bev_paper_lidar.png')
            radar_points_i = radar_points[i].cpu().detach().numpy()[:, :3]
            lidar_points_i = lidar_points[i].cpu().detach().numpy()[:, :3]
            pd_bbox_corners = pred_bboxes_3d.corners[:, [0,2,4,6],:2].numpy()[:, (0,1,3,2), :] if pred_bboxes_3d is not None else None
            gt_bbox_corners = gt_bboxes_3d.corners[:, [0,2,4,6],:2].numpy()[:, (0,1,3,2), :] if gt_bboxes_3d is not None else None
            draw_bev_pts_bboxes(radar_points_i, gt_bbox_corners, pd_bbox_corners, save_path=save_path_radar, xlim=self.xlim, ylim=self.ylim) 
            draw_bev_pts_bboxes(lidar_points_i, gt_bbox_corners, pd_bbox_corners, save_path=save_path_lidar, xlim=self.xlim, ylim=self.ylim) 
            # for paper figures
            tmp_img_true = custom_draw_lidar_bbox3d_on_img(gt_bboxes_3d, input_img, proj_mat, img_metas, color=(61, 102, 255), thickness=3, scale_factor=3)
            tmp_img_pred = custom_draw_lidar_bbox3d_on_img(pred_bboxes_3d, input_img, proj_mat, img_metas, color=(241, 101, 72), thickness=3, scale_factor=3)
            tmp_img_alls = custom_draw_lidar_bbox3d_on_img(pred_bboxes_3d, tmp_img_true, proj_mat, img_metas, color=(241, 101, 72), thickness=3, scale_factor=3)
            mmcv.imwrite(tmp_img_true, os.path.join(result_path, f'{filename}_gt.png'))
            mmcv.imwrite(tmp_img_pred, os.path.join(result_path, f'{filename}_pred.png'))
            mmcv.imwrite(tmp_img_alls, os.path.join(result_path, f'{filename}.png'))
            draw_paper_bboxes(radar_points_i, gt_bbox_corners, pd_bbox_corners, save_path=save_path_paper_radar, xlim=self.xlim, ylim=self.ylim)
            draw_paper_bboxes(lidar_points_i, gt_bbox_corners, pd_bbox_corners, save_path=save_path_paper_lidar, xlim=self.xlim, ylim=self.ylim)

    @master_only
    def draw_gt_pred_bev(self, gt_bev_mask_binary, bev_mask, bev_mask_logit_sigmoid, img_metas, suffix='former'):
        if suffix == 'ptsbinary': self.vis_time_bev2d += 1
        if not self.vis_time_bev2d % self.SAVE_INTERVALS == 0: return
        if self.training: figures_path_bev2d = self.figures_path_bev2d_train
        else: figures_path_bev2d = self.figures_path_bev2d_test
        
        bev1 = torch.rot90(gt_bev_mask_binary, k=1, dims=(2, 3))
        bev2 = torch.rot90(bev_mask, k=1, dims=(2, 3))
        bev3 = torch.rot90(bev_mask_logit_sigmoid, k=1, dims=(2, 3))
        b, _, h, w = bev1.shape
        frame_1 = 0.5*torch.ones((1, h, 5)).to(bev_mask_logit_sigmoid.device)
        for i in range(bev_mask.shape[0]):
            img_name = img_metas[i]['filename'].split('/')[-1].split('.')[0]
            save_bev = torch.cat([frame_1, bev1[i], frame_1, bev2[i], frame_1, bev3[i], frame_1], dim=2)*0.99
            frame_2 = 0.5*torch.ones((1, 5, save_bev.shape[2])).to(bev_mask_logit_sigmoid.device)
            save_bev = torch.cat([frame_2, save_bev, frame_2], dim=1)
            save_image(save_bev, os.path.join(figures_path_bev2d, str(self.vis_time_bev2d) + '_' + img_name + '_bev2d_'+ suffix +'.png'))     
         
    @master_only
    def draw_gt_pred_bev_semantic(self, gt_bev_mask_semant, bev_logit, img_metas, suffix='ptsbinary'):
        if suffix == 'ptsbinary': self.vis_time_bev2d += 1
        if not self.vis_time_bev2d % self.SAVE_INTERVALS == 0: return
        if self.training: figures_path_bev2d = self.figures_path_bev2d_train
        else: figures_path_bev2d = self.figures_path_bev2d_test
        
        bev_logit = bev_logit.softmax(dim=1)
        
        B, _, H, W = gt_bev_mask_semant.shape
        device = gt_bev_mask_semant.device
        dtype = gt_bev_mask_semant.dtype
        color_map = {
            0: [1.0, 1.0, 1.0],       # black Background
            1: [1.0, 0.0, 0.0],       # red Pedestrian
            2: [0.0, 1.0, 0.0],       # green Cyclist
            3: [0.0, 0.0, 1.0]}       # blue Car
        if self.num_classes == 4:
            color_map['4'] = [1.0, 1.0, 0.0] # yellow Truck

        gt_bev_mask_binary = torch.zeros(B, 3, H, W, dtype=dtype, device=device)
        pd_bev_mask = torch.zeros(B, 3, H, W, dtype=dtype, device=device)
        for class_idx, color in color_map.items():
            mask = (gt_bev_mask_semant == class_idx)  # (B, H, W)
            color = torch.tensor(color, dtype=dtype, device=device).view(1, 3, 1, 1)  # (1, 3, 1, 1)
            color = color.repeat(B, 1, H, W)  # (B, 3, H, W)
            gt_bev_mask_binary = torch.where(mask.view(B, 1, H, W), color, gt_bev_mask_binary)
            tmp_l = bev_logit[:,class_idx,:,:].unsqueeze(1).repeat(1,3,1,1)
            pd_bev_mask = pd_bev_mask + color * tmp_l
            
        bev1 = torch.rot90(gt_bev_mask_binary, k=1, dims=(2, 3))
        bev2 = torch.rot90(pd_bev_mask, k=1, dims=(2, 3))
        b, _, h, w = bev1.shape
        frame_1 = 0.5*torch.ones((3, h, 5)).to(device)
        for i in range(B):
            img_name = img_metas[i]['filename'].split('/')[-1].split('.')[0]
            save_bev = torch.cat([frame_1, bev1[i], frame_1, bev2[i], frame_1], dim=2)*0.99
            frame_2 = 0.5*torch.ones((3, 5, save_bev.shape[2])).to(pd_bev_mask.device)
            save_bev = torch.cat([frame_2, save_bev, frame_2], dim=1)
            save_image(save_bev, os.path.join(figures_path_bev2d, str(self.vis_time_bev2d) + '_' + img_name + '_bev2d_'+ suffix +'.png'))     
         
    @master_only
    def draw_bev_feature_map(self, bev_feats, img_metas, bev_feats_name='bev_feats'):
        if bev_feats_name=='bev_feats': self.vis_time_bevnd += 1
        if not self.vis_time_bevnd % self.SAVE_INTERVALS == 0: return
        if self.training: figures_path_bevnd = self.figures_path_bevnd_train
        else: figures_path_bevnd = self.figures_path_bevnd_test
            
        b, _, h, w = bev_feats.shape 
        bev_feats = F.interpolate(bev_feats.detach()[:,:,5:-5,5:-5], (h, w), mode='bilinear', align_corners=True)
        # bev_feats = bev_feats.mean(1).unsqueeze(1) # using mean
        bev_feats_show = bev_feats.max(1, keepdim=True).values # using max
        # if bev_feats_name == 'pts_bev_feats': bev_feats_show = bev_feats.mean(dim=1).unsqueeze(1) # using mean
        # else: bev_feats_show = bev_feats.max(1, keepdim=True).values # using max
        # bev_feats_show = torch.rot90(bev_feats_show, k=2, dims=(2, 3))\
        bev_feats_show = torch.flip(bev_feats_show, [2]) # horizontal flip for consistency to gt bev bbox
        for i in range(bev_feats.shape[0]):
            img_name = img_metas[i]['filename'].split('/')[-1].split('.')[0]
            bev_feats_tmp = bev_feats_show[i:i+1, :, :, :]
            bev_feats_tmp = (bev_feats_tmp - bev_feats_tmp.min())/(bev_feats_tmp.max() - bev_feats_tmp.min())
            # bev_feats_tmp = (bev_feats_tmp - 0.75)/(1.00 - 0.75)
            bev_feats_tmp_np = bev_feats_tmp.squeeze().cpu().detach().numpy()
            bev_feats_tmp_colored = plt.cm.viridis(bev_feats_tmp_np)[..., :3] 
            bev_feats_tmp_colored = torch.tensor(bev_feats_tmp_colored).permute(2, 0, 1).unsqueeze(0)
            save_image(bev_feats_tmp_colored, os.path.join(figures_path_bevnd, str(self.vis_time_bevnd) + '_' + img_name + '_' + bev_feats_name + '.png'))
    
    @master_only
    def draw_gt_pred_rangeview(self, img_metas, segs, gts, preds, sigmoids, eroded=False):
        if not eroded: self.vis_time_range += 1
        if not self.vis_time_range % self.SAVE_INTERVALS == 0: return
        if self.training: figures_path_range = self.figures_path_range_train
        else: figures_path_range = self.figures_path_range_test
        
        b, _, h, w = gts.shape
        frame_1 = 0.5*torch.ones((1, 5, w)).to(preds.device)
        for i in range(gts.shape[0]):
            img_name = img_metas[i]['filename'].split('/')[-1].split('.')[0]
            seg = segs[i]; gt = gts[i]; pred = preds[i]; sigmoid = sigmoids[i]
            save_range = torch.cat([frame_1, seg, frame_1, gt, frame_1, pred, frame_1, sigmoid, frame_1], dim=1)
            frame_2 = 0.5*torch.ones((1, save_range.shape[1], 5)).to(pred.device)
            save_range = torch.cat([frame_2, save_range, frame_2], dim=2)
            if not eroded: save_image(save_range, os.path.join(figures_path_range, str(self.vis_time_range) + '_' + img_name + '_range.png'))
            else: save_image(save_range, os.path.join(figures_path_range, str(self.vis_time_range) + '_' + img_name + '_range_eroded.png'))
    
    @master_only
    def draw_depth_estimation(self, img_metas, gt_depths, pd_depths, img, extra_depth, radar_depth, lidar_depth):
        self.vis_time_depth += 1
        if not self.vis_time_depth % self.SAVE_INTERVALS == 0: return
        if self.training: figures_path_depth = self.figures_path_depth_train
        else: figures_path_depth = self.figures_path_depth_test
        B, C, H, W = img.shape
        cam_depth_range = self.grid_config['dbound']
        depth_gaussian_labels, depth_values = generate_guassian_depth_target(gt_depths, self.downsample, cam_depth_range, constant_std=0.5)
        depth_gaussian_labels = depth_gaussian_labels.view(B, H//self.downsample, W//self.downsample, self.D)
        pd_depths = pd_depths.permute(0, 2, 3, 1).contiguous().view(B, H//self.downsample, W//self.downsample, self.D)
        for i in range(B):
            img_name = img_metas[i]['filename'].split('/')[-1].split('.')[0]
            batch_min_depth = torch.min(gt_depths[i:i+1].view(1, -1), dim=1).values.cpu().detach().numpy()
            batch_max_depth = torch.max(gt_depths[i:i+1].view(1, -1), dim=1).values.cpu().detach().numpy()
            batch_depth_gt = draw_sum_depth(depth_gaussian_labels[i:i+1], (H, W), cam_depth_range, [batch_min_depth, batch_max_depth])
            batch_depth_pred = draw_sum_depth(pd_depths[i:i+1], (H, W), cam_depth_range, [batch_min_depth, batch_max_depth])
            batch_depth_extra = draw_true_depth(extra_depth[i:i+1], [batch_min_depth, batch_max_depth])
            batch_depth_radar = draw_true_depth(radar_depth[i:i+1], [batch_min_depth, batch_max_depth]) # img array B H W C
            batch_depth_lidar = draw_true_depth(lidar_depth[i:i+1], [batch_min_depth, batch_max_depth]) # img array B H W C
            img_vis = np.array(img.cpu()).transpose(0, 2, 3, 1)
            img_vis = img_vis[:, :, :, (2,1,0)]
            img_vis = img_vis*self.std[None, None, None, :] + self.mean[None, None, None, :] # for visualization
            img_show = img_vis[i][:,:,(2,1,0)]
            out_depth_image_1 = np.concatenate([batch_depth_gt, batch_depth_pred, img_show], axis=0)
            out_depth_image_2 = np.concatenate([batch_depth_extra, batch_depth_radar, batch_depth_lidar], axis=0)
            out_depth_image = np.concatenate([out_depth_image_1, out_depth_image_2], axis=1).clip(0, 255).astype(np.uint8)
            cv2.imwrite(os.path.join(figures_path_depth, str(self.vis_time_depth) + '_' + img_name + '_depth.png'), out_depth_image)
    
    @master_only     
    def draw_pts_completion(self, img_metas, gt_points, pd_points, gt_bboxes_3d=None, gt_labels_3d=None, plot_mode='distance', points_type='pointpainting'):
        if points_type == 'pointseglogit': self.vis_time_point += 1
        if not self.vis_time_point % self.SAVE_INTERVALS == 0: return
        if self.training: figures_path_point = self.figures_path_point_train
        else: figures_path_point = self.figures_path_point_test
        
        for i in range(len(img_metas)):
            img_name = img_metas[i]['filename'].split('/')[-1].split('.')[0]
            fig, axes = plt.subplots(1, 2, figsize=(20, 10))
            
            for ax, points, title in zip(axes, [gt_points, pd_points], ['Raw Points', 'Virtual Points']):
                points = points[i][(self.xlim[0]<=points[i][:,0]) & (points[i][:,0]<=self.xlim[1]) & \
                    (self.ylim[0]<=points[i][:,1]) & (points[i][:,1]<=self.ylim[1])]
                ax.set_xlim(self.xlim[0], self.xlim[1])
                ax.set_ylim(self.ylim[0], self.ylim[1])
                ax.autoscale(False)

                # plot points
                points = points.cpu().detach().numpy()
                x = points[:, 0]
                y = points[:, 1]
                if plot_mode == 'distance': 
                    intensities = np.clip(np.sqrt(x**2 + y**2) / 60, 0, 1)
                    colors = plt.cm.gray(intensities)
                if plot_mode == 'RCS': 
                    norm_max = np.max(gt_points[i].cpu().detach().numpy()[:, 3])
                    norm_min = np.min(gt_points[i].cpu().detach().numpy()[:, 3])
                    intensities = np.clip((points[:, 3]-norm_min)/(norm_max-norm_min), 0, 1)
                    colors = plt.cm.jet(intensities)
                if plot_mode == 'v_r_compensated': 
                    norm_max = np.max(gt_points[i].cpu().detach().numpy()[:, 4])
                    norm_min = np.min(gt_points[i].cpu().detach().numpy()[:, 4])
                    intensities = np.clip((points[:, 4]-norm_min)/(norm_max-norm_min), 0, 1)
                    colors = plt.cm.jet(intensities)
                if plot_mode == 'logits': 
                    norm_max = np.max(points[:, -1])
                    norm_min = np.min(points[:, -1])
                    intensities = np.clip((points[:, -1]-norm_min)/(norm_max-norm_min), 0, 1)
                    intensities = 1 - intensities
                    intensities = intensities*0.6 + 0.2 # 0.2 - 0.8
                    colors = plt.cm.gray(intensities)
                if plot_mode == 'points_wise_seglogits':
                    index = np.argmax(points[:, 3:], axis=-1)
                    colors = np.ones((index.shape[0], 4))
                    colors[index == 0, :3] = [1, 0, 0] # red Pedestrian
                    colors[index == 1, :3] = [0, 1, 0] # green Cyclist
                    colors[index == 2, :3] = [0, 0, 1] # blue Car
                    if self.num_classes == 4:
                        colors[index == 3, :3] = [0, 0.5, 1] 
                    colors[index == self.num_classes, :3] = [0, 0, 0]
                if plot_mode == 'context_pointpainting':
                    context = np.sum(points[:, 5:], axis=-1)
                    norm_max = np.max(np.sum(gt_points[i].cpu().detach().numpy()[:, 5:], axis=-1))
                    norm_min = np.min(np.sum(gt_points[i].cpu().detach().numpy()[:, 5:], axis=-1))
                    intensities = np.clip((context-norm_min)/(norm_max-norm_min), 0, 1)
                    colors = plt.cm.jet(intensities)
                if plot_mode == 'context_voxelpainting': 
                    context = np.sum(points[:, 5:], axis=-1)
                    norm_max = np.max(np.sum(gt_points[i].cpu().detach().numpy()[:, 5:], axis=-1))
                    norm_min = np.min(np.sum(gt_points[i].cpu().detach().numpy()[:, 5:], axis=-1))
                    intensities = np.clip((context-norm_min)/(norm_max-norm_min), 0, 1)
                    intensities = 1 - intensities
                    intensities = intensities*0.6 + 0.2 # 0.2 - 0.8
                    colors = plt.cm.gray(intensities)
                    sorted_indices = np.argsort(-intensities)
                    x = x[sorted_indices]
                    y = y[sorted_indices]
                    colors = colors[sorted_indices]
                ax.scatter(x, y, c=colors, s=15) # alpha=0.5

                # plot bboxes
                if gt_bboxes_3d is not None:
                    if len(gt_bboxes_3d) != 0:
                        gt_bboxes_3d_filtered = gt_bboxes_3d[i][gt_labels_3d[i] != -1]
                        gt_bbox_corners = gt_bboxes_3d_filtered.corners[:, [0,2,4,6],:2]
                        gt_bbox_corners = gt_bbox_corners.cpu().detach().numpy()[:, (0,1,3,2), :] # clock_corners
                        for bbox in gt_bbox_corners:
                            polygon = patches.Polygon(bbox, closed=True, edgecolor='red', linewidth=1, fill=False)
                            ax.add_patch(polygon)
                        
                ax.set_xlabel('X (m)')
                ax.set_ylabel('Y (m)')
                ax.set_title(f'Point cloud and bboxes under BEV - {title}')
                ax.grid(True)
            
            save_path = os.path.join(figures_path_point, str(self.vis_time_point) + '_' + img_name + '_' + points_type +'.png')
            plt.savefig(save_path)
            plt.close(fig)
            
    @master_only      
    def draw_bboxes_on_image(self, img, pd_bboxes_2d, gt_bboxes_2d, img_metas, thickness=4, threshold=0.6):
        
        self.vis_time_det2d += 1
        if not self.vis_time_det2d % self.SAVE_INTERVALS == 0: return
        if self.training: figures_path_det2d = self.figures_path_det2d_train
        else: figures_path_det2d = self.figures_path_det2d_test
        
        device = img.device
        input_img = copy.deepcopy(img)
        std = torch.tensor(self.std[None, :, None, None]).to(img)
        mean = torch.tensor(self.mean[None, :, None, None]).to(img)
        input_img = (input_img * std + mean)[:, (2, 1, 0), :, :]
        input_img = (input_img / 255).clamp(0, 1)

        B, C, H, W = input_img.shape
        for b in range(B):
            img_name = img_metas[b]['filename'].split('/')[-1].split('.')[0]
            predict_bboxes_2d = pd_bboxes_2d[b][pd_bboxes_2d[b][:,4] > threshold]
            for bbox in gt_bboxes_2d[b].tensor:
                tl_x, tl_y, br_x, br_y = bbox.int()
                tl_x = torch.clamp(tl_x, 0, W - 1)
                tl_y = torch.clamp(tl_y, 0, H - 1)
                br_x = torch.clamp(br_x, 0, W - 1)
                br_y = torch.clamp(br_y, 0, H - 1)
                input_img[b, :, tl_y:tl_y + thickness, tl_x:br_x] = torch.tensor((61, 102, 255), device=device).view(-1, 1, 1)/255.0
                input_img[b, :, br_y - thickness:br_y, tl_x:br_x] = torch.tensor((61, 102, 255), device=device).view(-1, 1, 1)/255.0
                input_img[b, :, tl_y:br_y, tl_x:tl_x + thickness] = torch.tensor((61, 102, 255), device=device).view(-1, 1, 1)/255.0
                input_img[b, :, tl_y:br_y, br_x - thickness:br_x] = torch.tensor((61, 102, 255), device=device).view(-1, 1, 1)/255.0
            for bbox, class_index in zip(predict_bboxes_2d[:, :4], predict_bboxes_2d[:, 5:6]):
                if    class_index == 0: color = torch.tensor((241, 101, 72), device=device).view(-1, 1, 1)/255.0
                elif  class_index == 1: color = torch.tensor((241, 101, 72), device=device).view(-1, 1, 1)/255.0
                elif  class_index == 2: color = torch.tensor((241, 101, 72), device=device).view(-1, 1, 1)/255.0
                else: color = torch.tensor((241, 101, 72), device=device).view(-1, 1, 1)/255.0
                tl_x, tl_y, br_x, br_y = bbox.int()
                tl_x = torch.clamp(tl_x, 0, W - 1)
                tl_y = torch.clamp(tl_y, 0, H - 1)
                br_x = torch.clamp(br_x, 0, W - 1)
                br_y = torch.clamp(br_y, 0, H - 1)
                input_img[b, :, tl_y:tl_y + thickness, tl_x:br_x] = color
                input_img[b, :, br_y - thickness:br_y, tl_x:br_x] = color
                input_img[b, :, tl_y:br_y, tl_x:tl_x + thickness] = color
                input_img[b, :, tl_y:br_y, br_x - thickness:br_x] = color
        
            save_path = os.path.join(figures_path_det2d, str(self.vis_time_det2d) + '_' + img_name + '_det2d.png')
            save_image(input_img[b:b+1], save_path)
            
        return input_img
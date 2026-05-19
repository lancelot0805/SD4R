import torch, copy, time, os, mmcv, cv2
from mmcv.runner import force_fp32
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
from PIL import Image

@DETECTORS.register_module()
class SD4R_students(MVXFasterRCNN):
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
                distill_setts = dict(use=False, semi=False, freeze=True, adaptor=None, teacher_cfg=None, checkpoint=None), 
                architectures = dict(use_DCN=False, use_QFP=False, use_VPE=True, use_PBF=True),
                # training details
                use_grid_mask=False,
                freeze_depths=False,
                freeze_radars=False,
                freeze_images=True,
                freeze_ranges=False,
                freeze_propss=False,
                # framework config
                segmentor=None,
                depth_net=None,
                pvseg_net=None,
                fuses_net=None,
                bvseg_net=None,
                focus_net=None,
                warps_net=None,
                qfuse_net=None,
                p_BEV_net=None,
                graph_conv=None,
                vote_gnn=None,
                radar_voxel_layer=None,
                radar_voxel_encoder=None,
                radar_middle_encoder=None,
                radar_bev_backbone=None,
                radar_bev_neck=None,
                bev_fusion_net=None,
                point_net=None,
                meta_info=None,
                **kwargs):
        self.pts_bbox_head = kwargs.pop('pts_bbox_head')
        self.vote_pts_voxel_layer = kwargs['pts_voxel_layer']
        self.vote_pts_voxel_encoder = kwargs['pts_voxel_encoder']
        self.vote_pts_middle_encoder = kwargs['pts_middle_encoder']
        self.vote_pts_backbone = kwargs['pts_backbone']
        self.vote_pts_neck = kwargs['pts_neck']
        super(SD4R_students, self).__init__(**kwargs)

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
        # architecture
        self.aux_bbox_head = aux_bbox_head
        self.camera_stream = camera_stream
        self.depth_complet = depth_complet
        self.focus_supervision = focus_supervision
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
        
        # for teacher and student framework design
        self.segmentor = builder.build_detector(segmentor) if (segmentor and self.architectures['use_VPE']) else None
        # self.fuses_net = FUSION_LAYERS.build(fuses_net) if self.segmentor is not None else nn.Conv2d(self.rad_channels, self._dim_, kernel_size=1)
        self.fuses_net = nn.Conv2d(self.rad_channels, self._dim_, kernel_size=1)
        self.focus_net = FUSION_LAYERS.build(focus_net) if (focus_net and self.focus_supervision['use']) else None
        self.bvseg_net = FUSION_LAYERS.build(bvseg_net) if (bvseg_net and self.props_supervision['use']) else None
        self.warps_net = FUSION_LAYERS.build(warps_net) if (warps_net and self.architectures['use_DCN']) else None
        
        self.qfuse_net = FUSION_LAYERS.build(qfuse_net) if (qfuse_net and self.architectures['use_QFP']) else None
        self.p_BEV_net = FUSION_LAYERS.build(p_BEV_net) if (p_BEV_net and self.architectures['use_PBF']) else None
        
        # init weights and freeze if needed
        self.init_default_modules()
        self.init_weights()
        if self.distill_setts['use']: 
            self.init_distills_modules()
        if self.freeze_images: self.freeze_img_model()
        if self.freeze_radars: self.freeze_pts_model()
        if self.freeze_ranges: self.freeze_msk_model()
        self.record_fps = {'num': 0, 'time':0}
        self.init_visulization()
    
    def init_default_modules(self):
        # if self.segmentor is not None and self.qfuse_net is not None:
        #     channels = self.vote_pts_voxel_encoder['feat_channels'][0]
        #     self.vote_pts_voxel_encoder['type'] = 'PillarFeatureNet'
        #     self.vote_pts_voxel_encoder['in_channels'] = 3 + channels
        #     self.vote_pts_voxel_encoder.pop('with_velocity_snr_center')
        #     self.vote_pts_voxel_encoder.pop('num_classes')
        #     self.vote_pts_backbone['layer_nums'] = [2, 4, 4] # less parameters
        #     self.p_BEV_net.vote_pts_voxel_layer = Voxelization(**self.vote_pts_voxel_layer)
        #     self.p_BEV_net.vote_pts_voxel_encoder = builder.build_voxel_encoder(self.vote_pts_voxel_encoder)
        #     self.p_BEV_net.vote_pts_middle_encoder = builder.build_middle_encoder(self.vote_pts_middle_encoder)
        #     self.p_BEV_net.vote_pts_backbone = builder.build_backbone(self.vote_pts_backbone)
        #     self.p_BEV_net.vote_pts_neck = builder.build_backbone(self.vote_pts_neck)
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
    
    def init_distills_modules(self):
        cfg = Config.fromfile(self.distill_setts['teacher_cfg'])
        project_name_tmp = self.project_name.replace('student', 'teacher')
        figures_path_tmp = self.figures_path.replace('figures_path', 'figures_path_teacher')
        cfg.model.update(meta_info = {'figures_path': figures_path_tmp, 'project_name': project_name_tmp})
        self.teacher_input_size = cfg.ida_aug_conf['final_dim']
        self.teacher=build_model(
            cfg.model,
            train_cfg=cfg.get('train_cfg'),
            test_cfg=cfg.get('test_cfg'))
        checkpoint = torch.load(self.distill_setts['checkpoint'], map_location='cpu')['state_dict']
        missing_keys, unexpected_keys = self.teacher.load_state_dict(checkpoint, strict=True)
        if len(missing_keys) > 0: print("Teacher missing keys:", missing_keys)
        if len(unexpected_keys) > 0: print("Teacher unexpected keys:", unexpected_keys)
        self.teacher.SAVE_INTERVALS = self.SAVE_INTERVALS
        if self.distill_setts['freeze']:
            for param in self.teacher.parameters():
                param.requires_grad = False
        # adaptor for distillation
        # NOTE: None: no loss, 0: directly mse loss, 1: 1 layer adaptor, 2: 2 layer adaptor
        self.adaptor_fuses_feats = self.distill_setts['adaptor']['fuses_feats']
        self.adaptor_point_feats = self.distill_setts['adaptor']['point_feats']
        fuses_c, point_c = self._dim_, self.rad_channels
        if self.adaptor_fuses_feats==1: self.adaptor_fuses = nn.Conv2d(fuses_c, fuses_c, kernel_size=1, stride=1, padding=0)
        if self.adaptor_fuses_feats==2: 
            self.adaptor_fuses_1 = nn.Conv2d(fuses_c, fuses_c//2, kernel_size=1, stride=1, padding=0)
            self.adaptor_fuses_2 = nn.Conv2d(fuses_c, fuses_c//2, kernel_size=1, stride=1, padding=0)
        if self.adaptor_point_feats==1: self.adaptor_point = nn.Conv2d(fuses_c, point_c, kernel_size=1, stride=1, padding=0)   
        if self.adaptor_point_feats==2: 
            self.adaptor_point_1 = nn.Conv2d(point_c, point_c//2, kernel_size=1, stride=1, padding=0)
            self.adaptor_point_2 = nn.Conv2d(point_c, point_c//2, kernel_size=1, stride=1, padding=0)
            
    def init_visulization(self):
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

    def extract_pts_feat(self, pts, gt_bboxes_3d, gt_labels_3d, img_metas):
        """Extract features of input radar points."""
        time_start = time.time()
        
        if not self.with_pts_backbone: return None
        batch_size = len(pts)
        # VPE for feature extraction and qfuse for further enhancement
        if self.segmentor is not None and self.qfuse_net is not None:            
            # extract point-wise features using voxel-attended points feature extractor
            if self.training: segmt_dicts = self.segmentor(points=pts, img_metas=img_metas, gt_bboxes_3d=gt_bboxes_3d, gt_labels_3d=gt_labels_3d, as_subsegmentor=True)
            #如果是训练需要监督，测试就用simpletest推理
            # 得到的dict如下：
            #  output_dict = dict(
            #     seg_point=points, # raw points [13848,5]
            #     seg_feats=feats,  # VPE feats for points  [13848,128]
            #     seg_logits=preds_dict['seg_logits'],  [13848,4]
            #     seg_vote_preds=preds_dict['vote_preds'], [13848,9]
            #     offsets=offsets, # decode offset [13848,9]
            #     batch_idx=batch_idx, [13848]
            #     losses=losses, # seg and vote offset
            #     decoder_features=decoder_features, # None
            # )
            else: segmt_dicts = self.segmentor.simple_test(points=pts, img_metas=img_metas, rescale=False)
            # using segmentation logits and voting output to generate more (center) points
           
            if self.training: vote_sampled_out = self.sample(segmt_dicts, fg_threshold=0.7) # per cls list in sampled_out
            else: vote_sampled_out = self.sample(segmt_dicts, fg_threshold=0.4)
            # segmentation logits and voting output
            segmt_dicts['vote_sampled_out'] = vote_sampled_out
            all_vote_centers = vote_sampled_out['all_vote_centers']
            all_vote_logitss = vote_sampled_out['all_vote_logitss']
            all_vote_feature = vote_sampled_out['all_vote_feature']

            # add voted points and use qfusenet t extract finer features
            interpolate_vote_pts = self.vote_knn_interpolate(all_vote_centers, pts, k=5)
            
            # logits to cat with all_pts as supplemental features
            raws_pts_logits = [torch.softmax(segmt_dicts['seg_logits'][segmt_dicts['batch_idx']==i], dim=-1) for i in range(batch_size)]
           
            all_pts = [torch.cat([torch.cat([pts[i], raws_pts_logits[i]], dim=-1), torch.cat([interpolate_vote_pts[i], all_vote_logitss[i]], dim=-1)], dim=0) for i in range(len(pts))]
            
            # vanilla coventional radarpillarnet for global context extraction
            voxels, num_points, coors = self.voxelize(all_pts)
            #此时的voxels是[N,M,C]表示N个体素，每个体素内最多M个点，C通道 [9525,25,9]
            #coors:[9525,4] batchidx和z,y,x  
            
            voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,)
            #pillar的处理方法
            
            # using query-based prefusion, which utilize seg_feats*seg_logits to enhance the voxel_features
            voxel_features = self.qfuse_net(voxel_features, coors, segmt_dicts,voxels)
            global_x = self.pts_middle_encoder(voxel_features, coors, batch_size)
            global_x = self.pts_backbone(global_x)
            global_x = self.pts_neck(global_x)[0]
            x = global_x
            
            # BEV segmentation for radius selecting, car bigger radius, then pefrom point-BEV fusion
            if self.props_supervision['use']:
                #生成logits
                bev_binary_logit = self.bvseg_net.forward_binary(x.permute(0, 1, 3, 2).contiguous().detach())
                #生成logits
                bev_semant_logit = self.bvseg_net.forward_semant(x.permute(0, 1, 3, 2).contiguous().detach())
            else: bev_binary_logit, bev_semant_logit = None, None
            x, ins_points , ins_point_xyz= self.p_BEV_net(x, vote_sampled_out, segmt_dicts, bev_semant_logit)
            
        else: # vanilla conventional radarpillarnet
            voxels, num_points, coors = self.voxelize(pts)
            voxel_features = self.pts_voxel_encoder(voxels, num_points, coors,)
            x = self.pts_middle_encoder(voxel_features, coors, batch_size)
            x = self.pts_backbone(x)
            x = self.pts_neck(x)[0]
            
        time_end = time.time()
        FPS = 1 / (time_end - time_start)
        return x, segmt_dicts, FPS, ins_points, ins_point_xyz, bev_binary_logit, bev_semant_logit
    
    def vote_knn_interpolate(self, vote_pts, raws_pts, k=5):
        device = vote_pts[0].device
        dtype = vote_pts[0].dtype
        interpolate_vote_pts = []
        for bid in range(len(vote_pts)):
            tmp = torch.zeros((vote_pts[bid].shape[0], 5), device=device, dtype=dtype)
            this_vote_pts = vote_pts[bid][:,:2] # N, 3
            this_raws_pts = raws_pts[bid][:,:2] # M, 3
            this_raws_fts = raws_pts[bid][:,3:] # M, 2
            dist = torch.cdist(this_vote_pts, this_raws_pts)  # (N, M)
            knn_indices = dist.topk(k, largest=False).indices  # (N, k)

            # 提取 KNN 邻居点的特征
            knn_feats = this_raws_fts[knn_indices]  # (N, k, 2)
            knn_dists = dist.gather(1, knn_indices)  # (N, k)
            knn_weights = 1.0 / (knn_dists + 1e-8)  # 避免除 0
            knn_weights = knn_weights / knn_weights.sum(dim=1, keepdim=True)  # 归一化
            interpolated_feats = torch.sum(knn_feats * knn_weights.unsqueeze(-1), dim=1)  # (N, 2)
            tmp[:, 0:3] = vote_pts[bid][:,:3]
            tmp[:, 3:5] = interpolated_feats
            interpolate_vote_pts.append(tmp)
        return interpolate_vote_pts
    
    def sample(self, dict_to_sample, fg_threshold=0.1):

        offset = dict_to_sample['offsets']
        offset = offset.reshape(-1, self.num_classes, 3)
        seg_point = dict_to_sample['seg_point'][:, :3]
        seg_feats = dict_to_sample['seg_feats']

        batch_idx = dict_to_sample['batch_idx']
        assert batch_idx.numel() > 0
        batch_size = batch_idx.max().item() + 1
        seg_logits = dict_to_sample['seg_logits']
        seg_logits = torch.softmax(seg_logits, dim=-1)
        foreground_prob = 1 - seg_logits[:, -1]
        fg_mask = foreground_prob > fg_threshold
        while sum(fg_mask)==0 or len(torch.unique(batch_idx[fg_mask])) < batch_size:
            fg_threshold = fg_threshold - 0.01
            fg_mask = foreground_prob > fg_threshold

        votseg_feats_list = [] # predict foreground mask of each class
        center_preds_list = [] # center of each fg points, by class
        votseg_logit_list = [] # recording correspoing logits, by class
        for bid in range(batch_size):
            pred_cls = torch.argmax(seg_logits[:, :-1][(fg_mask) & (bid==batch_idx)], dim=-1)
            pred_pro = seg_logits[(fg_mask) & (bid==batch_idx), :]
            pred_off = offset[(fg_mask) & (bid==batch_idx)]
            pred_pts = seg_point[(fg_mask) & (bid==batch_idx), :]
            pred_fts = seg_feats[(fg_mask) & (bid==batch_idx), :]
            tmp_votseg_feats_list = [] # predict foreground mask of each class
            tmp_center_preds_list = [] # center of each fg points, by class
            tmp_votseg_logit_list = [] # recording correspoing logits, by class
            for cls in range(self.num_classes):
                this_segfts = pred_fts[pred_cls == cls, :]
                this_offset = pred_off[pred_cls == cls, cls, :]
                this_logits = pred_pro[pred_cls == cls, :]
                this_center = pred_pts[pred_cls == cls, :] + this_offset
                tmp_votseg_feats_list.append(this_segfts)
                tmp_center_preds_list.append(this_center)
                tmp_votseg_logit_list.append(this_logits)
            votseg_feats_list.append(tmp_votseg_feats_list)
            center_preds_list.append(tmp_center_preds_list)
            votseg_logit_list.append(tmp_votseg_logit_list)

        output_dict = dict()
        output_dict['votseg_feats_cls'] = votseg_feats_list
        output_dict['center_preds_cls'] = center_preds_list
        output_dict['votseg_logit_cls'] = votseg_logit_list
        output_dict['all_vote_feature'] = [torch.cat(x, dim=0) for x in votseg_feats_list] # batch all cat
        output_dict['all_vote_centers'] = [torch.cat(x, dim=0) for x in center_preds_list] # batch all cat
        output_dict['all_vote_logitss'] = [torch.cat(x, dim=0) for x in votseg_logit_list] # batch all cat
        return output_dict
    
    def point_segmentation_label_generation(self, points_list, gt_bboxes_3d_list, gt_labels_3d_list):
        gt_bboxes_3d_list = copy.deepcopy(gt_bboxes_3d_list)
        gt_labels_3d_list = copy.deepcopy(gt_labels_3d_list)
        point_seg_label_list = []
        for points, gt_bboxes_3d, gt_label_3d in zip(points_list, gt_bboxes_3d_list, gt_labels_3d_list):
            x = -points[:, 1].cpu()  # Swap x and y, negate new x to rotate clockwise
            y = points[:, 0].cpu()   # Swap x and y
            gt_bboxes_3d.tensor[:,3:4] = gt_bboxes_3d.tensor[:,3:4] # *1.05
            gt_bboxes_3d.tensor[:,4:5] = gt_bboxes_3d.tensor[:,4:5] # *1.15
            gt_bbox_corners = gt_bboxes_3d.corners[:, [0,2,4,6],:2].cpu().detach().numpy()[:, (0,1,3,2), :]
            point_seg_label = self.num_classes*np.ones(points.shape[0], dtype=bool) # init as background
            for i in range(self.num_classes):
                specific_gt_bbox_corners = gt_bbox_corners[gt_label_3d.cpu().detach().numpy() == i]
                for bbox in specific_gt_bbox_corners:
                    rotated_bbox = np.array([[-corner[1], corner[0]] for corner in bbox])
                    path = Path(rotated_bbox)
                    point_seg_label[path.contains_points(np.vstack((x, y)).T)] = i
            point_seg_label_list.append(point_seg_label)            
            # NOTE: not implemented yet
            # dims = gt_bboxes_3d.tensor[:,3:6].cpu().detach().numpy()
            # locs = gt_bboxes_3d.tensor[:,0:3].cpu().detach().numpy()
            # rots = gt_bboxes_3d.tensor[:,6:7].cpu().detach().numpy()
            # gt_boxes_lidar = np.concatenate([locs, dims, rots], axis=1)
            # indices = box_np_ops.points_in_rbbox(points[:,:3].cpu().detach().numpy(), gt_boxes_lidar) # [M, N] M is the number of points, N is the number of gt boxes
            # num_points_in_gt = indices.sum(0)     
            # point_seg_label = self.num_classes*np.ones(points.shape[0], dtype=bool) # init as background
            # point_seg_label_list.append(point_seg_label)
        return point_seg_label_list
    
    # NOTE: core model here, processing multi-modality feats
       
    def extract_feat(self, points, img, img_metas, processed_info):
        """Extract features from images and points."""

        # preparation of camera-geo-aware input
        if img is not None:
            if img.dim() == 3 and img.size(0)== 3: img = img.unsqueeze(0)
        if not isinstance(points, list): points = [points]
        
        B = len(points)
        img_metas, gt_bboxes_3d, gt_labels_3d, gt_bboxes_2d, gt_labels_2d, depth_comple, bbox_Mask, segmentation, radar_depth, cam_aware, \
            img_aug_matrix, lidar_aug_matrix, bda_rot, gt_depths, gt_bev_mask_binary, gt_bev_mask_semant, final_lidar2img, lidar_points = processed_info
        device = points[0].device

        # #创建投影矩阵
        # matrix = torch.eye(4).to(final_lidar2img.device)
        # matrix[0, 0] = matrix[0, 0] / self.downsample
        # matrix[1, 1] = matrix[1, 1] / self.downsample
        # projection = matrix @ final_lidar2img
        
        radar_points = copy.deepcopy(points)
        lidar_points = copy.deepcopy(points) if lidar_points is None else copy.deepcopy(lidar_points)
        # gt_point_segmentation = self.point_segmentation_label_generation(radar_points, gt_bboxes_3d, gt_labels_3d)
        
        # 1. pre-extract img and radar features of raw data | pts_bev_feats using R
        pts_bev_feats, segmt_dicts, FPS, ins_points, ins_point_xyz, bev_binary_logit, bev_semant_logit = self.extract_pts_feat(radar_points, gt_bboxes_3d, gt_labels_3d, img_metas)
        
        # print("----- FPS THIS FRAME IS: %5.2f" % FPS)
        if self.segmentor is not None and self.qfuse_net is not None:
            batch_idx, seg_logits = segmt_dicts['batch_idx'], segmt_dicts['seg_logits']
            all_vote_centers = segmt_dicts['vote_sampled_out']['all_vote_centers']
            all_vote_feature = segmt_dicts['vote_sampled_out']['all_vote_feature']
            points_seg_logits = []
            for i in range(len(points)):
                N, _ = points[i].shape
                x = seg_logits[batch_idx==i].view(N, -1)
                points_seg_logits.append(x)
            ins_points = [ins_points[i][:,:3] for i in range(B)]
            ins_points_xyz = [ins_point_xyz[i][:,:3] for i in range(B)]
            seg_logits = [torch.cat([points[i][:,:3], points_seg_logits[i]], dim=1) for i in range(B)]
            points_vis_point = [points[i][:,:3] for i in range(B)]
            points_vis_voted = [torch.cat([all_vote_centers[i][:,:3], torch.zeros(all_vote_centers[i].shape[0], self.num_classes+1, device=device)], dim=1) for i in range(B)]
            self.draw_pts_completion(img_metas, seg_logits, points_vis_voted, gt_bboxes_3d, gt_labels_3d, plot_mode='points_wise_seglogits', points_type='pointseglogit')
            self.draw_pts_completion(img_metas, ins_points, points_vis_point, gt_bboxes_3d, gt_labels_3d, plot_mode='distance', points_type='pointfiltered')
            #left is raw point segmentation logits, right is the virtual vote points
            
            #use these when generating point clouds on images
            #self.draw_ptsandmask_on_image(img_metas,ins_points_xyz, img, segmentation, points_type='instance')
            #self.draw_ptsandmask_on_image(img_metas,points_vis_point, img, segmentation, points_type='raw')
            #self.draw_ptsandmask_on_image(img_metas,points_vis_voted, img, segmentation, points_type='vote')
            
            
        # 4. BEV fusion radar and image feats
        bev_feats = self.fuses_net(pts_bev_feats)
        
        """ if self.architectures['use_DCN']: 
            bev_feats = self.warps_net(bev_feats) """
            
        bev_feats = bev_feats.permute(0, 1, 3, 2).contiguous()
        assert bev_feats.shape[2] == self.bev_h_
        assert bev_feats.shape[3] == self.bev_w_
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
        # self.draw_depth_estimation(img_metas, gt_depths, depth, img, depth_comple, radar_depth, gt_depths)
        self.draw_bev_feature_map(bev_feats, img_metas, bev_feats_name='bev_feats')
        if self.props_supervision['use']: self.draw_gt_pred_bev(gt_bev_mask_binary.bool(), bev_binary_logit.sigmoid()>self.bvseg_net.mask_thre, bev_binary_logit.sigmoid(), img_metas, 'ptsbinary')
        if self.props_supervision['use']: self.draw_gt_pred_bev_semantic(gt_bev_mask_semant, bev_semant_logit, img_metas, 'semantics')
        if self.focus_supervision['use']: self.draw_gt_pred_bev( gt_bev_mask_binary.bool(), logits_objects.sigmoid()>self.focus_net.objects_thre, logits_objects.sigmoid(), img_metas, 'foreground')
        if self.focus_supervision['use']: self.draw_gt_pred_bev(~gt_bev_mask_binary.bool(), logits_backgrd.sigmoid()>self.focus_net.backgrd_thre, logits_backgrd.sigmoid(), img_metas, 'background')
        
        return dict(pts_feats=[bev_feats],
                    pts_bev_feats=pts_bev_feats,
                    gt_bev_mask_binary=gt_bev_mask_binary,
                    gt_bev_mask_semant=gt_bev_mask_semant,
                    bev_binary_logit=bev_binary_logit,
                    bev_semant_logit=bev_semant_logit,
                    lidar_points=lidar_points,
                    focus_dicts=focus_dicts,
                    segmt_dicts=segmt_dicts)
        
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
                # img_metas[i]['gt_labels'] = gt_labels[i]
                # img_metas[i]['gt_bboxes'] = HorizontalBoxes(gt_bboxes[i], in_mode='xyxy')
                img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i].to(gt_labels_3d[i].device)
                img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]
        processed_info = self.preprocessing_information(img_metas, gt_labels_3d[i].device)
        feature_dict = self.extract_feat(points, img=img, img_metas=img_metas, processed_info=processed_info)
        lidar_points = feature_dict['lidar_points']
        pts_feats = feature_dict['pts_feats']

        bbox_list = [dict() for i in range(len(img_metas))]
        if pts_feats and self.with_pts_bbox: # pts means 3D detection
            bbox_pts, outs_pts = self.simple_test_pts(pts_feats, img_metas, rescale=rescale)
            for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
                result_dict['pts_bbox'] = pts_bbox
        
        # visualization for test stage 
        threshold = 0.3
        if gt_bboxes_3d is not None and self.box3d_supervision['use']:
            if img.dim() == 3 and img.size(0)== 3: img = img.unsqueeze(0)
            if not isinstance(points, list): points = [points]
            self.draw_gt_pred_figures_3d(points, lidar_points, img, gt_bboxes_3d, gt_labels_3d, img_metas, False, threshold, outs_pts=outs_pts)
            if self.distill_setts['use'] == True:
                with torch.no_grad():
                    feature_dict_teacher = self.teacher.extract_feat(points, img, img_metas, list(processed_info))
                    tch_outs = self.teacher.pts_bbox_head(feature_dict_teacher['pts_feats'])
                self.teacher.draw_gt_pred_figures_3d(points, lidar_points, img, gt_bboxes_3d, gt_labels_3d, img_metas, False, threshold, outs_pts=tch_outs)
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
        pts_feats = feature_dict['pts_feats']
        gt_bev_mask_binary = feature_dict['gt_bev_mask_binary']
        gt_bev_mask_semant = feature_dict['gt_bev_mask_semant']
        bev_binary_logit = feature_dict['bev_binary_logit']
        bev_semant_logit = feature_dict['bev_semant_logit']
        lidar_points = feature_dict['lidar_points']
        focus_dicts = feature_dict['focus_dicts']
        segmt_dicts = feature_dict['segmt_dicts']
        if feature_dict['lidar_points'] is None: 
            feature_dict['lidar_points'] = points

        # compute for all explicit losses
        losses = dict()
        outs_pts = None
        if self.box3d_supervision['use'] and gt_bboxes_3d is not None:
            losses_box3d, outs_pts = self.forward_pts_train(pts_feats, gt_bboxes_3d, gt_labels_3d, img_metas, points, lidar_points, img, gt_bboxes_ignore)
            losses.update(losses_box3d)
        if self.props_supervision['use'] and self.bvseg_net is not None:
            losses_props_1 = self.bvseg_net.get_bev_mask_loss(bev_binary_logit, gt_bev_mask_binary)
            losses_props_2 = self.bvseg_net.get_bev_mask_loss_semantic(bev_semant_logit, gt_bev_mask_semant)
            losses_props = {}; losses_props.update(losses_props_1); losses_props.update(losses_props_2); 
            losses.update(losses_props)
        if self.focus_supervision['use'] and self.focus_net is not None:
            losses_focus = self.focus_net.get_feature_focus_loss(focus_dicts, gt_bev_mask_binary)
            losses.update(losses_focus)
        if self.qfuse_net is not None and self.segmentor is not None:
            losses.update(segmt_dicts['losses'])
        return losses, outs_pts
    
    def distillation_loss(self, feature_dict_teacher, feature_dict_student, distill_mask):
        self.criterion = torch.nn.MSELoss(reduction='none')
        tch_fuses_feats = feature_dict_teacher['pts_feats'][0]
        std_fuses_feats = feature_dict_student['pts_feats'][0]
        tch_point_feats = feature_dict_teacher['pts_bev_feats']
        std_point_feats = feature_dict_student['pts_bev_feats']
        device, dtype = tch_fuses_feats.device, tch_fuses_feats.dtype
        tch_outs_cls_scores = feature_dict_teacher['outs_pts'][0][0]
        std_outs_cls_scores = feature_dict_student['outs_pts'][0][0]
        tch_outs_bbox_preds = feature_dict_teacher['outs_pts'][1][0]
        std_outs_bbox_preds = feature_dict_teacher['outs_pts'][1][0]
        # get fusion feats knowledge distillation loss
        if self.adaptor_fuses_feats is not None:
            if self.adaptor_fuses_feats == 1: 
                std_fuses_feats = self.adaptor_fuses(std_fuses_feats)
            if self.adaptor_fuses_feats == 2: 
                std_fuses_feats_1 = self.adaptor_fuses_1(std_fuses_feats)
                std_fuses_feats_2 = self.adaptor_fuses_2(std_fuses_feats)
                std_fuses_feats = torch.cat([std_fuses_feats_1, std_fuses_feats_2], dim=1)
            loss_distill_L2_fuses = self.criterion(tch_fuses_feats, std_fuses_feats)
            loss_distill_L2_fuses = (loss_distill_L2_fuses*distill_mask).mean()*2e4
        else: loss_distill_L2_fuses = torch.tensor(0.0, device=device, dtype=dtype)
        # get points feats knowledge distillation loss
        if self.adaptor_point_feats is not None:
            if self.adaptor_point_feats == 1: 
                std_point_feats = self.adaptor_point(std_point_feats)
            if self.adaptor_point_feats == 2: 
                std_point_feats_1 = self.adaptor_point_1(std_point_feats)
                std_point_feats_2 = self.adaptor_point_2(std_point_feats)
                std_point_feats = torch.cat([std_point_feats_1, std_point_feats_2], dim=1)        
            loss_distill_L2_point = self.criterion(tch_point_feats, std_point_feats)
            loss_distill_L2_point = (loss_distill_L2_point*distill_mask).mean()*2e4
        else: loss_distill_L2_point = torch.tensor(0.0, device=device, dtype=dtype)
        
        # class logits supervision
        if self.distill_setts['logits']:
            temperature, gamma = 2.0, 2.0
            student_probs = F.log_softmax(std_outs_cls_scores / temperature, dim=1)
            teacher_probs = F.softmax(tch_outs_cls_scores / temperature, dim=1).clamp(min=1e-6, max=1.0)
            loss_clskl = F.kl_div(student_probs, teacher_probs, reduction='none') * (temperature ** 2)
            loss_clskl = (loss_clskl*distill_mask).mean()*1e2
            student_probs = F.softmax(std_outs_cls_scores, dim=1)
            teacher_probs = F.softmax(tch_outs_cls_scores, dim=1)
            focal_weight = ((1 - teacher_probs).clamp(min=1e-6)) ** gamma  # 关注 teacher_probs 认为重要的部分
            loss_clsce = (-focal_weight * teacher_probs * torch.log((student_probs / teacher_probs).clamp(min=1e-6))).sum(dim=1, keepdim=True)
            loss_clsce = (loss_clsce*distill_mask).mean()*1e3
        else: loss_clskl = loss_clsce = torch.tensor(0.0, device=device, dtype=dtype)

        return dict(loss_distill_L2_fuses=loss_distill_L2_fuses, 
                    loss_distill_L2_point=loss_distill_L2_point*0.03, 
                    loss_clskl=loss_clskl, loss_clsce=loss_clsce)

    def get_teacher_result(self, points, img_metas, img, processed_info_teacher, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore):
        # experiments show that drop makes learn more representation from both radar and lidar features
        # self.teacher.overall_dropout=0.2 # 0.0 # if need drop
        # self.teacher.feats_1_dropout=0.8 # 0.0 # if need drop
        if self.distill_setts['freeze']:
            with torch.no_grad():
                feature_dict_teacher = self.teacher.extract_feat(points, img, img_metas, processed_info_teacher)
                tch_outs = self.teacher.pts_bbox_head(feature_dict_teacher['pts_feats'])
                losses_teacher = dict()
        else:
            feature_dict_teacher = self.teacher.extract_feat(points, img, img_metas, processed_info_teacher)
            losses_teacher, tch_outs = self.teacher.get_loss(feature_dict_teacher, gt_bboxes_3d, gt_labels_3d, img_metas, points, img, gt_bboxes_ignore)
            losses_teacher = {key + '_teacher': value*0.01 for key, value in losses_teacher.items()}
        return feature_dict_teacher, tch_outs, losses_teacher

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
            # img_metas[i]['gt_labels'] = gt_labels[i]
            # img_metas[i]['gt_bboxes'] = HorizontalBoxes(gt_bboxes[i], in_mode='xyxy')
            img_metas[i]['gt_bboxes_3d'] = gt_bboxes_3d[i].to(gt_labels_3d[i].device)
            img_metas[i]['gt_labels_3d'] = gt_labels_3d[i]
        gt_bboxes_3d = [gt_bboxes_3d[i][gt_labels_3d[i]!=-1] for i in range(len(img_metas))] # filter out the ignored labels
        gt_labels_3d = [gt_labels_3d[i][gt_labels_3d[i]!=-1] for i in range(len(img_metas))] # filter out the ignored labels
        processed_info = self.preprocessing_information(img_metas, points[0].device)
        feature_dict = self.extract_feat(points, img=img, img_metas=img_metas, processed_info=processed_info)
        
        # preparation for subsequent losses calculation
        gt_bev_mask_binary = feature_dict['gt_bev_mask_binary']
        feature_dict_student = feature_dict
        processed_info_teacher = list(processed_info)
        feature_dict_teacher, tch_outs = None, None
        
        # compute for 3D object detection losses
        if self.distill_setts['use'] and self.distill_setts['semi']:
            feature_dict_teacher, tch_outs, losses_teacher = self.get_teacher_result(points, img_metas, img, processed_info_teacher, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore)
            tch_bbox_list = self.teacher.pts_bbox_head.get_bboxes(*tch_outs, img_metas, rescale=False)
            tch_bbox_list = [bbox3d2result(bboxes, scores, labels)for bboxes, scores, labels in tch_bbox_list]
            gt_bboxes_3d_semi = [tch_bbox['boxes_3d'].to(img.device) for tch_bbox in tch_bbox_list]
            gt_labels_3d_semi = [tch_bbox['labels_3d'].to(img.device) for tch_bbox in tch_bbox_list]
            gt_scores_3d_semi = [tch_bbox['scores_3d'].to(img.device) for tch_bbox in tch_bbox_list]
            # if teacher detect nothing, then use the ground truth as the semi-supervised ground truth
            gt_bboxes_3d_semi = [gt_bboxes_3d[i] if len(gt_labels_3d_semi[i])==0 else gt_bboxes_3d_semi[i] for i in range(len(img_metas))]
            gt_labels_3d_semi = [gt_labels_3d[i] if len(gt_labels_3d_semi[i])==0 else gt_labels_3d_semi[i] for i in range(len(img_metas))]
            gt_bboxes_3d_semi, gt_labels_3d_semi = filter_boxes_by_iou(gt_bboxes_3d_semi, gt_bboxes_3d, gt_labels_3d, threshold=0.10)
            losses, outs_pts = self.get_loss(feature_dict, gt_bboxes_3d_semi, gt_labels_3d_semi, img_metas, points, img, gt_bboxes_ignore)
        else: losses, outs_pts = self.get_loss(feature_dict, gt_bboxes_3d, gt_labels_3d, img_metas, points, img, gt_bboxes_ignore)
        feature_dict_student.update(dict(outs_pts=outs_pts))
    
        # compute for all distillaion losses
        if self.distill_setts['use']:
            # NOTE: need to know that, during the forward processing procedure, 
            # feature_dict_teacher---all framework, but pts_bbox_head is in get_loss()
            if feature_dict_teacher is None and tch_outs is None: # self.distill_setts['semi']=False
                feature_dict_teacher, tch_outs, losses_teacher = self.get_teacher_result(points, img_metas, img, processed_info_teacher, gt_bboxes_3d, gt_labels_3d, gt_bboxes_ignore)
            feature_dict_teacher.update(dict(outs_pts=tch_outs))
            self.teacher.draw_gt_pred_figures_3d(points, feature_dict['lidar_points'], img, gt_bboxes_3d, gt_labels_3d, img_metas, False, 0.3, outs_pts=tch_outs)
            losses.update(losses_teacher) # if teacher not freeze, dynamic distillation loss generation
            # knowledge distillation loss generation
            distill_mask = gt_bev_mask_binary.permute(0,1,3,2) # torch.ones_like(gt_bev_mask_binary, device=gt_bev_mask_binary.device) # gt_bev_mask_binary.permute(0,1,3,2)
            loss_distill_L2 = self.distillation_loss(feature_dict_teacher, feature_dict_student, distill_mask)
            losses.update(loss_distill_L2)
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
        lidar_points = points if lidar_points is None else lidar_points
        if gt_bboxes_3d is not None and self.box3d_supervision['use']:
            self.draw_gt_pred_figures_3d(points, lidar_points, img, gt_bboxes_3d, gt_labels_3d, img_metas, False, 0.3, outs_pts=outs_pts)

        return losses, outs_pts
    
    # preprocessing for data and others
    
    def preprocessing_information(self, batch_img_metas, device):
        if self.training:
            # all important informations
            B = len(batch_img_metas)
            if 'img_shape' in batch_img_metas[0].keys():
                H, W = batch_img_metas[0]['img_shape']
            else: (H, W) = (800, 1280) if self.dataset_type == 'VoD' else (640, 800)
            h_down, w_down = H // self.downsample, W // self.downsample
            # gt lidar instances_3d, list of InstancesData
            gt_bboxes_3d = [img_meta['gt_bboxes_3d'] for img_meta in batch_img_metas]
            gt_labels_3d = [img_meta['gt_labels_3d'] for img_meta in batch_img_metas]
            gt_bboxes_2d = [img_meta['gt_bboxes'] for img_meta in batch_img_metas] if 'gt_bboxes' in batch_img_metas[0] else []
            gt_labels_2d = [img_meta['gt_labels'] for img_meta in batch_img_metas] if 'gt_labels' in batch_img_metas[0] else []
            gt_bboxes_3d = [gt_bboxes_3d[i][gt_labels_3d[i]!=-1] for i in range(B)]
            gt_labels_3d = [gt_labels_3d[i][gt_labels_3d[i]!=-1] for i in range(B)]
            gt_bboxes_2d = [gt_bboxes_2d[i][gt_labels_2d[i]!=-1] for i in range(B)] if 'gt_bboxes' in batch_img_metas[0] else []
            gt_labels_2d = [gt_labels_2d[i][gt_labels_2d[i]!=-1] for i in range(B)] if 'gt_labels' in batch_img_metas[0] else []
            if self.props_supervision['use']: 
                gt_bev_mask_binary, gt_bev_mask_semant = self.generate_bev_mask(gt_bboxes_3d, gt_labels_3d, B, device, occ_threshold=0.3) # B H W
                gt_bev_mask_binary = gt_bev_mask_binary.to(device)
                gt_bev_mask_semant = gt_bev_mask_semant.to(device)
            else: gt_bev_mask_binary, gt_bev_mask_semant = None, None
            
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
            else: 
                final_lidar2img = None 
                Warning("No cam_aware in batch_img_metas, can not project points to img") 
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
                #segmentation = F.interpolate(segmentation, (h_down, w_down), mode='bilinear', align_corners=True)
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
            batch_img_metas = batch_img_metas[0]
            if 'img_shape' in batch_img_metas.keys():
                H, W = batch_img_metas['img_shape']
            else: (H, W) = (800, 1280) if self.dataset_type == 'VoD' else (640, 800)
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
            if self.props_supervision['use']: 
                gt_bev_mask_binary, gt_bev_mask_semant = self.generate_bev_mask(gt_bboxes_3d, gt_labels_3d, 0, device, occ_threshold=0.3) # B H W
                gt_bev_mask_binary = gt_bev_mask_binary.to(device)
                gt_bev_mask_semant = gt_bev_mask_semant.to(device)
            else: gt_bev_mask_binary, gt_bev_mask_semant = None, None
            
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
            else:
                final_lidar2img = None 
                Warning("No cam_aware in batch_img_metas, can not project points to img")
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
                #segmentation = F.interpolate(segmentation, (h_down, w_down), mode='bilinear', align_corners=True)
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
                gt_bboxes_3d[bsid].tensor[:,3:4] = gt_bboxes_3d[bsid].tensor[:,3:4] # *1.2
                gt_bboxes_3d[bsid].tensor[:,4:5] = gt_bboxes_3d[bsid].tensor[:,4:5] # *1.8
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
        # if not self.vis_time_box3d % 1 == 0: return
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
        for i in range(len(radar_points)): # batch size
            # preparation
            if imgs is not None: input_img = np.array(imgs[i].cpu()).transpose(1,2,0)
            if imgs is not None: input_img = input_img*self.std[None, None, :] + self.mean[None, None, :]
            pred_bboxes_3d = bbox_list[i]['boxes_3d'] if bbox_list is not None else None
            pred_scores_3d = bbox_list[i]['scores_3d'] if bbox_list is not None else None
            pred_bboxes_3d = pred_bboxes_3d[pred_scores_3d>threshold].to('cpu') if bbox_list is not None else None
            gt_bboxes_3d = gt_bboxes_3ds[i].to('cpu')
            if "final_lidar2img" in img_metas[i]:
                proj_mat = img_metas[i]["final_lidar2img"] # update lidar2img
            img_name = img_metas[i]['pts_filename'].split('/')[-1].split('.')[0]
            # project 3D bboxes to image and get show figures
            if pred_bboxes_3d is not None:
                if len(pred_bboxes_3d) == 0: pred_bboxes_3d = None
                
            # draw in image view
            filename = str(self.vis_time_box3d) + '_' + img_name + '_det3d'
            result_path = figures_path_det3d; mmcv.mkdir_or_exist(result_path)
            # if imgs is not None: show_multi_modality_result(img=input_img, gt_bboxes=gt_bboxes_3d, pred_bboxes=pred_bboxes_3d, proj_mat=proj_mat, out_dir=figures_path_det3d, filename=filename, box_mode='lidar', show=False)
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
            if imgs is not None: tmp_img_true = custom_draw_lidar_bbox3d_on_img(gt_bboxes_3d, input_img, proj_mat, img_metas, color=(61, 102, 255), thickness=3, scale_factor=3)
            if imgs is not None: tmp_img_pred = custom_draw_lidar_bbox3d_on_img(pred_bboxes_3d, input_img, proj_mat, img_metas, color=(241, 101, 72), thickness=3, scale_factor=3)
            if imgs is not None: tmp_img_alls = custom_draw_lidar_bbox3d_on_img(pred_bboxes_3d, tmp_img_true, proj_mat, img_metas, color=(241, 101, 72), thickness=3, scale_factor=3)
            if imgs is not None: mmcv.imwrite(tmp_img_true, os.path.join(result_path, f'{filename}_gt.png'))
            if imgs is not None: mmcv.imwrite(tmp_img_pred, os.path.join(result_path, f'{filename}_pred.png'))
            if imgs is not None: mmcv.imwrite(tmp_img_alls, os.path.join(result_path, f'{filename}.png'))
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
            img_name = img_metas[i]['pts_filename'].split('/')[-1].split('.')[0]
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
            img_name = img_metas[i]['pts_filename'].split('/')[-1].split('.')[0]
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
            img_name = img_metas[i]['pts_filename'].split('/')[-1].split('.')[0]
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
            img_name = img_metas[i]['pts_filename'].split('/')[-1].split('.')[0]
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
            img_name = img_metas[i]['pts_filename'].split('/')[-1].split('.')[0]
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
            img_name = img_metas[i]['pts_filename'].split('/')[-1].split('.')[0]
            fig, axes = plt.subplots(1, 2, figsize=(40, 20))
            
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
                    # intensities = np.clip(np.sqrt(x**2 + y**2) / 60, 0, 1)
                    # colors = plt.cm.gray(intensities)
                    colors = 'black'
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
                    logits = np.array(torch.softmax(torch.from_numpy(points[:, 3:]), dim=-1))
                    colors = np.zeros((logits.shape[0], 3))
                    ped_weight = np.array([1.0, 0.0, 0.0])  # red Pedestrian
                    cyc_weight = np.array([0.0, 1.0, 0.0]) # green Cyclist
                    car_weight = np.array([0.0, 0.0, 1.0])  # blue Car
                    tck_weight = np.array([0.0, 0.5, 1.0])  # cyan Truck
                    bck_weight = np.array([0.0, 0.0, 0.0])  # red Pedestrian
                    for q in range(self.num_classes + 1):
                        class_probability = logits[:, q]  # 获取当前类别的概率
                        if   q == 0: colors[:, :3] += class_probability[:, None] * ped_weight
                        elif q == 1: colors[:, :3] += class_probability[:, None] * cyc_weight
                        elif q == 2: colors[:, :3] += class_probability[:, None] * car_weight
                        elif q == 3 and self.num_classes == 4: colors[:, :3] += class_probability[:, None] * tck_weight
                        elif q == self.num_classes: colors[:, :3] += class_probability[:, None] * bck_weight
                    
                    # index = np.argmax(points[:, 3:], axis=-1)
                    # colors = np.ones((index.shape[0], 4))
                    # colors[index == 0, :3] = [1, 0, 0] # red Pedestrian
                    # colors[index == 1, :3] = [0, 1, 0] # green Cyclist
                    # colors[index == 2, :3] = [0, 0, 1] # blue Car
                    # if self.num_classes == 4:
                    #     colors[index == 3, :3] = [0, 0.5, 1] 
                    # colors[index == self.num_classes, :3] = [0, 0, 0]
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
                        
                #ax.set_xlabel('X (m)')
                #ax.set_ylabel('Y (m)')
                ax.set_title(f'Point cloud and bboxes under BEV - {title}')
                ax.set_xticks([])  # 取消 x 轴刻度
                ax.set_yticks([]) 
                ax.grid(False)
                
            
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
    
    @master_only 
    def find_index(self,annos, file_index):
        for i in range(len(annos)):
            image_idx = annos[i]['image']['image_idx']
            if image_idx == int(file_index): return i
    
    def apply_3d_effect(self,image, mask):
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
        
    @master_only
    def draw_ptsandmask_on_image(self, img_metas, points, img, segmentation, points_type='instance'):
        if not self.vis_time_point % self.SAVE_INTERVALS == 0: return
        if self.training: figures_path_point = self.figures_path_point_train
        else: figures_path_point = self.figures_path_point_test
        #设置画图间隔和路径
        img = img.squeeze(0)
        input_img = np.array(img.cpu()).transpose(1,2,0)
        input_img = input_img*self.std[None, None, :] + self.mean[None, None, :]
        min_val = input_img.min()
        max_val = input_img.max()
        uint8_array = (input_img - min_val) / (max_val - min_val) * 255
        uint8_array = uint8_array.clip(0, 255).astype(np.uint8)       
        img = uint8_array
        #img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        # img = img.permute(1,2,0)
        # input_img = img*self.std[None, None, :] + self.mean[None, None, :]
        # min_val = input_img.min()
        # max_val = input_img.max()
        # new_img = (input_img - min_val) / (max_val - min_val) * 255
        # new_img = new_img.clip(0,255)
        # new_img = np.array(new_img.cpu())
        # new_img = new_img.astype(np.uint8)  
        # new_img = cv2.cvtColor(new_img,cv2.COLOR_BGR2RGB)
        
        for i in range(len(img_metas)):
            img_name = img_metas[i]['pts_filename'].split('/')[-1].split('.')[0]
            # points = points[i][(self.xlim[0]<=points[i][:,0]) & (points[i][:,0]<=self.xlim[1]) & \
            #         (self.ylim[0]<=points[i][:,1]) & (points[i][:,1]<=self.ylim[1])]
            points = points[i][:,:3]
            points = points.cpu()
            #把当前样本取出来
            
            #读取投影矩阵
            if "final_lidar2img" in img_metas[i]:
                proj_mat = img_metas[i]["final_lidar2img"] # update lidar2img
                
            lidar2img_rt = copy.deepcopy(proj_mat).reshape(4, 4)
            if isinstance(lidar2img_rt, torch.Tensor):
                lidar2img_rt = lidar2img_rt.cpu().numpy()
            #lidar2img_rt = projection[i]
            
            #device = lidar2img_rt.device
            
            # P2 = annos_ind['calib']['P2']
            # rect = annos_ind['calib']['R0_rect']
            # Trv2c = annos_ind['calib']['Tr_velo_to_cam']
            # radar2img = P2 @ rect @ Trv2c
            
            image = img
            H, W, C = image.shape
            
            # load segmentation mask
            segmentation = segmentation.squeeze()
            segmentation = segmentation.cpu()
            segmentation = segmentation.numpy()*255.0
            seg_mask_2 = segmentation
            
            # draw bbox_Mask
            alpha, gray = 0.80, 0
            bgdark_mask_2 = np.full_like(image, (gray, gray, gray), dtype=np.uint8)
            mask_2 = (seg_mask_2 == 255)
            bgdark_mask_2[mask_2] = image[mask_2]
            mask_with_img_2 = cv2.addWeighted(image, 1 - alpha, bgdark_mask_2, alpha, 0)
            # Edge detection
            edges_2 = cv2.Canny(seg_mask_2.astype(np.uint8), 100, 200)
            # Connect and thicken edges
            kernel = np.ones((3, 3), np.uint8)  # You can adjust the kernel size to change the thickness
            edges_2 = cv2.dilate(edges_2, kernel, iterations=3)
            # Create red edge overlay
            mask_3d_image = self.apply_3d_effect(image, seg_mask_2.astype(np.bool_))
            mask_with_img_3 = copy.copy(mask_3d_image)
            mask_with_img_3[edges_2 > 0] = [0, 69, 255]  # Red color for edges in mask 2
            
            
            
            #project points
            #把点齐次化
            pc_in_image = np.ascontiguousarray(copy.copy(mask_with_img_3))
            
            pts_hom = np.concatenate((points, np.ones((points.shape[0], 1))), axis=1)  # Convert to homogeneous coordinates
            #pts_hom = torch.cat((points, torch.ones((points.shape[0], 1), device=device)), dim=1)
            
            img_pts = np.matmul(lidar2img_rt, pts_hom.T).T  # Apply transformation
            #执行投影，矩阵相乘
            #img_pts = pts_hom @ lidar2img_rt.T
            
            #img_pts = torch.matmul(lidar2img_rt, pts_hom.t()).t()
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
                #cv2.circle(pc_in_image, (x, y), 7, tuple(int(c) for c in color), -1)
                #绘制白色实心圆（填充）
                cv2.circle(pc_in_image, (x, y), 7, tuple(int(c) for c in color), -1)
                # 绘制空心圆（描边）
                cv2.circle(pc_in_image, (x, y), 7, (255,255,255), 2)
            
            if(points_type == 'instance'):
                cv2.imwrite(os.path.join(figures_path_point, str(self.vis_time_point) + '_' + img_name + '_' + 'ptsandmask' + '_' + 'instance' +'.png'),pc_in_image)
            elif(points_type == 'raw'):
                cv2.imwrite(os.path.join(figures_path_point, str(self.vis_time_point) + '_' + img_name + '_' + 'ptsandmask' + '_' + 'raw' +'.png'),pc_in_image)
            else :
                cv2.imwrite(os.path.join(figures_path_point, str(self.vis_time_point) + '_' + img_name + '_' + 'ptsandmask' + '_' + 'vote' +'.png'),pc_in_image)
                
            
            
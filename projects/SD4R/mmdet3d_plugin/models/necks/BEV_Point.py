import torch
from torch import nn
from torch.nn import functional as F
from mmdet3d.ops import (ball_query, grouping_operation)
from mmdet3d.models.builder import FUSION_LAYERS
import math, copy
from torchvision.utils import save_image

@FUSION_LAYERS.register_module()
class BEV_Point(nn.Module):
    """ preFusion for center features & radar pillar features(before scatter to bev)
    
    """
    def __init__(self,
                 pts_channel=64,
                 bev_channel=384,
                 radius=0.2,
                 voxel_size=[0.16, 0.16, 5],
                 point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2],
                 sample_points=4):
        super(BEV_Point, self).__init__()
        self.voxel_size = voxel_size
        self.pc_range = point_cloud_range
        self.radius = radius
        self.sample_points = sample_points
        self.pts_channel = pts_channel
        self.bev_channel = bev_channel
        self.linear_project = nn.Sequential(
            nn.Linear(pts_channel, bev_channel // 2), 
            nn.LayerNorm(bev_channel // 2), 
            nn.ReLU(inplace=True),
            nn.Linear(bev_channel // 2, bev_channel))
        self.linear_mapping = nn.Linear(2*bev_channel, bev_channel)
        # self.conv_out = nn.Conv2d(2*bev_channel, bev_channel, 1, 1, 0)
        
    def forward(self, BEV_feats, vote_sampled_out, segmt_dicts, bev_semant_logit):

        B, C, H, W = BEV_feats.shape
        if bev_semant_logit is not None:
            bev_semant_logit = bev_semant_logit[:,:-1,:,:]
            num_class = bev_semant_logit.shape[1]
            BEV_logit = bev_semant_logit.view(B, num_class, -1) # BclassN
        BEV_feats, BEV_coors = self.get_BEV_center(BEV_feats)  # BCN B2N
        all_vote_centers = vote_sampled_out['all_vote_centers']
        all_vote_feature = vote_sampled_out['all_vote_feature']
        
        # NOTE: filter background points for easier feature aggregation
        filtered_point, filtered_feats = self.filter_background(segmt_dicts, theshold=0.2)
        #根据之前的BEV——logits过滤背景点
        
        all_vote_centers = [torch.cat([all_vote_centers[i], filtered_point[i]], dim=0) for i in range(B)]
        all_vote_centers_xyz = [center.clone() for center in all_vote_centers]
        all_vote_feature = [torch.cat([all_vote_feature[i], filtered_feats[i]], dim=0) for i in range(B)]
        # all_vote_posembd = self.positional_encoding_2d(all_vote_centers, self.pts_channel)
        # all_vote_feature = [torch.cat([all_vote_feature[i], all_vote_posembd[i]], dim=-1) for i in range(B)]
        # foreground_centers = [torch.cat([all_vote_centers[i], filtered_point[i]], dim=0) for i in range(B)]
        # foreground_feature = [torch.cat([all_vote_feature[i], filtered_feats[i]], dim=0) for i in range(B)]
        # foreground_pointss = [torch.cat([foreground_centers[i], foreground_feature[i]], dim=-1) for i in range(B)]
        # foreground_pointss = [foreground_pointss[i].detach() for i in range(B)] # no grad()
        # foreground_bevfeat = self.extract_foreground_feats(foreground_pointss)
        
        # preparation for channel expand
        all_vote_feature = [self.linear_project(all_vote_feature[i]) for i in range(B)]
        #对齐通道数
        
            
        for i in range(B):

            # preparation
            this_bevs_feats = BEV_feats[i:i+1,:,:].permute(0,2,1).contiguous() # [1, M, C]
            this_bevs_point = BEV_coors[i:i+1,:,:].permute(0,2,1).contiguous() # [1, M, 3]
            this_vote_feats = all_vote_feature[i].view(1, all_vote_feature[i].shape[0], -1).contiguous() # [1, N, C]
            this_vote_point = all_vote_centers[i].view(1, all_vote_centers[i].shape[0], -1).contiguous() # [1, N, 3]
            this_bevs_point[:, :, 2].fill_(0)     # (x, y, z) --> (x, y, 0)
            this_vote_point[:, :, 2].fill_(0)     # (x, y, z) --> (x, y, 0)
            #将点云投到2D平面

            # query for correspoding center pillar id
            if bev_semant_logit is not None:
                this_bevs_logit = BEV_logit[i:i+1,:,:].permute(0,2,1).contiguous() # [1, M, class]
                _, pred_classes = this_bevs_logit.max(dim=-1)  # [1, M]
                M = this_bevs_point.shape[1]  # BEV 点的数量 = H*W
                seg_point_idx = torch.full((1, M, self.sample_points), -1, device='cuda', dtype=torch.int32)
                
                #针对不同的类别用不同的半径
                for cls in range(len(self.radius)):  # 遍历所有类别
                    mask = (pred_classes == cls)  # 找到当前类别的点
                    if mask.any(): seg_point_idx[:, mask.squeeze()] = ball_query(0, self.radius[cls], self.sample_points, this_vote_point, this_bevs_point[:, mask.squeeze(), :]) 
            else: seg_point_idx = ball_query(0, self.radius[1], self.sample_points, this_vote_point, this_bevs_point)  # [1, M, self.sample_points(indice in N)]
            flag = seg_point_idx.sum(dim=2).permute(1,0); flag[flag > 0] = 1

            # group features
            input_seg_feats = this_vote_feats.permute(0,2,1).contiguous() # B C N
            grouped_feature = grouping_operation(input_seg_feats, seg_point_idx).squeeze(0).permute(1,0,2) # [M, C, nsample]
            grouped_feature = F.max_pool2d(grouped_feature, kernel_size=[1, grouped_feature.size(2)]) # [M, C, 1]
            grouped_feature = grouped_feature.squeeze(-1).contiguous() # [npoint, C]
            grouped_feature = grouped_feature*flag

            # upadate the BEV_feats
            pts_feats = grouped_feature[(flag > 0).squeeze()]
            bev_feats = this_bevs_feats[0][(flag > 0).squeeze()]
            tmp_feats = self.linear_mapping(torch.cat([pts_feats, bev_feats], dim=-1))
            grouped_feature[(flag > 0).squeeze()] = tmp_feats
            BEV_feats[i,:,:] = BEV_feats[i,:,:] + grouped_feature.permute(1,0).contiguous()
        
        output = BEV_feats.view(B, C, H, W)
        # output = torch.cat([foreground_bevfeat, output], dim=1)
        # output = self.conv_out(output) # conv_out
        return output, all_vote_centers, all_vote_centers_xyz

    def extract_foreground_feats(self, pts):
        batch_size = len(pts)
        
        voxels, coors, num_points = [], [], []
        for res in pts:
            res_voxels, res_coors, res_num_points = self.vote_pts_voxel_layer(res)
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
        voxel_features = self.vote_pts_voxel_encoder(voxels, num_points, coors_batch,)
        x = self.vote_pts_middle_encoder(voxel_features, coors_batch, batch_size)
        x = self.vote_pts_backbone(x)
        x = self.vote_pts_neck(x)[0]
        return x

    def positional_encoding_2d(self, all_vote_centers, d_model):
        """
        2D坐标的余弦编码 返回 [B, M, d_model] 形式的编码结果
        coords: 输入的坐标 [B, M, 3]，每个坐标是 (x, y, z)
        d_model: 输出的特征维度
        """
        pos_encoding_list = []
        for vote_center in all_vote_centers:
            M, _ = vote_center.shape
            position = vote_center.view(-1, 3)
            
            div_term = torch.exp(torch.arange(0, d_model, 4).float() * -(math.log(10000.0) / d_model)).to(vote_center.device)
            pos_encoding = torch.zeros(M, d_model, device=vote_center.device)
            
            pos_encoding[:, 0::4] = torch.sin(position[:, 0:1] * div_term)  # x
            pos_encoding[:, 1::4] = torch.cos(position[:, 0:1] * div_term)  # y
            pos_encoding[:, 2::4] = torch.sin(position[:, 1:2] * div_term)  # y
            pos_encoding[:, 3::4] = torch.cos(position[:, 1:2] * div_term)  # x

            pos_encoding = pos_encoding.view(M, d_model)  # 变回 [M, d_model]
            pos_encoding_list.append(pos_encoding)
        return pos_encoding_list


    def get_BEV_center(self, BEV_feats):
        B, C, H, W = BEV_feats.shape
        pc_range = torch.tensor(self.pc_range[0:2], device=BEV_feats.device).float()  # [x_min, x_max]
        voxel_size = torch.tensor(self.voxel_size[0:2], device=BEV_feats.device).float()  # [voxel_width, voxel_length]
        grid_y, grid_x = torch.meshgrid(torch.arange(0, W, device=BEV_feats.device), torch.arange(0, H, device=BEV_feats.device))
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0)  # (2, H * W)
        actual_coords = (coords + 0.5) * voxel_size[:, None] + pc_range[:, None]  # (2, H * W)
        BEV_feats_reshaped = BEV_feats.view(B, C, -1)  # (B, C, H * W)
        feats = BEV_feats_reshaped # B C N
        coords_out = actual_coords.unsqueeze(0).expand(B, -1, -1) # B 2 N
        zeros = torch.zeros(B, 1, actual_coords.shape[1], device=BEV_feats.device)
        coords_out = torch.cat([coords_out, zeros], dim=1)
        
        return feats, coords_out

    def filter_background(self, dict_to_filter, theshold=0.1):
        batch_idx = dict_to_filter['batch_idx']
        raw_point = dict_to_filter['seg_point'][:, :3]
        seg_logit = dict_to_filter['seg_logits']
        seg_feats = dict_to_filter['seg_feats']
        seg_logit = torch.softmax(seg_logit, dim=-1)
        foreground_prob = 1 - seg_logit[:, -1]
        B = batch_idx.max() + 1
        filtered_point = []
        filtered_feats = []
        for i in range(B):
            this_foreground_prob = foreground_prob[batch_idx==i]
            this_raw_point = raw_point[batch_idx==i]
            this_seg_feats = seg_feats[batch_idx==i]
            filtered_point.append(this_raw_point[this_foreground_prob>theshold])
            filtered_feats.append(this_seg_feats[this_foreground_prob>theshold])
        return filtered_point, filtered_feats

@FUSION_LAYERS.register_module()
class BEV_Point_cross(nn.Module):
    """ preFusion for center features & radar pillar features(before scatter to bev)
    
    """
    def __init__(self,
                 pts_channel=64,
                 bev_channel=384,
                 radius=0.2,
                 voxel_size=[0.16, 0.16, 5],
                 point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2],
                 sample_points=4):
        super(BEV_Point_cross, self).__init__()
        self.voxel_size = voxel_size
        self.pc_range = point_cloud_range
        self.radius = radius
        self.sample_points = sample_points
        self.pts_channel = pts_channel
        self.bev_channel = bev_channel
        self.linear_project = nn.Sequential(
            nn.Linear(pts_channel, bev_channel // 2), 
            nn.LayerNorm(bev_channel // 2), 
            nn.ReLU(inplace=True),
            nn.Linear(bev_channel // 2, bev_channel))
        self.MHSA = nn.MultiheadAttention(embed_dim=self.bev_channel, num_heads=8)
        self.linear_mapping = nn.Linear(self.bev_channel*2, self.bev_channel)
        
    def forward(self, BEV_feats, vote_sampled_out):

        B, C, H, W = BEV_feats.shape
        BEV_feats, BEV_coors = self.get_BEV_center(BEV_feats)  # BCN B2N
        all_vote_centers = vote_sampled_out['all_vote_centers']
        all_vote_feature = vote_sampled_out['all_vote_feature']
        # preparation for channel expand
        all_vote_feature = [self.linear_project(all_vote_feature[i]) for i in range(B)]
            
        for i in range(B):

            this_bevs_feats = BEV_feats[i:i+1,:,:].permute(0,2,1).contiguous() # [1, M, C]
            this_bevs_point = BEV_coors[i:i+1,:,:].permute(0,2,1).contiguous() # [1, M, 3]
            this_vote_feats = all_vote_feature[i].clone().view(1, all_vote_feature[i].shape[0], -1).contiguous() # [1, N, C]
            this_vote_point = all_vote_centers[i].clone().view(1, all_vote_centers[i].shape[0], -1).contiguous() # [1, N, 3]

            # query 数据准备
            this_bevs_point[:, :, 2].fill_(0)     # (x, y, z) --> (x, y, 0)
            this_vote_point[:, :, 2].fill_(0)     # (x, y, z) --> (x, y, 0)
            this_bevs_point[:, :, 0] = (this_bevs_point[:, :, 0] - self.pc_range[0])/(self.pc_range[3] - self.pc_range[0])
            this_bevs_point[:, :, 0] = (this_bevs_point[:, :, 0] - self.pc_range[1])/(self.pc_range[4] - self.pc_range[1])
            this_vote_point[:, :, 0] = (this_vote_point[:, :, 0] - self.pc_range[0])/(self.pc_range[3] - self.pc_range[0])
            this_vote_point[:, :, 0] = (this_vote_point[:, :, 0] - self.pc_range[1])/(self.pc_range[4] - self.pc_range[1]) 
            embd_bevs_point = self.positional_encoding_2d(this_bevs_point, self.bev_channel)
            embd_vote_point = self.positional_encoding_2d(this_vote_point, self.bev_channel)
            
            this_bevs_feats = torch.cat([this_bevs_feats, embd_bevs_point], dim=-1).permute(1,0,2).contiguous() # [M, 1, C+C]
            this_vote_feats = torch.cat([this_vote_feats, embd_vote_point], dim=-1).permute(1,0,2).contiguous() # [N, 1, C+C]
            this_bevs_feats = self.linear_mapping(this_bevs_feats.squeeze(1)).unsqueeze(1)
            this_vote_feats = self.linear_mapping(this_vote_feats.squeeze(1)).unsqueeze(1)
            
            this_bevs_feats, _ = self.MHSA(this_bevs_feats, this_vote_feats, this_vote_feats)
            this_bevs_feats = this_bevs_feats.permute(1,0,2).contiguous() # [1, M, C]
            
            BEV_feats[i,:,:] = BEV_feats[i,:,:] + this_bevs_feats.permute(0,2,1)

        return BEV_feats.view(B, C, H, W).contiguous()

    def get_BEV_center(self, BEV_feats):
        B, C, H, W = BEV_feats.shape
        pc_range = torch.tensor(self.pc_range[0:2], device=BEV_feats.device).float()  # [x_min, x_max]
        voxel_size = torch.tensor(self.voxel_size[0:2], device=BEV_feats.device).float()  # [voxel_width, voxel_length]
        grid_y, grid_x = torch.meshgrid(torch.arange(0, W, device=BEV_feats.device), torch.arange(0, H, device=BEV_feats.device))
        coords = torch.stack([grid_x.flatten(), grid_y.flatten()], dim=0)  # (2, H * W)
        actual_coords = (coords + 0.5) * voxel_size[:, None] + pc_range[:, None]  # (2, H * W)
        BEV_feats_reshaped = BEV_feats.view(B, C, -1)  # (B, C, H * W)
        feats = BEV_feats_reshaped # B C N
        coords_out = actual_coords.unsqueeze(0).expand(B, -1, -1) # B 2 N
        zeros = torch.zeros(B, 1, actual_coords.shape[1], device=BEV_feats.device)
        coords_out = torch.cat([coords_out, zeros], dim=1)
        
        return feats, coords_out


    def positional_encoding_2d(self, coords, d_model):
        """
        2D坐标的余弦编码 返回 [B, M, d_model] 形式的编码结果
        coords: 输入的坐标 [B, M, 3]，每个坐标是 (x, y, z)
        d_model: 输出的特征维度
        """
        B, M, _ = coords.shape
        position = coords.view(-1, 3)  # 变成 [B * M, 3]
        
        # 生成位置编码：每个维度的位置编码维度为 d_model
        # 使用 sin 和 cos 对每个维度进行编码
        div_term = torch.exp(torch.arange(0, d_model, 4).float() * -(math.log(10000.0) / d_model)).to(coords.device)
        pos_encoding = torch.zeros(M * B, d_model, device=coords.device)
        
        pos_encoding[:, 0::4] = torch.sin(position[:, 0:1] * div_term)  # x
        pos_encoding[:, 1::4] = torch.cos(position[:, 0:1] * div_term)  # y
        pos_encoding[:, 2::4] = torch.sin(position[:, 1:2] * div_term)  # y
        pos_encoding[:, 3::4] = torch.cos(position[:, 1:2] * div_term)  # x

        pos_encoding = pos_encoding.view(B, M, d_model)  # 变回 [B, M, d_model]
        return pos_encoding




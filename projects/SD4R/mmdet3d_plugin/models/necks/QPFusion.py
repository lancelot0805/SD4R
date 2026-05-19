import torch
from torch import nn
from torch.nn import functional as F
from mmdet3d.ops import (ball_query, grouping_operation)
from mmdet3d.models.builder import FUSION_LAYERS
import copy

@FUSION_LAYERS.register_module()
class QueryFusion(nn.Module):
    """ preFusion for center features & radar pillar features(before scatter to bev)
    
    """
    def __init__(self,
                 radius=0.2,
                 voxel_size=[0.16, 0.16, 5],
                 point_cloud_range = [0, -25.6, -3, 51.2, 25.6, 2],
                 sample_points=4,
                 use_vote=False,
                 channels=64,
                 num_classes=4):
        super(QueryFusion, self).__init__()
        self.voxel_size = voxel_size
        self.pc_range = point_cloud_range
        self.radius = radius
        self.sample_points = sample_points
        self.use_vote = use_vote
        self.channels = channels
        self.num_classes = num_classes
        if self.use_vote == True:
            self.MHSA_1 = nn.MultiheadAttention(embed_dim=self.channels, num_heads=8)
            self.MHSA_2 = nn.MultiheadAttention(embed_dim=self.channels, num_heads=8)

    def forward(self, pillar_feature, coors, seg_out_dict,voxels):
        """
        Args:
            pillar_feature (torch.tensor): [M, C], (feature_channels)
            coors (troch.tensor): [M, 4], (batch_id, z, y, x) pillar's coordinates before scatter to bev
            pred_center (torch.tensor): # [N, 4], (batch_idx, cls_id, x, y)
            center_feats (torch.tensor): [N, C] (feature_channel)
        
        Return:
            pillar_feature (torch.tensor): [M, C + self.add_channels]. pillar feature after enhanced.
        """
        
        pillar_xy_all, B = self.get_pillar_center(coors)       # [M', 3],  (batch_id, x, y)
        #计算pillar中心坐标
        pillar_batch_idx = pillar_xy_all[:, 0]

        for i in range(B):
            batch_idx = seg_out_dict['batch_idx']
            seg_feats = seg_out_dict['seg_feats'][batch_idx==i]
            seg_point = seg_out_dict['seg_point'][batch_idx==i][:,:3]  #仅取前三维xyz
            seg_logit = seg_out_dict['seg_logits'][batch_idx==i]
            seg_logit = torch.softmax(seg_logit, dim=-1)
            foreground_prob = (1 - seg_logit[:, -1]).view(1, -1, 1)
            #[1,N,1]
            N, C = seg_feats.shape
            seg_feats = seg_feats.view(1, N, C).contiguous()     # [1, N, C]
            seg_point = seg_point.view(1, N, 3).contiguous()     # [1, N, 3]
            seg_feats = seg_feats*foreground_prob # NOTE: modified here, re-weight foreground points for QPF
            #用前景概率加权点的特征，突出前景点
            
            probs = voxels[:, :, -(self.num_classes+1):][pillar_batch_idx==i]  # [N, 25, 4]
            # 找到最大概率的类别索引
            class_indices = probs.argmax(dim=-1)  # [N, 25]
            # 找出所有概率为 0 的点
            all_zero_mask = (probs == 0).all(dim=-1)  # [N, 25]
            # 将概率全为 0 的点标为类别3/4
            class_indices[all_zero_mask] = self.num_classes
            valid_mask = (class_indices != self.num_classes) #标记前景点索引
            masked_indices = class_indices.clone()
            masked_indices[~valid_mask] = -1  # 类别为3的都设为-1，避免进入统计
            if (self.num_classes == 3):
                category_counts = torch.stack([    #统计每一个piilar中0,1,2类别点的数量
                    (masked_indices == 0).sum(dim=1),
                    (masked_indices == 1).sum(dim=1),
                    (masked_indices == 2).sum(dim=1),
                ], dim=1)  # [N, 3]
            else:
                category_counts = torch.stack([    #统计每一个piilar中0,1,2类别点的数量
                    (masked_indices == 0).sum(dim=1),
                    (masked_indices == 1).sum(dim=1),
                    (masked_indices == 2).sum(dim=1),
                    (masked_indices == 3).sum(dim=1),
                ], dim=1)  # [N, 3]
            majority_class = category_counts.argmax(dim=1)  # [N] 找出每行中计数最多的类别
            all_invalid_mask = ~valid_mask.any(dim=1)  # [N]
            majority_class[all_invalid_mask] = self.num_classes  #如果整行都是类别3（即 valid_mask 全为 False），则将 majority_class 设置为3
            pillar_class = majority_class #[N]
            pillar_class = pillar_class.unsqueeze(0)
            
            if self.use_vote == True:
                all_vote_centers = seg_out_dict['vote_sampled_out']['all_vote_centers'][i].view(1, -1, 3)
                all_vote_feature = seg_out_dict['vote_sampled_out']['all_vote_feature'][i].view(1, -1, C)
                all_vote_logitss = seg_out_dict['vote_sampled_out']['all_vote_logitss'][i]
                all_vote_foreground_prob = (1 - all_vote_logitss[:, -1]).view(1, -1, 1)
                all_vote_feature = all_vote_feature*all_vote_foreground_prob # B N C
                all_vote_feature = all_vote_feature.permute(1, 0, 2) # N B C
                all_vote_feature, _ = self.MHSA_1(all_vote_feature, all_vote_feature, all_vote_feature)
                all_vote_feature, _ = self.MHSA_2(all_vote_feature, all_vote_feature, all_vote_feature)
                all_vote_feature = all_vote_feature.permute(1, 0, 2) # B N C
                seg_feats = torch.cat([seg_feats, all_vote_feature], dim=1)
                seg_point = torch.cat([seg_point, all_vote_centers], dim=1)
                foreground_prob = torch.cat([foreground_prob, all_vote_foreground_prob], dim=1)
                #同上，只是加上投票点
            # query 数据准备
            pillar_xy = pillar_xy_all[pillar_xy_all[:, 0]==i][:, [1, 2, 0]] # [M, 3], (x, y, batch_id)
            M, _ = pillar_xy.shape
            pillar_xy = pillar_xy.view(1, M, 3).contiguous()
            pillar_xy[:, :, 2].fill_(0)     # [1, M, 3], (x, y, batch_id) --> (x, y, 0)
            seg_point[:, :, 2].fill_(0)     # [1, N, 3], (x, y,        z) --> (x, y, 0)
            #将点云的z轴忽略，投影到柱子2D平面
            
            # query 得到每个中心点对应的 pillar id
            #seg_point_idx = ball_query(0, self.radius, self.sample_points, seg_point, pillar_xy)  # [1, M, 4(在 N 中的indice)]
            #输出 idx: 形状 [1, 180, 8]，
            seg_point_idx = torch.full((1, M, self.sample_points), -1, device='cuda', dtype=torch.int32)
            
            for cls in range(len(self.radius)): 
                mask = (pillar_class == cls)
                if mask.any(): seg_point_idx[:, mask.squeeze()] = ball_query(0, self.radius[cls], self.sample_points, seg_point, pillar_xy[:, mask.squeeze(), :])
            
            flag = seg_point_idx.sum(dim=2).permute(1,0)         
            flag[flag > 0] = 1
            #有效柱子掩码，判断柱子周围有没有可查询的点

            input_seg_feats = seg_feats.permute(0,2,1).contiguous() # B C N
            input_seg_logit = foreground_prob.permute(0,2,1).contiguous() # B 1 N
            grouped_feature = grouping_operation(input_seg_feats, seg_point_idx).squeeze(0).permute(1,0,2) # [M, C, nsample]
            #每个柱子的邻近点特征
            grouped_forelgt = grouping_operation(input_seg_logit, seg_point_idx).squeeze(0).permute(1,0,2) # [M, C, nsample]
            #每个柱子的邻近点前景概率
            
            grouped_feature = F.max_pool2d(grouped_feature, kernel_size=[1, grouped_feature.size(2)]) # [M, C, 1]
            grouped_forelgt = F.max_pool2d(grouped_forelgt, kernel_size=[1, grouped_forelgt.size(2)]) # [M, 1, 1]
            #对周围的点采用最大池化
            
            grouped_feature = grouped_feature.squeeze(-1).contiguous() # [npoint, C]
            grouped_forelgt = grouped_forelgt.squeeze(-1).contiguous() # [npoint, 1]
            grouped_feature = grouped_feature*flag
            grouped_forelgt = grouped_forelgt*flag
            #用掩码过滤

            # tmp_features = (pillar_feature[pillar_xy_all[:, 0]==i] + grouped_feature) * grouped_forelgt
            tmp_features = (pillar_feature[pillar_xy_all[:, 0]==i] + grouped_feature) # * grouped_forelgt
            #将周围点特征直接加到pillar feature上
            pillar_feature[pillar_xy_all[:, 0]==i] = tmp_features

        return pillar_feature

    def get_pillar_center(self, coors):
        """
        Args:
            coors (troch.tensor): [M, 4], (batch_id, z, y, x) pillar's coordinates before scatter to bev
        
        Return:
            pillar_center (torch.tensor): [M, 3],  (batch_id, x, y)
        """
        pillar_centers = torch.zeros(coors.shape[0], 3, dtype=coors.dtype, device=coors.device)
        pillar_centers = coors[:, [0, 3, 2]].float()  # (batch_id, x, y)

        pc_range = torch.tensor(self.pc_range[0:2], device=pillar_centers.device).float()
        voxel_size = torch.tensor(self.voxel_size[0:2], device=pillar_centers.device).float()
        pillar_centers[:, 1:3] = (pillar_centers[:, 1:3] + 0.5) * voxel_size + pc_range
        
        batch_size = pillar_centers[-1, 0].int().item() + 1

        # 计算每个 batch 中有的 pillar 数
        pillar_batch_cnt = pillar_centers.new_zeros(batch_size)
        for bs_idx in range(batch_size):
            pillar_batch_cnt[bs_idx] = (coors[:, 0] == bs_idx).sum()

        return pillar_centers, batch_size



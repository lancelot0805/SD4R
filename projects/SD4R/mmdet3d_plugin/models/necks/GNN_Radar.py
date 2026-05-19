import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable, Optional, Union
from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.typing import Adj, OptTensor, PairOptTensor, PairTensor
from torch_geometric.nn import GCNConv
from torchvision import models
try: from torch_cluster import knn
except ImportError: knn = None
from torch_geometric.data import Data, Batch
from torch_cluster import knn_graph
from mmdet3d.models.builder import FUSION_LAYERS
from ...utils.depth_tools import generate_guassian_depth_target
from ...models.img2bev.forward_projection.GeometryDepth_Net import DepthVolumeEncoder, DepthAggregation
import numpy as np
import copy
from mmcv.utils import build_from_cfg
from mmcv.cnn import MODELS as NN_REGISTRY

if 'Linear' not in NN_REGISTRY._module_dict:
    @NN_REGISTRY.register_module()
    class Linear(nn.Linear):
        pass
if 'ReLU' not in NN_REGISTRY._module_dict:
    @NN_REGISTRY.register_module('ReLU')
    class ReLU(nn.ReLU):
        pass   
class custom_resnet34(nn.Module):
    def __init__(self, downsample=8):
        super(custom_resnet34, self).__init__()
        resnet34 = models.resnet34(pretrained=True)
        self.resnet34 = resnet34
        self.downsample = downsample

        self.layer0 = nn.Conv2d(1, 3, kernel_size=3, stride=1, padding=1)
        self.layer1 = nn.Sequential(*list(resnet34.children())[:3])  # 前3层是conv1和layer1
        self.layer2 = resnet34.layer1
        self.layer3 = resnet34.layer2
        self.layer4 = resnet34.layer3
        self.layer5 = resnet34.layer4
        self.fusion_neck = nn.Sequential( # 64 64 128 256 512
            nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256 , 256, kernel_size=3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(256 , 256, kernel_size=3, stride=1, padding=1))

    def forward(self, depth_images):
        feat0 = self.layer0(depth_images)
        feat1 = self.layer1(feat0)
        feat2 = self.layer2(feat1)
        feat3 = self.layer3(feat2)
        feat4 = self.layer4(feat3)
        feat5 = self.layer5(feat4)
        rad_feats_list = [feat1, feat2, feat3, feat4, feat5]
        B, C, H, W = depth_images.shape
        h, w = H // self.downsample, W // self.downsample
        rad_align_feats = [F.interpolate(feat, (h, w), mode='bilinear', align_corners=True) for feat in rad_feats_list]
        rad_align_feats = torch.cat(rad_align_feats, dim=1)
        rad_align_feats = self.fusion_neck(rad_align_feats)
        
        return rad_feats_list, rad_align_feats

class MultiHeadCrossAttention(nn.Module):
    def __init__(self, feature_dim1, feature_dim2, hidden_dim, num_heads):
        super(MultiHeadCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim

        assert hidden_dim % num_heads == 0, 'Hidden dimension must be divisible by the number of heads'
        self.head_dim = hidden_dim // num_heads

        self.query_transform = nn.Linear(feature_dim1, hidden_dim)
        self.key_transform = nn.Linear(feature_dim2, hidden_dim)
        self.value_transform = nn.Linear(feature_dim2, hidden_dim)

        # output linear layer
        self.out_transform = nn.Linear(hidden_dim, hidden_dim)
        
        # scaling factor
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim]))
    
    def forward(self, tensor1, tensor2):
        device = tensor1.device
        scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)
        # scale = torch.sqrt(torch.tensor([self.head_dim], dtype=torch.float32)).to(device)
        batch_size = tensor1.shape[0]

        # transform queries, keys, values
        queries = self.query_transform(tensor1).view(batch_size, self.num_heads, self.head_dim)
        keys = self.key_transform(tensor2).view(batch_size, self.num_heads, self.head_dim)
        values = self.value_transform(tensor2).view(batch_size, self.num_heads, self.head_dim)

        # compute scaled dot product attention for each head
        # attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / self.scale
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / scale

        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_output = torch.matmul(attention_weights, values)

        # concatenate the output of all heads
        concatenated = attention_output.view(batch_size, self.hidden_dim)

        # final linear transformation
        output = self.out_transform(concatenated)

        return output
    
class EdgeConv(MessagePassing):
    r"""The edge convolutional operator from the `"Dynamic Graph CNN for
    Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper

    .. math::
        \mathbf{x}^{\prime}_i = \sum_{j \in \mathcal{N}(i)}
        h_{\mathbf{\Theta}}(\mathbf{x}_i \, \Vert \,
        \mathbf{x}_j - \mathbf{x}_i),

    where :math:`h_{\mathbf{\Theta}}` denotes a neural network, *.i.e.* a MLP.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            :obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, points_outs_channels]`,
            *e.g.*, defined by :class:`torch.nn.Sequential`.
        aggr (str, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V}|, F_{in}), (|\mathcal{V}|, F_{in}))`
          if bipartite,
          edge indices :math:`(2, |\mathcal{E}|)`
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn: Callable, aggr: str = 'max', **kwargs):
        super().__init__(aggr=aggr, **kwargs)
        self.nn = nn
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()
        reset(self.nn)

    def forward(self, x: Union[Tensor, PairTensor], edge_index: Adj) -> Tensor:
        if isinstance(x, Tensor):
            x: PairTensor = (x, x)
        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)

    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn})'

@FUSION_LAYERS.register_module()
class DynamicEdgeConv(MessagePassing):
    r"""The dynamic edge convolutional operator from the `"Dynamic Graph CNN
    for Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper
    (see :class:`torch_geometric.nn.conv.EdgeConv`), where the graph is
    dynamically constructed using nearest neighbors in the feature space.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            `:obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, points_outs_channels]`,
            *e.g.* defined by :class:`torch.nn.Sequential`.
        k (int): Number of nearest neighbors.
        aggr (str, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
        num_workers (int): Number of workers to use for k-NN computation.
            Has no effect in case :obj:`batch` is not :obj:`None`, or the input
            lies on the GPU. (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V}|, F_{in}), (|\mathcal{V}|, F_{in}))`
          if bipartite,
          batch vector :math:`(|\mathcal{V}|)` or
          :math:`((|\mathcal{V}|), (|\mathcal{V}|))`
          if bipartite *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn_config: Callable, k: int, aggr: str = 'max',
                 num_workers: int = 1, **kwargs):
        super().__init__(aggr=aggr, flow='source_to_target', **kwargs)
        """ layers = [nn.Linear(**layer) for layer in mynn['layers']]
        self.nn = nn.Sequential(*layers) """
        
        layers = [build_from_cfg(layer, NN_REGISTRY) for layer in nn_config['layers']]
        self.nn = nn.Sequential(*layers)
        
        if knn is None:
            raise ImportError('`DynamicEdgeConv` requires `torch-cluster`.')

        #self.nn = nn
        self.k = k
        self.num_workers = num_workers
        self.reset_parameters()

    def reset_parameters(self):
        reset(self.nn)

    def forward(
            self, x: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None) -> Tensor:
        # type: (Tensor, OptTensor) -> Tensor  # noqa
        # type: (PairTensor, Optional[PairTensor]) -> Tensor  # noqa

        if isinstance(x, Tensor):  # True
            x: PairTensor = (x, x)

        if x[0].dim() != 2:
            raise ValueError("Static graphs not supported in DynamicEdgeConv")

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):  # True
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        # 确保输入张量是连续的
        x0_contiguous = x[0].contiguous()
        x1_contiguous = x[1].contiguous()

        # 使用连续张量计算 k-NN
        edge_index = knn(x0_contiguous, x1_contiguous, self.k, b[0], b[1]).flip([0])
        print(f"Input x0 mean={x0_contiguous.mean()}, std={x0_contiguous.std()}")
        out = self.propagate(edge_index, x=x, size=None)
        print(f"Output mean={out.mean()}, std={out.std()}")
        return out
        '''edge_index = knn(x[0], x[1], self.k, b[0], b[1]).flip([0])

        # propagate_type: (x: PairTensor)
        return self.propagate(edge_index, x=x, size=None)
        '''
    def message(self, x_i: Tensor, x_j: Tensor) -> Tensor:
        return self.nn(torch.cat([x_i, x_j - x_i], dim=-1))

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(nn={self.nn}, k={self.k})'

@FUSION_LAYERS.register_module()
class Attention_enhanced_DGCNN(nn.Module):
    def __init__(self, 
                 num_classes=3,
                 data_config=None,
                 grid_config=None,
                 radar_supervision=None,
                 point_supervision=None,
                 downsample=8, 
                 points_outs_channels=128,
                 fusion_base_channels=128,
                 depth_feats_channels=256, 
                 knn_radar=6, 
                 aggr='max'):
        super(Attention_enhanced_DGCNN, self).__init__()
        
        # config settings
        self.radar_supervision = radar_supervision
        self.point_supervision = point_supervision
        self.num_classes = num_classes
        self.data_config = data_config
        self.grid_config = grid_config
        ds = torch.arange(*self.grid_config['dbound'], dtype=torch.float).view(-1, 1, 1)
        D, _, _ = ds.shape
        self.D = D
        self.cam_depth_range = self.grid_config['dbound']
        self.D_bins = int((grid_config['dbound'][1]-grid_config['dbound'][0])//grid_config['dbound'][2])
        self.downsample = downsample
        
        # for point GNN and fusion with radar_sparse_depth features
        self.knn_radar = knn_radar
        self.points_outs_channels = points_outs_channels
        self.fusion_base_channels = fusion_base_channels
        self.depth_feats_channels = depth_feats_channels
        self.DEdgeConv_1 = DynamicEdgeConv(nn.Sequential(nn.Linear(3 * 2, self.fusion_base_channels)), k=knn_radar, aggr=aggr)
        self.DEdgeConv_2 = DynamicEdgeConv(nn.Sequential(nn.Linear(self.fusion_base_channels * 2, self.fusion_base_channels)), k=knn_radar, aggr=aggr)
        self.DEdgeConv_3 = DynamicEdgeConv(nn.Sequential(nn.Linear(self.fusion_base_channels * 2, self.fusion_base_channels)), k=knn_radar, aggr=aggr)
        self.attention = MultiHeadCrossAttention(self.fusion_base_channels, self.depth_feats_channels, self.fusion_base_channels, num_heads=8)
        self.linear = nn.Linear(self.fusion_base_channels, self.points_outs_channels)
        self.act = nn.ELU(inplace=True)
        if self.point_supervision['use']:
            self.points_seg_net = nn.Sequential(
            nn.Linear(self.points_outs_channels, self.points_outs_channels // 2), nn.LayerNorm(self.points_outs_channels // 2), nn.ReLU(inplace=True),
            nn.Linear(self.points_outs_channels // 2, self.points_outs_channels // 4), nn.LayerNorm(self.points_outs_channels // 4), nn.ReLU(inplace=True),
            nn.Linear(self.points_outs_channels // 4, self.num_classes + 1))
        
        # feature extraction of radar_sparse_depth
        self.point_depth_volume_encoder = DepthVolumeEncoder(in_channels=D, out_channels=D)
        self.depth_conv_mapping = nn.Conv2d(D, self.depth_feats_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, radar_points, depth_images, lidar2img):
        
        B, _, raw_H, raw_W = depth_images.shape
        H, W = raw_H//self.downsample, raw_W//self.downsample
        # feature extraction of depth_images, B C H W
        _, point_depth_volume = self.get_downsampled_gt_depth(depth_images)
        point_depth_volume = point_depth_volume.view(B, H, W, -1).permute(0, 3, 1, 2)
        point_depth_volume = self.point_depth_volume_encoder(point_depth_volume)
        depth_features = self.depth_conv_mapping(point_depth_volume)
        depth_volume = self.get_depth_dist(point_depth_volume) # supervised by lidar depth
        
        # points graph data preparation
        edge_index_list = []
        for i in range(B):
            edge_index = knn_graph(radar_points[i], k=self.knn_radar)
            edge_index_list.append(edge_index)
        graph_radar_points = [Data(x=radar_points[i], edge_index=edge_index_list[i]) for i in range(B)]
        graph_radar_points = Batch.from_data_list(graph_radar_points)
        xyz = graph_radar_points.x # [N, 3], N = N1 + N2 + N3 + ...
        batch = graph_radar_points.batch # [N]
        device, dtype = xyz.device, xyz.dtype
        epsilon = 1e-6

        # point projection to image plane preparation
        img_pts_int_list = []
        for i in range(batch.max() + 1):
            pts = xyz[batch == i]
            pts_hom = torch.cat((pts, torch.ones((pts.shape[0], 1), device=device)), dim=1)  # (N, 4)
            img_pts = torch.matmul(lidar2img[i], pts_hom.t()).t()  # (N, 4)
            img_pts[:, :2] = img_pts[:, :2] / (img_pts[:, 2:3] + epsilon)
            # valid_mask = (img_pts[:, 0] >= 0) & (img_pts[:, 0] < W) & (img_pts[:, 1] >= 0) & (img_pts[:, 1] < H)
            # img_pts_int = img_pts[:, :2].long()
            # valid_img_pts_int = img_pts_int[valid_mask]
            img_pts_int = img_pts[:, :2].long()
            valid_img_pts_int = copy.copy(img_pts_int)
            valid_img_pts_int[valid_img_pts_int[:, 0]<= 0, 0] = 0
            valid_img_pts_int[valid_img_pts_int[:, 0] >= W, 0] = W - 1
            valid_img_pts_int[valid_img_pts_int[:, 1] <= 0, 1] = 0
            valid_img_pts_int[valid_img_pts_int[:, 1] >= H, 1] = H - 1
            img_pts_int_list.append(valid_img_pts_int)
        point_idx = torch.cat(img_pts_int_list, dim=0) # [N, 2]
        
        # point grapth feats fuse with depth_feature
        corresponding_depth_feats = depth_features[batch, :, point_idx[:, 1], point_idx[:, 0]]
        xyz_out = self.DEdgeConv_1(xyz, batch)
        xyz_out = self.DEdgeConv_2(xyz_out, batch)
        xyz_out = self.DEdgeConv_3(xyz_out, batch)
        
        # re-mapping to list[N, C], depth_volume [B, H, W, D]
        seg_feats_list = []
        for i in range(B):
            xyz_out_batch = xyz_out[batch==i]
            depth_batch = corresponding_depth_feats[batch==i]
            outputs = self.attention(xyz_out_batch, depth_batch) + xyz_out_batch
            outputs = self.act(self.linear(outputs))
            seg_feats_list.append(outputs)
        return seg_feats_list, depth_volume
    
    def get_klv_depth_loss(self, depth_labels, depth_preds):
        B, bin_size, H, W = depth_preds.shape
        depth_gaussian_labels, depth_values = generate_guassian_depth_target(depth_labels,
            self.downsample, self.cam_depth_range, constant_std=0.5)
        
        depth_values = depth_values.view(B, H, W)
        fg_mask = (depth_values >= self.cam_depth_range[0]) & (depth_values <= (self.cam_depth_range[1] - self.cam_depth_range[2]))        
        
        depth_gaussian_labels = depth_gaussian_labels.view(B, H, W, self.D)
        depth_preds = depth_preds.permute(0, 2, 3, 1).contiguous().view(B, H, W, self.D)
        
        # saving train depth estimating images
        # self.draw_depth(depth_labels, depth_preds, depth_gaussian_labels, img, extra_depth, precise_depth, radar_depth, show = B)
        
        depth_gaussian_labels = depth_gaussian_labels[fg_mask]
        depth_preds = depth_preds[fg_mask]
        depth_loss = F.kl_div(torch.log(depth_preds + 1e-4), depth_gaussian_labels, reduction='batchmean', log_target=False)
        loss_radar_depth = self.radar_supervision['weight']*depth_loss
        return dict(loss_radar_depth=loss_radar_depth)

    def get_downsampled_gt_depth(self, gt_depths):
        """
        Input:
            gt_depths: [B, N, H, W]
        Output:
            gt_depths: [B*N*h*w, d]
        """
        B, N, H, W = gt_depths.shape
        # may be different resolution
        if self.training: final_dim = self.data_config['final_dim']
        else: final_dim = self.data_config['final_dim_test']
        target_H = final_dim[0] // self.downsample
        target_W = final_dim[1] // self.downsample
        assert H // target_H == W // target_W
        down_here = H // target_H
        
        gt_depths = gt_depths.view(B * N,
                                   H // down_here, down_here,
                                   W // down_here, down_here, 1)
        gt_depths = gt_depths.permute(0, 1, 3, 5, 2, 4).contiguous()
        gt_depths = gt_depths.view(-1, down_here * down_here)
        gt_depths_tmp = torch.where(gt_depths == 0.0, 1e5 * torch.ones_like(gt_depths), gt_depths)
        gt_depths = torch.min(gt_depths_tmp, dim=-1).values
        # NOTE here, we re assgine back since image augmetation may cause some regions to be empty
        gt_depths = torch.where(gt_depths == 1e5, torch.zeros_like(gt_depths), gt_depths)
        gt_depths = gt_depths.view(B * N, H // down_here, W // down_here)
        
        # [min - step / 2, min + step / 2] creates min depth
        gt_depths = (gt_depths - (self.grid_config['dbound'][0] - self.grid_config['dbound'][2] / 2)) / self.grid_config['dbound'][2]
        gt_depths_vals = gt_depths.clone()
        
        # original implementation
        # gt_depths = torch.where((gt_depths < self.D + 1) & (gt_depths >= 0.0), gt_depths, torch.zeros_like(gt_depths))
        # my newly implementation
        gt_depths = torch.where(gt_depths < 0.0, torch.zeros_like(gt_depths), gt_depths)
        gt_depths = torch.where(gt_depths > self.D, torch.full_like(gt_depths, self.D), gt_depths)
        gt_depths = F.one_hot(gt_depths.long(), num_classes=self.D + 1).view(-1, self.D + 1)[:, 1:]
        
        return gt_depths_vals, gt_depths.float()
    
    def get_depth_dist(self, x):
        return x.softmax(dim=1)
    
    def get_point_seg_loss(self, gt_labels, pred_logits):
        total_loss = 0.0
        for i in range(len(pred_logits)):
            logits = pred_logits[i]  # (N, C)
            labels = gt_labels[i].squeeze()  # (N,)

            # 确保 labels 是 Tensor 类型，如果是 numpy.ndarray 则转换为 Tensor
            if isinstance(labels, np.ndarray):
                labels = torch.tensor(labels, dtype=torch.long, device=logits.device)
            # 计算类别的权重
            class_counts = torch.bincount(labels)
            total_points = labels.size(0)
            background_class_weight = 1.0 / (class_counts[self.num_classes] / total_points)  # 假设背景类为 class 0
            other_class_weight = 1.0 / (class_counts[:self.num_classes].sum() / total_points)  # 其他类别权重

            # 为每个类别设置不同的权重
            class_weights = torch.ones_like(class_counts, dtype=torch.float32)
            class_weights[self.num_classes] = background_class_weight
            class_weights[:self.num_classes] = other_class_weight  # 其他类别的权重

            # 计算交叉熵损失，并加入类别权重
            loss = F.cross_entropy(logits, labels, weight=class_weights.to(logits.device))
            total_loss += loss

        loss_point_seg = self.point_supervision['weight']*total_loss
        return dict(loss_point_seg=loss_point_seg)

@FUSION_LAYERS.register_module()
class VoteFeatureGNN(nn.Module):
    def __init__(self, in_channels, hidden_channels, k=8):
        """
        初始化 GNN 模块。
        
        参数：
            in_channels (int): 输入特征的通道数(如 all_vote_feature 的 C)。
            hidden_channels (int): GNN 隐藏层的通道数。
            k (int): k 近邻的数量，用于构建图结构。
        """
        super(VoteFeatureGNN, self).__init__()
        self.k = k
        # 定义两层 GCN，第一层扩展特征，第二层恢复到原始维度
        self.gcn1 = GCNConv(in_channels, hidden_channels)
        self.gcn2 = GCNConv(hidden_channels, in_channels)
        self.relu = nn.ReLU()  # 非线性激活函数

    def forward(self, vote_features, vote_centers, batch_size=None):
        """
        前向传播，增强投票中心特征。
        
        参数：
            vote_features (torch.Tensor): [N_vote, C] - 投票中心特征
            vote_centers (torch.Tensor): [N_vote, 3] - 投票中心坐标
            batch_idx (torch.Tensor, optional): [N_vote] - 批次索引，用于多样本处理
            
        返回：
            enhanced_features (torch.Tensor): [N_vote, C] - 增强后的特征
        """
        # 处理多批次数据的情况
        
        enhanced_features = []

        for b in range(batch_size):
            feats_b = vote_features[b]        # [N_b, C]
            centers_b = vote_centers[b]       # [N_b, 3]
            # 构建 k-NN 图
            edge_index_b = knn(centers_b, centers_b, self.k)
            edge_index_b = edge_index_b.flip([0])  # 调整为 [2, num_edges]
            # 应用 GCN
            feats_b = self._apply_gcn(feats_b, edge_index_b)
            enhanced_features.append(feats_b)
        
        return enhanced_features

    def _apply_gcn(self, features, edge_index):
        """
        应用 GCN 层并返回增强特征。
        
        参数：
            features (torch.Tensor): 输入特征
            edge_index (torch.Tensor): 图的边索引
            
        返回：
            torch.Tensor: 增强后的特征
        """
        # 第一层 GCN
        x = self.gcn1(features, edge_index)
        x = self.relu(x)
        # 第二层 GCN
        x = self.gcn2(x, edge_index)
        # 残差连接，保留原始特征信息
        return features + x
if __name__ == "__main__":

    B = 2
    N1, N2 = 373, 211
    H, W = 128, 128
    knn_radar = 6
    radar_points = [torch.randn(N1, 3), torch.randn(N2, 3)]
    depth_images = torch.randn(B, 1, H, W)
    lidar2img = torch.randn(B, 4, 4)

    model = Attention_enhanced_DGCNN(points_outs_channels=64, fusion_base_channels=64, depth_feats_channels=256, knn_radar=knn_radar, aggr='max')
    out = model(radar_points, depth_images, lidar2img)

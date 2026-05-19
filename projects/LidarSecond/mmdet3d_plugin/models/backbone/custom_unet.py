from typing import Set

try:
    import spconv.pytorch as spconv
except:
    import spconv as spconv

import torch.nn as nn
from functools import partial
import torch
import torch.nn as nn
import logging
import os
import pickle
import random
import shutil
import subprocess
import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

from mmcv.runner import BaseModule, auto_fp16
 
# from mmdet3d.ops import spconv as spconv
from mmdet3d.ops import  make_sparse_convmodule
from mmdet3d.models.builder import BACKBONES
from mmdet3d.models import builder

class SparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None):
        super(SparseBasicBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=bias, indice_key=indice_key
        )
        self.bn2 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = replace_feature(out, self.bn1(out.features))
        out = replace_feature(out, self.relu(out.features))

        out = self.conv2(out)
        out = replace_feature(out, self.bn2(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = replace_feature(out, out.features + identity.features)
        out = replace_feature(out, self.relu(out.features))

        return out

def find_all_spconv_keys(model: nn.Module, prefix="") -> Set[str]:
    """
    Finds all spconv keys that need to have weight's transposed
    """
    found_keys: Set[str] = set()
    for name, child in model.named_children():
        new_prefix = f"{prefix}.{name}" if prefix != "" else name

        if isinstance(child, spconv.conv.SparseConvolution):
            new_prefix = f"{new_prefix}.weight"
            found_keys.add(new_prefix)

        found_keys.update(find_all_spconv_keys(child, prefix=new_prefix))

    return found_keys

def replace_feature(out, new_features):
    if "replace_feature" in out.__dir__():
        # spconv 2.x behaviour
        return out.replace_feature(new_features)
    else:
        out.features = new_features
        return out

def post_act_block(in_channels, out_channels, kernel_size, indice_key=None, stride=1, padding=0,
                   conv_type='SubMConv3d', norm_fn=None):

    if conv_type == 'SubMConv3d':
        conv = spconv.SubMConv3d(in_channels, out_channels, kernel_size, bias=False, indice_key=indice_key)
    elif conv_type == 'SparseConv3d':
        conv = spconv.SparseConv3d(in_channels, out_channels, kernel_size, stride=stride, padding=padding,
                                   bias=False, indice_key=indice_key)
    elif conv_type == 'SparseInverseConv3d':
        conv = spconv.SparseInverseConv3d(in_channels, out_channels, kernel_size, indice_key=indice_key, bias=False)
    else:
        raise NotImplementedError

    m = spconv.SparseSequential(
        conv,
        norm_fn(out_channels),
        nn.ReLU(),
    )

    return m

def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False

def limit_period(val, offset=0.5, period=np.pi):
    val, is_numpy = check_numpy_to_torch(val)
    ans = val - torch.floor(val / period + offset) * period
    return ans.numpy() if is_numpy else ans

def drop_info_with_name(info, name):
    ret_info = {}
    keep_indices = [i for i, x in enumerate(info['name']) if x != name]
    for key in info.keys():
        ret_info[key] = info[key][keep_indices]
    return ret_info

def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:

    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

def mask_points_by_range(points, limit_range):
    mask = (points[:, 0] >= limit_range[0]) & (points[:, 0] <= limit_range[3]) \
           & (points[:, 1] >= limit_range[1]) & (points[:, 1] <= limit_range[4])
    return mask

def get_voxel_centers(voxel_coords, downsample_times, voxel_size, point_cloud_range):
    """
    Args:
        voxel_coords: (N, 3)
        downsample_times:
        voxel_size:
        point_cloud_range:

    Returns:

    """
    assert voxel_coords.shape[1] == 3
    voxel_centers = voxel_coords[:, [2, 1, 0]].float()  # (xyz)
    voxel_size = torch.tensor(voxel_size, device=voxel_centers.device).float() * downsample_times
    pc_range = torch.tensor(point_cloud_range[0:3], device=voxel_centers.device).float()
    voxel_centers = (voxel_centers + 0.5) * voxel_size + pc_range
    return voxel_centers

def create_logger(log_file=None, rank=0, log_level=logging.INFO):
    logger = logging.getLogger(__name__)
    logger.setLevel(log_level if rank == 0 else 'ERROR')
    formatter = logging.Formatter('%(asctime)s  %(levelname)5s  %(message)s')
    console = logging.StreamHandler()
    console.setLevel(log_level if rank == 0 else 'ERROR')
    console.setFormatter(formatter)
    logger.addHandler(console)
    if log_file is not None:
        file_handler = logging.FileHandler(filename=log_file)
        file_handler.setLevel(log_level if rank == 0 else 'ERROR')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    logger.propagate = False
    return logger

def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_pad_params(desired_size, cur_size):
    """
    Get padding parameters for np.pad function
    Args:
        desired_size: int, Desired padded output size
        cur_size: int, Current size. Should always be less than or equal to cur_size
    Returns:
        pad_params: tuple(int), Number of values padded to the edges (before, after)
    """
    assert desired_size >= cur_size

    # Calculate amount to pad
    diff = desired_size - cur_size
    pad_params = (0, diff)

    return pad_params

def keep_arrays_by_name(gt_names, used_classes):
    inds = [i for i, x in enumerate(gt_names) if x in used_classes]
    inds = np.array(inds, dtype=np.int64)
    return inds

def init_dist_slurm(tcp_port, local_rank, backend='nccl'):
    """
    modified from https://github.com/open-mmlab/mmdetection
    Args:
        tcp_port:
        backend:

    Returns:

    """
    proc_id = int(os.environ['SLURM_PROCID'])
    ntasks = int(os.environ['SLURM_NTASKS'])
    node_list = os.environ['SLURM_NODELIST']
    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(proc_id % num_gpus)
    addr = subprocess.getoutput('scontrol show hostname {} | head -n1'.format(node_list))
    os.environ['MASTER_PORT'] = str(tcp_port)
    os.environ['MASTER_ADDR'] = addr
    os.environ['WORLD_SIZE'] = str(ntasks)
    os.environ['RANK'] = str(proc_id)
    dist.init_process_group(backend=backend)

    total_gpus = dist.get_world_size()
    rank = dist.get_rank()
    return total_gpus, rank

def init_dist_pytorch(tcp_port, local_rank, backend='nccl'):
    if mp.get_start_method(allow_none=True) is None:
        mp.set_start_method('spawn')

    num_gpus = torch.cuda.device_count()
    torch.cuda.set_device(local_rank % num_gpus)
    dist.init_process_group(
        backend=backend,
        # init_method='tcp://127.0.0.1:%d' % tcp_port,
        # rank=local_rank,
        # world_size=num_gpus
    )
    rank = dist.get_rank()
    return num_gpus, rank

def get_dist_info(return_gpu_per_machine=False):
    if torch.__version__ < '1.0':
        initialized = dist._initialized
    else:
        if dist.is_available():
            initialized = dist.is_initialized()
        else:
            initialized = False
    if initialized:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    if return_gpu_per_machine:
        gpu_per_machine = torch.cuda.device_count()
        return rank, world_size, gpu_per_machine

    return rank, world_size

def merge_results_dist(result_part, size, tmpdir):
    rank, world_size = get_dist_info()
    os.makedirs(tmpdir, exist_ok=True)

    dist.barrier()
    pickle.dump(result_part, open(os.path.join(tmpdir, 'result_part_{}.pkl'.format(rank)), 'wb'))
    dist.barrier()

    if rank != 0:
        return None

    part_list = []
    for i in range(world_size):
        part_file = os.path.join(tmpdir, 'result_part_{}.pkl'.format(i))
        part_list.append(pickle.load(open(part_file, 'rb')))

    ordered_results = []
    for res in zip(*part_list):
        ordered_results.extend(list(res))
    ordered_results = ordered_results[:size]
    shutil.rmtree(tmpdir)
    return ordered_results

def scatter_point_inds(indices, point_inds, shape):
    ret = -1 * torch.ones(*shape, dtype=point_inds.dtype, device=point_inds.device)
    ndim = indices.shape[-1]
    flattened_indices = indices.view(-1, ndim)
    slices = [flattened_indices[:, i] for i in range(ndim)]
    ret[slices] = point_inds
    return ret

def generate_voxel2pinds(sparse_tensor):
    device = sparse_tensor.indices.device
    batch_size = sparse_tensor.batch_size
    spatial_shape = sparse_tensor.spatial_shape
    indices = sparse_tensor.indices.long()
    point_indices = torch.arange(indices.shape[0], device=device, dtype=torch.int32)
    output_shape = [batch_size] + list(spatial_shape)
    v2pinds_tensor = scatter_point_inds(indices, point_indices, output_shape)
    return v2pinds_tensor

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class SparseUNet(BaseModule):
    r"""SparseUNet for PartA^2.

    See the `paper <https://arxiv.org/abs/1907.03670>`_ for more details.

    Args:
        in_channels (int): The number of input channels.
        sparse_shape (list[int]): The sparse shape of input tensor.
        norm_fn (dict): Config of normalization layer.
        base_channels (int): Out channels for conv_input layer.
        output_channels (int): Out channels for conv_out layer.
        encoder_channels (tuple[tuple[int]]):
            Convolutional channels of each encode block.
        encoder_paddings (tuple[tuple[int]]): Paddings of each encode block.
        decoder_channels (tuple[tuple[int]]):
            Convolutional channels of each decode block.
        decoder_paddings (tuple[tuple[int]]): Paddings of each decode block.
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_fn=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
                                                                        64)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
                                                                 1)),
                 decoder_channels=((64, 64, 64), (64, 64, 32), (32, 32, 16),
                                   (16, 16, 16)),
                 decoder_paddings=((1, 0), (1, 0), (0, 0), (0, 1)),
                 init_cfg=None):
        super().__init__(init_cfg=init_cfg)
        self.sparse_shape = sparse_shape
        self.in_channels = in_channels
        self.order = order
        self.base_channels = base_channels
        self.output_channels = output_channels
        self.encoder_channels = encoder_channels
        self.encoder_paddings = encoder_paddings
        self.decoder_channels = decoder_channels
        self.decoder_paddings = decoder_paddings
        self.stage_num = len(self.encoder_channels)
        self.fp16_enabled = False
        # Spconv init all weight on its own

        assert isinstance(order, tuple) and len(order) == 3
        assert set(order) == {'conv', 'norm', 'act'}
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        
        if self.order[0] != 'conv':  # pre activate
            self.conv_input = post_act_block(
                in_channels,
                self.base_channels,
                3,
                norm_fn=norm_fn,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d',
                order=('conv', ))
        else:  # post activate
            self.conv_input = post_act_block(
                in_channels,
                self.base_channels,
                3,
                norm_fn=norm_fn,
                padding=1,
                indice_key='subm1',
                conv_type='SubMConv3d')

        encoder_out_channels = self.make_encoder_layers(
            post_act_block, norm_fn, self.base_channels)
        self.make_decoder_layers(post_act_block, norm_fn,
                                 encoder_out_channels)

        self.conv_out = post_act_block(
            encoder_out_channels,
            self.output_channels,
            kernel_size=(3, 1, 1),
            stride=(2, 1, 1),
            norm_fn=norm_fn,
            padding=0,
            indice_key='spconv_down2',
            conv_type='SparseConv3d')

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_features, coors, batch_size):
        """Forward of SparseUNet.

        Args:
            voxel_features (torch.float32): Voxel features in shape [N, C].
            coors (torch.int32): Coordinates in shape [N, 4],
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict[str, torch.Tensor]: Backbone features.
        """
        coors = coors.int()
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors,
                                                  self.sparse_shape,
                                                  batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(encode_features[-1])
        spatial_features = out.dense()

        N, C, D, H, W = spatial_features.shape
        spatial_features = spatial_features.view(N, C * D, H, W)

        # for segmentation head, with output shape:
        # [400, 352, 11] <- [200, 176, 5]
        # [800, 704, 21] <- [400, 352, 11]
        # [1600, 1408, 41] <- [800, 704, 21]
        # [1600, 1408, 41] <- [1600, 1408, 41]
        decode_features = []
        x = encode_features[-1]
        for i in range(self.stage_num, 0, -1):
            x = self.decoder_layer_forward(encode_features[i - 1], x,
                                           getattr(self, f'lateral_layer{i}'),
                                           getattr(self, f'merge_layer{i}'),
                                           getattr(self, f'upsample_layer{i}'))
            decode_features.append(x)

        seg_features = decode_features[-1].features

        ret = dict(
            spatial_features=spatial_features, seg_features=seg_features)

        return ret

    def decoder_layer_forward(self, x_lateral, x_bottom, lateral_layer,
                              merge_layer, upsample_layer):
        """Forward of upsample and residual block.

        Args:
            x_lateral (:obj:`SparseConvTensor`): Lateral tensor.
            x_bottom (:obj:`SparseConvTensor`): Feature from bottom layer.
            lateral_layer (SparseBasicBlock): Convolution for lateral tensor.
            merge_layer (SparseSequential): Convolution for merging features.
            upsample_layer (SparseSequential): Convolution for upsampling.

        Returns:
            :obj:`SparseConvTensor`: Upsampled feature.
        """
        x = lateral_layer(x_lateral)
        # x.features = torch.cat((x_bottom.features, x.features), dim=1)
        x = replace_feature(x, torch.cat((x_bottom.features, x.features), dim=1))
        x_merge = merge_layer(x)
        x = self.reduce_channel(x, x_merge.features.shape[1])
        # x.features = x_merge.features + x.features
        x = replace_feature(x, x_merge.features + x.features)
        x = upsample_layer(x)
        return x

    @staticmethod
    def reduce_channel(x, out_channels):
        """reduce channel for element-wise addition.

        Args:
            x (:obj:`SparseConvTensor`): Sparse tensor, ``x.features``
                are in shape (N, C1).
            out_channels (int): The number of channel after reduction.

        Returns:
            :obj:`SparseConvTensor`: Channel reduced feature.
        """
        features = x.features
        n, in_channels = features.shape
        assert (in_channels % out_channels
                == 0) and (in_channels >= out_channels)

        # x.features = features.view(n, out_channels, -1).sum(dim=2)
        x = replace_feature(x, features.view(n, out_channels, -1).sum(dim=2))
        return x

    def make_encoder_layers(self, make_block, norm_fn, in_channels):
        """make encoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_fn (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.

        Returns:
            int: The number of encoder output channels.
        """
        self.encoder_layers = spconv.SparseSequential()

        for i, blocks in enumerate(self.encoder_channels):
            blocks_list = []
            for j, out_channels in enumerate(tuple(blocks)):
                padding = tuple(self.encoder_paddings[i])[j]
                # each stage started with a spconv layer
                # except the first stage
                if i != 0 and j == 0:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_fn=norm_fn,
                            stride=2,
                            padding=padding,
                            indice_key=f'spconv{i + 1}',
                            conv_type='SparseConv3d'))
                else:
                    blocks_list.append(
                        make_block(
                            in_channels,
                            out_channels,
                            3,
                            norm_fn=norm_fn,
                            padding=padding,
                            indice_key=f'subm{i + 1}',
                            conv_type='SubMConv3d'))
                in_channels = out_channels
            stage_name = f'encoder_layer{i + 1}'
            stage_layers = spconv.SparseSequential(*blocks_list)
            self.encoder_layers.add_module(stage_name, stage_layers)
        return out_channels

    def make_decoder_layers(self, make_block, norm_fn, in_channels):
        """make decoder layers using sparse convs.

        Args:
            make_block (method): A bounded function to build blocks.
            norm_fn (dict[str]): Config of normalization layer.
            in_channels (int): The number of encoder input channels.

        Returns:
            int: The number of encoder output channels.
        """
        block_num = len(self.decoder_channels)
        for i, block_channels in enumerate(self.decoder_channels):
            paddings = self.decoder_paddings[i]
            setattr(
                self, f'lateral_layer{block_num - i}',
                SparseBasicBlock(
                    in_channels,
                    block_channels[0],
                    # conv_cfg=dict(type='SubMConv3d', indice_key=f'subm{block_num - i}'),
                    indice_key=f'subm{block_num - i}',
                    norm_fn=norm_fn))
            setattr(
                self, f'merge_layer{block_num - i}',
                make_block(
                    in_channels * 2,
                    block_channels[1],
                    3,
                    norm_fn=norm_fn,
                    padding=paddings[0],
                    indice_key=f'subm{block_num - i}',
                    conv_type='SubMConv3d'))
            if block_num - i != 1:
                setattr(
                    self, f'upsample_layer{block_num - i}',
                    make_block(
                        in_channels,
                        block_channels[2],
                        3,
                        norm_fn=norm_fn,
                        indice_key=f'spconv{block_num - i}',
                        conv_type='SparseInverseConv3d'))
            else:
                # use submanifold conv instead of inverse conv
                # in the last block
                setattr(
                    self, f'upsample_layer{block_num - i}',
                    make_block(
                        in_channels,
                        block_channels[2],
                        3,
                        norm_fn=norm_fn,
                        padding=paddings[1],
                        indice_key='subm1',
                        conv_type='SubMConv3d'))
            in_channels = block_channels[2]

@BACKBONES.register_module()
class SimpleSparseUNet_modified(SparseUNet):
    r""" A simpler SparseUNet, removing the densify part
    """

    def __init__(self,
                 in_channels,
                 sparse_shape,
                 order=('conv', 'norm', 'act'),
                 norm_fn=dict(type='BN1d', eps=1e-3, momentum=0.01),
                 base_channels=16,
                 output_channels=128,
                 encoder_channels=((16, ), (32, 32, 32), (64, 64, 64), (64, 64,
                                                                        64)),
                 encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1,
                                                                 1)),
                 decoder_channels=((64, 64, 64), (64, 64, 32), (32, 32, 16),
                                   (16, 16, 16)),
                 decoder_paddings=((1, 0), (1, 0), (0, 0), (0, 1)),
                 keep_coors_dims=None,
                #  act_type='relu',
                 return_multiscale_features=False,
                 init_cfg=None,
                 ):
        super().__init__(
            in_channels=in_channels,
            sparse_shape=sparse_shape,
            order=order,
            norm_fn=norm_fn,
            base_channels=base_channels,
            output_channels=output_channels,
            encoder_channels=encoder_channels,
            encoder_paddings=encoder_paddings,
            decoder_channels=decoder_channels,
            decoder_paddings=decoder_paddings,
            # act_type=act_type,
            init_cfg=init_cfg,
        )
        self.conv_out = None # override
        self.keep_coors_dims = keep_coors_dims
        self.return_multiscale_features = return_multiscale_features

    @auto_fp16(apply_to=('voxel_features', ))
    def forward(self, voxel_info):
        """Forward of SparseUNet.

        Args:
            voxel_features (torch.float32): Voxel features in shape [N, C].
            coors (torch.int32): Coordinates in shape [N, 4],
                the columns in the order of (batch_idx, z_idx, y_idx, x_idx).
            batch_size (int): Batch size.

        Returns:
            dict[str, torch.Tensor]: Backbone features.
        """
        coors = voxel_info['voxel_coors']
        if self.keep_coors_dims is not None:
            coors = coors[:, self.keep_coors_dims]
        voxel_features = voxel_info['voxel_feats']
        coors = coors.int()
        batch_size = coors[:, 0].max().item() + 1
        input_sp_tensor = spconv.SparseConvTensor(voxel_features, coors,
                                                  self.sparse_shape,
                                                  batch_size)
        x = self.conv_input(input_sp_tensor)

        encode_features = []
        decode_features = []
        for encoder_layer in self.encoder_layers:
            x = encoder_layer(x)
            encode_features.append(x)

        x = encode_features[-1]
        for i in range(self.stage_num, 0, -1):
            x = self.decoder_layer_forward(encode_features[i - 1], x,
                                           getattr(self, f'lateral_layer{i}'),
                                           getattr(self, f'merge_layer{i}'),
                                           getattr(self, f'upsample_layer{i}'))
            if self.return_multiscale_features:
                decode_features.append(x)

        seg_features = x.features
        ret = {'voxel_feats':x.features, 'voxel_coors': x.indices, 'sparse_shape':x.spatial_shape, 'batch_size':x.batch_size, 'decoder_features':decode_features}
        ret = [ret,] # keep consistent with SSTv2

        return ret
    
if __name__ == '__main__':
    # 假设 bev_w_ 和 bev_h_ 是给定的值
    bev_w_ = 160  # 你可以根据实际设置修改
    bev_h_ = 160  # 同样需要修改
    # 1. 创建模型
    model = SimpleSparseUNet_modified(in_channels=64,
        sparse_shape=[32, bev_w_ * 4, bev_h_ * 4],  # 根据 bev_w_ 和 bev_h_ 设置
        order=('conv', 'norm', 'act'),
        norm_fn=dict(type='BN1d', eps=1e-3, momentum=0.01),
        base_channels=64,
        output_channels=128,
        encoder_channels=((64, ), (64, 64, 64), (64, 64, 64), (128, 128, 128), (256, 256, 256)),
        encoder_paddings=((1, ), (1, 1, 1), (1, 1, 1), ((0, 1, 1), 1, 1), (1, 1, 1)),
        decoder_channels=((256, 256, 128), (128, 128, 64), (64, 64, 64), (64, 64, 64), (64, 64, 64)),
        decoder_paddings=((1, 1), (1, 0), (1, 0), (0, 0), (0, 1))).to('cuda')

    # 2. 生成输入数据
    batch_size = 2
    num_voxels = 200  # 假设有 10000 个体素
    # 假设 voxel_feats 每个体素有 64 个特征
    voxel_feats = torch.rand(num_voxels, 64)

    # 假设稀疏网格的尺寸为 sparse_shape=[32, bev_w_*4, bev_h_*4]
    sparse_shape = [32, bev_w_ * 4, bev_h_ * 4]
    max_z, max_x, max_y = sparse_shape

    # 随机生成坐标 (batch_idx, z_idx, x_idx, y_idx)
    # batch_idx 取值 [0, batch_size-1]
    # z_idx 取值 [0, max_z-1]
    # x_idx 取值 [0, max_x-1]
    # y_idx 取值 [0, max_y-1]
    voxel_coors = torch.stack([
        torch.randint(0, batch_size, (num_voxels,)),  # batch_idx
        torch.randint(0, max_z, (num_voxels,)),      # z_idx
        torch.randint(0, max_x, (num_voxels,)),      # x_idx
        torch.randint(0, max_y, (num_voxels,))       # y_idx
        ], dim=-1)

    # 将数据包装在字典中
    voxel_info = {
    'voxel_feats': voxel_feats.to('cuda'),
    'voxel_coors': voxel_coors.to('cuda'),
    }

    # 3. 进行前向传播
    model.eval()  # 切换到评估模式
    with torch.no_grad():
        output = model(voxel_info)

    # 4. 输出结果
    print("Output:")
    print(output)

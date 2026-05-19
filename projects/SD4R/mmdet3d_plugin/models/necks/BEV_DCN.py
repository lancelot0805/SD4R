import torch
import torch.nn as nn
import torchvision.ops as ops
from mmdet3d.models.builder import FUSION_LAYERS

class DeformableConvLayer(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True):
        super(DeformableConvLayer, self).__init__()
        
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.in_channels = in_channels
        self.out_channels = out_channels

        # 普通卷积层
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=False)
        
        # 用于计算偏移量的卷积层
        self.offset_conv = nn.Conv2d(in_channels, 2 * kernel_size * kernel_size, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)

    def forward(self, x):
        # 计算偏移量
        offset = self.offset_conv(x)
        offset = torch.sigmoid(offset) * 2 - 1
        
        # 使用 deform_conv 实现 Deformable Convolution
        out = ops.deform_conv2d(input=x, offset=offset, weight=self.conv.weight, stride=self.stride, padding=self.padding, dilation=self.dilation)
        
        return out

@FUSION_LAYERS.register_module()    
class DeepDeformableConvNet(nn.Module):
    def __init__(self, in_channels=3, base_channels=256, num_layers=4):
        super(DeepDeformableConvNet, self).__init__()
        
        layers = []
        current_channels = in_channels  # 初始化为输入通道数

        # 动态构建Deformable Conv 层
        for i in range(num_layers):
            out_channels = base_channels  # 每一层的输出通道数固定为base_channels
            layers.append(DeformableConvLayer(current_channels, out_channels, kernel_size=3, stride=1, padding=1))
            current_channels = out_channels  # 当前层输出的通道数作为下一层的输入通道数
            layers.append(nn.ReLU())  # 每个卷积后添加ReLU激活
        
        # 最后一层调整通道数为输入的通道数
        layers.append(DeformableConvLayer(current_channels, in_channels, kernel_size=3, stride=1, padding=1))
        
        # 将所有层放入Sequential中
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


# 测试模型
if __name__ == '__main__':
    x = torch.randn(8, 64, 128, 128)  # 假设输入图像尺寸为 (B, C, H, W)，比如 B=8, C=64, H=128, W=128
    model = DeepDeformableConvNet(in_channels=64, base_channels=32, num_layers=4)  # 设置输入通道数64，基础通道数32，层数4
    output = model(x)
    print(output.shape, sum(p.numel() for p in model.parameters()))  # 输出形状和参数数量

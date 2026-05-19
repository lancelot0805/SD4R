import torch
import torch.nn as nn
import torch.nn.functional as F
from mmdet3d.models.builder import FUSION_LAYERS
from torchvision.utils import save_image
from .MRF3Net import MRF3Net

@FUSION_LAYERS.register_module()    
class FeatureFocus(nn.Module):
    def __init__(self, c, loss_weight, objects_thre=0.4, backgrd_thre=0.4):
        super(FeatureFocus, self).__init__()
        self.objects_thre = objects_thre
        self.backgrd_thre = backgrd_thre
        self.back_encoder = nn.Sequential(Conv(c, c, 3, p=1), Conv(c, c, 1))
        self.objt_encoder = nn.Sequential(Conv(c, c, 3, p=1), Conv(c, c, 1))
        self.back_aware = TargetAwareFusion(c)
        self.objt_aware = TargetAwareFusion(c)
        self.reshape_back = nn.Sequential(nn.Conv2d(c, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.reshape_objt = nn.Sequential(nn.Conv2d(c, 1, kernel_size=1, stride=1, padding=0), nn.Sigmoid())
        self.feature_loss = Feature_loss(weight=loss_weight)
    def get_feature_focus_loss(self, input_dict, gt_bev_mask):
        pred_mask_objt = input_dict['pred_mask_objt']
        similarity_logits_objt = input_dict['similarity_logits_objt']
        pred_mask_back = input_dict['pred_mask_back']
        similarity_logits_back = input_dict['similarity_logits_back']
        contrastive = input_dict['contrastive']
        loss = self.feature_loss((pred_mask_objt, similarity_logits_objt, pred_mask_back, similarity_logits_back, contrastive), gt_bev_mask)
        return loss
        
    def forward(self, x, tokens, training):
        x_back_feat, pred_mask_back, similarity_logits_back, ccs_back = self.back_aware(self.back_encoder(x), tokens, training)
        x_objt_feat, pred_mask_objt, similarity_logits_objt, ccs_objt = self.objt_aware(self.objt_encoder(x), tokens, training)
        contrastive = self.reshape_back(x_back_feat) * self.reshape_objt(x_objt_feat)
        output = dict(x_back_feat=x_back_feat, pred_mask_back=pred_mask_back, similarity_logits_back=similarity_logits_back, ccs_back=ccs_back,
                      x_objt_feat=x_objt_feat, pred_mask_objt=pred_mask_objt, similarity_logits_objt=similarity_logits_objt, ccs_objt=ccs_objt,
                      contrastive=contrastive)
        return output
    
class Feature_loss(nn.Module):
    def __init__(self, weight=0.1):
        super(Feature_loss, self).__init__()
        self.weight = weight
        self.diceBCE = DiceBCELoss(1.0)
        self.negEnt = NegEnt(1.0)
        self.contra = contrastive_loss(1.0)

    def forward(self, inputs, mask):
        mask = mask.to(torch.float32)
        x_pred_mask, x_ccs_logit, bac_pred_mask, bac_ccs_logit, contra = inputs
    
        loss_ojct_dice = self.diceBCE(x_pred_mask, mask)
        loss_obct_negEnt = self.negEnt(x_ccs_logit)
        loss_back_dice = self.diceBCE(bac_pred_mask, 1. - mask)
        loss_back_negEnt = self.negEnt(bac_ccs_logit)
        loss_contra = self.contra(contra)

        loss = dict(loss_ojct_dice=loss_ojct_dice, loss_obct_negEnt=loss_obct_negEnt,
                    loss_back_dice=loss_back_dice, loss_back_negEnt=loss_back_negEnt,
                    loss_contra=loss_contra)
        loss_scaled = {key: value * self.weight for key, value in loss.items()}
        return loss_scaled

def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p

class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

class SpatialGate(nn.Module):
    def __init__(self):
        super().__init__()
        kernel_size = 7
        self.compress = ChannelPool()
        self.spatial = Conv(2, 1, kernel_size, 1, (kernel_size-1)//2, act=nn.Sigmoid())

    def forward(self, x):
        x_compress = self.compress(x)
        scale = self.spatial(x_compress)
        return x * scale

class ChannelPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.cat((torch.max(x, 1)[0].unsqueeze(1), torch.mean(x, 1).unsqueeze(1)), dim=1)
        
# iccv version
class TargetAwareFusion(nn.Module):
    # zxue defined fusion module
    def __init__(self, c):
        super(TargetAwareFusion, self).__init__()
        self.cos = torch.nn.functional.cosine_similarity
        self.sigmoid = torch.nn.Sigmoid()
        # self.masklayer = nn.Sequential(
        #                     Conv(c, int(c * 0.5), 3, p=1),
        #                     Conv(int(c * 0.5), int(c * 0.5), 3, p=1),
        #                     Conv(int(c * 0.5), 1, act=False))
        # self.masklayer = nn.Sequential(
        #     nn.Conv2d(c, c//2, kernel_size=3, padding=1, stride=1),
        #     nn.BatchNorm2d(c//2),
        #     nn.ReLU(),
        #     nn.Conv2d(c//2, 1, kernel_size=3, padding=1, stride=1))
        # self.masklayer = MRF3Net(input_channel=c, base_channel=c//8, output_channel=1)
        self.cosVConvLayer = nn.Sequential(
            nn.Conv2d(c, c // 16, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(c // 16, c, 1, bias=False)
        )
        self.spatialAttLayer = SpatialGate()

    def forward(self, x, tokens, training):
        # pred_mask = self.masklayer(x)
        B, C, H, W = x.shape
        tokens = tokens.reshape(B, C, 1, 1).repeat(1, 1, H, W)
        pred_mask = (x * tokens).sum(dim=1, keepdim=True)
        similarity_value_b = self.cos(self.sigmoid(pred_mask).flatten(2), x.flatten(2), 2).unsqueeze(-1).unsqueeze(-1)
        similarity_logits = self.cosVConvLayer(similarity_value_b)
        # similarity_logits = similarity_value_b
        if training:
            f_fusion = self.sigmoid(similarity_logits) * x
        else: 
            f_fusion = x
        f_fusion = self.spatialAttLayer(f_fusion)

        return f_fusion, pred_mask, similarity_logits, similarity_value_b
    
class DiceBCELoss(nn.Module):
    def __init__(self, weight):
        super(DiceBCELoss, self).__init__()
        # self.bce = nn.BCEWithLogitsLoss(reduction='mean')
        self.weight = weight
        
        self.dice_loss = FUSION_LAYERS.build(dict(type='CustomDiceLoss', use_sigmoid=True, loss_weight=1.))
        self.ce_loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([2.13]))  # From lss

    def forward(self, inputs, target, smooth=1):
        # Mitigating instability
        # inputs = torch.nan_to_num(inputs, nan=0.0, posinf=1e6, neginf=-1e6)
        # inputs = torch.clamp(inputs, min=-10, max=10)
        
        # flatten label and prediction tensors
        # inputs = torch.sigmoid(inputs).view(-1)
        # target = target.view(-1).to(torch.float32)
        # intersection = (inputs * targets).sum()
        # dice_loss = 1 - (2. * intersection + smooth) / (inputs.sum() + target.sum() + smooth)

        # BCE = self.bce(inputs.view(-1), target)
        # Dice_BCE = BCE + dice_loss
        bs, _, bev_h, bev_w = target.shape
        b = target.reshape(bs, bev_w * bev_h).permute(1, 0).to(torch.float)
        a = inputs.reshape(bs, bev_w * bev_h).permute(1, 0).to(torch.float)
        
        mask_ce_loss = self.ce_loss(a, b)*self.weight
        mask_dc_loss = self.dice_loss(inputs.reshape(bs, -1), target.reshape(bs, -1))*self.weight
        
        # return Dice_BCE * self.weight
        return mask_dc_loss + mask_ce_loss

class NegEnt(nn.Module):
    def __init__(self, weight):
        super(NegEnt, self).__init__()
        self.weight = weight
        self.bce = nn.BCEWithLogitsLoss(reduction='mean')

    def forward(self, logits):
        logits = torch.nan_to_num(logits, nan=0.0)
        return self.bce(logits, torch.ones_like(logits)) * self.weight

class contrastive_loss(nn.Module):
    def __init__(self, weight):
        super(contrastive_loss, self).__init__()
        self.weight = weight

    def forward(self, inputs):
        bs, ch, w, h = inputs.size()
        return torch.mean(torch.norm(inputs, p=1) / (w * h * ch)) * self.weight


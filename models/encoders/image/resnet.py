
from pyexpat import features
import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights
import torch
import torch.nn.functional as F
import sys
import os
from typing import List, Union, Tuple
import math
import warnings
from itertools import repeat
import collections.abc
# from torch.nn.init import _calculate_fan_in_and_fan_out

def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))
    return parse
to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)
to_3tuple = _ntuple(3)
to_4tuple = _ntuple(4)
to_ntuple = _ntuple

class AttentionPool2d(nn.Module):
    """ Attention based 2D feature pooling w/ learned (absolute) pos embedding.
    This is a multi-head attention based replacement for (spatial) average pooling in NN architectures.
    It was based on impl in CLIP by OpenAI
    https://github.com/openai/CLIP/blob/3b473b0e682c091a9e53623eebc1ca1657385717/clip/model.py
    NOTE: This requires feature size upon construction and well prevent adaptive sizing of the network.
    """
    def __init__(
            self,
            in_features: int,
            feat_size: Union[int, Tuple[int, int]],
            out_features: int = None,
            embed_dim: int = None,
            num_heads: int = 4,
            qkv_bias: bool = True,
    ):
        super().__init__()

        embed_dim = embed_dim or in_features
        out_features = out_features or in_features
        assert embed_dim % num_heads == 0
        self.feat_size = to_2tuple(feat_size)
        self.qkv = nn.Linear(in_features, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, out_features)
        self.num_heads = num_heads # 8
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        spatial_dim = self.feat_size[0] * self.feat_size[1]
        self.pos_embed = nn.Parameter(torch.zeros(spatial_dim + 1, in_features))# 7*7+1, 2048
        trunc_normal_(self.pos_embed, std=in_features ** -0.5)
        trunc_normal_(self.qkv.weight, std=in_features ** -0.5)
        nn.init.zeros_(self.qkv.bias)

    def forward(self, x):
        B, _, H, W = x.shape # [1, 2048, 7, 7]
        N = H * W
        assert self.feat_size[0] == H
        assert self.feat_size[1] == W
        x = x.reshape(B, -1, N).permute(0, 2, 1) # [1, 49, 2048]
        x = torch.cat([x.mean(1, keepdim=True), x], dim=1) # [1, 1+49, 2048]
        x = x + self.pos_embed.unsqueeze(0).to(x.dtype)
        # attention
        x = self.qkv(x).reshape(B, N + 1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4) # [3, 1, 8, 50, 256]
        q, k, v = x[0], x[1], x[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale 
        attn = attn.softmax(dim=-1)# [1, 8, 50, 50]
        # attn @ v: [1, 8, 50, 256]
        x = (attn @ v).transpose(1, 2).reshape(B, N + 1, -1) # [1, 50, 2048]
        x = self.proj(x)
        return x[:, 0] # [1, 2048]
    

# class AttentionPool2d(nn.Module):
#     def __init__(self, spacial_dim: int, embed_dim: int, num_heads: int, output_dim: int = None):
#         super().__init__()
#         self.positional_embedding = nn.Parameter(torch.randn(spacial_dim ** 2 + 1, embed_dim) / embed_dim ** 0.5)
#         self.k_proj = nn.Linear(embed_dim, embed_dim)
#         self.q_proj = nn.Linear(embed_dim, embed_dim)
#         self.v_proj = nn.Linear(embed_dim, embed_dim)
#         self.c_proj = nn.Linear(embed_dim, output_dim or embed_dim)
#         self.num_heads = num_heads

#     def forward(self, x):
#         x = x.reshape(x.shape[0], x.shape[1], x.shape[2] * x.shape[3]).permute(2, 0, 1)  # NCHW -> (HW)NC
#         x = torch.cat([x.mean(dim=0, keepdim=True), x], dim=0)  # (HW+1)NC
#         x = x + self.positional_embedding[:, None, :].to(x.dtype)  # (HW+1)NC
#         x, _ = F.multi_head_attention_forward(
#             query=x, key=x, value=x,
#             embed_dim_to_check=x.shape[-1],
#             num_heads=self.num_heads,
#             q_proj_weight=self.q_proj.weight,
#             k_proj_weight=self.k_proj.weight,
#             v_proj_weight=self.v_proj.weight,
#             in_proj_weight=None,
#             in_proj_bias=torch.cat([self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
#             bias_k=None,
#             bias_v=None,
#             add_zero_attn=False,
#             dropout_p=0,
#             out_proj_weight=self.c_proj.weight,
#             out_proj_bias=self.c_proj.bias,
#             use_separate_proj_weight=True,
#             training=self.training,
#             need_weights=False
#         )
#         return x[0]


class ResNet(nn.Module):
    def __init__(self, in_channels=1, name=None, pool_method='mean', pretrained=True) -> None:
        super().__init__()
        self.name = name
        if name == 'resnet50':
            self.encoder = resnet50_224(in_channels=in_channels, pretrained=pretrained)
        else:
            raise NotImplementedError
        if pool_method == 'mean':
            self.pool = lambda x: torch.mean(x, dim=[2, 3])
        else:
            self.pool = lambda x: x
        self._modify_forward()
        

    def _modify_forward(self):
        if self.name == 'resnet50':
            def forward_wrapper(x, return_features=False):
                features = []
                x = self.encoder.conv1(x)
                x = self.encoder.bn1(x)
                x = self.encoder.relu(x)
                if return_features:
                    features.append(x)
                x = self.encoder.maxpool(x)
                x = self.encoder.layer1(x)
                if return_features:
                    features.append(x)
                x = self.encoder.layer2(x)
                if return_features:
                    features.append(x)
                x = self.encoder.layer3(x)
                if return_features:
                    features.append(x)
                x = self.encoder.layer4(x)
                local_features = x
                if return_features:
                    features.append(x)
                global_x = self.pool(x)
                global_x = global_x.view(global_x.size(0), -1)
                # local_features = local_features.view(local_features.shape[0], -1, local_features.shape[1])
                if return_features:
                    return local_features,  global_x, features
                return local_features, global_x
        else:
            raise NotImplementedError
        self.encoder.forward = forward_wrapper

        
    def forward(self, x, return_features=False):
        try:
            return self.encoder.forward_features(x)
        except:
            return self.encoder.forward(x, return_features=return_features)
    

    def get_global_width(self):
        try:
            return self.encoder.num_features
        except:
            return 512 * 4

    def get_width(self):
        try:
            return self.encoder.num_features 
        except:
            return 512 * 4 

    def get_local_width(self):
        try:
            return self.encoder.num_features 
        except:
            return 512 * 4 
        
    def get_name(self):
        return self.name
    
    def get_last_spatial_info(self):
        if  self.name == 'resnet50':
            return [7, 7]

          

def resnet50_224(in_channels=3, pretrained=False):
    if pretrained:
        model =  resnet50(weights=ResNet50_Weights.DEFAULT)
    else:
        model =  resnet50()
        
    if in_channels != 3:
        old_conv = model.conv1
        out_channels = model.conv1.out_channels
        model.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=7, stride=2, padding=3,
                               bias=False)
        model.conv1.weight = torch.nn.Parameter(old_conv.weight.sum(dim=1, keepdim=True))
        model.fc = nn.Identity()
    return model 


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.",
                      stacklevel=2)

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


if __name__ == '__main__':
    # model = ResNet(name='resnet50', in_channels=1, pretrained=True, pool_method='attention', img_size=224)
    model = ResNet(name='resnet50', in_channels=1, pretrained=True, pool_method='mean')
    CHECKPOINT = '/media/brownradai/ssd_2t/covid_cxr/region_surv/checkpoints/prior_resnet50.pt'
    checkpoint = torch.load(CHECKPOINT, map_location=torch.device("cpu"))
    from collections import OrderedDict
    new_state_dict = OrderedDict()        
    for k, v in checkpoint.items():
        new_state_dict['encoder.'+k] = v
    model.load_state_dict(new_state_dict)
    
    
    # model.load_state_dict(checkpoint)
    x = torch.randn(1, 1, 224, 224)
    local_features, global_x = model(x)    
    local_features, global_x, features = model.encoder(x, return_features=True)
    for f in features:
        print(f.shape)
        
    global_attention_pooler = AttentionPool2d(in_features=2048, feat_size=7, embed_dim=2048, num_heads=8)
    
    global_features = global_attention_pooler(local_features)
# ------------------------------------------------------------------------------------------------
# Deformable DETR
# Copyright (c) 2020 SenseTime. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------------------------------
# Modified from https://github.com/chengdazhi/Deformable-Convolution-V2-PyTorch/tree/pytorch_1.0.0
# ------------------------------------------------------------------------------------------------

from __future__ import absolute_import
from __future__ import print_function
from __future__ import division

import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction
from torch.nn.functional import grid_sample
import numpy as np
from utils.misc import inverse_sigmoid

def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0

def dynamic_local_filtering(x, depth, dilated=1): # x:(B, n, 256)
    padding = nn.ReflectionPad2d(dilated)  # ConstantPad2d(1, 0)
    pad_depth = padding(depth)
    n, c, h, w = x.size()
    y = torch.cat((x[:, -1:, :, :], x[:, :-1, :, :]), dim=1)
    z = torch.cat((x[:, -2:, :, :], x[:, :-2, :, :]), dim=1)
    x = (x + y + z) / 3
    pad_x = padding(x)
    filter = (pad_depth[:, :, dilated: dilated + h, dilated: dilated + w] * pad_x[:, :, dilated: dilated + h, dilated: dilated + w]).clone()
    for i in [-dilated, 0, dilated]:
        for j in [-dilated, 0, dilated]:
            if i != 0 or j != 0:
                filter += (pad_depth[:, :, dilated + i: dilated + i + h, dilated + j: dilated + j + w] *
                           pad_x[:, :, dilated + i: dilated + i + h, dilated + j: dilated + j + w]).clone()
    return filter / 9

class MSDeformAttn(nn.Module):
    def __init__(self, d_model=256, n_levels=4, n_heads=8, n_points=4):
        """
        Multi-Scale Deformable Attention Module
        :param d_model      hidden dimension
        :param n_levels     number of feature levels
        :param n_heads      number of attention heads
        :param n_points     number of sampling points per attention head per feature level
        """
        super().__init__()
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        _d_per_head = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(_d_per_head):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")

        self.im2col_step = 64

        self.d_model = d_model
        self.n_levels = n_levels
        self.n_heads = n_heads
        self.n_points = n_points

        self.sampling_offsets = nn.Linear(d_model, n_heads * n_levels * n_points * 2)
        self.attention_weights = nn.Linear(d_model, n_heads * n_levels * n_points)
        self.value_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)

        self._reset_parameters()

        self.adaptive_softmax = nn.Softmax(dim=-1)
        self.adaptive_layers = nn.Sequential(
            nn.AdaptiveMaxPool2d(3),
            nn.Conv2d(256, 256 * 3, 3, padding=0),
        )
        self.my_pool = nn.AdaptiveMaxPool2d(5)
        self.my_adaptive = nn.Linear(256, 5)
        self.my_adaptive_d = nn.Linear(256, 1)
        self.conv0 = torch.nn.Conv1d(256, 256, 1)
        self.conv1 = torch.nn.Conv1d(512, 256, 1)
        self.bn1 = torch.nn.BatchNorm1d(256)
        self.adaptive_bn = nn.BatchNorm1d(256)
        self.adaptive_relu = nn.ReLU(inplace=True)
        self.my_multi_f1 = nn.Linear(9, 1)
        self.my_multi_f2 = nn.Linear(25, 1)
        self.my_multi_f3 = nn.Linear(49, 1)
        self.my_multi_f4 = nn.Linear(81, 1)

    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)

    def depth_guided_offset(self, img, dep, loc, q):
        dep1 = self.interpolate_img_by_xy_d(dep, loc)
        weight1 = self.my_adaptive(dep1)
        weight = self.adaptive_softmax(weight1)
        img0, img1, img2, img3, img4 = self.interpolate_img_by_xy(img, loc)
        final_weight, index = torch.max(weight, -1)
        index1 = np.eye(5)[index.cpu().numpy().astype(np.int64)]
        index2 = torch.from_numpy(index1.astype(np.float32)).cuda()
        img_all = img0 * index2[:, :, 0:1] + img1 * index2[:, :, 1:2] + img2 * index2[:, :, 2:3] + img3 * index2[:, :,3:4] + img4 * index2[:, :,4:5]
        f0 = F.relu(self.bn1(self.conv0(img_all.permute(0, 2, 1)))).permute(0, 2, 1)
        f1 = f0 * q
        return f1, weight

    def interpolate_img_by_xy_d(self, img, xy):
        """
        :param img:(H,W,c)
        :param xy:(N,2) (x,y)->(w,h) N=50
        :param normal_shape:(2),H_size and W_size
        :return:interpolated features (N,3)
        """
        # (B,C,H,W)
        xy1 = xy.cpu().numpy().copy()
        normal_shape = np.array([80., 24.])
        xy1[:, :, 0] *= 80.
        xy1[:, :, 1] *= 24.
        xy11 = torch.from_numpy(np.float32(xy1 * 2 / (normal_shape - 1.) - 1.)).unsqueeze(1).cuda()
        ret_img = grid_sample(img, xy11, padding_mode='reflection', mode='bilinear')
        ret_img = ret_img.squeeze(2).permute(0, 2, 1)
        return ret_img

    def auto_gen_kernel(self, l):
        p_list=[]
        for i in range(-l, l+1):
            for j in range(-l, l+1):
                p_list.append((i,j))
        return np.array(p_list)

    def interpolate_img_by_xy(self, img, xy):
        """
        :param img:(H,W,c)
        :param xy:(N,2) (x,y)->(w,h) N=50
        :param normal_shape:(2),H_size and W_size
        :return:interpolated features (N,3)
        """
        # (B,C,H,W)
        xy0 = xy.cpu().numpy().copy()
        normal_shape = np.array([80., 24.])
        xy0[:, :, 0] *= 80
        xy0[:, :, 1] *= 24

        k1 = self.auto_gen_kernel(1) # 3x3
        k2 = self.auto_gen_kernel(2) # 5x5
        k3 = self.auto_gen_kernel(3) # 7x7
        k4 = self.auto_gen_kernel(4) # 9x9

        xy1 = np.expand_dims(xy0, 2).repeat(k1.shape[0], 2)
        xy11 = xy1[:, :,] + k1
        xy2 = np.expand_dims(xy0, 2).repeat(k2.shape[0], 2)
        xy22 = xy2[:, :, ] + k2
        xy3 = np.expand_dims(xy0, 2).repeat(k3.shape[0], 2)
        xy33 = xy3[:, :, ] + k3
        xy4 = np.expand_dims(xy0, 2).repeat(k4.shape[0], 2)
        xy44 = xy4[:, :, ] + k4

        xy000 = torch.from_numpy(np.float32(np.expand_dims(xy0, 2) * 2 / (normal_shape - 1.) - 1.)).cuda()
        xy111 = torch.from_numpy(np.float32(xy11 * 2 / (normal_shape - 1.) - 1.)).cuda()
        xy222 = torch.from_numpy(np.float32(xy22 * 2 / (normal_shape - 1.) - 1.)).cuda()
        xy333 = torch.from_numpy(np.float32(xy33 * 2 / (normal_shape - 1.) - 1.)).cuda()
        xy444 = torch.from_numpy(np.float32(xy44 * 2 / (normal_shape - 1.) - 1.)).cuda()

        ret_img0 = grid_sample(img, xy000, padding_mode='reflection', mode='bilinear').permute(0, 2, 1, 3)
        ret_img00 = torch.sum(ret_img0, -1)
        ret_img1 = grid_sample(img, xy111, padding_mode='reflection', mode='bilinear').permute(0,2,1,3)
        ret_img11 = torch.sum(ret_img1, -1)/9
        ret_img2 = grid_sample(img, xy222, padding_mode='reflection', mode='bilinear').permute(0,2,1,3)
        ret_img22 = torch.sum(ret_img2, -1)/25
        ret_img3 = grid_sample(img, xy333, padding_mode='reflection', mode='bilinear').permute(0,2,1,3)
        ret_img33 = torch.sum(ret_img3, -1)/49
        ret_img4 = grid_sample(img, xy444, padding_mode='reflection', mode='bilinear').permute(0, 2, 1, 3)
        ret_img44 = torch.sum(ret_img4, -1)/81

        return ret_img00, ret_img11, ret_img22, ret_img33, ret_img44

    def forward(self, query, reference_points, input_flatten, s1, depth_map, input_spatial_shapes, input_level_start_index, input_padding_mask=None):
        """
        :param query                       (N, Length_{query}, C)
        :param reference_points            (N, Length_{query}, n_levels, 2), range in [0, 1], top-left (0,0), bottom-right (1, 1), including padding area
                                        or (N, Length_{query}, n_levels, 4), add additional (w, h) to form reference boxes
        :param input_flatten               (N, \sum_{l=0}^{L-1} H_l \cdot W_l, C)
        :param input_spatial_shapes        (n_levels, 2), [(H_0, W_0), (H_1, W_1), ..., (H_{L-1}, W_{L-1})]
        :param input_level_start_index     (n_levels, ), [0, H_0*W_0, H_0*W_0+H_1*W_1, H_0*W_0+H_1*W_1+H_2*W_2, ..., H_0*W_0+H_1*W_1+...+H_{L-1}*W_{L-1}]
        :param input_padding_mask          (N, \sum_{l=0}^{L-1} H_l \cdot W_l), True for padding elements, False for non-padding elements

        :return output                     (N, Length_{query}, C)
        """
        N, Len_q, _ = query.shape
        N, Len_in, _ = input_flatten.shape
        assert (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum() == Len_in

        value0 = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value0.masked_fill(input_padding_mask[..., None], float(0))
        value = value0.view(N, Len_in, self.n_heads, self.d_model // self.n_heads)
        attention_weights = self.attention_weights(query).view(N, Len_q, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, -1).view(N, Len_q, self.n_heads, self.n_levels, self.n_points)
        # N, Len_q, n_heads, n_levels, n_points, 2
        if reference_points.shape[-1] == 2:
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_offsets = self.sampling_offsets(query).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
            sampling_locations = reference_points[:, :, None, :, None, :] + sampling_offsets / offset_normalizer[None, None, None, :, None, :]
            my_weight=None
        elif reference_points.shape[-1] == 6:
            fusion_f, my_weight = self.depth_guided_offset(s1[1], depth_map, reference_points[:,:,0,:2], query) # (B, 256, 24, 80) (B, 50, 4, 6)
            sampling_offsets = self.sampling_offsets(fusion_f).view(N, Len_q, self.n_heads, self.n_levels, self.n_points, 2)
            a = reference_points[:, :, None, :, None, 2::2]
            b = reference_points[:, :, None, :, None, 3::2]
            sampling_locations = reference_points[:, :, None, :, None, :2] + sampling_offsets / self.n_points * (a + b) * 0.5


        else:
            raise ValueError('Last dim of reference_points must be 2 or 4, but get {} instead.'.format(reference_points.shape[-1]))

        output = MSDeformAttnFunction.apply(
            value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights,
            self.im2col_step)
        output = self.output_proj(output)
        return output, my_weight

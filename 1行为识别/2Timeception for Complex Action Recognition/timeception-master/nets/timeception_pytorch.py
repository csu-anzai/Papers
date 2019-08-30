#!/usr/bin/env python
# -*- coding: UTF-8 -*-

########################################################################
# GNU General Public License v3.0
# GNU GPLv3
# Copyright (c) 2019, Noureldien Hussein
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
########################################################################

"""
Definitio of Timeception as pytorch model.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import logging

import torch
import torch.nn
import torchvision
import torchviz
import torchsummary

from torch.nn import Module, Conv3d, BatchNorm3d, MaxPool3d, ReLU
from torch.nn import functional as F

from nets.layers_pytorch import ChannelShuffleLayer, DepthwiseConv1DLayer #这个是自定义的操作

# region Timeception as Module

class Timeception(Module):
    """
    Timeception is defined as a keras model.
    """

    def __init__(self, input_shape, n_layers=4, n_groups=8, is_dilated=True):
        '''
         self.timeception = timeception_pytorch.Timeception(input_shape, n_tc_layers, n_groups, is_dilated)
        :param input_shape: (32, 1024, 128, 7, 7)
        :param n_layers: 2
        :param n_groups: 8
        :param is_dilated: True
        '''

        super(Timeception, self).__init__()

        # TODO: Add support for multi-scale using dilation rates
        # current, for pytorch, we only support multi-scale using kernel sizes，源代码目前只支持多尺度kernel_sizes
        is_dilated = False

        expansion_factor = 1.25
        self.expansion_factor = expansion_factor
        self.n_layers = n_layers #2
        self.is_dilated = is_dilated
        self.n_groups = n_groups
        self.n_channels_out = None

        # convert it as a list
        input_shape = list(input_shape)

        # define timeception layers
        # 在这里已经将需要的操作都进行了定义，包括timeception层的内部具体操作，返回该层timeception的输出通道的个数，该通道数将被作为下一层timeception的输入通道数
        n_channels_out = self.__define_timeception_layers(input_shape, n_layers, n_groups, expansion_factor, is_dilated)

        # set the output channels
        self.n_channels_out = n_channels_out

    def forward(self, input):

        n_layers = self.n_layers #2
        n_groups = self.n_groups #8
        expansion_factor = self.expansion_factor #1.25

        output = self.__call_timeception_layers(input, n_layers, n_groups, expansion_factor)

        return output

    def __define_timeception_layers(self, input_shape, n_layers, n_groups, expansion_factor, is_dilated):
        '''
        Define layers inside the timeception layers. 定义timeception层的内部操作结构
        :param input_shape: (32, 1024, 128, 7, 7)：1024输入的通道数
        :param n_layers: 2
        :param n_groups: 8
        :param expansion_factor: 1.25
        :param is_dilated: True
        :return: n_channels_in :返回该层timeception的输出通道数，作为下一层timeception的输入通道数
        '''
        n_channels_in = input_shape[1]

        # how many layers of timeception
        for i in range(n_layers):
            # i表示层数
            layer_num = i + 1

            # get details about grouping
            n_channels_per_branch, n_channels_out = self.__get_n_channels_per_branch(n_groups, expansion_factor, n_channels_in)

            # temporal conv per group，在这里将timeception中用到的操作进行了定义
            self.__define_grouped_convolutions(input_shape, n_groups, n_channels_per_branch, is_dilated, layer_num)

            # downsample over time，定义在时间域上的下采样操作，即在时间域上进行最大池化操作，来满足第二个子空间平衡的原则
            layer_name = 'maxpool_tc%d' % (layer_num)
            layer = MaxPool3d(kernel_size=(2, 1, 1))
            layer._name = layer_name
            setattr(self, layer_name, layer)

            n_channels_in = n_channels_out
            input_shape[1] = n_channels_in #下一层输入时的通道个数[1280, 1600, 2000, 2480]

        return n_channels_in

    def __define_grouped_convolutions(self, input_shape, n_groups, n_channels_per_branch, is_dilated, layer_num):
        '''
        Define layers inside grouped convolutional block. 定义timeception中的操作
        :param input_shape: (32, 1024, 128, 7, 7)：1024输入的通道数
        :param n_groups: 8
        :param n_channels_per_branch: [32,40,50,62]
        :param is_dilated:
        :param layer_num: 当前的层数
        :return:
        '''

        n_channels_in = input_shape[1]

        n_branches = 5
        n_channels_per_group_in = int(n_channels_in / n_groups) #每个group输入通道的个数 = 128
        n_channels_out = int(n_groups * n_branches * n_channels_per_branch) #整个temporal层的输出通道数
        n_channels_per_group_out = int(n_channels_out / n_groups) #每个group的输出通道的个数

        assert n_channels_in % n_groups == 0 #断言表达式，否的话抛出异常
        assert n_channels_out % n_groups == 0

        # type of multi-scale kernels to use: either multi_kernel_sizes or multi_dilation_rates
        # 设置膨胀与核大小已应对多变的复杂动作时长
        if is_dilated:
            kernel_sizes = (3, 3, 3)
            dilation_rates = (1, 2, 3)
        else:
            kernel_sizes = (3, 5, 7)
            dilation_rates = (1, 1, 1)

        input_shape_per_group = list(input_shape)
        input_shape_per_group[1] = n_channels_per_group_in #(32, 128, 128, 7, 7)

        # loop on groups, and define convolutions in each group
        # 对每一层的每一个group进行定义具体的操作
        for idx_group in range(n_groups):
            group_num = idx_group + 1 #当前的group数
            #定义每个group的操作
            self.__define_temporal_convolutional_block(input_shape_per_group, n_channels_per_branch, kernel_sizes, dilation_rates, layer_num, group_num)

        # activation 定义激活操作
        layer_name = 'relu_tc%d' % (layer_num)
        layer = ReLU()
        layer._name = layer_name
        setattr(self, layer_name, layer)

        # shuffle channels 定义混洗操作
        layer_name = 'shuffle_tc%d' % (layer_num)
        layer = ChannelShuffleLayer(n_channels_out, n_groups)
        layer._name = layer_name
        setattr(self, layer_name, layer)

    def __define_temporal_convolutional_block(self, input_shape, n_channels_per_branch_out, kernel_sizes, dilation_rates, layer_num, group_num):
        '''
        Define 5 branches of convolutions that operate of channels of each group.
        定义每组通道的具体处理操作
        :param input_shape: (32, 128, 128, 7, 7)：1024输入的通道数
        :param n_channels_per_branch_out: [32,40,50,62]
        :param kernel_sizes: dilated-[3,3,3],no dilated-[3,5,7]
        :param dilation_rates: dilated-[1,2,3],no dilated-[1,1,1]
        :param layer_num: 当前的层数
        :param group_num: 当前的group数
        :return: 定义temporal卷积的block
        '''

        n_channels_in = input_shape[1]

        dw_input_shape = list(input_shape)
        dw_input_shape[1] = n_channels_per_branch_out #[32,40,50,62]

        # setattr()函数对应函数getattr()，用于设置属性值，该属性不一定是存在的
        '''
        示例：
        layer_name = 'conv_b1_g%d_tc%d' % (1, 1)
        layer_name
        Out[13]: 'conv_b1_g1_tc1'
        '''
        # branch 1: dimension reduction only and no temporal conv (kernel-size 1)
        layer_name = 'conv_b1_g%d_tc%d' % (group_num, layer_num)
        layer = Conv3d(n_channels_in, n_channels_per_branch_out, kernel_size=(1, 1, 1))
        layer._name = layer_name
        setattr(self, layer_name, layer)
        layer_name = 'bn_b1_g%d_tc%d' % (group_num, layer_num)
        layer = BatchNorm3d(n_channels_per_branch_out)
        layer._name = layer_name
        setattr(self, layer_name, layer)

        # branch 2: dimension reduction followed by depth-wise temp conv (kernel-size 3)
        # 缩减通道
        layer_name = 'conv_b2_g%d_tc%d' % (group_num, layer_num)
        layer = Conv3d(n_channels_in, n_channels_per_branch_out, kernel_size=(1, 1, 1))
        # layer = Conv3d(n_channels_in = 128, n_channels_per_branch_out = 32, kernel_size=(1, 1, 1))
        layer._name = layer_name
        setattr(self, layer_name, layer)

        #卷积
        layer_name = 'convdw_b2_g%d_tc%d' % (group_num, layer_num)
        layer = DepthwiseConv1DLayer(dw_input_shape, kernel_sizes[0], dilation_rates[0], layer_name)
        # layer = DepthwiseConv1DLayer(dw_input_shape=(32, 32, 128, 7, 7) , kernel_sizes[0], dilation_rates[0], layer_name)
        setattr(self, layer_name, layer)

        #BN操作
        layer_name = 'bn_b2_g%d_tc%d' % (group_num, layer_num)
        layer = BatchNorm3d(n_channels_per_branch_out)
        layer._name = layer_name
        setattr(self, layer_name, layer)

        # branch 3: dimension reduction followed by depth-wise temp conv (kernel-size 5)
        layer_name = 'conv_b3_g%d_tc%d' % (group_num, layer_num)
        layer = Conv3d(n_channels_in, n_channels_per_branch_out, kernel_size=(1, 1, 1))
        layer._name = layer_name
        setattr(self, layer_name, layer)
        layer_name = 'convdw_b3_g%d_tc%d' % (group_num, layer_num)
        layer = DepthwiseConv1DLayer(dw_input_shape, kernel_sizes[1], dilation_rates[1], layer_name)
        setattr(self, layer_name, layer)
        layer_name = 'bn_b3_g%d_tc%d' % (group_num, layer_num)
        layer = BatchNorm3d(n_channels_per_branch_out)
        layer._name = layer_name
        setattr(self, layer_name, layer)

        # branch 4: dimension reduction followed by depth-wise temp conv (kernel-size 7)
        layer_name = 'conv_b4_g%d_tc%d' % (group_num, layer_num)
        layer = Conv3d(n_channels_in, n_channels_per_branch_out, kernel_size=(1, 1, 1))
        layer._name = layer_name
        setattr(self, layer_name, layer)
        layer_name = 'convdw_b4_g%d_tc%d' % (group_num, layer_num)
        layer = DepthwiseConv1DLayer(dw_input_shape, kernel_sizes[2], dilation_rates[2], layer_name)
        setattr(self, layer_name, layer)
        layer_name = 'bn_b4_g%d_tc%d' % (group_num, layer_num)
        layer = BatchNorm3d(n_channels_per_branch_out)
        layer._name = layer_name
        setattr(self, layer_name, layer)

        # branch 5: dimension reduction followed by temporal max pooling
        layer_name = 'conv_b5_g%d_tc%d' % (group_num, layer_num)
        layer = Conv3d(n_channels_in, n_channels_per_branch_out, kernel_size=(1, 1, 1))
        layer._name = layer_name
        setattr(self, layer_name, layer)
        layer_name = 'maxpool_b5_g%d_tc%d' % (group_num, layer_num)
        layer = MaxPool3d(kernel_size=(2, 1, 1), stride=(1, 1, 1))
        layer._name = layer_name
        setattr(self, layer_name, layer)
        layer_name = 'padding_b5_g%d_tc%d' % (group_num, layer_num)
        layer = torch.nn.ReplicationPad3d((0, 0, 0, 0, 1, 0))  # left, right, top, bottom, front, back
        layer._name = layer_name
        setattr(self, layer_name, layer)
        layer_name = 'bn_b5_g%d_tc%d' % (group_num, layer_num)
        layer = BatchNorm3d(n_channels_per_branch_out)
        layer._name = layer_name
        setattr(self, layer_name, layer)

    def __call_timeception_layers(self, tensor, n_layers, n_groups, expansion_factor):
        # output = self.__call_timeception_layers(input, n_layers, n_groups, expansion_factor)
        input_shape = tensor.size() #原始的输入
        n_channels_in = input_shape[1]

        # how many layers of timeception
        for i in range(n_layers):
            layer_num = i + 1 #第几层

            # get details about grouping
            n_channels_per_branch, n_channels_out = self.__get_n_channels_per_branch(n_groups, expansion_factor, n_channels_in)
            '''
            n_channels_per_branch：得到每个batch要输入的通道数
            n_groups = 8, expansion_factor = 1.25, n_channels_in = 1024
            n_channels_per_branch|每个brach的输入: [32,40,50,62], 
            n_channels_out|该层timeception的输出通道的个数: [1280, 1600, 2000, 2480]
            '''

            # temporal conv per group 每个group的时域卷积
            tensor = self.__call_grouped_convolutions(tensor, n_groups, layer_num)

            # downsample over time
            tensor = getattr(self, 'maxpool_tc%d' % (layer_num))(tensor)
            n_channels_in = n_channels_out

        return tensor

    def __call_grouped_convolutions(self, tensor_input, n_groups, layer_num):
        '''
        对某一层的timeception进行处理，在这里面是对group进行循环
        :param tensor_input: (32, 1024, 128, 7, 7)
        :param n_groups: 8
        :param layer_num: 4
        :return: 返回该层timeception的输出
        '''

        n_channels_in = tensor_input.size()[1]
        n_channels_per_group_in = int(n_channels_in / n_groups) #每个group的通道：128个channels

        # loop on groups
        t_outputs = []
        for idx_group in range(n_groups):
            #对每个group进行操作
            group_num = idx_group + 1 #第几个group

            # slice maps to get maps per group
            idx_start = idx_group * n_channels_per_group_in
            idx_end = (idx_group + 1) * n_channels_per_group_in
            tensor = tensor_input[:, idx_start:idx_end] #取出第group_num个输入(32, 128, 128, 7, 7)

            tensor = self.__call_temporal_convolutional_block(tensor, layer_num, group_num)

            t_outputs.append(tensor)

        # concatenate channels of groups 将该层timeception的group输出进行串联
        tensor = torch.cat(t_outputs, dim=1)
        # activation 激活
        tensor = getattr(self, 'relu_tc%d' % (layer_num))(tensor)
        # shuffle channels 混洗
        tensor = getattr(self, 'shuffle_tc%d' % (layer_num))(tensor)

        return tensor

    def __call_temporal_convolutional_block(self, tensor, layer_num, group_num):
        '''
        将某一层的某一个group的tensor传进来进行处理
        Feedforward for 5 branches of convolutions that operate of channels of each group.
        5个分支的前向传播
        :param tensor: (32, 128, 128, 7, 7) #某一层的某一个group的tensor
        :param layer_num: 4
        :param group_num: 第几个，当前的分组group数
        :return: 返回该group的处理结果
        '''

        # layer_num in range(4)+1
        # group_num in range(5)+1
        # getattr() 函数用于返回一个对象属性值
        # 'conv_b1_g%d_tc%d' % (1, 1) -->  'conv_b1_g1_tc1'
        # branch 1: dimension reduction only and no temporal conv
        t_1 = getattr(self, 'conv_b1_g%d_tc%d' % (group_num, layer_num))(tensor) #第group_num组，第layer_num层#
        # getattr()得到对象self的'conv_b1_g%d_tc%d'的属性，得到类似于Conv3d(250, 62, kernel_size=(1, 1, 1), stride=(1, 1, 1))的结果，由上面的setattr()函数设置
        t_1 = getattr(self, 'bn_b1_g%d_tc%d' % (group_num, layer_num))(t_1)

        # branch 2: dimension reduction followed by depth-wise temp conv (kernel-size 3)
        t_2 = getattr(self, 'conv_b2_g%d_tc%d' % (group_num, layer_num))(tensor)
        t_2 = getattr(self, 'convdw_b2_g%d_tc%d' % (group_num, layer_num))(t_2)
        t_2 = getattr(self, 'bn_b2_g%d_tc%d' % (group_num, layer_num))(t_2)

        # branch 3: dimension reduction followed by depth-wise temp conv (kernel-size 5)
        t_3 = getattr(self, 'conv_b3_g%d_tc%d' % (group_num, layer_num))(tensor)
        t_3 = getattr(self, 'convdw_b3_g%d_tc%d' % (group_num, layer_num))(t_3)
        t_3 = getattr(self, 'bn_b3_g%d_tc%d' % (group_num, layer_num))(t_3)

        # branch 4: dimension reduction followed by depth-wise temp conv (kernel-size 7)
        t_4 = getattr(self, 'conv_b4_g%d_tc%d' % (group_num, layer_num))(tensor)
        t_4 = getattr(self, 'convdw_b4_g%d_tc%d' % (group_num, layer_num))(t_4)
        t_4 = getattr(self, 'bn_b4_g%d_tc%d' % (group_num, layer_num))(t_4)

        # branch 5: dimension reduction followed by temporal max pooling
        t_5 = getattr(self, 'conv_b5_g%d_tc%d' % (group_num, layer_num))(tensor)
        t_5 = getattr(self, 'maxpool_b5_g%d_tc%d' % (group_num, layer_num))(t_5)
        t_5 = getattr(self, 'padding_b5_g%d_tc%d' % (group_num, layer_num))(t_5)
        t_5 = getattr(self, 'bn_b5_g%d_tc%d' % (group_num, layer_num))(t_5)

        # concatenate channels of branches
        tensors = (t_1, t_2, t_3, t_4, t_5)
        tensor = torch.cat(tensors, dim=1)

        return tensor

    def __get_n_channels_per_branch(self, n_groups, expansion_factor, n_channels_in):
        '''
        :param n_groups: 8
        :param expansion_factor: 1.25
        :param n_channels_in: 1024，当前整个timeception层的输入通道的个数
        :return: n_channels_per_branch|每个brach的输入: [32,40,50,62], n_channels_out|该层timeception的输出通道的个数: [1280, 1600, 2000, 2480]
        '''
        n_branches = 5
        n_channels_per_branch = int(n_channels_in * expansion_factor / float(n_branches * n_groups))

        n_channels_per_branch = int(n_channels_per_branch) #[32,40,50,62]

        n_channels_out = int(n_channels_per_branch * n_groups * n_branches)
        n_channels_out = int(n_channels_out) #[1280, 1600, 2000, 2480]

        return n_channels_per_branch, n_channels_out

# endregion

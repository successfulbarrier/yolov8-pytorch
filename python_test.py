# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   python_test.py
# @Time    :   2024/09/02 14:26:36
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   测试模块
import sys
sys.path.append("/media/lht/LHT/DL_code/yolov8-pytorch-my")
import torch
from nets.backbone import ChannelSplitModule

#-------------------------------------------------#
#   main函数
#-------------------------------------------------#
if __name__ == '__main__':
    # 生成一个大小为[1,192,80,80]的tensor矩阵
    input_tensor = torch.rand(1, 192, 80, 80)
    # 创建ChannelSplitModule实例
    channel_split_module = ChannelSplitModule(input_channels=192, feat_size=80)
    # 将输入tensor传入ChannelSplitModule
    output = channel_split_module(input_tensor)
    # 打印输出结果的大小
    print("输出结果的大小:", output.size())
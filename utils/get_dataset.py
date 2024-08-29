# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   get_dataset.py
# @Time    :   2024/08/29 20:11:56
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   制作经过DCT变换的数据集，离线制作好数据集，加快训练速度

import os
import cv2
from tqdm import tqdm
import numpy as np

#-------------------------------------------------#
#   参数
#-------------------------------------------------#
image_size = (640, 640)

#-------------------------------------------------#
#   以存在路径，获取现有数据集中的图像数据
#-------------------------------------------------#
yolo_dataset_root_path = "/media/lht/D_Project/datasets/TS_dataset5"
yolo_dataset_train_path = os.path.join(yolo_dataset_root_path, "train")
yolo_dataset_val_path = os.path.join(yolo_dataset_root_path, "val")
yolo_dataset_train_images_path = os.path.join(yolo_dataset_train_path, "images")
yolo_dataset_val_images_path = os.path.join(yolo_dataset_val_path, "images")

#-------------------------------------------------#
#   县增加的数据，经过DCT变换之后的数据[W/8,H/8,192]
#-------------------------------------------------#
yolo_dataset_train_dct_path = os.path.join(yolo_dataset_train_path, "dct")
yolo_dataset_val_dct_path = os.path.join(yolo_dataset_val_path, "dct")

if not os.path.exists(yolo_dataset_train_dct_path):
    os.makedirs(yolo_dataset_train_dct_path)
    print(f"已创建路径: {yolo_dataset_train_dct_path}")
else:
    print(f"路径已存在: {yolo_dataset_train_dct_path}")

if not os.path.exists(yolo_dataset_val_dct_path):
    os.makedirs(yolo_dataset_val_dct_path)
    print(f"已创建路径: {yolo_dataset_val_dct_path}")
else:
    print(f"路径已存在: {yolo_dataset_val_dct_path}")

#-------------------------------------------------#
#   处理数据
#-------------------------------------------------#
def save_dct_data(images_path, dct_path):
    train_image_files = [f for f in os.listdir(images_path) if f.endswith(('.jpg', '.jpeg', '.png', '.JPG'))]
    # 使用tqdm创建进度条
    for image_file in tqdm(train_image_files, desc="处理训练集图像"):
        # 构建完整的图像路径
        image_path = os.path.join(images_path, image_file)
        
        # 读取图像
        image = cv2.imread(image_path)
        
        if image is not None:
            if len(image.shape) == 2:  # 灰度图像
                rgb_image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
            elif image.shape[2] == 4:  # RGBA图像
                rgb_image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
            else:
                rgb_image = image

            rgb_image = cv2.resize(rgb_image, image_size)
            
            # 检查图像是否为RGB图像
            if len(rgb_image.shape) != 3 or rgb_image.shape[2] != 3:
                raise ValueError("输入的图像不是RGB图像")

            # 将RGB图像转换为YCBCR图像
            ycbcr_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2YCrCb)
            
            # 对YCBCR图像进行DCT变换,先分块再对每个块进行单独的DCT变换
            height, width = ycbcr_image.shape[0], ycbcr_image.shape[1]
            dct_image = np.zeros((height // 8, width // 8, 64 * 3), dtype=np.float32)
            for i in range(0, height, 8):
                for j in range(0, width, 8):
                    for k in range(3):
                        block = ycbcr_image[i:i+8, j:j+8, k]
                        dct_block = cv2.dct(np.float32(block))
                        dct_image[i//8, j//8, k*64:(k+1)*64] = dct_block.flatten()
            # 构建保存路径
            save_name = os.path.splitext(image_file)[0] + ".npy" # 去掉后缀
            save_path = os.path.join(dct_path, save_name)
            
            # 使用np.save()保存DCT变换后的数据
            np.save(save_path, dct_image)
            # print(f"已保存DCT数据: {save_path}.npy")
        else:
            print(f"无法读取图像: {image_path}")

#-------------------------------------------------#
#   处理训练集数据
#-------------------------------------------------#
save_dct_data(yolo_dataset_train_images_path, yolo_dataset_train_dct_path)
#-------------------------------------------------#
#   处理验证集数据
#-------------------------------------------------#
save_dct_data(yolo_dataset_val_images_path, yolo_dataset_val_dct_path)
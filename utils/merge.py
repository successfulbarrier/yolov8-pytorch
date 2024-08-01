# !/usr/bin/env python
# -*- coding: utf-8 -*-
# @File    :   merge.py
# @Time    :   2024/07/31 20:54:18
# @Author  :   机灵巢穴_WitNest
# @Version :   1.0
# @Site    :   https://space.bilibili.com/484950664?spm_id_from=333.1007.0.0
# @Desc    :   实现类别合并
import sys
sys.path.append("/media/lht/LHT/DL_code/yolov8-pytorch-my")
import os
import cv2
import random
import string
import shutil
from tqdm import tqdm
from utils import get_classes

#-------------------------------------------------#
#   这个四个变量分别存储
#   合并前的类别名称和顺序，以及类别数量
#-------------------------------------------------#
before_classes, before_classes_len  = get_classes("model_data/voc_classes.txt")
after_classes, after_classes_len    = get_classes("model_data/voc_classes_merge1.txt")
before_classes_map  = {class_name: idx for idx, class_name in enumerate(before_classes)}
after_classes_map   = {class_name: idx for idx, class_name in enumerate(after_classes)}

#-------------------------------------------------#
#   键为，合成后的类名称；
#   值为，要和合并的类别；
#-------------------------------------------------#
merge_class         = {"quadrupeds":["cat", "dog", "cow", "sheep", "horse"], 
                        "two_car": ["bicycle", "motorbike"], "four_car": ["car", "bus"]}

#-------------------------------------------------#
#   要合并的文件路径
#-------------------------------------------------#
train_txt_path      = "2007_train.txt"
val_txt_path        = "2007_val.txt"

#-------------------------------------------------#
#   合并后的文件名
#-------------------------------------------------#
train_txt_path_merge= "2007_train_merge1.txt"
val_txt_path_merge  = "2007_val_merge1.txt"

#-------------------------------------------------#
#   提取分类数据集
#-------------------------------------------------#
classification_root_path = "/media/lht/LHT/code/datasets/voc2007_merge1"
classification_class_name = ["cat", "dog", "cow", "sheep", "horse", "bicycle", "motorbike", "car", "bus"]

#-------------------------------------------------#
#   合并类
#-------------------------------------------------#
def merge_class_function(input_path, output_path):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    merged_lines = []
    for line in lines:
        parts = line.strip().split(' ')
        new_parts = []
        new_parts.append(' '.join(str(parts[0]).split()))
        for part in parts[1:]:
            data = part.split(',')
            class_id = int(data[-1])
            class_name = before_classes[class_id]
            
            # 查找合并后的类别
            new_class_name = class_name
            for merged_class, original_classes in merge_class.items():
                if class_name in original_classes:
                    new_class_name = merged_class
                    break
            
            new_class_id = after_classes_map[new_class_name]
            data[-1] = str(new_class_id)
            new_parts.append(','.join(data))
        
        merged_lines.append(' '.join(new_parts))
    
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in merged_lines:
            f.write(line + '\n')


#-------------------------------------------------#
#   检查txt文件中包含的类别
#-------------------------------------------------#
def query_classes(input_path, class_names):
    with open(input_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    class_set = set()
    for line in lines:
        parts = line.strip().split(' ')
        for part in parts[1:]:
            data = part.split(',')
            class_id = int(data[-1])
            class_name = class_names[class_id]
            class_set.add(class_name)
    
    return class_set


#-------------------------------------------------#
#   裁减数据集
#-------------------------------------------------#
def get_classification_dataset(train=True):
    #-------------------------------------------------#
    #   模式选择
    #-------------------------------------------------#
    if train:
        output_path = classification_root_path + "/train"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        txt_file_path = train_txt_path
    else:
        output_path = classification_root_path + "/val"
        if not os.path.exists(output_path):
            os.makedirs(output_path)
        txt_file_path = val_txt_path

    #-------------------------------------------------#
    #   清除上一次生成的图片
    #-------------------------------------------------#
    for root, dirs, files in os.walk(output_path):
        for file in files:
            os.remove(os.path.join(root, file))
        for dir in dirs:
            shutil.rmtree(os.path.join(root, dir))  # 递归删除文件夹
            
    #-------------------------------------------------#
    #   创建类别文件夹
    #-------------------------------------------------#
    for class_name in classification_class_name:
        if not os.path.exists(os.path.join(output_path, class_name)):
            os.makedirs(os.path.join(output_path, class_name))
    
    #-------------------------------------------------#
    #   遍历裁剪
    #-------------------------------------------------#
    with open(txt_file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    #-------------------------------------------------#
    #   生成一个包含所有字母和数字的列表
    #-------------------------------------------------#
    all_chars = string.ascii_letters + string.digits

    for line in tqdm(lines):
        parts = line.strip().split(' ')
        for part in parts[1:]:
            data = part.split(',')
            class_id = int(data[-1])
            class_name = before_classes[class_id]
            
            # 查找合并后的类别
            for merge_class in classification_class_name:
                if class_name == merge_class:
                    image = cv2.imread(parts[0])
                    # x, y, w, h = int(data[0]), int(data[1]), int(data[2]), int(data[3])
                    # cropped_image = image[int(y - h / 2):int(y + h / 2), int(x - w / 2):int(x + w / 2)]  # 截取box区域
                    cropped_image = image[int(data[1]):int(data[3]), int(data[0]):int(data[2])]  # 截取box区域
                    if cropped_image is not None and cropped_image.size != 0:
                        # 生成长度为10的随机字符串
                        random_string = ''.join(random.sample(all_chars, 10))
                        cv2.imwrite(os.path.join(output_path, merge_class, random_string+".jpg"), cropped_image)  # 保存截取的区域
                    else:
                        print("次文件出现异常："+parts[0])
    
    if train:
        print("-->训练集提取完毕！！！")
    else:
        print("-->验证集提取完毕！！！")     
                       
                       
#-------------------------------------------------#
#   main函数
#-------------------------------------------------#
if __name__ == '__main__':
    # # 查询训练集和验证集中的类别
    # train_classes = query_classes(train_txt_path, before_classes)
    # val_classes = query_classes(val_txt_path, before_classes)
    # print(train_classes)
    # print(val_classes)
    # # 合并训练集和验证集的类别
    # merge_class_function(train_txt_path, train_txt_path_merge)
    # merge_class_function(val_txt_path, val_txt_path_merge)
    # # 查询合并后训练集和验证集中的类别
    # train_classes = query_classes(train_txt_path_merge, after_classes)
    # val_classes = query_classes(val_txt_path_merge, after_classes)
    # print(train_classes)
    # print(val_classes)
    
    # 提取分类数据集
    get_classification_dataset(train=True)
    get_classification_dataset(train=False)
    
    
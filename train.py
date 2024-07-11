import os
import logging
import copy
import random
import math
import csv
from types import SimpleNamespace as sns
from dotwiz import DotWiz
from datetime import datetime
import time
from tqdm import tqdm

from comet_ml import Experiment
# from comet_ml.integration.pytorch import log_metric

import torch
import torch.nn as nn
from torch.nn import Conv2d, Dropout, Linear, ReLU, LayerNorm, Tanh
from torch.nn.functional import relu
from torch.nn.utils import clip_grad_norm_
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.transforms import transforms as T
from torch.utils.data.dataloader import DataLoader
from torch.utils.data import random_split
from torchvision.utils import save_image

import dataset
import dataset.kitti_odometry_remote

DATASET_FILEPATH = "../ETRI_Project_Auto_Calib/datasets/KITTI-Odometry/"
REMOTE_DATASET_FILEPATH = "/home/wicomai/dataset/KITTI-Odometry/"
TRAIN_SEQUENCE = list(range(1,22))
VAL_SEQUENCE = [0]
RESIZE_IMG = (192, 640)
CAM_ID = "2"

def check_data(dataset):
    sampled_data = dataset[random.randint(0,len(dataset))]
    count = 1
    for key,value in sampled_data.items():
        if isinstance(value,torch.Tensor):
            shape = value.size()
            
            # if len(shape) == 3:
            #     # print(shape, value)
            #     save_image(value, './output/check_img/rand_check_'+str(count)+'.png')
        else:
            shape = value
        print('{key}: {shape}'.format(key=key,shape=shape))
        
        count += 1

if __name__ == "__main__":
    ### Image Preprocessing
    rgb_transform = T.Compose([T.Resize((RESIZE_IMG[0], RESIZE_IMG[1])),
                               T.ToTensor(),
                               T.Normalize(mean=[0.33, 0.36, 0.33], 
                                           std=[0.30, 0.31, 0.32])])
    
    depth_transform = T.Compose([T.Resize((RESIZE_IMG[0], RESIZE_IMG[1])),
                                 T.Normalize(mean=[0.0404], 
                                             std=[6.792])])
    
    ### Load the data from csv file
    ## local dataset
    # ds_train = dataset.KITTI_Odometry(rootdir=DATASET_FILEPATH,
    #                           sequences=TRAIN_SEQUENCE,
    #                           camera_id=CAM_ID,
    #                           frame_step=2,
    #                           n_scans=None,
    #                           voxel_size=None,
    #                           max_trans = [1.5, 1.0, 0.5, 0.2, 0.1],
    #                           max_rot = [20, 10, 5, 2, 1],
    #                           rgb_transform=rgb_transform,
    #                           depth_transform=depth_transform,
    #                           device='cpu'
    #                           )
    
    ## remote dataset
    ds_train = dataset.KITTI_Odometry_remote(rootdir=REMOTE_DATASET_FILEPATH,
                              sequences=TRAIN_SEQUENCE,
                              camera_id=CAM_ID,
                              frame_step=2,
                              n_scans=None,
                              voxel_size=None,
                              max_trans = [1.5, 1.0, 0.5, 0.2, 0.1],
                              max_rot = [20, 10, 5, 2, 1],
                              rgb_transform=rgb_transform,
                              depth_transform=depth_transform,
                              device='cpu'
                              )
    
    
    # logging.info('Successfully loaded the dataset with length of: ', str(len(load_ds)))
    print('Successfully loaded the training dataset with length of: ', str(len(ds_train)))
    
    ## local dataset
    # ds_val = dataset.KITTI_Odometry(rootdir=DATASET_FILEPATH,
    #                           sequences=VAL_SEQUENCE,
    #                           camera_id=CAM_ID,
    #                           frame_step=2,
    #                           n_scans=None,
    #                           voxel_size=None,
    #                           max_trans = [1.5, 1.0, 0.5, 0.2, 0.1],
    #                           max_rot = [20, 10, 5, 2, 1],
    #                           rgb_transform=rgb_transform,
    #                           depth_transform=depth_transform,
    #                           device='cpu'
    #                           )
    
    ## remote dataset
    ds_val = dataset.KITTI_Odometry_remote(rootdir=REMOTE_DATASET_FILEPATH,
                              sequences=VAL_SEQUENCE,
                              camera_id=CAM_ID,
                              frame_step=2,
                              n_scans=None,
                              voxel_size=None,
                              max_trans = [1.5, 1.0, 0.5, 0.2, 0.1],
                              max_rot = [20, 10, 5, 2, 1],
                              rgb_transform=rgb_transform,
                              depth_transform=depth_transform,
                              device='cpu'
                              )
    
    print('Successfully loaded the validation dataset with length of: ', str(len(ds_val)))
    
    ### Check loaded dataset
    print("checking training dataset")
    check_data(ds_train)
    print("checking validation dataset")
    check_data(ds_val)
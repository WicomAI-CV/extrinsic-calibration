import random
from types import SimpleNamespace as sns
from dotwiz import DotWiz
from datetime import datetime

import torch
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torchvision.transforms import transforms as T
from torch.utils.data.dataloader import DataLoader

import dataset
import dataset.kitti_odometry_remote
import models
import criterion

from config.model_config import config_transcalib_LVT_efficientnet_june17
from checkpoint import load_checkpoint
from trainer import train_model

DATASET_FILEPATH = "/mnt/hdd/Dataset/KITTI-Odometry/"
REMOTE_DATASET_FILEPATH = "/home/wicomai/dataset/KITTI-Odometry/"
TRAIN_SEQUENCE = list(range(1,22))
VAL_SEQUENCE = [0]
RESIZE_IMG = (192, 640)
CAM_ID = "2"
LOAD_MODEL = False
BATCH_SIZE = 16

MODEL_CONFIG = config_transcalib_LVT_efficientnet_june17
MODEL_CONFIG_CL = DotWiz(MODEL_CONFIG)

DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4
PIN_MEMORY = True
GRAD_CLIP = 1.0

MODEL_NAME = MODEL_CONFIG_CL.model_name
MODEL_DATE = datetime.now().strftime('%Y%m%d_%H%M%S')
if LOAD_MODEL == False:
    LOAD_CHECKPOINT_DIR = None
else:
    LOAD_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_20240626_205457_best_trans.pth.tar"
    
training_config = {'n_epochs': 10,
                'learning_rate': 1e-5,
                'momentum': 0.8,
                'loss_treshold' : 0.001,
                'early_stop_patience' : 5}

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
    
    ### Create a dataLoader object for the loaded data
    # train_loader, val_loader, test_loader = create_DataLoader(load_ds)
    train_loader = DataLoader(dataset = ds_train, 
                              batch_size = BATCH_SIZE, 
                              shuffle = True, 
                              collate_fn = list, 
                              pin_memory = PIN_MEMORY, 
                              num_workers = NUM_WORKERS)
    val_loader = DataLoader(dataset = ds_val, 
                            batch_size = 8, 
                            shuffle = True, 
                            collate_fn = list, 
                            pin_memory = PIN_MEMORY, 
                            num_workers = NUM_WORKERS)
    
    model_ = models.TransCalib_lvt_efficientnet_june2(MODEL_CONFIG_CL).to(DEVICE)
    print(model_)
    pytorch_total_params = sum(p.numel() for p in model_.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model_.parameters() if p.requires_grad)
    print(f'[INFO] Model total parameters: {pytorch_total_params} | Model total trainable parameters {pytorch_total_params_trainable}')
    
    if LOAD_MODEL:
        model_, last_epoch, last_val_loss, last_error_t, last_error_r = load_checkpoint(LOAD_CHECKPOINT_DIR, model_)
    else:
        last_epoch, last_val_loss = 0, None
        last_error_t, last_error_r = None, None
        
    sns_training_config = sns(**training_config)
    
    # Criteria
    reg_loss = criterion.regression_loss().to(DEVICE)
    rot_loss = criterion.rotation_loss().to(DEVICE)
    pcd_loss = criterion.chamfer_distance_loss().to(DEVICE)
    criterion_ = [reg_loss, rot_loss, pcd_loss]
    
    optimizer = optim.AdamW(model_.parameters(), lr=training_config['learning_rate'], weight_decay=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True, eps=1e-15)
    
    ### Start training the model
    train_model(model_, 
                train_loader, val_loader, criterion_, optimizer, sns_training_config, 
                last_epoch, last_best_loss=last_val_loss, 
                last_best_error_t=last_error_t, last_best_error_r=last_error_r,
                scheduler=scheduler, DEVICE=DEVICE, GRAD_CLIP=GRAD_CLIP,
                MODEL_CONFIG_CL=MODEL_CONFIG_CL, LOAD_MODEL=LOAD_MODEL)
    
    # experiment.end()
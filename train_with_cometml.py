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
import torchvision.transforms.functional as F

from models.realignment_layer import realignment_layer

import dataset
import dataset.kitti_odometry_remote
from models.lvt_effnet import TransCalib_lvt_efficientnet_june2
from models.lvt_effnet_light_v1 import TransCalib_lvt_efficientnet_july18
import criteria

from config.model_config import *

from utils.helpers import qua2rot_torch
import numpy as np

RESCALE_IMG_RANGE = False
RESCALE_TARGET = False
RESCALE_PCD = False

DATASET_FILEPATH = "../ETRI_Project_Auto_Calib/datasets/KITTI-Odometry/"
TRAIN_SEQUENCE = list(range(1,22))
VAL_SEQUENCE = [0]
TEST_SEQUENCE = [0]
SKIP_FRAMES = 2
RESIZE_IMG = (192, 640)
BATCH_SIZE = 16
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_CONFIG = config_lvt_effnet_light_v1_july18
MODEL_CONFIG_CL = DotWiz(MODEL_CONFIG)
MODEL_CLASS = TransCalib_lvt_efficientnet_july18(MODEL_CONFIG_CL)
LOAD_MODEL = True
MODEL_NAME = MODEL_CONFIG_CL.model_name
MODEL_DATE = datetime.now().strftime('%Y%m%d_%H%M%S')
CAM_ID = "2"

# DIRECTORY
if LOAD_MODEL == False:
    LOAD_CHECKPOINT_DIR = None
    BEST_TRAIN_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_{MODEL_DATE}_best_train.pth.tar"
    BEST_VAL_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_{MODEL_DATE}_best_val.pth.tar"
    BEST_ROT_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_{MODEL_DATE}_best_rot.pth.tar"
    BEST_TRANS_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_{MODEL_DATE}_best_trans.pth.tar"
else:
    LOAD_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_20240718_151139_best_val.pth.tar"
    BEST_TRAIN_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_{MODEL_DATE}_best_train.pth.tar"
    BEST_VAL_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_{MODEL_DATE}_best_val.pth.tar"
    BEST_ROT_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_{MODEL_DATE}_best_rot.pth.tar"
    BEST_TRANS_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_{MODEL_DATE}_best_trans.pth.tar"

GRAD_CLIP = 1.0
NUM_WORKERS = 4
PIN_MEMORY = True

torch.cuda.empty_cache()

# logging.basicConfig(filename='vanilla_CNN-debug.log', level=logging.INFO)
torch.autograd.set_detect_anomaly(True)

training_config = {'n_epochs': 30,
                   'learning_rate': 1e-5,
                   'momentum': 0.8,
                   'loss_treshold' : 0.001,
                   'early_stop_patience' : 5}

### MLOps Setup
experiment = Experiment(
    api_key=os.environ.get("COMETML_API_KEY"),
    project_name="online-extrinsic-calibration",
    workspace="krishnayoga",
    auto_metric_logging=True,
    auto_param_logging=True,
    auto_histogram_weight_logging=False,
    auto_histogram_gradient_logging=False,
    auto_histogram_activation_logging=False,
)

experiment.set_name(f'{MODEL_NAME}_{MODEL_DATE}')

hyper_params = {
    "model_name": MODEL_NAME,
    "learning_rate": float(training_config['learning_rate']),
    "batch_size": BATCH_SIZE,
    "training_sequence": TRAIN_SEQUENCE,
    "val_sequence": VAL_SEQUENCE,
    "early_stop_patience": int(training_config['early_stop_patience']),
    "device": DEVICE,
    "load_checkpoint": LOAD_CHECKPOINT_DIR,
    "train_checkpoint":BEST_TRAIN_CHECKPOINT_DIR,
    "val_checkpoint":BEST_VAL_CHECKPOINT_DIR,
    "trans_checkpoint": BEST_TRANS_CHECKPOINT_DIR,
    "rot_checkpoint": BEST_ROT_CHECKPOINT_DIR
}
experiment.log_parameters(hyper_params)

experiment.add_tag('seq_num ' + str(experiment.get_parameter('training_sequence')))

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

def validate(model, loader, crit):
    model.eval()
    val_loss = 0
    ex_epoch, ey_epoch, ez_epoch, et_epoch = 0., 0., 0., 0.
    eyaw_epoch, eroll_epoch, epitch_epoch, er_epoch = 0., 0., 0., 0.
    dR_epoch = 0.0
    
    reg_loss, rot_loss, pcd_loss = crit
    # reg_loss, rot_loss = crit

    process = tqdm(loader, unit='batch')

    for _, batch_data in enumerate(process):
        # print(i, batch_data)

        T_gt = [sample["T_gt"].to(DEVICE) for sample in batch_data]
        rgb_img = [sample["img"].to(DEVICE) for sample in batch_data]
        depth_img = [sample["depth_img_error"].to(DEVICE) for sample in batch_data]
        delta_q_gt = [sample["delta_q_gt"].to(DEVICE) for sample in batch_data]
        delta_t_gt = [sample["delta_t_gt"].to(DEVICE) for sample in batch_data]
        pcd_mis = [sample["pcd_mis"].to(DEVICE) for sample in batch_data]
        pcd_gt = [sample["pcd_gt"].to(DEVICE) for sample in batch_data]

        T_gt = torch.stack(T_gt, dim=0)
        rgb_img = torch.stack(rgb_img, dim=0) # correct shape
        depth_img = torch.stack(depth_img, dim=0) # correct shape
        delta_q_gt = torch.stack(delta_q_gt, dim=0)
        delta_t_gt = torch.stack(delta_t_gt, dim=0)
        targets = torch.cat((delta_q_gt, delta_t_gt), 1) # correct shape
        # print('targets shape: ', targets.shape)

        T_mis_batch = torch.tensor([]).to(DEVICE)

        for i in range(targets.shape[0]):
            delta_R_gt  = qua2rot_torch(delta_q_gt[i])
            delta_tr_gt = torch.reshape(delta_t_gt[i],(3,1))
            delta_T_gt  = torch.hstack((delta_R_gt, delta_tr_gt)) 
            delta_T_gt  = torch.vstack((delta_T_gt, torch.Tensor([0., 0., 0., 1.]).to(DEVICE)))

            T_mis = torch.unsqueeze(torch.matmul(delta_T_gt, T_gt[i]), 0)
            T_mis_batch = torch.cat((T_mis_batch, T_mis), 0)
        # print(rgb_img.shape, i)

        pcd_pred, batch_T_pred, delta_q_pred, delta_t_pred = model(rgb_img, depth_img,  pcd_mis, T_mis_batch)
        
        translational_loss = reg_loss(delta_q_gt, delta_t_gt, delta_q_pred, delta_t_pred)
        rotational_loss = rot_loss(delta_q_gt, delta_q_pred)
        pointcloud_loss = pcd_loss(pcd_gt, pcd_pred)
        loss = translational_loss + rotational_loss + pointcloud_loss
        # loss = crit(output, targets)

        val_loss += loss.item()
        experiment.log_metric('val batch loss', loss.item()/targets.shape[0])

        # print(f'L1 = {translational_loss}| L2 = {rotational_loss} | L3 = {pointcloud_loss}')
        e_x, e_y, e_z, e_t, e_yaw, e_pitch, e_roll, e_r, dR = criteria.test_metrics(batch_T_pred, T_gt)

        ex_epoch += e_x.item()
        ey_epoch += e_y.item()
        ez_epoch += e_z.item()
        et_epoch += e_t.item()
        eyaw_epoch += e_yaw.item()
        eroll_epoch += e_roll.item()
        epitch_epoch += e_pitch.item()
        er_epoch += e_r.item()
        dR_epoch += dR.item()

        process.set_description('Validation: ')
        process.set_postfix(loss=loss.item())
    
    ex_epoch /= len(loader)
    ey_epoch /= len(loader)
    ez_epoch /= len(loader)
    et_epoch /= len(loader)
    eyaw_epoch /= len(loader)
    eroll_epoch /= len(loader)
    epitch_epoch /= len(loader)
    er_epoch /= len(loader)
    dR_epoch /= len(loader)

    # val_loss /= len(loader)
    
    print(f'Ex = {ex_epoch}| Ey = {ey_epoch} | Ez = {ez_epoch} | Et = {et_epoch}') 
    print(f'yaw = {eyaw_epoch} | pitch = {epitch_epoch} | roll = {eroll_epoch} | er = {er_epoch} | Dg = {dR_epoch}')

    experiment.log_metrics({'Ex': ex_epoch, 'Ey': ey_epoch, 'Ez': ez_epoch, 'Et': et_epoch})
    experiment.log_metrics({'yaw': eyaw_epoch, 'pitch': epitch_epoch, 'roll': eroll_epoch, 'Er': er_epoch, 'Dg': dR_epoch})

    return val_loss, et_epoch, er_epoch

def train_model(model, 
                train_loader, val_loader, 
                crit, opt, train_conf, last_epoch, 
                last_best_loss = None, 
                last_best_error_t = None, last_best_error_r = None,
                scheduler = None):
    
    reg_loss, rot_loss, pcd_loss = crit
    # reg_loss, rot_loss = crit

    loss_treshold = train_conf.loss_treshold
    early_stop_patience = train_conf.early_stop_patience
    treshold_count = 0
    
    # initialize val loss milestone
    if last_best_loss is None:
        best_val_loss = 1500000.0
    else:
        best_val_loss = last_best_loss

    # initialize train loss milestone
    if last_best_loss is None:
        best_train_loss = 1500000.0
    else:
        best_train_loss = last_best_loss

    # initialize translational error milestone
    if last_best_error_t is None:
        best_error_t = 50.0
    else:
        best_error_t = last_best_error_t

    # initialize rotational error milestone
    if last_best_error_r is None:
        best_error_r = 10.0
    else:
        best_error_r = last_best_error_r

    start_time = time.time()
    st_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print('Start time: ', datetime.now())

    model.train()
    for j in range(train_conf.n_epochs):
        print("===================================")
        print(f'Epoch: {j+1+last_epoch} or {j+1}')
        epoch_start = time.time()

        loss_epoch = 0
        trans_loss_epoch = 0
        rot_loss_epoch = 0
        pcd_loss_epoch = 0

        process = tqdm(train_loader, unit='batch')

        for _, batch_data in enumerate(process):
        # print(i, batch_data)

            T_gt = [sample["T_gt"].to(DEVICE) for sample in batch_data]
            rgb_img = [sample["img"].to(DEVICE) for sample in batch_data]
            depth_img = [sample["depth_img_error"].to(DEVICE) for sample in batch_data]
            delta_q_gt = [sample["delta_q_gt"].to(DEVICE) for sample in batch_data]
            delta_t_gt = [sample["delta_t_gt"].to(DEVICE) for sample in batch_data]
            pcd_mis = [sample["pcd_mis"].to(DEVICE) for sample in batch_data]
            pcd_gt = [sample["pcd_gt"].to(DEVICE) for sample in batch_data]

            T_gt = torch.stack(T_gt, dim=0)
            rgb_img = torch.stack(rgb_img, dim=0) # correct shape
            depth_img = torch.stack(depth_img, dim=0) # correct shape
            delta_q_gt = torch.stack(delta_q_gt, dim=0)
            delta_t_gt = torch.stack(delta_t_gt, dim=0)
            targets = torch.cat((delta_q_gt, delta_t_gt), 1) # correct shape
            # print('targets shape: ', targets.shape)

            T_mis_batch = torch.tensor([]).to(DEVICE)

            for i in range(targets.shape[0]):
                delta_R_gt  = qua2rot_torch(delta_q_gt[i])
                delta_tr_gt = torch.reshape(delta_t_gt[i],(3,1))
                delta_T_gt  = torch.hstack((delta_R_gt, delta_tr_gt)) 
                delta_T_gt  = torch.vstack((delta_T_gt, torch.Tensor([0., 0., 0., 1.]).to(DEVICE)))

                T_mis = torch.unsqueeze(torch.matmul(delta_T_gt, T_gt[i]), 0)
                T_mis_batch = torch.cat((T_mis_batch, T_mis), 0)

            # print(rgb_img.shape, i)

            opt.zero_grad()

            pcd_pred, _, delta_q_pred, delta_t_pred = model(rgb_img, depth_img, pcd_mis, T_mis_batch)

            # print(rgb_img.shape, depth_img.shape, len(pcd), T_gt.shape, T_mis_batch.shape)
            # print(len(pcd_true), len(pcd_pred), delta_q_pred.shape, delta_t_pred.shape)
            # print(pcd[1].shape, pcd_true[1].shape, pcd_pred[1].shape)
            # print(pcd[2].shape, pcd_true[2].shape, pcd_pred[2].shape)
            # print(pcd[3].shape, pcd_true[3].shape, pcd_pred[3].shape)
            
            # Loss calculation and backprop
            translational_loss = reg_loss(delta_q_gt, delta_t_gt, delta_q_pred, delta_t_pred)
            rotational_loss = rot_loss(delta_q_gt, delta_q_pred)
            pointcloud_loss = pcd_loss(pcd_gt, pcd_pred)
            loss = translational_loss + rotational_loss + pointcloud_loss

            # print(f'L1 = {translational_loss}| L2 = {rotational_loss} | L3 = {pointcloud_loss}')
            # loss = crit(output, targets)
            # # print('loss shape: ', loss.shape)
            loss.backward()

            # gradient clipping
            if GRAD_CLIP is not None:
                clip_grad_norm_(model.parameters(), GRAD_CLIP)

            opt.step()

            loss_epoch += loss.item()
            trans_loss_epoch += translational_loss.item()
            rot_loss_epoch += rotational_loss.item()
            pcd_loss_epoch += pointcloud_loss.item()  
            # print('current loss: ', loss_epoch)

            experiment.log_metric("batch loss", loss.item()/targets.shape[0])
            experiment.log_metric("batch trans loss",  translational_loss.item()/targets.shape[0])
            experiment.log_metric("batch rot loss", rotational_loss.item()/targets.shape[0])
            experiment.log_metric("batch pcd loss", pointcloud_loss.item()/targets.shape[0])

            process.set_postfix(loss=loss.item())
            # print('batch_no: ', i)
        train_time = time.time() - epoch_start
        
        # loss_epoch /= len(train_loader)
        trans_loss_epoch /= len(train_loader)
        rot_loss_epoch /= len(train_loader)
        pcd_loss_epoch /= len(train_loader)
        
        val_start = time.time()
        val_loss, error_t, error_r = validate(model, val_loader, crit)
        val_time = time.time() - val_start

        print(f'L1 = {trans_loss_epoch} | L2 = {rot_loss_epoch} | L3 = {pcd_loss_epoch}')
        print(f'Loss Train: {loss_epoch} | Loss Val: {val_loss} | LR: {opt.param_groups[0]["lr"]}')
        print(f'Training time: {train_time} | Validation time: {val_time}')

        if scheduler is not None:
            scheduler.step(val_loss)
        
        # save checkpoint with the best validation score
        if val_loss/len(val_loader) < best_val_loss:
            print("best val loss achieved")
            checkpoint = {"state_dict": model.state_dict(),
                          "optimizer:": opt.state_dict(),
                          "epoch": j+last_epoch,
                          "loss": val_loss/len(val_loader),
                          "rot_error": error_r,
                          "trans_error": error_t}
            save_checkpoint(checkpoint, filename=BEST_VAL_CHECKPOINT_DIR)
            best_val_loss = val_loss/len(val_loader)

        # save checkpoint with the best training score
        if loss_epoch/len(train_loader) < best_train_loss:
            print("best train loss achieved")
            checkpoint = {"state_dict": model.state_dict(),
                          "optimizer:": opt.state_dict(),
                          "epoch": j+last_epoch,
                          "loss": loss_epoch/len(train_loader),
                          "rot_error": error_r,
                          "trans_error": error_t}
            save_checkpoint(checkpoint, filename=BEST_TRAIN_CHECKPOINT_DIR)
            best_train_loss = loss_epoch/len(train_loader)

        # save checkpoint with the best translational alignment metric
        if error_t < best_error_t:
            print("best translation alignment achieved")
            checkpoint = {"state_dict": model.state_dict(),
                          "optimizer:": opt.state_dict(),
                          "epoch": j+last_epoch,
                          "loss": val_loss,
                          "rot_error": error_r,
                          "trans_error": error_t}
            save_checkpoint(checkpoint, filename=BEST_TRANS_CHECKPOINT_DIR)
            best_error_t = error_t

        # save checkpoint with the best rotational alignment metric
        if error_r < best_error_r:
            print("best rotation alignment achieved")
            checkpoint = {"state_dict": model.state_dict(),
                          "optimizer:": opt.state_dict(),
                          "epoch": j+last_epoch,
                          "loss": val_loss,
                          "rot_error": error_r,
                          "trans_error": error_t}
            save_checkpoint(checkpoint, filename=BEST_ROT_CHECKPOINT_DIR)
            best_error_r = error_r

        experiment.log_metric("Training total loss", loss_epoch)
        experiment.log_metric("Training avg loss", loss_epoch/len(train_loader))
        experiment.log_metric("Validation total loss", val_loss)
        experiment.log_metric("Validation avg loss", val_loss/len(val_loader))
        experiment.log_metric('learning rate', opt.param_groups[0]["lr"])

        if val_loss <= loss_treshold:
            treshold_count += 1
        
        if treshold_count == early_stop_patience:
            break

    print('Start time: ', st_time, '\nFinished time: ', datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print('Overall training time: ', (time.time()-start_time)/3600, ' hours')

def save_checkpoint(state, filename="ViT_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)

def load_checkpoint(checkpoint_path, model):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    last_epoch = checkpoint["epoch"]
    last_epoch_loss = checkpoint["loss"]
    last_best_error_t = checkpoint["trans_error"] 
    last_best_error_r = checkpoint["rot_error"] 

    return model, last_epoch, last_epoch_loss, last_best_error_t, last_best_error_r

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
    ds_train = dataset.KITTI_Odometry(rootdir=DATASET_FILEPATH,
                                        sequences=TRAIN_SEQUENCE,
                                        camera_id=CAM_ID,
                                        frame_step=2,
                                        n_scans=None,
                                        voxel_size=None,
                                        max_trans = [1.5, 0.5, 0.1, 0.1],
                                        max_rot = [20, 10, 5, 1],
                                        rgb_transform=rgb_transform,
                                        depth_transform=depth_transform,
                                        device='cpu'
                                        )
                
    # logging.info('Successfully loaded the dataset with length of: ', str(len(load_ds)))
    print('Successfully loaded the training dataset with length of: ', str(len(ds_train)))
    
    ds_val = dataset.KITTI_Odometry(rootdir=DATASET_FILEPATH,
                                        sequences=VAL_SEQUENCE,
                                        camera_id=CAM_ID,
                                        frame_step=2,
                                        n_scans=None,
                                        voxel_size=None,
                                        max_trans = [1.5, 0.5, 0.1],
                                        max_rot = [20, 5, 1],
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
    train_loader = DataLoader(dataset = ds_train, 
                              batch_size = experiment.get_parameter('batch_size'), 
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
    
    ## Build the model
    model = MODEL_CLASS.to(DEVICE)
    print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[INFO] Model total parameters: {pytorch_total_params:,} | Model total trainable parameters {pytorch_total_params_trainable:,}')

    if LOAD_MODEL:
        model, last_epoch, last_val_loss, last_error_t, last_error_r = load_checkpoint(LOAD_CHECKPOINT_DIR, model)
    else:
        last_epoch, last_val_loss = 0, None
        last_error_t, last_error_r = None, None

    ### Prepare training stuffs
    sns_training_config = sns(**training_config)

    # Criteria
    reg_loss = criteria.regression_loss().to(DEVICE)
    rot_loss = criteria.rotation_loss().to(DEVICE)
    pcd_loss = criteria.chamfer_distance_loss(scale=.1).to(DEVICE)
    criterion = [reg_loss, rot_loss, pcd_loss]

    # criterion = nn.MSELoss(reduction='mean').cuda()
    # optimizer = optim.SGD(model_vit.parameters(), lr=sns_training_config.learning_rate, momentum=sns_training_config.momentum)
    optimizer = optim.AdamW(model.parameters(), lr=experiment.get_parameter('learning_rate'), weight_decay=0.1)
    scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=10, verbose=True, eps=1e-15)
    
    ### Start training the model
    train_model(model, 
                train_loader, val_loader, criterion, optimizer, sns_training_config, 
                last_epoch, last_best_loss=last_val_loss, 
                last_best_error_t=last_error_t, last_best_error_r=last_error_r,
                scheduler=scheduler)
    
    experiment.end()
    # train_results = trainViT(model_vit, train_loader, val_loader, criterion, optimizer, sns_training_config)
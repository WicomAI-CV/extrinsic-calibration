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
import matplotlib.pyplot as plt
from PIL import Image
import mathutils

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

from models.realignment_layer import realignment_layer

import dataset
import dataset.kitti_odometry_remote
from models.LVT_models.lvt_effnet import TransCalib_lvt_efficientnet_june2
from models.LVT_models.lvt_effnet_light_v1 import TransCalib_lvt_efficientnet_july18
import criteria

from config.model_config import *

from utils.helpers import qua2rot_torch, rot2qua, pcd_extrinsic_transform

import numpy as np
# import matplotlib.pyplot as plt

# from load_dataset_1 import KITTI360_dataset
RESCALE_IMG_RANGE = False
RESCALE_TARGET = False
RESCALE_PCD = False

DATASET_FILEPATH = "./kitti360_transcalib_new_dataset/dataset.csv"
TRAIN_SEQUENCE = [3,4,5,6,7,10]
VAL_SEQUENCE = [0,2]
TEST_SEQUENCE = [0,2,9]
SKIP_FRAMES = 1
RESIZE_IMG = (192, 640)
TRAIN_LEN = 0.9
TEST_LEN = 0.1
VAL_LEN = 0.2
BATCH_SIZE = 1
INFER_ITER = 4
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
MODEL_CONFIG = config_transcalib_LVT_efficientnet_june17
# MODEL_CONFIG = config_lvt_effnet_light_v1_july18
MODEL_CONFIG_CL = DotWiz(MODEL_CONFIG)
MODEL_CLASS = TransCalib_lvt_efficientnet_june2(MODEL_CONFIG_CL)
# MODEL_CLASS = TransCalib_lvt_efficientnet_july18(MODEL_CONFIG_CL)
MODEL_NAME = MODEL_CONFIG_CL.model_name
MODEL_DATE = datetime.now().strftime('%Y%m%d_%H%M%S')
CAM_ID = "2"
LOAD_DSET_FROM_FOLDER = False 

# DIRECTORY
LOAD_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_20240704_002115_best_val.pth.tar"
# LOAD_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_20240718_151139_best_val.pth.tar"

GRAD_CLIP = 1.0
NUM_WORKERS = 4
PIN_MEMORY = True

torch.cuda.empty_cache()

# logging.basicConfig(filename='vanilla_CNN-debug.log', level=logging.INFO)
torch.autograd.set_detect_anomaly(True)

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
        
def load_checkpoint(checkpoint_path, model):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    last_epoch = checkpoint["epoch"]
    last_epoch_loss = checkpoint["loss"]
    last_best_error_t = checkpoint["trans_error"] 
    last_best_error_r = checkpoint["rot_error"] 

    return model, last_epoch, last_epoch_loss, last_best_error_t, last_best_error_r


def generate_misalignment(max_rot = 20, max_trans = 0.5, rot_order='XYZ'):
        rot_z = np.random.uniform(-max_rot, max_rot) * (3.141592 / 180.0)
        rot_y = np.random.uniform(-max_rot, max_rot) * (3.141592 / 180.0)
        rot_x = np.random.uniform(-max_rot, max_rot) * (3.141592 / 180.0)
        trans_x = np.random.uniform(-max_trans, max_trans)
        trans_y = np.random.uniform(-max_trans, max_trans)
        trans_z = np.random.uniform(-max_trans, max_trans)

        R_perturb = mathutils.Euler((rot_x, rot_y, rot_z)).to_matrix()
        t_perturb = np.array([trans_x, trans_y, trans_z])
            
        return np.array(R_perturb), t_perturb

def depth_img_gen(point_cloud, T_ext, K_int, im_size, range_mode=False):
        W, H = im_size

        n_points = point_cloud.shape[0]

        if T_ext is not None: 
            pcd_cam = np.matmul(T_ext, np.hstack((point_cloud, np.ones((n_points, 1)))).T).T  # (P_velo -> P_cam)
        else:
            pcd_cam = point_cloud

        pcd_cam = pcd_cam[:,:3]
        z_axis = pcd_cam[:,2]  # Extract the z_axis to determine 3D points that located in front of the camera.

        pixel_proj = np.matmul(K_int, pcd_cam.T).T

        # normalize pixel coordinates
        pixel_proj = np.array([x/x[2] for x in pixel_proj])

        u = np.array(pixel_proj[:, 0], dtype=np.int32)
        v = np.array(pixel_proj[:, 1], dtype=np.int32)

        # depth calculation of each point
        depth = np.array([np.linalg.norm(x) for x in pcd_cam]) if range_mode else z_axis

        condition = (0<=u)*(u<W)*(0<=v)*(v<H)*(depth>0)*(z_axis>=0)
        # print(np.min(z_axis))

        u_proj = u[condition]
        v_proj = v[condition]
        d_proj = depth[condition]

        # image array generation
        image_tensor = torch.zeros(H, W, dtype=torch.float32)

        if d_proj.shape[0] > 0:
            max_depth = np.max(d_proj)
            d_proj = np.array([np.interp(d, [0, max_depth], [1, 0]) for d in d_proj]) # convert depth values to [0, 1]
            image_tensor[v_proj,u_proj] = torch.from_numpy(d_proj).type(torch.float32) #(1400, 1400, )

        image_tensor = torch.unsqueeze(image_tensor, 0)

        return image_tensor


def test(model, loader, crit, depth_transform):
    # if CAM_ID == "00" or CAM_ID == "01":
    #     im_size, K_int = load_cam_intrinsic("../ETRI_Project_Auto_Calib/datasets/KITTI-360/", CAM_ID)
    # else:
    #     im_size, K_int, distort_params = load_fisheye_intrinsic("../ETRI_Project_Auto_Calib/datasets/KITTI-360/", CAM_ID)
    #     distort_params = (distort_params['xi'], 
    #                             distort_params['k1'],
    #                             distort_params['k2'],
    #                             distort_params['p1'],
    #                             distort_params['p2'])
    model.eval()
    val_loss = 0
    ex_epoch, ey_epoch, ez_epoch, et_epoch = 0., 0., 0., 0.
    eyaw_epoch, eroll_epoch, epitch_epoch, er_epoch = 0., 0., 0., 0.
    trans_loss_epoch, rot_loss_epoch, pcd_loss_epoch = 0., 0., 0.
    dR_epoch = 0.0
    
    # reg_loss, rot_loss, pcd_loss = crit
    reg_loss, rot_loss = crit

    process = tqdm(loader, unit='batch')

    for _, batch_data in enumerate(process):
        # print(i, batch_data)
        img_path = [sample["img_path"] for sample in batch_data]
        K_int = [sample["K_int"] for sample in batch_data]
        im_size = [sample["im_size"] for sample in batch_data]

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

        pcd_mis_ori = pcd_mis

        for _ in range(INFER_ITER):
            # print("infer iter:", z)
            # print(depth_img.shape)
            pcd_pred, batch_T_pred, delta_q_pred, delta_t_pred = model(rgb_img, depth_img,  pcd_mis, T_mis_batch)

            if depth_transform is not None:
                depth_img = [depth_transform(
                            depth_img_gen(point_cloud=pcd.detach().cpu().numpy(), 
                                          T_ext=None, 
                                          K_int=Kint, 
                                          im_size=im_sz)
                                       ) 
                            for pcd, Kint, im_sz in zip(pcd_pred, K_int, im_size)]
            else: 
                depth_img = [depth_img_gen(point_cloud=pcd.detach().cpu().numpy(), 
                                          T_ext=None, 
                                          K_int=Kint, 
                                          im_size=im_sz)
                            for pcd, Kint, im_sz in zip(pcd_pred, K_int, im_size)]
            
            # update for next inference iteration
            depth_img = torch.stack(depth_img, dim=0).to(DEVICE)
            pcd_mis = pcd_pred
            T_mis_batch = batch_T_pred
        
        translational_loss = reg_loss(delta_q_gt, delta_t_gt, delta_q_pred, delta_t_pred)
        rotational_loss = rot_loss(delta_q_gt, delta_q_pred)
        # pointcloud_loss = pcd_loss(pcd_gt, pcd_pred)
        loss = translational_loss + rotational_loss # + pointcloud_loss
        # loss = crit(output, targets)

        val_loss += loss.item()
        trans_loss_epoch += translational_loss.item()
        rot_loss_epoch += rotational_loss.item()
        # pcd_loss_epoch += pointcloud_loss.item()

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

        process.set_description('Testing: ')
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

    trans_loss_epoch /= len(loader)
    rot_loss_epoch /= len(loader)
    pcd_loss_epoch /= len(loader)
    
    print(f"Total loss = {val_loss}")
    print(f'L1 = {trans_loss_epoch}| L2 = {rot_loss_epoch} | L3 = {pcd_loss_epoch}')
    print(f'Ex = {ex_epoch*100:.3f} cm | Ey = {ey_epoch*100:.3f} cm | Ez = {ez_epoch*100:.3f} cm | Et = {et_epoch*100:.3f} cm') 
    print(f'yaw = {eyaw_epoch*180/torch.pi:.3f} \N{DEGREE SIGN} | pitch = {epitch_epoch*180/torch.pi:.3f} \N{DEGREE SIGN} | roll = {eroll_epoch*180/torch.pi:.3f} \N{DEGREE SIGN} | er = {er_epoch*180/torch.pi:.3f} \N{DEGREE SIGN} | Dg = {dR_epoch}')
    
    # proj_img = rgb_mono_projection(image_size=im_size,
    #                                 K_int = K_int)
    
    new_img = Image.open(img_path[-1]).convert("RGB")
    new_img = (T.ToTensor())(new_img)
    
    rgb_mono_projection_(im_size[-1],
                         K_int[-1],
                         pcd_mis_ori[-1].detach().cpu().numpy(),
                         pcd_pred[-1].detach().cpu().numpy(),
                         pcd_gt[-1].detach().cpu().numpy(), 
                         new_img)

def load_checkpoint(checkpoint_path, model):
    print("=> Loading checkpoint")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint["state_dict"])
    last_epoch = checkpoint["epoch"]
    last_epoch_loss = checkpoint["loss"]
    last_best_error_t = checkpoint["trans_error"] 
    last_best_error_r = checkpoint["rot_error"] 

    return model, last_epoch, last_epoch_loss, last_best_error_t, last_best_error_r

def rgb_mono_projection_(image_size, K_int, pcd_mis, pcd_pred, pcd_gt, rgb_im, range_mode=False):
    W, H = image_size
    rgb_img = (T.ToPILImage())(rgb_im).resize((W, H), Image.Resampling.BILINEAR)
        
    pcd_cam = pcd_pred[:,:3]
    pcd_cam_gt = pcd_gt[:,:3]
    pcd_cam_mis = pcd_mis[:,:3]

    # Extract the z_axis to determine 3D points that located in front of the camera.
    z_axis = pcd_cam[:,2] 
    z_axis_gt = pcd_cam_gt[:,2]
    z_axis_mis = pcd_cam_mis[:,2]

    pixel_proj = np.matmul(K_int, pcd_cam.T).T
    pixel_proj_gt = np.matmul(K_int, pcd_cam_gt.T).T
    pixel_proj_mis = np.matmul(K_int, pcd_cam_mis.T).T

    # normalize pixel coordinates
    pixel_proj = np.array([x/x[2] for x in pixel_proj])
    pixel_proj_gt = np.array([x/x[2] for x in pixel_proj_gt])
    pixel_proj_mis = np.array([x/x[2] for x in pixel_proj_mis])

    u = np.array(pixel_proj[:, 0]).astype(np.int32)
    v = np.array(pixel_proj[:, 1]).astype(np.int32)
    u_gt = np.array(pixel_proj_gt[:, 0]).astype(np.int32)
    v_gt = np.array(pixel_proj_gt[:, 1]).astype(np.int32)
    u_mis = np.array(pixel_proj_mis[:, 0]).astype(np.int32)
    v_mis = np.array(pixel_proj_mis[:, 1]).astype(np.int32)

    # depth calculation of each point
    depth = np.array([np.linalg.norm(x) for x in pcd_cam]) if range_mode else z_axis
    depth_gt = np.array([np.linalg.norm(x) for x in pcd_cam_gt]) if range_mode else z_axis_gt
    depth_mis = np.array([np.linalg.norm(x) for x in pcd_cam_mis]) if range_mode else z_axis_mis

    condition = (0<=u)*(u<W)*(0<=v)*(v<H)*(depth>0)*(z_axis>=0)
    condition_gt = (0<=u_gt)*(u_gt<W)*(0<=v_gt)*(v_gt<H)*(depth_gt>0)*(z_axis_gt>=0)
    condition_mis = (0<=u_mis)*(u_mis<W)*(0<=v_mis)*(v_mis<H)*(depth_mis>0)*(z_axis_mis>=0)
    # print(np.min(z_axis))

    u_proj = u[condition]
    v_proj = v[condition]
    d_proj = depth[condition]

    u_proj_gt = u_gt[condition_gt]
    v_proj_gt = v_gt[condition_gt]
    d_proj_gt = depth_gt[condition_gt]

    u_proj_mis = u_mis[condition_mis]
    v_proj_mis = v_mis[condition_mis]
    d_proj_mis = depth_mis[condition_mis]

    if d_proj.shape[0] > 0:
        max_depth = np.max(d_proj)
        d_proj = np.array([np.interp(d, [0, max_depth], [0, 255]) for d in d_proj]) # convert depth values to [0, 255]
    
    if d_proj_gt.shape[0] > 0:
        max_depth_gt = np.max(d_proj_gt)
        d_proj_gt = np.array([np.interp(d, [0, max_depth_gt], [0, 255]) for d in d_proj_gt])
    
    if d_proj_mis.shape[0] > 0:
        max_depth_mis = np.max(d_proj_mis)
        d_proj_mis = np.array([np.interp(d, [0, max_depth_mis], [0, 255]) for d in d_proj_mis])


    # fig = plt.figure(figsize=(14.08,3.76),dpi=100,frameon=False)
    # # fig.set_size(self.W, self.H)
    # ax = plt.Axes(fig, [0., 0., 1., 1.])
    # ax.set_axis_off()
    # fig.add_axes(ax)

    # ax.imshow(rgb_img)
    # ax.scatter([u_proj],[v_proj],c=[d_proj],cmap='rainbow_r',alpha=0.6,s=2)
    # # fig.show()
    # fig.savefig('test_mono_rgb.png')

    fig1 = plt.figure(figsize=(16 , 14),dpi=100) #,frameon=False)
    fig1.add_subplot(3,1,1)
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.title("Initial Misalignment", fontdict = {'fontsize' : 18})
    plt.scatter([u_proj_mis],[v_proj_mis],c=[d_proj_mis],cmap='rainbow_r',alpha=0.6,s=3)
    fig1.add_subplot(3,1,2)
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.title("Prediction", fontdict = {'fontsize' : 18})
    plt.scatter([u_proj],[v_proj],c=[d_proj],cmap='rainbow_r',alpha=0.6,s=3)
    fig1.add_subplot(3,1,3)
    plt.imshow(rgb_img)
    plt.axis('off')
    plt.title("Ground Truth", fontdict = {'fontsize' : 18})
    plt.scatter([u_proj_gt],[v_proj_gt],c=[d_proj_gt],cmap='rainbow_r',alpha=0.6,s=3)
    fig1.tight_layout()
    fig1.savefig('test_comparison_KITTI_odometry_real.png', transparent=False)

if __name__ == "__main__":
    ### Image Preprocessing
    rgb_transform = T.Compose([T.Resize((RESIZE_IMG[0], RESIZE_IMG[1])),
                               T.ToTensor(),
                               T.Normalize(mean=[0.33, 0.36, 0.33], 
                                           std=[0.30, 0.31, 0.32])])
    
    depth_transform = T.Compose([T.Resize((RESIZE_IMG[0], RESIZE_IMG[1])),
                                 T.Normalize(mean=[0.0404], 
                                             std=[6.792])])
    
    ds_test = dataset.KITTI_Odometry(rootdir="../ETRI_Project_Auto_Calib/datasets/KITTI-Odometry/",
                            sequences=[0],
                            camera_id=CAM_ID,
                            frame_step=SKIP_FRAMES,
                            n_scans=300,
                            voxel_size=None,
                            max_trans=[1.5],
                            max_rot=[20],
                            rgb_transform=rgb_transform,
                            depth_transform=depth_transform,
                            device='cpu'
                            )

    print('Successfully loaded the test dataset with length of: ', str(len(ds_test)))

    ### Check loaded dataset
    print("checking test dataset")
    check_data(ds_test)
    
    ### Create a dataLoader object for the loaded data
    # train_loader, val_loader, test_loader = create_DataLoader(load_ds)
    test_loader = DataLoader(dataset = ds_test, 
                             batch_size = BATCH_SIZE, 
                             shuffle = True, 
                             collate_fn = list, 
                             pin_memory = PIN_MEMORY, 
                             num_workers = NUM_WORKERS)

    # max_bounds, min_bounds = get_bounds(load_ds)
    # print('max bound:', max_bounds)
    # print('min bound:', min_bounds)

    ### Set model config
    # sns_model_config = sns(**model_config)
    # sns_model_config_512 = sns(**model_config_512)
    # sns_agg_config = sns(**agg_config)
    # print(sns_model_config.pe_image_size)
    
    ## Build the vanilla ViT model
    model = MODEL_CLASS.to(DEVICE)
    # print(model)
    pytorch_total_params = sum(p.numel() for p in model.parameters())
    pytorch_total_params_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'[INFO] Model total parameters: {pytorch_total_params:,} | Model total trainable parameters {pytorch_total_params_trainable:,}')

    model, last_epoch, last_val_loss, last_error_t, last_error_r = load_checkpoint(LOAD_CHECKPOINT_DIR, model)

    # Criteria
    reg_loss = criteria.regression_loss().to(DEVICE)
    rot_loss = criteria.rotation_loss().to(DEVICE)
    # pcd_loss = criteria.chamfer_distance_loss().to(DEVICE)
    criterion = [reg_loss, rot_loss] #, pcd_loss]
    
    test(model, test_loader, criterion, depth_transform)
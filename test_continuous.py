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
import open3d as o3d
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
from models.LVT_models.lvt_effnet_ablation import TransCalib_lvt_efficientnet_ablation
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
SKIP_FRAMES = 10
RESIZE_IMG = (192, 640)
TRAIN_LEN = 0.9
TEST_LEN = 0.1
VAL_LEN = 0.2
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_CONFIG = config_transcalib_LVT_efficientnet_june17 # 291M
# MODEL_CONFIG = config_lvt_effnet_light_v1_july18 # 70M
# MODEL_CONFIG = config_transcalib_LVT_efficientnet_ablation # ablation
MODEL_CONFIG_CL = DotWiz(MODEL_CONFIG)
MODEL_CLASS = TransCalib_lvt_efficientnet_june2(MODEL_CONFIG_CL) # 291M
# MODEL_CLASS = TransCalib_lvt_efficientnet_july18(MODEL_CONFIG_CL) # 70M
# MODEL_CLASS = TransCalib_lvt_efficientnet_ablation(MODEL_CONFIG_CL) # ablation
MODEL_NAME = MODEL_CONFIG_CL.model_name
MODEL_DATE = datetime.now().strftime('%Y%m%d_%H%M%S')
CAM_ID = "2"
LOAD_DSET_FROM_FOLDER = False 

# DIRECTORY
LOAD_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_20240704_002115_best_val.pth.tar" # 291M
# LOAD_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_20240718_151139_best_val.pth.tar" # 70M
# LOAD_CHECKPOINT_DIR = f"checkpoint_weights/{MODEL_NAME}_20240823_202426_best_val.pth.tar" # ablation

GRAD_CLIP = 1.0
NUM_WORKERS = 4
PIN_MEMORY = False

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
    checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
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
        pcd_cam = np.matmul(T_ext, np.hstack((point_cloud, np.ones((n_points, 1)))).T).T  # (P_velo -> P_cam)
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

def test(model, loader, crit, max_rot=10, max_trans=0.25):
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
    ex_epoch, ey_epoch, ez_epoch, et_epoch = [], [], [], []
    eyaw_epoch, eroll_epoch, epitch_epoch, er_epoch = [], [], [], []
    trans_loss_epoch, rot_loss_epoch, pcd_loss_epoch = [], [], []
    time_iter = []
    dR_epoch = []
    T_mis_list = []

    rotate_pcd = pcd_extrinsic_transform(crop=False)
    depth_transform = T.Compose([T.Resize((RESIZE_IMG[0], RESIZE_IMG[1])),
                                 T.Normalize(mean=[0.0404], 
                                             std=[6.792])])

    T_prev = np.eye(4)
    
    # reg_loss, rot_loss, pcd_loss = crit

    delta_R_init, delta_t_init = generate_misalignment(max_rot, max_trans)
    delta_q_init = rot2qua(delta_R_init)
    delta_T_init = np.hstack((delta_R_init, np.expand_dims(delta_t_init, axis=1)))
    delta_T_init = np.vstack((delta_T_init, np.array([0., 0., 0., 1.])))
    
    process = tqdm(loader, unit='batch')

    for k, batch_data in enumerate(process):
        T_gt = batch_data[0]["T_gt"]
        # print(T_gt.shape)
        
        if k == 0:
            T_mis = np.matmul(delta_T_init, T_gt)
        else:
            T_mis = T_prev

        T_mis_list.append(T_mis)

        img_path = batch_data[0]["img_path"]
        rgb_img = batch_data[0]["img"]
        K_int = batch_data[0]["K_int"]
        im_size = batch_data[0]["im_size"]

        pcd_gt = batch_data[0]["pcd_gt"]
        pcd = batch_data[0]["pcd"]

        depth_img = depth_img_gen(pcd, T_mis, K_int, im_size)
        if depth_transform is not None:
            depth_img = depth_transform(depth_img)

        pcd_mis = rotate_pcd(pcd, T_mis)

        pcd_gt = [torch.FloatTensor(pcd_gt).to(DEVICE)]
        pcd_mis = [torch.FloatTensor(pcd_mis).to(DEVICE)]
        rgb_img = torch.unsqueeze(rgb_img.to(DEVICE), dim=0)
        depth_img = torch.unsqueeze(depth_img.to(DEVICE), dim=0)
        T_mis = torch.unsqueeze(torch.Tensor(T_mis).to(DEVICE), dim=0)
        T_gt = torch.unsqueeze(torch.Tensor(T_gt).to(DEVICE), dim=0)

        # print(T_mis.shape, T_gt.shape, len(pcd_gt), len(pcd_mis))

        st_inf_time = time.time()
        pcd_pred, T_pred, delta_q_pred, delta_t_pred = model(rgb_img, 
                                                             depth_img,  
                                                             pcd_mis, 
                                                             torch.Tensor(T_mis).to(DEVICE))
        end_inf_time = time.time() - st_inf_time

        # translational_loss = reg_loss(delta_q_gt, delta_t_gt, delta_q_pred, delta_t_pred)
        # rotational_loss = rot_loss(delta_q_gt, delta_q_pred)
        # pointcloud_loss = pcd_loss(pcd_gt, pcd_pred)
        # loss = translational_loss + rotational_loss + pointcloud_loss
        # loss = crit(output, targets)

        # val_loss += loss.item()
        # trans_loss_epoch += translational_loss.item()
        # rot_loss_epoch += rotational_loss.item()
        # pcd_loss_epoch += pointcloud_loss.item()

        # print(f'L1 = {translational_loss}| L2 = {rotational_loss} | L3 = {pointcloud_loss}')
        e_x, e_y, e_z, e_t, e_yaw, e_pitch, e_roll, e_r, dR = criteria.test_metrics(T_pred, T_gt)

        ex_epoch.append(e_x.item())
        ey_epoch.append(e_y.item())
        ez_epoch.append(e_z.item())
        et_epoch.append(e_t.item())
        eyaw_epoch.append(e_yaw.item()*180/torch.pi)
        eroll_epoch.append(e_roll.item()*180/torch.pi)
        epitch_epoch.append(e_pitch.item()*180/torch.pi)
        er_epoch.append(e_r.item()*180/torch.pi)
        dR_epoch.append(dR.item())
        time_iter.append(end_inf_time*1000)

        # update
        T_prev = torch.squeeze(T_pred).detach().cpu().numpy()

        process.set_description('Testing: ')
        process.set_postfix(et=e_t.item())
    
    # ex_epoch /= len(loader)
    # ey_epoch /= len(loader)
    # ez_epoch /= len(loader)
    # et_epoch /= len(loader)
    # eyaw_epoch /= len(loader)
    # eroll_epoch /= len(loader)
    # epitch_epoch /= len(loader)
    # er_epoch /= len(loader)
    # dR_epoch /= len(loader)

    print("== X AXIS ==")
    print("x-axis error mean:", f"{100*np.asarray(ex_epoch).mean():.4f}", "cm")
    print("x-axis error median:", f"{np.median(100*np.asarray(ex_epoch)):.4f}", "cm")
    print("x-axis error std:", f"{np.std(100*np.asarray(ex_epoch)):.4f}", "cm")

    print("== Y AXIS ==")
    print("y-axis error mean:", f"{100*np.asarray(ey_epoch).mean():.4f}", "cm")
    print("y-axis error median:", f"{np.median(100*np.asarray(ey_epoch)):.4f}", "cm")
    print("y-axis error std:", f"{np.std(100*np.asarray(ey_epoch)):.4f}", "cm")

    print("== Z AXIS ==")
    print("z-axis error mean:", f"{100*np.asarray(ez_epoch).mean():.4f}", "cm")
    print("z-axis error median:", f"{np.median(100*np.asarray(ez_epoch)):.4f}", "cm")
    print("z-axis error std:", f"{np.std(100*np.asarray(ez_epoch)):.4f}", "cm")

    print("== ET ==")
    print("Translation error:", f"{100*np.asarray(et_epoch).mean():.4f}", "cm")
    print("Translation median:", f"{np.median(100*np.asarray(et_epoch)):.4f}", "cm")
    print("Translation std:", f"{np.std(100*np.asarray(et_epoch)):.4f}", "cm")

    print("== YAW ==")
    print("yaw error mean:", f"{np.asarray(eyaw_epoch).mean():.4f}", f"\N{DEGREE SIGN}")
    print("yaw error median:", f"{np.median(100*np.asarray(eyaw_epoch)):.4f}", f"\N{DEGREE SIGN}")
    print("yaw error std:", f"{np.std(100*np.asarray(eyaw_epoch)):.4f}", f"\N{DEGREE SIGN}")

    print("== PITCH ==")
    print("pitch error mean:", f"{np.asarray(epitch_epoch).mean():.4f}", f"\N{DEGREE SIGN}")
    print("pitch error median:", f"{np.median(100*np.asarray(epitch_epoch)):.4f}", f"\N{DEGREE SIGN}")
    print("pitch error std:", f"{np.std(100*np.asarray(epitch_epoch)):.4f}", f"\N{DEGREE SIGN}")
    
    print("== ROLL ==")
    print("roll error mean:", f"{np.asarray(eroll_epoch).mean():.4f}", f"\N{DEGREE SIGN}")
    print("roll error median:", f"{np.median(100*np.asarray(eroll_epoch)):.4f}", f"\N{DEGREE SIGN}")
    print("roll error std:", f"{np.std(100*np.asarray(eroll_epoch)):.4f}", f"\N{DEGREE SIGN}")
    
    print("== ER ==")
    print("Rotation error mean:", f"{np.asarray(er_epoch).mean():.4f}", f"\N{DEGREE SIGN}")
    print("Rotation error median:", f"{np.median(100*np.asarray(er_epoch)):.4f}", f"\N{DEGREE SIGN}")
    print("Rotation error std:", f"{np.std(100*np.asarray(er_epoch)):.4f}", f"\N{DEGREE SIGN}")

    print("== GD ==")
    print("geodesic dist:", f"{np.asarray(dR_epoch).mean():.4f}")
    
    print("== Inference Time ==")
    print("\nInference Time mean:", f"{np.asarray(time_iter).mean():.2f}", "ms")
    print("\nInference Time min:", f"{np.asarray(time_iter).min():.2f}", "ms")
    print("\nInference Time max:", f"{np.asarray(time_iter).max():.2f}", "ms")

    fig = plt.figure()
    plt.plot(np.asarray(ex_epoch), label='ex')
    plt.plot(np.asarray(ey_epoch), label='ey')
    plt.plot(np.asarray(ez_epoch), label='ez')
    plt.plot(np.asarray(et_epoch), label='et')
    plt.legend()
    fig.savefig('plot_test_trans.png')

    fig1 = plt.figure()
    plt.plot(np.asarray(eyaw_epoch), label='yaw')
    plt.plot(np.asarray(epitch_epoch), label='pitch')
    plt.plot(np.asarray(eroll_epoch), label='roll')
    plt.plot(np.asarray(er_epoch), label='er')
    plt.legend()
    fig1.savefig('plot_test_rot.png')

    fig2 = plt.figure()
    plt.plot(np.asarray(dR_epoch), label='dR')
    fig2.savefig('plot_test_dr.png')

    new_img = Image.open(img_path).convert("RGB")
    new_img = (T.ToTensor())(new_img)

    pcd_mis_vis = rotate_pcd(pcd, T_mis_list[0])

    rgb_mono_projection_(im_size,
                         K_int,
                         pcd_mis_vis,
                         pcd_pred[0].detach().cpu().numpy(),
                         pcd_gt[0].detach().cpu().numpy(), 
                         new_img)

    # trans_loss_epoch /= len(loader)
    # rot_loss_epoch /= len(loader)
    # pcd_loss_epoch /= len(loader)

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
    fig1.savefig('test_comparison_KITTI_odometry_iter.png', transparent=False)


if __name__ == "__main__":
    ### Image Preprocessing
    rgb_transform = T.Compose([T.Resize((RESIZE_IMG[0], RESIZE_IMG[1])),
                               T.ToTensor(),
                               T.Normalize(mean=[0.33, 0.36, 0.33], 
                                           std=[0.30, 0.31, 0.32])])
    
    depth_transform = T.Compose([T.Resize((RESIZE_IMG[0], RESIZE_IMG[1])),
                                 T.Normalize(mean=[0.0404], 
                                             std=[6.792])])
    
    ds_test = dataset.KITTI_Odometry_TestOnly(rootdir="../ETRI_Project_Auto_Calib/datasets/KITTI-Odometry/",
                            sequences=[0],
                            camera_id=CAM_ID,
                            frame_step=1,
                            n_scans=30,
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
                             shuffle = False, 
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
    
    test(model, test_loader, criterion)
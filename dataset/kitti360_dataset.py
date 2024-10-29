import os
import glob
import yaml
import torch
from torch.utils.data.dataset import Dataset 
from torchvision.transforms import transforms as T
from torchvision.utils import save_image
import numpy as np
from numpy.linalg import inv
import open3d as o3d
from PIL import Image
import mathutils
import csv
import pickle
import pandas as pd
import ast
import re
import random
import warnings
warnings.filterwarnings("ignore")

from utils.helpers import (pcd_extrinsic_transform,
                           pcd_extrinsic_transform_torch,
                            depth_image_projection_monocular, 
                            depth_image_projection_fisheye,
                            load_pcd, 
                            load_calib_gt, 
                            load_cam_intrinsic,
                            load_fisheye_intrinsic,
                            rot2qua, qua2rot) 

class KITTI360_Dataset(Dataset):
    def __init__(self,
                 rootdir="./datasets/KITTI-360/", 
                 sequences = [0], 
                 camera_id = "00",
                 frame_step = 1,
                 n_scans = None,
                 voxel_size = None,
                 max_trans = [1.5, 1.0, 0.5, 0.2, 0.1],
                 max_rot = [20, 10, 5, 2, 1],
                 range_img = False, # depth image (z-buffer) or range image (euclidian distance)
                 rgb_transform = None,
                 depth_transform = None,
                 device = 'cuda'):
        super(KITTI360_Dataset, self).__init__()

        self.rootdir = rootdir
        self.sequences = sequences
        self.camera_id = camera_id
        self.scans = []
        self.voxel_size = voxel_size
        self.max_trans = max_trans
        self.max_rot = max_rot
        self.rgb_transform = T.ToTensor() if rgb_transform is None else rgb_transform
        self.depth_transform = depth_transform
        self.device = device

        if self.camera_id == "00":
            self.T_velo_to_cam_gt = load_calib_gt(self.rootdir)[0]
        elif self.camera_id == "01":
            self.T_velo_to_cam_gt = load_calib_gt(self.rootdir)[1]
        elif self.camera_id == "02":
            self.T_velo_to_cam_gt = load_calib_gt(self.rootdir)[2]
        else:
            self.T_velo_to_cam_gt = load_calib_gt(self.rootdir)[3]

        if self.camera_id == "00" or self.camera_id == "01":
            self.im_size, self.K_int = load_cam_intrinsic(self.rootdir, camera_id)
            self.depth_img_gen = depth_image_projection_monocular(image_size=self.im_size,
                                                                  K_int=self.K_int, range_mode=range_img)
        else:
            self.im_size, self.K_int, distort_params = load_fisheye_intrinsic(self.rootdir, self.camera_id)
            self.distort_params = (distort_params['xi'], 
                                    distort_params['k1'],
                                    distort_params['k2'],
                                    distort_params['p1'],
                                    distort_params['p2'])
            self.depth_img_gen = depth_image_projection_fisheye(image_size=self.im_size,
                                                                K_int=self.K_int,
                                                                distort_params=self.distort_params,
                                                                range_mode=range_img)

        for i in range(len(max_trans)):
            for sequence in sequences:
                if self.camera_id == "00" or self.camera_id == "01":
                    img_path = sorted(glob.glob(os.path.join(
                        self.rootdir, 
                        "data_2d_raw", 
                        "2013_05_28_drive_%04d_sync"%(sequence), 
                        "image_%s/data_rect"%(camera_id), 
                        "*.png"
                    )))
                else:
                    img_path = sorted(glob.glob(os.path.join(
                        self.rootdir, 
                        "data_2d_raw", 
                        "2013_05_28_drive_%04d_sync"%(sequence), 
                        "image_%s/data_rgb"%(camera_id), 
                        "*.png"
                    )))

                pcl_path = sorted(glob.glob(os.path.join(
                    self.rootdir, 
                    "data_3d_raw", 
                    "2013_05_28_drive_%04d_sync"%(sequence), 
                    "velodyne_points/data", 
                    "*.bin"
                )))

                for img, pcl in zip(img_path, pcl_path):
                    self.scans.append({"img_path": img,
                                        "pcl_path": pcl, 
                                        "sequence": sequence,
                                        "max_trans": max_trans[i], 
                                        "max_rot": max_rot[i]})

        scan_len = len(self.scans)
        scan_idx = list(range(0, scan_len, frame_step))
        self.scans = [self.scans[i] for i in scan_idx]

        # limit the data length
        if n_scans is not None:
            self.scans = self.scans[:n_scans]

        self.rotate_pcd = pcd_extrinsic_transform(crop=False) 
        
    def __len__(self):
    # print('scans', len(self.scans))
        return len(self.scans)
    
    def __getitem__(self, index):
        data = {'T_gt': torch.Tensor(self.T_velo_to_cam_gt).to(self.device)}
        scan = self.scans[index]

        img_path = scan["img_path"]
        pcl_path = scan["pcl_path"]
        sequence = scan["sequence"]
        max_trans = scan["max_trans"]
        max_rot = scan["max_rot"]

        # filename and frame_id
        filename = os.path.basename(img_path)
        frame_id = os.path.splitext(filename)[0]

        # transform 2d fisheye image
        img_raw = Image.open(img_path).convert("RGB")
        img = self.rgb_transform(img_raw).float().to(self.device)

        # load and preprocess point cloud data (outlier removal & voxel downsampling)
        pcd = load_pcd(pcl_path)[:,:3]
        pcd = self.voxel_downsampling(pcd, self.voxel_size) if self.voxel_size is not None else pcd

        # generate misalignment in extrinsic parameters (labels)
        while True:
            delta_R_gt, delta_t_gt = self.generate_misalignment(max_rot, max_trans)
            delta_q_gt = rot2qua(delta_R_gt)
            delta_T = np.hstack((delta_R_gt, np.expand_dims(delta_t_gt, axis=1)))
            delta_T = np.vstack((delta_T, np.array([0., 0., 0., 1.])))
            T_mis = np.matmul(delta_T, self.T_velo_to_cam_gt)

            # generate 2d depth image from point cloud
            depth_img_error = self.depth_img_gen(pcd, T_mis)

            # check if the depth image is totally blank or not
            if torch.count_nonzero(depth_img_error) > 0.03*torch.numel(depth_img_error) or self.camera_id == '02' or self.camera_id == '03':
                break
        
        if self.depth_transform is not None:
            depth_img_error = self.depth_transform(depth_img_error).to(self.device)
        else:
            depth_img_error = depth_img_error.to(self.device)

        pcd_gt = self.rotate_pcd(pcd, self.T_velo_to_cam_gt)
        pcd_mis = self.rotate_pcd(pcd_gt, delta_T)
        # pcd_mis2 = self.rotate_pcd(pcd, T_mis)

        # print(np.all(np.round(pcd_mis,9) == np.round(pcd_mis2,9)))

        pcd_gt = torch.FloatTensor(pcd_gt).to(self.device)
        pcd_mis = torch.FloatTensor(pcd_mis).to(self.device)

        delta_t_gt = torch.Tensor(delta_t_gt).to(self.device)
        delta_q_gt = torch.Tensor(delta_q_gt).to(self.device)

        # sample for dataloader
        data["frame_id"] = frame_id
        data["img_path"] = img_path
        data["sequence"] = sequence
        data["img"] = img
        data["pcd_gt"] = pcd_gt                # target point cloud (ground truth) if necessary
        data['pcd_mis'] = pcd_mis           # misaligned point cloud
        # data["pcd_error"] = pcd_error
        # data["depth_img_true"] = depth_img_true     # target depth image (ground truth) if necessary
        data["depth_img_error"] = depth_img_error
        data["delta_t_gt"] = delta_t_gt             # translation error ground truth
        data["delta_q_gt"] = delta_q_gt             # rotation error ground truth
        
        return data

    def voxel_downsampling(self, points, voxel_size):
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd = pcd.voxel_down_sample(voxel_size)
        out_points = np.array(pcd.points, dtype=np.float32)

        return out_points

    def generate_misalignment(self, max_rot = 30, max_trans = 0.5, rot_order='XYZ'):
        rot_z = np.random.uniform(-max_rot, max_rot) * (3.141592 / 180.0)
        rot_y = np.random.uniform(-max_rot, max_rot) * (3.141592 / 180.0)
        rot_x = np.random.uniform(-max_rot, max_rot) * (3.141592 / 180.0)
        trans_x = np.random.uniform(-max_trans, max_trans)
        trans_y = np.random.uniform(-max_trans, max_trans)
        trans_z = np.random.uniform(-max_trans, max_trans)

        R_perturb = mathutils.Euler((rot_x, rot_y, rot_z)).to_matrix()
        t_perturb = np.array([trans_x, trans_y, trans_z])
            
        return np.array(R_perturb), t_perturb

def normalize_pc(points):
	centroid = np.mean(points, axis=0)
	points -= centroid
	furthest_distance = np.max(np.sqrt(np.sum(abs(points)**2,axis=-1)))
	points /= furthest_distance

	return points

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
    # torch.cuda.empty_cache()

    ALL_SEQUENCE = [0,2,3,4,5,6,7,9,10]
    DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"

    RESIZE_IMG = (192, 640)

    rgb_transform = T.Compose([T.Resize((RESIZE_IMG[0], RESIZE_IMG[1])),
                               T.ToTensor()])#,
                            #    T.Normalize(mean=[0.33, 0.36, 0.33], 
                            #                std=[0.30, 0.31, 0.32])])
    depth_transform = T.Compose([T.Resize((RESIZE_IMG[0], RESIZE_IMG[1]))])

    dset = KITTI360_Dataset(rootdir="/home/indowicom/umam/ext_auto_calib_camlid/ETRI_Project_Auto_Calib/datasets/KITTI-360/",
                            sequences=ALL_SEQUENCE,
                            camera_id="00",
                            frame_step=1,
                            n_scans=None,
                            voxel_size=None,
                            max_trans=[0.5],
                            max_rot=[10],
                            rgb_transform=rgb_transform,
                            depth_transform=depth_transform,
                            device=DEVICE
                            )
    
    check_data(dset)

    sample = dset[random.randint(0,len(dset))]

    depth_img = (T.ToPILImage())(sample["depth_img_error"])
    rgb_img = (T.ToPILImage())(sample["img"])
    depth_img.save("test_depth_z.png")
    rgb_img.save("test_rgb.png")

    

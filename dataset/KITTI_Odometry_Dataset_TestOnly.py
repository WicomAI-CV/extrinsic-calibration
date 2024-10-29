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
import math
import csv
import pickle
import pandas as pd
import ast
import re
import random
import warnings
from pykitti import odometry
warnings.filterwarnings("ignore")

from utils.helpers import pcd_extrinsic_transform, load_pcd

class KITTI_Odometry_TestOnly(Dataset):
    def __init__(self,
                 rootdir="../ETRI_Project_Auto_Calib/datasets/KITTI-Odometry/", 
                 sequences = [0, 1, 2], 
                 camera_id = "2",
                 frame_step = 1,
                 n_scans = None,
                 voxel_size = None,
                 max_trans = [1.5, 1.0, 0.5, 0.2, 0.1],
                 max_rot = [20, 10, 5, 2, 1],
                 range_img = False, # depth image (z-buffer) or range image (euclidian distance)
                 rgb_transform = None,
                 depth_transform = None,
                 device = 'cpu'):
        super(KITTI_Odometry_TestOnly, self).__init__()

        self.rootdir = rootdir
        self.sequences = sequences
        self.camera_id = camera_id
        self.scans = []
        self.voxel_size = voxel_size
        self.max_trans = max_trans
        self.max_rot = max_rot
        self.range_img = range_img
        self.rgb_transform = T.ToTensor() if rgb_transform is None else rgb_transform
        self.depth_transform = depth_transform
        self.device = device

        assert len(max_rot) == len(max_trans)

        for i in range(len(max_trans)):
            for sequence in sequences:
                calib_path = os.path.join(
                        self.rootdir, 
                        "dataset", 
                        "sequences",
                        "%02d"%(sequence), 
                        "calib.txt"
                )

                img_path = sorted(glob.glob(os.path.join(
                        self.rootdir, 
                        "dataset", 
                        "sequences",
                        "%02d"%(sequence), 
                        "image_%s"%(camera_id), 
                        "*.png"
                )))
                
                pcl_path = sorted(glob.glob(os.path.join(
                        self.rootdir, 
                        "dataset", 
                        "sequences",
                        "%02d"%(sequence),
                        "velodyne", 
                        "*.bin"
                )))

                (T_cam_lidar_gt, K_cam, im_size) = self.load_calib(calib_path, sequence)

                assert len(img_path) == len(pcl_path)

                for img, pcl in zip(img_path, pcl_path):
                    self.scans.append({"img_path": img,
                                        "pcl_path": pcl, 
                                        "sequence": sequence,
                                        "max_trans": max_trans[i], 
                                        "max_rot": max_rot[i],
                                        "T_gt": T_cam_lidar_gt,
                                        "K_int": K_cam,
                                        "im_size": im_size})
                    
        scan_len = len(self.scans)
        scan_idx = list(range(0, scan_len, frame_step))
        self.scans = [self.scans[i] for i in scan_idx]

        # limit the data length
        if n_scans is not None:
            self.scans = self.scans[:n_scans]

        self.rotate_pcd = pcd_extrinsic_transform(crop=False) 

    def __len__(self):
        return len(self.scans)

    def __getitem__(self, index):
        scan = self.scans[index]

        img_path = scan["img_path"]
        pcl_path = scan["pcl_path"]
        sequence = scan["sequence"]
        max_trans = scan["max_trans"]
        max_rot = scan["max_rot"]
        T_gt = scan["T_gt"]
        K_int = scan["K_int"] 
        im_size = scan["im_size"]

        data = {'T_gt': T_gt}

        # filename and frame_id
        filename = os.path.basename(img_path)
        frame_id = os.path.splitext(filename)[0]

        # transform 2d fisheye image
        img_raw = Image.open(img_path).convert("RGB")
        img = self.rgb_transform(img_raw).float().to(self.device)

        # load and preprocess point cloud data (outlier removal & voxel downsampling)
        pcd = load_pcd(pcl_path)[:,:3]
        pcd = self.voxel_downsampling(pcd, self.voxel_size) if self.voxel_size is not None else pcd

        # # generate misalignment in extrinsic parameters (labels)
        # while True:
        #     delta_R_gt, delta_t_gt = self.generate_misalignment(max_rot, max_trans)
        #     delta_q_gt = rot2qua(delta_R_gt)
        #     delta_T = np.hstack((delta_R_gt, np.expand_dims(delta_t_gt, axis=1)))
        #     delta_T = np.vstack((delta_T, np.array([0., 0., 0., 1.])))
        #     T_mis = np.matmul(delta_T, T_gt)

        #     # generate 2d depth image from point cloud
        #     depth_img_error = self.depth_img_gen(pcd, T_mis, K_int, im_size, self.range_img)

        #     # check if the depth image is totally blank or not
        #     if torch.count_nonzero(depth_img_error) > 0.03*torch.numel(depth_img_error):
        #         break
        
        # if self.depth_transform is not None:
        #     depth_img_error = self.depth_transform(depth_img_error).to(self.device)
        # else:
        #     depth_img_error = depth_img_error.to(self.device)

        pcd_gt = self.rotate_pcd(pcd, T_gt)
        # pcd_mis = self.rotate_pcd(pcd_gt, delta_T)
        # pcd_mis2 = self.rotate_pcd(pcd, T_mis)

        # print("pcd_check:", np.all(np.round(pcd_mis,8) == np.round(pcd_mis2,8)))

        # pcd_gt = torch.FloatTensor(pcd_gt).to(self.device)
        # pcd = torch.FloatTensor(pcd).to(self.device)

        # delta_t_gt = torch.Tensor(delta_t_gt).to(self.device)
        # delta_q_gt = torch.Tensor(delta_q_gt).to(self.device)

        # sample for dataloader
        data["frame_id"] = frame_id
        data["img_path"] = img_path
        data["sequence"] = sequence
        data["img"] = img
        data["pcd_gt"] = pcd_gt                # target point cloud (ground truth) if necessary
        data['pcd'] = pcd                      # misaligned point cloud
        data['K_int'] = K_int
        data['im_size'] = im_size
        data["max_trans"] = max_trans
        data["max_rot"] = max_rot
        # data["pcd_error"] = pcd_error
        # data["depth_img_true"] = depth_img_true     # target depth image (ground truth) if necessary
        # data["depth_img_error"] = depth_img_error
        # data["delta_t_gt"] = delta_t_gt             # translation error ground truth
        # data["delta_q_gt"] = delta_q_gt             # rotation error ground truth

        return data

    def load_calib(self, calib_file, sequence):
        read_data = {}

        with open(calib_file, 'r') as f:
            for line in f.readlines():
                try:
                    key, value = line.split(':', 1)
                except ValueError:
                    key, value = line.split(' ', 1)
                # The only non-float values in these files are dates, which
                # we don't care about anyway
                try:
                    read_data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass

        # Create 3x4 projection matrices
        P_rect_00 = np.reshape(read_data['P0'], (3, 4))
        P_rect_10 = np.reshape(read_data['P1'], (3, 4))
        P_rect_20 = np.reshape(read_data['P2'], (3, 4))
        P_rect_30 = np.reshape(read_data['P3'], (3, 4))

        data = {}

        data['P_rect_00'] = P_rect_00
        data['P_rect_10'] = P_rect_10
        data['P_rect_20'] = P_rect_20
        data['P_rect_30'] = P_rect_30

        # Compute the rectified extrinsics from cam0 to camN
        T1 = np.eye(4)
        T1[0, 3] = P_rect_10[0, 3] / P_rect_10[0, 0]
        T2 = np.eye(4)
        T2[0, 3] = P_rect_20[0, 3] / P_rect_20[0, 0]
        T3 = np.eye(4)
        T3[0, 3] = P_rect_30[0, 3] / P_rect_30[0, 0]

        # Compute the velodyne to rectified camera coordinate transforms
        data['T_cam0_velo'] = np.reshape(read_data['Tr'], (3, 4))
        data['T_cam0_velo'] = np.vstack([data['T_cam0_velo'], [0, 0, 0, 1]])
        data['T_cam1_velo'] = T1.dot(data['T_cam0_velo'])
        data['T_cam2_velo'] = T2.dot(data['T_cam0_velo'])
        data['T_cam3_velo'] = T3.dot(data['T_cam0_velo'])

        # Compute the camera intrinsics
        data['K_cam0'] = P_rect_00[0:3, 0:3]
        data['K_cam1'] = P_rect_10[0:3, 0:3]
        data['K_cam2'] = P_rect_20[0:3, 0:3]
        data['K_cam3'] = P_rect_30[0:3, 0:3]

        # im size (for projection)
        if sequence in [0, 1, 2, 13, 14, 15, 16, 17, 18, 19, 20, 21]:
            im_size = (1241, 376)
        elif sequence in [4, 5, 6, 7, 8, 9, 10, 11, 12]:
            im_size = (1226, 370)
        else:
            im_size = (1242, 375)

        if self.camera_id == "0":
            return (data['T_cam0_velo'], data['K_cam0'], im_size)
        elif self.camera_id == "1":
            return (data['T_cam1_velo'], data['K_cam1'], im_size)
        elif self.camera_id == "2":
            return (data['T_cam2_velo'], data['K_cam2'], im_size)
        else:
            return (data['T_cam3_velo'], data['K_cam3'], im_size)
        
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
    
    def depth_img_gen(self, point_cloud, T_ext, K_int, im_size, range_mode=False):
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

    dset = KITTI_Odometry_TestOnly(rootdir="../ETRI_Project_Auto_Calib/datasets/KITTI-Odometry/",
                            sequences=list(range(1,22)),
                            camera_id="2",
                            frame_step=2,
                            n_scans=None,
                            voxel_size=None,
                            max_trans = [1.5, 1.0, 0.5, 0.2, 0.1],
                            max_rot = [20, 10, 5, 2, 1],
                            rgb_transform=rgb_transform,
                            depth_transform=depth_transform,
                            device='cpu'
                            )
    print("dset len:",len(dset))

    check_data(dset)

    # sample = dset[random.randint(0,len(dset))]

    # depth_img = (T.ToPILImage())(sample["depth_img_error"])
    # rgb_img = (T.ToPILImage())(sample["img"])
    # depth_img.save("test_depth_z.png")
    # rgb_img.save("test_rgb.png")


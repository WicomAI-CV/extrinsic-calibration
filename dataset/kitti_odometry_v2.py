import os
import glob
import torch
from torch.utils.data.dataset import Dataset 
from torchvision.transforms import transforms as T
import numpy as np
import open3d as o3d
from PIL import Image
import mathutils
import warnings
warnings.filterwarnings("ignore")

import utils

class KITTI_Odometry(Dataset):
    def __init__(self,
                 rootdir="../ETRI_Project_Auto_Calib/datasets/KITTI-Odometry/", 
                 sequences = [0, 1, 2], 
                 camera_id = "2",
                 frame_step = 1,
                 n_scans = None,
                 voxel_size = None,
                 depth_grid = 4 , # densify depth map
                 max_trans = [1.5, 1.0, 0.5, 0.2, 0.1],
                 max_rot = [20, 10, 5, 2, 1],
                 range_img = False, # depth image (z-buffer) or range image (euclidian distance)
                 rgb_transform = None,
                 depth_transform = None,
                 device = 'cpu'):
        super(KITTI_Odometry, self).__init__()

        self.rootdir = rootdir
        self.sequences = sequences
        self.camera_id = camera_id
        self.scans = []
        self.voxel_size = voxel_size
        self.grid = depth_grid
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

        self.rotate_pcd = utils.pcd_extrinsic_transform(crop=False) 

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

        data = {'T_gt': torch.Tensor(T_gt).to(self.device)}

        # filename and frame_id
        filename = os.path.basename(img_path)
        frame_id = os.path.splitext(filename)[0]

        # transform 2d fisheye image
        img_raw = Image.open(img_path).convert("RGB")
        img = self.rgb_transform(img_raw).float().to(self.device)

        # load and preprocess point cloud data (outlier removal & voxel downsampling)
        pcd = utils.load_pcd(pcl_path)[:,:3]
        pcd = self.voxel_downsampling(pcd, self.voxel_size) if self.voxel_size is not None else pcd

        # generate misalignment in extrinsic parameters (labels)
        depth_img_true = self.depth_img_gen(pcd, T_gt, K_int, im_size, self.range_img)

        while True:
            delta_R_gt, delta_t_gt = self.generate_misalignment(max_rot, max_trans)
            delta_q_gt = utils.rot2qua(delta_R_gt)
            delta_T = np.hstack((delta_R_gt, np.expand_dims(delta_t_gt, axis=1)))
            delta_T = np.vstack((delta_T, np.array([0., 0., 0., 1.])))
            T_mis = np.matmul(delta_T, T_gt)

            # generate 2d depth image from point cloud
            depth_img_error = self.depth_img_gen(pcd, T_mis, K_int, im_size, self.range_img)

            # check if the depth image is totally blank or not
            if torch.count_nonzero(depth_img_error) > 0.03*torch.numel(depth_img_error):
                break
        
        if self.depth_transform is not None:
            depth_img_error = self.depth_transform(depth_img_error).to(self.device)
            depth_img_true = self.depth_transform(depth_img_true).to(self.device)
        else:
            depth_img_error = depth_img_error.to(self.device)
            depth_img_true = depth_img_true.to(self.device)

        pcd_gt = self.rotate_pcd(pcd, T_gt)
        pcd_mis = self.rotate_pcd(pcd_gt, delta_T)
        # pcd_mis2 = self.rotate_pcd(pcd, T_mis)

        # print("pcd_check:", np.all(np.round(pcd_mis,8) == np.round(pcd_mis2,8)))

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
        data["depth_img_true"] = depth_img_true     # target depth image (ground truth) if necessary
        data["depth_img_error"] = depth_img_error
        data["delta_t_gt"] = delta_t_gt             # translation error ground truth
        data["delta_q_gt"] = delta_q_gt             # rotation error ground truth
        data['K_int'] = K_int
        data['im_size'] = im_size

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
        if d_proj.shape[0] > 0:
            max_depth = np.max(d_proj)
            d_proj = np.array([np.interp(d, [0, max_depth], [0, 1]) for d in d_proj]) # convert depth values to [0, 1]


        point_proj = np.array([u_proj, v_proj, d_proj])
        
        image_tensor = self.depth_densifier(point_proj.T, im_size, grid = self.grid)
        image_tensor = torch.from_numpy(image_tensor).type(torch.float32)
        image_tensor = torch.unsqueeze(image_tensor, 0)

        return image_tensor
    
    def depth_densifier(self, Pts, im_size, grid=5):
        m, n = im_size

        # from: https://github.com/BerensRWU/DenseMap/issues/7
        ng = 2 * grid + 1

        # Initialize matrices
        mX = np.full((n, m), np.inf)
        mY = np.full((n, m), np.inf)
        mD = np.zeros((n, m))

        # Compute linear indices
        linearindex = np.ravel_multi_index((np.round(Pts[:, 1]).astype(int), np.round(Pts[:, 0]).astype(int)), (n, m))

        # Populate matrices
        mX.flat[linearindex] = Pts[:, 0] - np.round(Pts[:, 0])
        mY.flat[linearindex] = Pts[:, 1] - np.round(Pts[:, 1])
        mD.flat[linearindex] = Pts[:, 2]

        # Prepare KmX, KmY, KmD
        KmX = np.zeros((ng, ng, n - ng, m - ng))
        KmY = np.zeros((ng, ng, n - ng, m - ng))
        KmD = np.zeros((ng, ng, n - ng, m - ng))

        for i in range(ng):
            for j in range(ng):
                KmX[i, j] = mX[i:n - ng + i, j:m - ng + j] - grid - 1 + i
                KmY[i, j] = mY[i:n - ng + i, j:m - ng + j] - grid - 1 + j
                KmD[i, j] = mD[i:n - ng + i, j:m - ng + j]

        S = np.zeros(KmD[0, 0].shape)
        Y = np.zeros(KmD[0, 0].shape)

        for i in range(ng):
            for j in range(ng):
                s = 1.0 / np.sqrt(KmX[i, j] ** 2 + KmY[i, j] ** 2)
                Y += s * KmD[i, j]
                S += s

        S[S == 0] = 1
        out = np.zeros((n, m))
        out[grid + 1: -grid, grid + 1: -grid] = Y / S

        return out
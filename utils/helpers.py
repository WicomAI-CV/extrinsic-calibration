import os
import numpy as np
import math
import torch
from numpy.linalg import inv
import yaml

class pcd_extrinsic_transform: # transform PCD into fisheye camera reference frame.
    def __init__(self, crop = True):
        self.crop = crop

    def __call__(self, point_cloud, T_ext):
        # extrinsic transformation from velodyne reference frame to fisheye camera reference frame
        n_points = point_cloud.shape[0]
        pcd_cam = np.matmul(T_ext, np.hstack((point_cloud, np.ones((n_points, 1)))).T).T  # (P_velo -> P_fisheye)
        pcd_cam = pcd_cam[:,:3]
        z_axis = pcd_cam[:,2]  # Extract the z_axis to determine 3D points that located in front of the camera.

        if self.crop:
            condition = (z_axis>=0)
            new_pcd = pcd_cam[condition]
        else:
            new_pcd = pcd_cam

        # print(point_cloud.shape)
        # print(new_pcd.shape)

        return new_pcd
    
class pcd_extrinsic_transform_torch: # transform PCD into fisheye camera reference frame.
    def __init__(self, crop = True):
        self.crop = crop

    def __call__(self, point_cloud, T_ext):
        # extrinsic transformation from velodyne reference frame to fisheye camera reference frame
        device_ = point_cloud.device
        n_points = point_cloud.shape[0]
        pcd_fisheye = torch.matmul(T_ext, torch.hstack((point_cloud, torch.ones(n_points, 1).to(device_))).T).T  # (P_velo -> P_fisheye)
        pcd_fisheye = pcd_fisheye[:,:3]
        z_axis = pcd_fisheye[:,2]  # Extract the z_axis to determine 3D points that located in front of the camera.

        if self.crop:
            condition = (z_axis>=0)
            new_pcd = pcd_fisheye[condition]
        else:
            new_pcd = pcd_fisheye

        # print(point_cloud.shape)
        # print(new_pcd.shape)

        return new_pcd

class depth_image_projection_fisheye:
    def __init__(self, image_size, K_int, distort_params, expand=False, range_mode=False):
        self.H, self.W = image_size
        
        # intrinsic parameters
        self.K_int = K_int 

        # mirror and distortion parameters
        self.xi, self.k1, self.k2, self.p1, self.p2 = distort_params

        # if true, return depth image as 3 channel image
        self.expand = expand

        # depth (z-buffer) or range (distance)
        self.range_mode = range_mode
        
    def __call__(self, point_cloud, T_ext):
        # extrinsic transformation from velodyne reference frame to fisheye camera reference frame
        n_points = point_cloud.shape[0]
        pcd_fisheye = np.matmul(T_ext, np.hstack((point_cloud, np.ones((n_points, 1)))).T).T  # (P_velo -> P_fisheye)
        pcd_fisheye = pcd_fisheye[:,:3]
        z_axis = pcd_fisheye[:,2]  # Extract the z_axis to determine 3D points that located in front of the camera.

        # Project to unit sphere (every vector magnitude is 1)
        pcd_sphere = np.array([x/np.linalg.norm(x) for x in pcd_fisheye])

        # Reference frame center moved to Cp = xi => (X, Y, Z) --> (X, Y, Z + Xi)
        pcd_sphere[:,2] += self.xi

        # Project into normalized plane => (X/(Z + Xi), Y/(Z + Xi), 1)
        norm_plane = np.array([x/x[2] for x in pcd_sphere])

        # distortion ((u',v') = radial(u,v) + tangential(u,v))
        distorted_plane = np.array([self.distortion(pixel[0], pixel[1]) for pixel in norm_plane])

        # final projection using generalized camera projection (intrinsic matrix K)
        distorted_pixels = np.hstack((distorted_plane, np.expand_dims(norm_plane[:,2], axis=1)))
        pixel_proj = np.matmul(self.K_int, distorted_pixels.T).T

        u = np.asarray(pixel_proj[:, 0], dtype=np.int32)
        v = np.asarray(pixel_proj[:, 1], dtype=np.int32)

        # depth calculation of each point
        depth = np.array([np.linalg.norm(x) for x in point_cloud]) if self.range_mode else z_axis

        condition = (0<=u)*(u<self.W)*(0<=v)*(v<self.H)*(depth>0)*(z_axis>=0)
        # print(np.min(z_axis))

        u_proj = u[condition]
        v_proj = v[condition]
        d_proj = depth[condition]

        max_depth = np.max(d_proj)
        d_proj = np.array([np.interp(d, [0, max_depth], [1, 0]) for d in d_proj]) # convert depth values to [0, 255]

        # image array generation
        image_tensor = torch.zeros(self.H, self.W, dtype=torch.float32)
        image_tensor[v_proj,u_proj] = torch.from_numpy(d_proj).type(torch.float32) #(1400, 1400, )

        # expand dimension to (1400, 1400, 3)
        image_tensor = torch.unsqueeze(image_tensor, 0)
        if self.expand:
            image_tensor = image_tensor.expand(3, -1, -1)

        return image_tensor

    def distortion(self, x, y):
        r = math.sqrt(x**2 + y**2)

        # radial distortion
        D = 1 + (self.k1)*(r**2) + (self.k2)*(r**4)

        # tangential distortion
        dx = 2*self.p1*x*y + self.p2*((r**2) + 2*(x**2))
        dy = self.p1*((r**2) + 2*(y**2)) + 2*self.p2*x*y

        # total distortion
        x_distorted = D*x + dx
        y_distorted = D*y + dy

        return np.array([x_distorted, y_distorted])

def load_calib_gt(rootdir = "./datasets/KITTI-360/"):
    # camera extrinsic and intrinsic parameter transformation
    cam_to_pose_path = os.path.join(rootdir, "calibration", "calib_cam_to_pose.txt")
    cam_to_velo_path = os.path.join(rootdir, "calibration", "calib_cam_to_velo.txt")

    data_cam_to_pose = {}
    data_cam_to_velo = {}

    with open(cam_to_pose_path, 'r') as f:
        for line in f.readlines():
            key, value = line.split(':',1)

            try:
                data_cam_to_pose[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    with open(cam_to_velo_path, 'r') as f:
        for line in f.readlines():
            try:
                data_cam_to_velo = np.array([float(x) for x in line.split()])
            except ValueError:
                pass

    T_cam0_to_pose = np.reshape(data_cam_to_pose['image_00'], (3, 4))
    T_cam1_to_pose = np.reshape(data_cam_to_pose['image_01'], (3, 4))
    T_cam2_to_pose = np.reshape(data_cam_to_pose['image_02'], (3, 4))
    T_cam3_to_pose = np.reshape(data_cam_to_pose['image_03'], (3, 4))
    T_cam0_to_velo = np.reshape(data_cam_to_velo, (3, 4))

    last_row = np.array([0, 0, 0, 1])

    T_cam0_to_pose = np.vstack((T_cam0_to_pose, last_row))
    T_cam1_to_pose = np.vstack((T_cam1_to_pose, last_row))
    T_cam2_to_pose = np.vstack((T_cam2_to_pose, last_row))
    T_cam3_to_pose = np.vstack((T_cam3_to_pose, last_row))
    T_cam0_to_velo = np.vstack((T_cam0_to_velo, last_row))

    T_velo_to_cam0 = inv(T_cam0_to_velo)
    T_velo_to_cam1 = inv(T_cam1_to_pose).dot(T_cam0_to_pose).dot(inv(T_cam0_to_velo))
    T_velo_to_cam2 = inv(T_cam2_to_pose).dot(T_cam0_to_pose).dot(inv(T_cam0_to_velo))
    T_velo_to_cam3 = inv(T_cam3_to_pose).dot(T_cam0_to_pose).dot(inv(T_cam0_to_velo))

    return T_velo_to_cam0, T_velo_to_cam1, T_velo_to_cam2, T_velo_to_cam3

def load_fisheye_intrinsic(rootdir = "./datasets/KITTI-360/", camera_id = '02'):
    intrinsic_path = os.path.join(rootdir, 
                                    "calibration", 
                                    "image_%s.yaml"%(camera_id))
    
    with open(intrinsic_path, 'r') as file:
        intrinsic = yaml.safe_load(file)

    # 2D image dimension
    H = intrinsic['image_height']
    W = intrinsic['image_width']
    img_size = (H, W)
    
    # mirror parameters
    xi = intrinsic['mirror_parameters']['xi']

    # distortion parameters
    k1 = intrinsic['distortion_parameters']['k1']
    k2 = intrinsic['distortion_parameters']['k2']
    p1 = intrinsic['distortion_parameters']['p1']
    p2 = intrinsic['distortion_parameters']['p2']

    # projection parameters
    gamma1 = intrinsic['projection_parameters']['gamma1']
    gamma2 = intrinsic['projection_parameters']['gamma2']
    u0 = intrinsic['projection_parameters']['u0']
    v0 = intrinsic['projection_parameters']['v0']
    
    # intrinsic matrix (K) formulation
    K_int = np.array([[gamma1, 0.0, u0],[0.0, gamma2, v0], [0.0, 0.0, 1.0]])

    # mirror & distortion parameters dictionary
    distort_params = {'xi': xi, 'k1': k1, 'k2': k2, 'p1': p1, 'p2': p2}

    return img_size, K_int, distort_params

def load_cam_intrinsic(rootdir = "./datasets/KITTI-360/", camera_id = '01'):
    calib_params = {}
    with open(os.path.join(rootdir, 'calibration', 'perspective.txt'), 'r') as file:
        for line in file.readlines():
            try:
                key, value = line.split(':', 1)
            except ValueError:
                key, value = line.split(' ', 1)
            try:
                calib_params[key] = np.array([float(x) for x in value.split()])
            except ValueError:
                pass

    im_size = (int(calib_params['S_rect_'+ camera_id][0]), 
                    int(calib_params['S_rect_'+ camera_id][1]))
    
    K_int = (np.resize(calib_params['P_rect_' + camera_id], (3,4)))[:3,:3]

    return im_size, K_int


def load_pcd(path):
    pcd = np.fromfile(path, dtype=np.float32)
    return pcd.reshape((-1, 4))

def rot2qua(matrix):
    """
    Convert a rotation matrix to quaternion.
    Args:
        matrix (torch.Tensor): [4x4] transformation matrix or [3,3] rotation matrix.

    Returns:
        torch.Tensor: shape [4], normalized quaternion
    """
    if matrix.shape == (4, 4):
        R = matrix[:-1, :-1]
    elif matrix.shape == (3, 3):
        R = matrix
    else:
        raise TypeError("Not a valid rotation matrix")
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    q = np.zeros(4)
    if tr > 0.:
        S = math.sqrt(tr+1.0) * 2
        q[0] = 0.25 * S
        q[1] = (R[2, 1] - R[1, 2]) / S
        q[2] = (R[0, 2] - R[2, 0]) / S
        q[3] = (R[1, 0] - R[0, 1]) / S
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        S = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q[0] = (R[2, 1] - R[1, 2]) / S
        q[1] = 0.25 * S
        q[2] = (R[0, 1] + R[1, 0]) / S
        q[3] = (R[0, 2] + R[2, 0]) / S
    elif R[1, 1] > R[2, 2]:
        S = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q[0] = (R[0, 2] - R[2, 0]) / S
        q[1] = (R[0, 1] + R[1, 0]) / S
        q[2] = 0.25 * S
        q[3] = (R[1, 2] + R[2, 1]) / S
    else:
        S = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q[0] = (R[1, 0] - R[0, 1]) / S
        q[1] = (R[0, 2] + R[2, 0]) / S
        q[2] = (R[1, 2] + R[2, 1]) / S
        q[3] = 0.25 * S
    return q / np.linalg.norm(q)

def qua2rot_torch(q):
    """
    Convert a quaternion to a rotation matrix
    Args:
        q (torch.Tensor): shape [4], input quaternion

    Returns:
        torch.Tensor: [3x3] homogeneous rotation matrix
    """
    assert q.shape == torch.Size([4]), "Not a valid quaternion"
    if q.norm() != 1.:
        q = q / q.norm()
    mat = torch.zeros((3, 3), device=q.device)
    mat[0, 0] = 1 - 2*q[2]**2 - 2*q[3]**2
    mat[0, 1] = 2*q[1]*q[2] - 2*q[3]*q[0]
    mat[0, 2] = 2*q[1]*q[3] + 2*q[2]*q[0]
    mat[1, 0] = 2*q[1]*q[2] + 2*q[3]*q[0]
    mat[1, 1] = 1 - 2*q[1]**2 - 2*q[3]**2
    mat[1, 2] = 2*q[2]*q[3] - 2*q[1]*q[0]
    mat[2, 0] = 2*q[1]*q[3] - 2*q[2]*q[0]
    mat[2, 1] = 2*q[2]*q[3] + 2*q[1]*q[0]
    mat[2, 2] = 1 - 2*q[1]**2 - 2*q[2]**2
    return mat
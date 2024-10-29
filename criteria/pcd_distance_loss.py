import torch
import torch.nn.functional as F
from torch import nn
# from ..utils import helpers
# from ..utils.helpers import pcd_extrinsic_transform
# from ..utils.helpers import rot2qua, qua2rot_torch

## point cloud loss functions
class point_cloud_distance_loss(nn.Module):
    def __init__(self, crop_pcd = False):
        super(point_cloud_distance_loss, self).__init__()
        self.pcd_ext_trans = pcd_extrinsic_transform(crop = crop_pcd)

    def forward(self, T_gt, pcd, delta_t_gt, delta_q_gt, delta_t_pred, delta_q_pred):
        total_loss = torch.tensor([0.0]).to(delta_t_pred.device)
        for i in range(len(pcd)):
            pcd_gt    = pcd[i].to(delta_t_pred.device)
            pcd_pred  = pcd[i].clone()

            delta_R_gt = qua2rot_torch(delta_q_gt[i])
            delta_t_gt = delta_t_gt[i].reshape(1,3)
            delta_T_gt = torch.hstack((delta_R_gt, delta_t_gt)) 
            delta_T_gt = torch.vstack((delta_T_gt, torch.Tensor([0., 0., 0., 1.])))

            delta_R_pred = qua2rot_torch(delta_q_pred[i])
            delta_t_pred = delta_t_pred[i].reshape(1,3)
            delta_T_pred = torch.hstack((delta_R_pred, delta_t_pred)) 
            delta_T_pred = torch.vstack((delta_T_pred, torch.Tensor([0., 0., 0., 1.])))  

            T_pred = torch.matmul(T_gt, torch.matmul(delta_T_gt, torch.inverse(delta_T_pred)))

            pcd_gt = self.pcd_ext_trans(pcd_gt, T_gt)
            pcd_pred = self.pcd_ext_trans(pcd_pred, T_pred)

            error = (pcd_pred - pcd_gt).norm(dim=0) 
            error.clamp(100.)
            total_loss += error.mean()
        
        return total_loss/delta_t_gt.shape[0]

# alternative version
class pcd_distance_loss(nn.Module):
    def __init__(self, scale = 1.0):
        super(pcd_distance_loss, self).__init__()
        self.scale = scale

    def forward(self, p1, p2):
        total_loss = torch.tensor([0.0])
        for i in range(len(p1)):
            pcd1 = p1[i]*self.scale
            pcd2 = p2[i]*self.scale

            error = (pcd1 - pcd2).norm(dim=0) 
            error.clamp(100.)
            total_loss += error.mean()
        
        return total_loss/len(p1)
    
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

class pcd_extrinsic_transform: # transform PCD into fisheye camera reference frame.
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
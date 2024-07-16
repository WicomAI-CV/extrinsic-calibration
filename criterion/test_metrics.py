import torch
from . import so3
import torch.nn.functional as F

def test_metrics(T_pred, T_gt):
    R_pred = T_pred[:,:3,:3]  # (B,3,3) rotation
    t_pred = T_pred[:,:3,3]   # (B,3) translation
    R_gt = T_gt[:,:3,:3]  # (B,3,3) rotation
    t_gt = T_gt[:,:3,3]   # (B,3) translation

    # Euclidian Distance / Absolute Distance Error Rate
    t_error = F.l1_loss(t_pred, t_gt, reduction='none')
    # t_error = torch.sqrt(t_error)
    e_x = t_error[:,0].mean(dim=0)
    e_y = t_error[:,1].mean(dim=0)
    e_z = t_error[:,2].mean(dim=0)
    e_t = t_error.mean(dim=1).mean(dim=0)

    # Euler Angles Error Rate
    RIR = torch.bmm(torch.inverse(R_pred), R_gt)

    yaws = torch.atan2(RIR[:,1,0], RIR[:,0,0])
    pitches = torch.atan2(-RIR[:,2,0], torch.sqrt(RIR[:,2,0]*RIR[:,2,0] + RIR[:,2,2]*RIR[:,2,2]))
    rolls = torch.atan2(RIR[:,2,1], RIR[:,2,2])

    e_yaw = (torch.abs(yaws)).mean(dim=0)
    e_pitch = (torch.abs(pitches)).mean(dim=0)
    e_roll = (torch.abs(rolls)).mean(dim=0)
    e_r = (e_yaw + e_roll + e_pitch)/3.0

    # Geodesic Error Rate
    RTR = torch.bmm(torch.transpose(R_pred, 1, 2), R_gt)
    dR = so3.log(RTR)
    dR = F.mse_loss(dR,torch.zeros_like(dR).to(dR),reduction='none').mean(dim=1)  # (B,3) -> (B,1)
    dR = torch.sqrt(dR).mean(dim=0)


    return e_x, e_y, e_z, e_t, e_yaw, e_pitch, e_roll, e_r, dR
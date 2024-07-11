import torch
import torch.nn.functional as F
from torch import nn

import utils
    
class realignment_layer(nn.Module):
    def __init__(self, crop_pcd = False):
        super(realignment_layer, self).__init__()
        self.rotate_pcd = utils.pcd_extrinsic_transform_torch(crop = crop_pcd)

    def forward(self, pcd_mis, T_mis, delta_q_pred, delta_t_pred):
        device_ = delta_q_pred.device
        batch_size = delta_q_pred.shape[0]
        batch_T_pred = torch.tensor([]).to(device_)
        batch_pcd_realigned = []

        # print("pcd: ",pcd_mis.shape, "T_mis: ", T_mis.shape, "q: ", delta_q_pred.shape, "t: ", delta_t_pred.shape)

        for i in range(batch_size):
            delta_R_pred = utils.qua2rot_torch(delta_q_pred[i])
            delta_tr_pred = torch.reshape(delta_t_pred[i],(3,1))
            delta_T_pred = torch.hstack((delta_R_pred, delta_tr_pred)) 
            delta_T_pred = torch.vstack((delta_T_pred, torch.Tensor([0., 0., 0., 1.]).to(device_)))

            T_act_pred = torch.unsqueeze(torch.matmul(torch.linalg.inv(delta_T_pred), T_mis[i]), 0)

            # print(torch.linalg.inv(delta_T_pred).shape)
            # print(pcd_mis[i].shape)
            pcd_pred = self.rotate_pcd(pcd_mis[i], torch.linalg.inv(delta_T_pred))

            batch_T_pred = torch.cat((batch_T_pred, T_act_pred), 0)
            batch_pcd_realigned.append(pcd_pred)
        
        return batch_T_pred, batch_pcd_realigned
    

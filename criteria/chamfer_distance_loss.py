# requirement: pip install git+'https://github.com/otaheri/chamfer_distance'

import torch
import torch.nn as nn
from chamfer_distance import ChamferDistance as chamfer_dist

class chamfer_distance_loss(nn.Module):
    def __init__(self, scale=1.0):
        super(chamfer_distance_loss, self).__init__()
        self.loss = chamfer_dist()
        self.scale = scale
    
    def forward(self, p1, p2):
        total_loss = torch.tensor([0.0]).to(p1[0].device)
        for i in range(len(p1)):
            pcd1 = torch.unsqueeze(p1[i], dim=0)  
            pcd2 = torch.unsqueeze(p2[i], dim=0)
              
            pcd1 = pcd1*self.scale
            pcd2 = pcd2*self.scale

            dist1, dist2, _, _ = self.loss(pcd1,pcd2)
            loss = (torch.mean(dist1)) + (torch.mean(dist2))
            total_loss += loss

        return total_loss/len(p1)
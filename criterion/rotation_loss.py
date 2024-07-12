from torch import nn
from .quaternion_distance import quaternion_distance

class rotation_loss(nn.Module):  # quaternion angular distance
    def __init__(self, weight_rot = 1.0):
        super(rotation_loss, self).__init__()
        self.weight_rot = weight_rot

    def forward(self, target_q, pred_q):
        loss_rot = quaternion_distance(pred_q, target_q, pred_q.device).mean()
        return self.weight_rot*loss_rot
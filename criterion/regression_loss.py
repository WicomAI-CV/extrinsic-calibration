from torch import nn

## regression loss functions (for predicted t and R)
class regression_loss(nn.Module):
    def __init__(self, trans_weight = 1.0, rot_weight = 0.0, reduction='sum'):
        super(regression_loss, self).__init__()
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight
        self.reduction = reduction
        self.reg_loss = nn.SmoothL1Loss(reduction=self.reduction)

    def forward(self, delta_q_gt, delta_t_gt, delta_q_pred, delta_t_pred):
        if self.reduction == 'none':
            trans_loss = self.reg_loss(delta_t_pred, delta_t_gt).sum(1).mean()
            rot_loss = self.reg_loss(delta_q_pred, delta_q_gt).sum(1).mean()
        else:
            trans_loss = self.reg_loss(delta_t_pred, delta_t_gt)
            rot_loss = self.reg_loss(delta_q_pred, delta_q_gt)
        total_loss = self.trans_weight*trans_loss + self.rot_weight*rot_loss
        return total_loss
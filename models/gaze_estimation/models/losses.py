import torch


class HeatmapLoss(torch.nn.Module):
    def __init__(self):
        super(HeatmapLoss, self).__init__()

    def forward(self, pred, gt):
        loss = ((pred - gt)**2)
        loss = torch.mean(loss, dim=(1, 2, 3))
        return loss


class AngularError(torch.nn.Module):
    def __init__(self):
        super(AngularError, self).__init__()

    def forward(self, gaze_pred, gaze):
        loss = ((gaze_pred - gaze)**2)
        loss = torch.mean(loss, dim=(1, 2, 3))
        return loss

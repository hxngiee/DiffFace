import torch
from torch import nn
from models.gaze_estimation.models.layers import Conv, Hourglass, Pool, Residual
from models.gaze_estimation.models.losses import HeatmapLoss
from models.gaze_estimation.util.softargmax import softargmax2d


class Merge(nn.Module):
    def __init__(self, x_dim, y_dim):
        super(Merge, self).__init__()
        self.conv = Conv(x_dim, y_dim, 1, relu=False, bn=False)

    def forward(self, x):
        return self.conv(x)


class EyeNet(nn.Module):
    def __init__(self, nstack, nfeatures, nlandmarks, bn=False, increase=0, **kwargs):
        super(EyeNet, self).__init__()

        self.img_w = 160
        self.img_h = 96
        self.nstack = nstack
        self.nfeatures = nfeatures
        self.nlandmarks = nlandmarks

        self.heatmap_w = self.img_w / 2
        self.heatmap_h = self.img_h / 2

        self.nstack = nstack
        self.pre = nn.Sequential(
            Conv(1, 64, 7, 1, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, nfeatures)
        )

        self.pre2 = nn.Sequential(
            Conv(nfeatures, 64, 7, 2, bn=True, relu=True),
            Residual(64, 128),
            Pool(2, 2),
            Residual(128, 128),
            Residual(128, nfeatures)
        )

        self.hgs = nn.ModuleList([
            nn.Sequential(
                Hourglass(4, nfeatures, bn, increase),
            ) for i in range(nstack)])

        self.features = nn.ModuleList([
            nn.Sequential(
                Residual(nfeatures, nfeatures),
                Conv(nfeatures, nfeatures, 1, bn=True, relu=True)
            ) for i in range(nstack)])

        self.outs = nn.ModuleList([Conv(nfeatures, nlandmarks, 1, relu=False, bn=False) for i in range(nstack)])
        self.merge_features = nn.ModuleList([Merge(nfeatures, nfeatures) for i in range(nstack - 1)])
        self.merge_preds = nn.ModuleList([Merge(nlandmarks, nfeatures) for i in range(nstack - 1)])

        self.gaze_fc1 = nn.Linear(in_features=int(nfeatures * self.img_w * self.img_h / 64 + nlandmarks*2), out_features=256)
        self.gaze_fc2 = nn.Linear(in_features=256, out_features=2)

        self.nstack = nstack
        self.heatmapLoss = HeatmapLoss()
        self.landmarks_loss = nn.MSELoss()
        self.gaze_loss = nn.MSELoss()

    def forward(self, imgs):
        # imgs of size 1,ih,iw
        x = imgs.unsqueeze(1)
        x = self.pre(x)

        gaze_x = self.pre2(x)
        gaze_x = gaze_x.flatten(start_dim=1)

        combined_hm_preds = []
        for i in torch.arange(self.nstack):
            hg = self.hgs[i](x)
            feature = self.features[i](hg)
            preds = self.outs[i](feature)
            combined_hm_preds.append(preds)
            if i < self.nstack - 1:
                x = x + self.merge_preds[i](preds) + self.merge_features[i](feature)

        heatmaps_out = torch.stack(combined_hm_preds, 1)

        # preds = N x nlandmarks * heatmap_w * heatmap_h
        landmarks_out = softargmax2d(preds)  # N x nlandmarks x 2

        # Gaze
        gaze = torch.cat((gaze_x, landmarks_out.flatten(start_dim=1)), dim=1)
        gaze = self.gaze_fc1(gaze)
        gaze = nn.functional.relu(gaze)
        gaze = self.gaze_fc2(gaze)

        return heatmaps_out, landmarks_out, gaze

    def calc_loss(self, combined_hm_preds, heatmaps, landmarks_pred, landmarks, gaze_pred, gaze):
        combined_loss = []
        for i in range(self.nstack):
            combined_loss.append(self.heatmapLoss(combined_hm_preds[:, i, :], heatmaps))

        heatmap_loss = torch.stack(combined_loss, dim=1)
        landmarks_loss = self.landmarks_loss(landmarks_pred, landmarks)
        gaze_loss = self.gaze_loss(gaze_pred, gaze)

        return torch.sum(heatmap_loss), landmarks_loss, 1000 * gaze_loss

import torch
from torch import nn
import torch.nn.functional as F


class ATDHConv(nn.Module):
    def __init__(self, in_channels, out_channels, M=2, G=32, r=16, stride=1, L=32):
        """ Constructor
        Args:
            in_channels: input channel dimensionality.
            out_channels: output chanel dimensionality.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            stride: stride, default 1.
            L: the minimum dim of the vector z in paper, default 32.
        """
        super(ATDHConv, self).__init__()
        d = max(in_channels // r, L)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.fc = nn.Sequential(
            nn.Conv2d(out_channels, d, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(d, out_channels * M, kernel_size=1)
        )
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.softmax = nn.Softmax(dim=1)
        self.out_channels = out_channels
        self.M = M
        self.G = G

    def forward(self, x):
        b, c, _, _ = x.shape
        U = self.conv(x)
        avg_pool = self.avg_pool(U)
        max_pool = self.max_pool(U)

        z = avg_pool + max_pool
        z = z.view(b, self.M, self.G, -1)
        z = self.softmax(z)
        z = z.view(b, -1, 1, 1)
        out = U * z.expand_as(U)
        return out


class ATDHUnit(nn.Module):
    def __init__(self, in_features, out_features, WH, M, G, r, mid_features=None, stride=1, L=32):
        """ Constructor
        Args:
            in_features: input channel dimensionality.
            out_features: output channel dimensionality.
            WH: input spatial dimensionality, used for GAP kernel size.
            M: the number of branchs.
            G: num of convolution groups.
            r: the radio for compute d, the length of z.
            mid_features: the channle dim of the middle conv with stride not 1, default out_features/2.
            stride: stride.
            L: the minimum dim of the vector z in paper.
        """
        super(ATDHUnit, self).__init__()
        if mid_features is None:
            mid_features = int(out_features / 2)
        self.feas = nn.Sequential(
            nn.Conv2d(in_features, mid_features, 1, stride=1),
            nn.BatchNorm2d(mid_features),
            ATDHConv(mid_features, WH, M, G, r, stride=stride, L=L),
            nn.BatchNorm2d(mid_features),
            nn.Conv2d(mid_features, out_features, 1, stride=1),
            nn.BatchNorm2d(out_features)
        )
        if in_features == out_features:  # when dim not change, in could be added diectly to out
            self.shortcut = nn.Sequential()
        else:  # when dim not change, in should also change dim to be added to out
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_features, out_features, 1, stride=stride),
                nn.BatchNorm2d(out_features)
            )

    def forward(self, x):
        fea = self.feas(x)
        return fea + self.shortcut(x)


class ATDHNet(nn.Module):
    def __init__(self, class_num):
        super(ATDHNet, self).__init__()
        self.basic_conv = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64)
        )  # 32x32
        self.stage_1 = nn.Sequential(
            ATDHUnit(64, 256, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            ATDHUnit(256, 256, 32, 2, 8, 2),
            nn.ReLU(),
            ATDHUnit(256, 256, 32, 2, 8, 2),
            nn.ReLU()
        )  # 32x32
        self.stage_2 = nn.Sequential(
            ATDHUnit(256, 512, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            ATDHUnit(512, 512, 32, 2, 8, 2),
            nn.ReLU(),
            ATDHUnit(512, 512, 32, 2, 8, 2),
            nn.ReLU()
        )  # 16x16
        self.stage_3 = nn.Sequential(
            ATDHUnit(512, 1024, 32, 2, 8, 2, stride=2),
            nn.ReLU(),
            ATDHUnit(1024, 1024, 32, 2, 8, 2),
            nn.ReLU(),
            ATDHUnit(1024, 1024, 32, 2, 8, 2),
            nn.ReLU()
        )  # 8x8
        self.pool = nn.AvgPool2d(8)
        self.classifier = nn.Sequential(
            nn.Linear(1024, class_num),
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        fea = self.basic_conv(x)
        fea = self.stage_1(fea)
        fea = self.stage_2(fea)
        fea = self.stage_3(fea)
        fea = self.pool(fea)
        fea = torch.squeeze(fea)
        fea = self.classifier(fea)
        return fea


if __name__ == '__main__':
    x = torch.rand(8, 64, 32, 32)
    conv = ATDHConv(64, 32, 3, 8, 2)
    out = conv(x)
    criterion = nn.L1Loss()
    loss = criterion(out, x)
    loss.backward()
    print('out shape : {}'.format(out.shape))
    print('loss value : {}'.format(loss))

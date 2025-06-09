import math
import torch
from torch import nn
from models.head.ATDHNet import ATDHConv


class RegHead(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, feature_size=256, attention='atdh'):
        super(RegHead, self).__init__()
        self.attention = attention
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)

        self.output = nn.Conv2d(feature_size, num_anchors * 4, kernel_size=3, padding=1)
        self.act = nn.ReLU()

        if self.attention == 'atdh':
            self.atdh1 = ATDHConv(feature_size, feature_size)
            self.atdh2 = ATDHConv(feature_size, feature_size)
            self.atdh3 = ATDHConv(feature_size, feature_size)
            self.atdh4 = ATDHConv(feature_size, feature_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        self.output.weight.data.fill_(0)
        self.output.bias.data.fill_(0)

    def forward(self, x):
        if self.attention == 'atdh':
            out = self.conv1(x)
            out = self.atdh1(out) + out
            out = self.act(out)

            out = self.conv2(out)
            out = self.atdh2(out) + out
            out = self.act(out)

            out = self.conv3(out)
            out = self.atdh3(out) + out
            out = self.act(out)

            out = self.conv4(out)
            out = self.atdh4(out) + out
            out = self.act(out)

            out = self.output(out)


        else:
            out = self.conv1(x)
            out = self.act(out)

            out = self.conv2(out)
            out = self.act(out)

            out = self.conv3(out)
            out = self.act(out)

            out = self.conv4(out)
            out = self.act(out)

            out = self.output(out)
        # out is B x C x W x H, with C = 4*num_anchors
        out = out.permute(0, 2, 3, 1)
        # shape : (batch_size, H*W*num_anchors, 4)
        out = out.contiguous().view(out.shape[0], -1, 4)
        del x
        return out


class ClsHead(nn.Module):
    def __init__(self, num_features_in, num_anchors=9, num_classes=80, feature_size=256, attention='atdh'):
        super(ClsHead, self).__init__()
        self.attention = attention
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        self.conv1 = nn.Conv2d(num_features_in, feature_size, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.conv4 = nn.Conv2d(feature_size, feature_size, kernel_size=3, padding=1)
        self.output = nn.Conv2d(feature_size, num_anchors * num_classes, kernel_size=3, padding=1)

        self.sigmoid = nn.Sigmoid()
        self.act = nn.ReLU()

        if self.attention == 'atdh':
            self.atdh1 = ATDHConv(feature_size, feature_size)
            self.atdh2 = ATDHConv(feature_size, feature_size)
            self.atdh3 = ATDHConv(feature_size, feature_size)
            self.atdh4 = ATDHConv(feature_size, feature_size)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[1] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        self.output.weight.data.fill_(0)
        prior = 0.01
        b = -math.log((1 - prior) / prior)
        self.output.bias.data.fill_(b)

    def forward(self, x):
        if self.attention == 'atdh':
            out = self.conv1(x)
            out = self.atdh1(out) + out
            out = self.act(out)

            out = self.conv2(out)
            out = self.atdh2(out) + out
            out = self.act(out)

            out = self.conv3(out)
            out = self.atdh3(out) + out
            out = self.act(out)

            out = self.conv4(out)
            out = self.atdh4(out) + out
            out = self.act(out)

            out = self.output(out)

        else:
            out = self.conv1(x)
            out = self.act(out)

            out = self.conv2(out)
            out = self.act(out)

            out = self.conv3(out)
            out = self.act(out)

            out = self.conv4(out)
            out = self.act(out)

            out = self.output(out)

        out = self.sigmoid(out)
        # out is B x C x H x W, with C = n_classes * n_anchors
        out = out.permute(0, 2, 3, 1)
        b, h, w, c = out.shape
        out = out.view(b, h, w, self.num_anchors, self.num_classes)
        out = out.contiguous().view(x.shape[0], -1, self.num_classes)
        del x
        return out


if __name__ == "__main__":
    # B,C,H,W
    C = torch.randn([2, 256, 512, 512])
    RegHead = RegHead(256)
    out = RegHead(C)
    print("RegHead out.shape:")
    print(out.shape)
    # torch.Size([2, 2359296, 4])

    print("********************************")

    C1 = torch.randn([2, 256, 64, 64])
    C2 = torch.randn([2, 256, 32, 32])
    C3 = torch.randn([2, 256, 16, 16])
    C4 = torch.randn([2, 256, 8, 8])
    C5 = torch.randn([2, 256, 4, 4])

    print("ClsHead out.shape:")
    ClsHead = ClsHead(256)
    print(ClsHead(C1).shape)  # torch.Size([2, 36864, 80])
    print(ClsHead(C2).shape)  # torch.Size([2, 9216, 80])
    print(ClsHead(C3).shape)  # torch.Size([2, 2304, 80])
    print(ClsHead(C4).shape)  # torch.Size([2, 576, 80])
    print(ClsHead(C5).shape)  # torch.Size([2, 144, 80])

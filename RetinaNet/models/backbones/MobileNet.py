import math
import torch
import torch.nn as nn
from torchvision.models import mobilenet_v2, mobilenet_v3_small, mobilenet_v3_large


class MobileNet(nn.Module):
    def __init__(self, mobilenet_type="mobilenet_v2", pretrained=False):
        super(MobileNet, self).__init__()
        self.mobilenet_type = mobilenet_type
        if mobilenet_type == "mobilenet_v2":
            self.model = mobilenet_v2(pretrained=pretrained)
            self.conv_v2 = nn.Conv2d(in_channels=160, out_channels=128, kernel_size=1)
            self.out_channels = [32, 64, 128]

            del self.model.classifier
            del self.model.features[15]
            del self.model.features[16]
            del self.model.features[-2]
            del self.model.features[-1]

        elif mobilenet_type == "mobilenet_v3_small":
            self.model = mobilenet_v3_small(pretrained=pretrained)
            self.conv_v30 = nn.Conv2d(in_channels=24, out_channels=48, kernel_size=1)
            self.conv_v31 = nn.Conv2d(in_channels=48, out_channels=96, kernel_size=1)
            self.conv_v32 = nn.Conv2d(in_channels=96, out_channels=192, kernel_size=1)
            self.out_channels = [48, 96, 192]

            del self.model.avgpool
            del self.model.classifier
            del self.model.features[-2]
            del self.model.features[-1]

        elif mobilenet_type == "mobilenet_v3_large":
            self.model = mobilenet_v3_large(pretrained=pretrained)
            self.out_channels = [40, 80, 160]
            del self.model.avgpool
            del self.model.classifier
            del self.model.features[-1]

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        if self.mobilenet_type == "mobilenet_v2":
            x = self.model.features[0](x)
            x = self.model.features[1](x)
            x = self.model.features[2](x)
            x = self.model.features[3](x)
            x = self.model.features[4](x)
            x = self.model.features[5](x)
            C3 = self.model.features[6](x)

            C4 = self.model.features[7](C3)
            C4 = self.model.features[8](C4)
            C4 = self.model.features[9](C4)
            C4 = self.model.features[10](C4)

            C5 = self.model.features[11](C4)
            C5 = self.model.features[12](C5)
            C5 = self.model.features[13](C5)
            C5 = self.model.features[14](C5)
            C5 = self.conv_v2(C5)

        elif self.mobilenet_type == "mobilenet_v3_small":
            x = self.model.features[0](x)
            x = self.model.features[1](x)
            x = self.model.features[2](x)
            x = self.model.features[3](x)
            C3 = self.conv_v30(x)

            x = self.model.features[4](x)
            x = self.model.features[5](x)
            x = self.model.features[6](x)
            x = self.model.features[7](x)
            x = self.model.features[8](x)
            C4 = self.conv_v31(x)

            x = self.model.features[9](x)
            x = self.model.features[10](x)
            C5 = self.conv_v32(x)

        elif self.mobilenet_type == "mobilenet_v3_large":
            x = self.model.features[0](x)
            x = self.model.features[1](x)
            x = self.model.features[2](x)
            x = self.model.features[3](x)
            x = self.model.features[4](x)
            x = self.model.features[5](x)
            C3 = self.model.features[6](x)

            C4 = self.model.features[7](C3)
            C4 = self.model.features[8](C4)
            C4 = self.model.features[9](C4)
            C4 = self.model.features[10](C4)

            C5 = self.model.features[11](C4)
            C5 = self.model.features[12](C5)
            C5 = self.model.features[13](C5)
            C5 = self.model.features[14](C5)
            C5 = self.model.features[15](C5)
        del x
        return [C3, C4, C5]


if __name__ == "__main__":
    mobilenet = MobileNet(mobilenet_type="mobilenet_v3_large", pretrained=False)
    x = torch.randn([16, 3, 512, 512])
    mobilenet = mobilenet(x)

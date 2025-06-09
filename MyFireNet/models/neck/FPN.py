import torch
import torch.nn as nn
import torch.nn.functional as F


class FPN(nn.Module):
    '''only for resnet50,101,152'''

    def __init__(self, C3_inplanes, C4_inplanes, C5_inplanes, features=256, use_p5=True):
        super(FPN, self).__init__()
        self.prj_5 = nn.Conv2d(C5_inplanes, features, kernel_size=1)
        self.prj_4 = nn.Conv2d(C4_inplanes, features, kernel_size=1)
        self.prj_3 = nn.Conv2d(C3_inplanes, features, kernel_size=1)
        self.conv_5 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_4 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv_3 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        if use_p5:
            self.conv_out6 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        else:
            self.conv_out6 = nn.Conv2d(C5_inplanes, features, kernel_size=3, padding=1, stride=2)
        self.conv_out7 = nn.Conv2d(features, features, kernel_size=3, padding=1, stride=2)
        self.use_p5 = use_p5
        self.apply(self.init_conv_kaiming)

    def upsamplelike(self, inputs):
        src, target = inputs
        return F.interpolate(src, size=(target.shape[2], target.shape[3]),
                             mode='nearest')

    def init_conv_kaiming(self, module):
        if isinstance(module, nn.Conv2d):
            nn.init.kaiming_uniform_(module.weight, a=1)

            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

    def forward(self, x):
        C3, C4, C5 = x
        P5 = self.prj_5(C5)
        P4 = self.prj_4(C4)
        P3 = self.prj_3(C3)

        P4 = P4 + self.upsamplelike([P5, C4])
        P3 = P3 + self.upsamplelike([P4, C3])

        P3 = self.conv_3(P3)
        P4 = self.conv_4(P4)
        P5 = self.conv_5(P5)

        P5 = P5 if self.use_p5 else C5
        P6 = self.conv_out6(P5)
        P7 = self.conv_out7(F.relu(P6))
        return [P3, P4, P5, P6, P7]


if __name__ == "__main__":
    # Img size 672*640 -> C1 168*160 -> C2 168*160
    # -> C3 84*80 -> C4 42*40 -> C5 21*20
    # -> P3 84*80 -> P4 42*40 -> P5 21*20 -> P6 11*10 -> P7 6*5
    C3 = torch.randn([2, 128 * 4, 84, 80])
    C4 = torch.randn([2, 256 * 4, 42, 40])
    C5 = torch.randn([2, 512 * 4, 21, 20])

    model = FPN(128 * 4, 256 * 4, 512 * 4)
    out = model([C3, C4, C5])
    print("len(out):", len(out))
    for i in range(len(out)):
        print(i + 1, out[i].shape)
        # print(out[i])
    # torch.Size([2, 256, 84, 80])
    # torch.Size([2, 256, 42, 40])
    # torch.Size([2, 256, 21, 20])
    # torch.Size([2, 256, 11, 10])
    # torch.Size([2, 256, 6, 5])

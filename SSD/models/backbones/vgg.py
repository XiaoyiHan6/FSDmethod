import torch
from torch import nn
from models.utils.l2norm import L2Norm
from torchvision.models.vgg import vgg11, vgg11_bn, vgg13, vgg13_bn, vgg16, vgg16_bn, vgg19, vgg19_bn


# This function is derived from torchvision VGG make_layers()
# https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
class vgg(nn.Module):
    def __init__(self, cfg, pretrained=False):
        super(vgg, self).__init__()
        self.num_classes = cfg['Data']['num_classes']
        self.batch_norm = cfg['Models']['batch_norm']
        self.vgg_type = cfg['Backbones']['name'] + str(cfg['Backbones']['depth'])
        relu = nn.ReLU(inplace=True)
        conv6 = nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding=6, dilation=6)
        conv7 = nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1)

        self.l2norm = L2Norm(512, self.num_classes)
        if self.batch_norm == True:
            bn = nn.BatchNorm2d(1024)
        if self.vgg_type == 'vgg11' and self.batch_norm == False:
            self.vggs = vgg11(pretrained=pretrained)
            del self.vggs.avgpool
            del self.vggs.classifier
            self.vggs.features[10] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.vggs.features[-1] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.vggs.features.add_module('21', conv6)
            self.vggs.features.add_module('22', relu)
            self.vggs.features.add_module('23', conv7)
            self.vggs.features.add_module('24', relu)
            self.out_features = 15


        elif self.vgg_type == 'vgg11' and self.batch_norm == True:
            self.vggs = vgg11_bn(pretrained=pretrained)
            del self.vggs.avgpool
            del self.vggs.classifier
            self.vggs.features[14] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.vggs.features[-1] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.vggs.features.add_module('29', conv6)
            self.vggs.features.add_module('30', bn)
            self.vggs.features.add_module('31', relu)
            self.vggs.features.add_module('32', conv7)
            self.vggs.features.add_module('33', bn)
            self.vggs.features.add_module('34', relu)
            self.out_features = 21

        elif self.vgg_type == 'vgg13' and self.batch_norm == False:
            self.vggs = vgg13(pretrained=pretrained)
            del self.vggs.avgpool
            del self.vggs.classifier
            self.vggs.features[14] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.vggs.features[-1] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.vggs.features.add_module('25', conv6)
            self.vggs.features.add_module('26', relu)
            self.vggs.features.add_module('27', conv7)
            self.vggs.features.add_module('28', relu)
            self.out_features = 19

        elif self.vgg_type == 'vgg13' and self.batch_norm == True:
            self.vggs = vgg13_bn(pretrained=pretrained)
            del self.vggs.avgpool
            del self.vggs.classifier
            self.vggs.features[20] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.vggs.features[-1] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.vggs.features.add_module('35', conv6)
            self.vggs.features.add_module('36', bn)
            self.vggs.features.add_module('37', relu)
            self.vggs.features.add_module('38', conv7)
            self.vggs.features.add_module('39', bn)
            self.vggs.features.add_module('40', relu)
            self.out_features = 27

        elif self.vgg_type == 'vgg16' and self.batch_norm == False:
            self.vggs = vgg16(pretrained=pretrained)
            del self.vggs.avgpool
            del self.vggs.classifier
            self.vggs.features[16] = nn.MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=True)
            self.vggs.features[-1] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1, dilation=1, ceil_mode=False)
            self.vggs.features.add_module('31', conv6)
            self.vggs.features.add_module('32', relu)
            self.vggs.features.add_module('33', conv7)
            self.vggs.features.add_module('34', relu)
            self.out_features = 23

        elif self.vgg_type == 'vgg16' and self.batch_norm == True:
            self.vggs = vgg16_bn(pretrained=pretrained)
            del self.vggs.avgpool
            del self.vggs.classifier
            self.vggs.features[23] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.vggs.features[-1] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.vggs.features.add_module('44', conv6)
            self.vggs.features.add_module('45', bn)
            self.vggs.features.add_module('46', relu)
            self.vggs.features.add_module('47', conv7)
            self.vggs.features.add_module('48', bn)
            self.vggs.features.add_module('49', relu)
            self.out_features = 33

        elif self.vgg_type == 'vgg19' and self.batch_norm == False:
            self.vggs = vgg19(pretrained=pretrained)
            del self.vggs.avgpool
            del self.vggs.classifier
            self.vggs.features[18] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.vggs.features[-1] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.vggs.features.add_module('37', conv6)
            self.vggs.features.add_module('38', relu)
            self.vggs.features.add_module('39', conv7)
            self.vggs.features.add_module('40', relu)
            self.out_features = 27

        elif self.vgg_type == 'vgg19' and self.batch_norm == True:
            self.vggs = vgg19_bn(pretrained=pretrained)
            del self.vggs.avgpool
            del self.vggs.classifier
            self.vggs.features[26] = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
            self.vggs.features[-1] = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
            self.vggs.features.add_module('53', conv6)
            self.vggs.features.add_module('54', bn)
            self.vggs.features.add_module('55', relu)
            self.vggs.features.add_module('56', conv7)
            self.vggs.features.add_module('57', bn)
            self.vggs.features.add_module('58', relu)
            self.out_features = 39

    def forward(self, x):
        sources = []
        for k in range(self.out_features):
            x = self.vggs.features[k](x)

        s = self.l2norm(x)
        sources.append(s)

        for k in range(self.out_features, len(self.vggs.features)):
            x = self.vggs.features[k](x)
        sources.append(x)
        del x
        return sources


class add_extras(nn.Module):
    def __init__(self, cfg):
        super(add_extras, self).__init__()
        # Extra layers added to VGG for feature scaling
        layers = []
        in_channels = cfg['Backbones']['out_channels'][1]
        flag = False
        extras = cfg['Backbones']['extras']
        relu = nn.ReLU(inplace=True)
        self.bn = cfg['Models']['batch_norm']
        for k, v in enumerate(extras):
            if in_channels != 'S':
                if v == 'S' and self.bn == False:
                    layers += [nn.Conv2d(in_channels=in_channels, out_channels=extras[k + 1], kernel_size=(1, 3)[flag],
                                         stride=2, padding=1), relu]
                elif v == 'S' and self.bn == True:
                    layers += [nn.Conv2d(in_channels=in_channels, out_channels=extras[k + 1], kernel_size=(1, 3)[flag],
                                         stride=2, padding=1), nn.BatchNorm2d(extras[k + 1]), relu]
                elif v != 'S' and self.bn == False:
                    layers += [nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=(1, 3)[flag]), relu]
                elif v != 'S' and self.bn == True:
                    layers += [nn.Conv2d(in_channels=in_channels, out_channels=v, kernel_size=(1, 3)[flag]),
                               nn.BatchNorm2d(v), relu]
                flag = not flag
            in_channels = v
        if cfg['Data']['size'] == 512 and self.bn == True:
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1),
                       nn.BatchNorm2d(128), relu]
            layers += [nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=1),
                       nn.BatchNorm2d(256), relu]
        if cfg['Data']['size'] == 512 and self.bn == False:
            layers += [nn.Conv2d(in_channels=in_channels, out_channels=128, kernel_size=1, stride=1), relu]
            layers += [nn.Conv2d(in_channels=128, out_channels=256, kernel_size=4, stride=1, padding=1), relu]
        self.layers = nn.Sequential(*layers)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, x):
        sources = []
        for k, v in enumerate(self.layers):
            x = self.layers[k](x)
            if k % 4 == 3 and self.bn == False:
                sources.append(x)
            elif k % 6 == 5 and self.bn == True:
                sources.append(x)
        del x
        return sources


if __name__ == '__main__':
    from options.train import cfg

    bn = True
    Vgg = vgg(cfg)
    print("vgg:", Vgg)
    img = torch.randn(16, 3, 300, 300)
    vgg_output = Vgg(img)

    layers = add_extras(cfg)
    print("layers:", layers)
    features = torch.rand(16, 1024, 19, 19)
    extras_output = layers(features)

    for i in range(len(vgg_output)):
        print("vgg base output[{}].shape:{}".format(i, vgg_output[i].shape))
    for i in range(len(extras_output)):
        print("extras output[{}].shape:{}".format(i, extras_output[i].shape))

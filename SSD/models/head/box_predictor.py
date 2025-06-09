import torch
from torch import nn
from models.head.ATDHNet import ATDHConv


class ClsHead(nn.Module):
    def __init__(self, cfg):
        super(ClsHead, self).__init__()
        num_classes = cfg['Data']['num_classes']
        cls_heads = []
        self.attention = cfg['Head']['name']
        for i in range(len(cfg['Backbones']['out_channels'])):
            cls_heads += [nn.Conv2d(in_channels=cfg['Backbones']['out_channels'][i],
                                    out_channels=cfg['Backbones']['boxes_per_location'][i] * num_classes, kernel_size=3,
                                    padding=1)]
        self.cls_heads = nn.Sequential(*cls_heads)
        if self.attention == 'atdh':
            atdhs = []
            if cfg['Data']['size'] == 300:
                atdhs = [ATDHConv(512, 512), ATDHConv(1024, 1024), ATDHConv(512, 512), ATDHConv(256, 256),
                         ATDHConv(256, 256), ATDHConv(256, 256)]
            elif cfg['Data']['size'] == 512:
                atdhs = [ATDHConv(512, 512), ATDHConv(1024, 1024), ATDHConv(512, 512), ATDHConv(256, 256),
                         ATDHConv(256, 256), ATDHConv(256, 256), ATDHConv(256, 256)]
            self.adths = nn.Sequential(*atdhs)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, features):
        out = []
        if self.attention == 'atdh':
            for feature, cls_head, atdh in zip(features, self.cls_heads, self.adths):
                feature = atdh(feature) + feature
                out.append(cls_head(feature).permute(0, 2, 3, 1).contiguous())
        else:
            for feature, cls_head in zip(features, self.cls_heads):
                out.append(cls_head(feature).permute(0, 2, 3, 1).contiguous())
        out = torch.cat([o.view(o.size(0), -1) for o in out], 1)
        del features

        return out


class RegHead(nn.Module):
    def __init__(self, cfg):
        super(RegHead, self).__init__()
        self.attention = cfg['Head']['name']
        reg_heads = []
        for i in range(len(cfg['Backbones']['out_channels'])):
            reg_heads += [nn.Conv2d(in_channels=cfg['Backbones']['out_channels'][i],
                                    out_channels=cfg['Backbones']['boxes_per_location'][i] * 4, kernel_size=3,
                                    padding=1)]
        self.reg_heads = nn.Sequential(*reg_heads)
        if self.attention=='atdh':
            atdhs = []
            if cfg['Data']['size'] == 300:
                atdhs = [ATDHConv(512, 512), ATDHConv(1024, 1024), ATDHConv(512, 512), ATDHConv(256, 256),
                         ATDHConv(256, 256), ATDHConv(256, 256)]
            elif cfg['Data']['size'] == 512:
                atdhs = [ATDHConv(512, 512), ATDHConv(1024, 1024), ATDHConv(512, 512), ATDHConv(256, 256),
                         ATDHConv(256, 256), ATDHConv(256, 256), ATDHConv(256, 256)]
            self.adths = nn.Sequential(*atdhs)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, std=0.1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, val=0)

    def forward(self, features):
        out = []
        if self.attention=='atdh':
            for feature, reg_head, atdh in zip(features, self.reg_heads, self.adths):
                feature = atdh(feature) + feature
                out.append(reg_head(feature).permute(0, 2, 3, 1).contiguous())
        else:
            for feature, reg_head in zip(features, self.reg_heads):
                out.append(reg_head(feature).permute(0, 2, 3, 1).contiguous())
        out = torch.cat([o.view(o.size(0), -1) for o in out], 1)
        del features
        return out


if __name__ == '__main__':
    from options.train import cfg

    reg_head = RegHead(cfg)
    print("loc layers:\n", reg_head)
    print('---------------------------')
    cls_head = ClsHead(cfg)
    print("conf layers:\n", cls_head)

import torch.nn as nn
from models.head.inference import Detect
from models.loss.loss import MultiBoxLoss
from models.anchor.prior_box import PriorBox
from models.backbones.vgg import vgg, add_extras
from models.head.box_predictor import RegHead, ClsHead


class SSD(nn.Module):
    def __init__(self, cfg, training=False, pretrained=False):
        super(SSD, self).__init__()
        self.cfg = cfg
        self.backbones_type = cfg['Backbones']['name'] + str(cfg['Backbones']['depth'])
        self.num_classes = cfg['Data']['num_classes']
        self.batch_norm = cfg['Models']['batch_norm']
        self.training = training
        if self.backbones_type[:3] == 'vgg':
            self.backbones = vgg(cfg=cfg, pretrained=pretrained)
            self.extras = add_extras(cfg=cfg)
        self.anchor = PriorBox(cfg=cfg)
        self.reghead = RegHead(cfg=cfg)
        self.clshead = ClsHead(cfg=cfg)

        if self.training:
            self.loss = MultiBoxLoss(cfg=cfg)
        else:
            self.softmax = nn.Softmax(dim=-1)
            self.detect = Detect(cfg=cfg)

    def forward(self, inputs):
        features = []
        if self.training:
            img_batch, annots = inputs
        else:
            img_batch = inputs

        sources = self.backbones(img_batch)
        for i in range(len(sources)):
            features.append(sources[i])
        sources = self.extras(sources[-1])
        for i in range(len(sources)):
            features.append(sources[i])
        anchor = self.anchor.forward()
        regs = self.reghead(features)
        cls = self.clshead(features)

        if self.training:
            output = (
                regs.view(regs.size(0), -1, 4),
                cls.view(cls.size(0), -1, self.num_classes),
                anchor
            )
            return self.loss(output, annots)
        else:
            output = self.detect.forward(
                regs.view(regs.size(0), -1, 4),
                self.softmax(cls.view(cls.size(0), -1, self.num_classes)),
                anchor.type(type(features[-1].data))
            )
            return output

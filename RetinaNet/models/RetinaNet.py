import torch
import torch.nn as nn
from torchvision.ops import nms
from models.neck.FPN import FPN
from models.anchor.Anchor import Anchor
from models.backbones.ResNet import ResNet
from models.loss.FocalLoss import FocalLoss
from models.backbones.DarkNet import DarkNet
from models.head.Head import RegHead, ClsHead
from models.backbones.MobileNet import MobileNet
from models.utils import BBoxTransform, ClipBoxes


class RetinaNet(nn.Module):
    def __init__(self,
                 backbones_type="resnet50",
                 num_classes=80,
                 planes=256,
                 attention='atdh',
                 pretrained=False,
                 training=False):
        super(RetinaNet, self).__init__()

        self.backbones_type = backbones_type

        # coco 80 voc 20
        self.num_classes = num_classes
        self.planes = planes
        self.training = training
        if backbones_type[:6] == 'resnet':
            self.backbones = ResNet(resnet_type=self.backbones_type, pretrainted=pretrained)
        elif backbones_type[:9] == 'mobilenet':
            self.backbones = MobileNet(mobilenet_type=self.backbones_type, pretrained=pretrained)
        elif backbones_type[:7] == 'darknet':
            self.backbones = DarkNet(darknet_type=self.backbones_type)
        expand_ratio = {
            "resnet18": 1,
            "resnet34": 1,
            "resnet50": 4,
            "resnet101": 4,
            "resnet152": 4,
            "darknettiny": 0.5,
            "darknet19": 1,
            "darknet53": 2,
            "mobilenet_v2": 0.25,
            "mobilenet_v3_small": 0.375,
            "mobilenet_v3_large": 0.3125
        }
        C3_inplanes, C4_inplanes, C5_inplanes = \
            int(128 * expand_ratio[self.backbones_type]), \
            int(256 * expand_ratio[self.backbones_type]), \
            int(512 * expand_ratio[self.backbones_type])

        self.fpn = FPN(C3_inplanes, C4_inplanes, C5_inplanes, planes=self.planes)

        self.reghead = RegHead(planes, attention=attention)
        self.clshead = ClsHead(planes, num_classes=num_classes, attention=attention)

        self.anchors = Anchor()

        self.regressBoxes = BBoxTransform()

        self.clipBoxes = ClipBoxes()

        self.focalLoss = FocalLoss()

        self.freeze_bn()

    def freeze_bn(self):
        '''Freeze BatchNorm layers.'''
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()

    def forward(self, inputs):
        if self.training:
            img_batch, annotations = inputs
        else:
            img_batch = inputs
        [C3, C4, C5] = self.backbones(img_batch)
        del inputs

        features = self.fpn([C3, C4, C5])
        del C3, C4, C5
        regression = torch.cat([self.reghead(feature) for feature in features], dim=1)
        classification = torch.cat([self.clshead(feature) for feature in features], dim=1)
        del features

        anchors = self.anchors(img_batch)

        if self.training:
            return self.focalLoss(classification, regression, anchors, annotations)
        else:
            transformed_anchors = self.regressBoxes(anchors, regression)
            transformed_anchors = self.clipBoxes(transformed_anchors, img_batch)

            finalResult = [[], [], []]

            finalScores = torch.Tensor([])
            finalAnchorBoxesIndexes = torch.Tensor([]).long()
            finalAnchorBoxesCoordinates = torch.Tensor([])

            if torch.cuda.is_available():
                finalScores = finalScores.cuda()
                finalAnchorBoxesIndexes = finalAnchorBoxesIndexes.cuda()
                finalAnchorBoxesCoordinates = finalAnchorBoxesCoordinates.cuda()

            for i in range(classification.shape[2]):
                scores = torch.squeeze(classification[:, :, i])
                scores_over_thresh = (scores > 0.05)
                if scores_over_thresh.sum() == 0:
                    # no boxes to NMS, just continue
                    continue

                scores = scores[scores_over_thresh]
                anchorBoxes = torch.squeeze(transformed_anchors)
                anchorBoxes = anchorBoxes[scores_over_thresh]
                anchors_nms_idx = nms(anchorBoxes, scores, 0.5)

                finalResult[0].extend(scores[anchors_nms_idx])
                finalResult[1].extend(torch.tensor([i] * anchors_nms_idx.shape[0]))
                finalResult[2].extend(anchorBoxes[anchors_nms_idx])

                finalScores = torch.cat((finalScores, scores[anchors_nms_idx]))
                finalAnchorBoxesIndexesValue = torch.tensor([i] * anchors_nms_idx.shape[0])
                if torch.cuda.is_available():
                    finalAnchorBoxesIndexesValue = finalAnchorBoxesIndexesValue.cuda()

                finalAnchorBoxesIndexes = torch.cat((finalAnchorBoxesIndexes, finalAnchorBoxesIndexesValue))
                finalAnchorBoxesCoordinates = torch.cat((finalAnchorBoxesCoordinates, anchorBoxes[anchors_nms_idx]))

            return [finalScores, finalAnchorBoxesIndexes, finalAnchorBoxesCoordinates]


if __name__ == "__main__":
    C = torch.randn([8, 3, 512, 512])
    annot = torch.randn([8, 15, 5])
    model = RetinaNet(backbones_type="darknet19", num_classes=80, pretrained=True, training=True)
    model = model.cuda()
    C = C.cuda()
    annot = annot.cuda()
    model = torch.nn.DataParallel(model).cuda()
    model.training = True
    out = model([C, annot])
    # if model.training == True out==loss
    # out = model([C, annot])
    # if model.training == False out== scores
    # out = model(C)
    for i in range(len(out)):
        print(out[i])

import torch
import torch.nn as nn
from .neck import FPN
from .head import ClsCntRegHead
from .backbones.ResNet import ResNet
from .backbones.DarkNet import DarkNet
from .backbones.MobileNet import MobileNet
from .loss.Loss import GenTargets, LOSS, coords_fmap2orig


class FCOS(nn.Module):

    def __init__(self, cfg=None):
        super().__init__()
        if cfg is None:
            self.backbones_type = "resnet50"
            self.pretrained = False
            self.fpn_out_channels = 256
            self.use_p5 = True
            self.class_num = 20
            self.use_GN_head = True
            self.cnt_on_reg = True
            self.prior = 0.01
            self.freeze_bn = True
            self.freeze_stage_1 = True
            self.attention = 'atdh'
        else:
            self.backbones_type = cfg['Backbones']['name'] + str(cfg['Backbones']['depth'])
            self.pretrained = cfg['Models']['pretrained']
            self.fpn_out_channels = cfg['Neck']['fpn_out_channels']
            self.use_p5 = cfg['Neck']['use_p5']
            self.class_num = cfg['Data']['class_num']
            self.use_GN_head = cfg['Head']['use_GN_head']
            self.cnt_on_reg = cfg['Head']['cnt_on_reg']
            self.prior = cfg['Head']['prior']
            self.freeze_bn = cfg['Models']['freeze_bn']
            self.freeze_stage_1 = cfg['Models']['freeze_stage_1']
            self.attention = cfg['Head']['name']

        if self.backbones_type[:6] == "resnet":
            self.backbones = ResNet(resnet_type=self.backbones_type, pretrained=self.pretrained)

        elif self.backbones_type[:9] == "mobilenet":
            self.backbones = MobileNet(mobilenet_type=self.backbones_type, pretrained=self.pretrained)

        elif self.backbones_type[:7] == "darknet":
            self.backbones = DarkNet(darknet_type=self.backbones_type)

        expand_ratio = {
            "resnet18": 1,
            "resnet34": 1,
            "resnet50": 4,
            "resnet101": 4,
            "resnet152": 4,
            "darknetttiny": 0.5,
            "darknet19": 1,
            "darknet53": 2,
            "mobilenet_v2": 1,
            "mobilenet_v3_small": 2,
            "mobilent_v3_large": 2,
        }
        C3_inplanes, C4_inplanes, C5_inplanes = \
            int(128 * expand_ratio[self.backbones_type]), \
            int(256 * expand_ratio[self.backbones_type]), \
            int(512 * expand_ratio[self.backbones_type])
        self.fpn = FPN(C3_inplanes, C4_inplanes, C5_inplanes, self.fpn_out_channels, use_p5=self.use_p5)
        self.head = ClsCntRegHead(self.fpn_out_channels, self.class_num,
                                  self.use_GN_head, self.cnt_on_reg, self.prior, self.attention)

    def train(self, mode=True):
        '''
        set module training mode, and frozen bn
        '''
        super().train(mode=True)

        def freeze_bn(module):
            if isinstance(module, nn.BatchNorm2d):
                module.eval()
            classname = module.__class__.__name__
            if classname.find('BatchNorm') != -1:
                for p in module.parameters(): p.requires_grad = False

        if self.freeze_bn:
            self.apply(freeze_bn)
            print("INFO===>success frozen BN")
        if self.freeze_stage_1:
            self.backbones.freeze_stages(1)
            print("INFO===>success frozen backbone stage1")

    def forward(self, x):
        '''
        Returns
        list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        C3, C4, C5 = self.backbones(x)
        all_P = self.fpn([C3, C4, C5])
        cls_logits, cnt_logits, reg_preds = self.head(all_P)
        return [cls_logits, cnt_logits, reg_preds]


class DetectHead(nn.Module):
    def __init__(self, score_threshold, nms_iou_threshold, max_detection_boxes_num, strides, cfg=None):
        super().__init__()
        self.score_threshold = score_threshold
        self.nms_iou_threshold = nms_iou_threshold
        self.max_detection_boxes_num = max_detection_boxes_num
        self.strides = strides
        if cfg is None:
            self.add_centerness = True
        else:
            self.add_centerness = cfg['Loss']['add_centerness']

    def forward(self, inputs):
        '''
        inputs  list [cls_logits,cnt_logits,reg_preds]
        cls_logits  list contains five [batch_size,class_num,h,w]
        cnt_logits  list contains five [batch_size,1,h,w]
        reg_preds   list contains five [batch_size,4,h,w]
        '''
        cls_logits, coords = self._reshape_cat_out(inputs[0], self.strides)  # [batch_size,sum(_h*_w),class_num]
        cnt_logits, _ = self._reshape_cat_out(inputs[1], self.strides)  # [batch_size,sum(_h*_w),1]
        reg_preds, _ = self._reshape_cat_out(inputs[2], self.strides)  # [batch_size,sum(_h*_w),4]

        cls_preds = cls_logits.sigmoid_()
        cnt_preds = cnt_logits.sigmoid_()

        coords = coords.cuda() if torch.cuda.is_available() else coords

        cls_scores, cls_classes = torch.max(cls_preds, dim=-1)  # [batch_size,sum(_h*_w)]
        if self.add_centerness:
            cls_scores = torch.sqrt(cls_scores * (cnt_preds.squeeze(dim=-1)))  # [batch_size,sum(_h*_w)]
        cls_classes = cls_classes + 1  # [batch_size,sum(_h*_w)]

        boxes = self._coords2boxes(coords, reg_preds)  # [batch_size,sum(_h*_w),4]

        # select topk
        max_num = min(self.max_detection_boxes_num, cls_scores.shape[-1])
        topk_ind = torch.topk(cls_scores, max_num, dim=-1, largest=True, sorted=True)[1]  # [batch_size,max_num]
        _cls_scores = []
        _cls_classes = []
        _boxes = []
        for batch in range(cls_scores.shape[0]):
            _cls_scores.append(cls_scores[batch][topk_ind[batch]])  # [max_num]
            _cls_classes.append(cls_classes[batch][topk_ind[batch]])  # [max_num]
            _boxes.append(boxes[batch][topk_ind[batch]])  # [max_num,4]
        cls_scores_topk = torch.stack(_cls_scores, dim=0)  # [batch_size,max_num]
        cls_classes_topk = torch.stack(_cls_classes, dim=0)  # [batch_size,max_num]
        boxes_topk = torch.stack(_boxes, dim=0)  # [batch_size,max_num,4]
        assert boxes_topk.shape[-1] == 4
        return self._post_process([cls_scores_topk, cls_classes_topk, boxes_topk])

    def _post_process(self, preds_topk):
        '''
        cls_scores_topk [batch_size,max_num]
        cls_classes_topk [batch_size,max_num]
        boxes_topk [batch_size,max_num,4]
        '''
        _cls_scores_post = []
        _cls_classes_post = []
        _boxes_post = []
        cls_scores_topk, cls_classes_topk, boxes_topk = preds_topk
        for batch in range(cls_classes_topk.shape[0]):
            mask = cls_scores_topk[batch] >= self.score_threshold
            _cls_scores_b = cls_scores_topk[batch][mask]  # [?]
            _cls_classes_b = cls_classes_topk[batch][mask]  # [?]
            _boxes_b = boxes_topk[batch][mask]  # [?,4]
            nms_ind = self.batched_nms(_boxes_b, _cls_scores_b, _cls_classes_b, self.nms_iou_threshold)
            _cls_scores_post.append(_cls_scores_b[nms_ind])
            _cls_classes_post.append(_cls_classes_b[nms_ind])
            _boxes_post.append(_boxes_b[nms_ind])
        scores, classes, boxes = torch.stack(_cls_scores_post, dim=0), torch.stack(_cls_classes_post,
                                                                                   dim=0), torch.stack(_boxes_post,
                                                                                                       dim=0)

        return scores, classes, boxes

    @staticmethod
    def box_nms(boxes, scores, thr):
        '''
        boxes: [?,4]
        scores: [?]
        '''
        if boxes.shape[0] == 0:
            return torch.zeros(0, device=boxes.device).long()
        assert boxes.shape[-1] == 4
        x1, y1, x2, y2 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.sort(0, descending=True)[1]
        keep = []
        while order.numel() > 0:
            if order.numel() == 1:
                i = order.item()
                keep.append(i)
                break
            else:
                i = order[0].item()
                keep.append(i)

            xmin = x1[order[1:]].clamp(min=float(x1[i]))
            ymin = y1[order[1:]].clamp(min=float(y1[i]))
            xmax = x2[order[1:]].clamp(max=float(x2[i]))
            ymax = y2[order[1:]].clamp(max=float(y2[i]))
            inter = (xmax - xmin).clamp(min=0) * (ymax - ymin).clamp(min=0)
            iou = inter / (areas[i] + areas[order[1:]] - inter)
            idx = (iou <= thr).nonzero().squeeze()
            if idx.numel() == 0:
                break
            order = order[idx + 1]
        return torch.LongTensor(keep)

    def batched_nms(self, boxes, scores, idxs, iou_threshold):

        if boxes.numel() == 0:
            return torch.empty((0,), dtype=torch.int64, device=boxes.device)
        # strategy: in order to perform NMS independently per class.
        # we add an offset to all the boxes. The offset is dependent
        # only on the class idx, and is large enough so that boxes
        # from different classes do not overlap
        max_coordinate = boxes.max()
        offsets = idxs.to(boxes) * (max_coordinate + 1)
        boxes_for_nms = boxes + offsets[:, None]
        keep = self.box_nms(boxes_for_nms, scores, iou_threshold)
        return keep

    def _coords2boxes(self, coords, offsets):
        '''
        Args
        coords [sum(_h*_w),2]
        offsets [batch_size,sum(_h*_w),4] ltrb
        '''
        x1y1 = coords[None, :, :] - offsets[..., :2]
        x2y2 = coords[None, :, :] + offsets[..., 2:]  # [batch_size,sum(_h*_w),2]
        boxes = torch.cat([x1y1, x2y2], dim=-1)  # [batch_size,sum(_h*_w),4]
        return boxes

    def _reshape_cat_out(self, inputs, strides):
        '''
        Args
        inputs: list contains five [batch_size,c,_h,_w]
        Returns
        out [batch_size,sum(_h*_w),c]
        coords [sum(_h*_w),2]
        '''
        batch_size = inputs[0].shape[0]
        c = inputs[0].shape[1]
        out = []
        coords = []
        for pred, stride in zip(inputs, strides):
            pred = pred.permute(0, 2, 3, 1)
            coord = coords_fmap2orig(pred, stride).to(device=pred.device)
            pred = torch.reshape(pred, [batch_size, -1, c])
            out.append(pred)
            coords.append(coord)
        return torch.cat(out, dim=1), torch.cat(coords, dim=0)


class ClipBoxes(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, batch_imgs, batch_boxes):
        batch_boxes = batch_boxes.clamp_(min=0)
        h, w = batch_imgs.shape[2:]
        batch_boxes[..., [0, 2]] = batch_boxes[..., [0, 2]].clamp_(max=w - 1)
        batch_boxes[..., [1, 3]] = batch_boxes[..., [1, 3]].clamp_(max=h - 1)
        return batch_boxes


class FCOSDetector(nn.Module):
    def __init__(self, mode="training", cfg=None):
        super().__init__()
        if cfg is None:
            self.strides = [8, 16, 32, 64, 128]
            self.limit_range = [[-1, 64], [64, 128], [128, 256], [256, 512], [512, 999999]]
            self.score_threshold = 0.05
            self.nms_iou_threshold = 0.6
            self.max_detection_boxes_num = 1000
        else:
            self.strides = cfg['Training']['strides']
            self.limit_range = cfg['Training']['limit_range']
            self.score_threshold = cfg['Inference']['score_threshold']
            self.nms_iou_threshold = cfg['Inference']['nms_iou_threshold']
            self.max_detection_boxes_num = cfg['Inference']['max_detection_boxes_num']

        self.mode = mode
        self.fcos_body = FCOS(cfg=cfg)
        if mode == "training":
            self.target_layer = GenTargets(strides=self.strides, limit_range=self.limit_range)
            self.loss_layer = LOSS()
        elif mode == "inference":
            self.detection_head = DetectHead(self.score_threshold, self.nms_iou_threshold,
                                             self.max_detection_boxes_num, self.strides, cfg)
            self.clip_boxes = ClipBoxes()

    def forward(self, inputs):
        '''
        inputs
        [training] list  batch_imgs,batch_boxes,batch_classes
        [inference] img
        '''

        if self.mode == "training":
            batch_imgs, batch_boxes, batch_classes = inputs
            out = self.fcos_body(batch_imgs)
            targets = self.target_layer([out, batch_boxes, batch_classes])
            losses = self.loss_layer([out, targets])
            return losses
        elif self.mode == "inference":
            # raise NotImplementedError("no implement inference model")
            '''
            for inference mode, img should preprocessed before feeding in net 
            '''
            batch_imgs = inputs
            out = self.fcos_body(batch_imgs)
            scores, classes, boxes = self.detection_head(out)
            boxes = self.clip_boxes(batch_imgs, boxes)
            return scores, classes, boxes

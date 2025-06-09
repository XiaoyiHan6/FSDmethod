import torch
from torchvision.ops import nms
from torch.autograd import Function
from models.utils.box_utils import decode


class Detect(Function):
    """At test time, Detect is the final layer of SSD.  Decode location preds,
    apply non-maximum suppression to location predictions based on conf
    scores and threshold to a top_k number of output predictions for both
    confidence score and locations.
    """

    def __init__(self, cfg, bkg_label=0):
        self.num_classes = cfg['Data']['num_classes']
        self.top_k = cfg['Test']['top_k']
        self.conf_thresh = cfg['Test']['conf_thresh']
        # Parameters used in nms.
        self.nms_thresh = cfg['Test']['nms_thresh']
        if cfg['Test']['nms_thresh'] <= 0:
            raise ValueError('nms_threshold must be non negative.')
        self.background_label = bkg_label
        self.variance = cfg['Prior_box']['variance']

    def forward(self, loc_data, conf_data, prior_data):
        """
        Args:
            loc_data: (tensor) Loc preds from loc layers
                Shape: [batch,num_priors,4]
            conf_data: (tensor) Shape: Conf preds from conf layers
                Shape: [batch,num_priors,num_classes]
            prior_data: (tensor) Prior boxes and variances from priorbox layers
                Shape: [num_priors,4]
        """
        device = loc_data.device
        prior_data = prior_data.to(device)
        num = loc_data.size(0)  # batch size
        num_priors = prior_data.size(0)  #  small 8732,large 24564
        output = torch.zeros(num, self.num_classes, self.top_k, 5)
        conf_preds = conf_data.view(num, num_priors,
                                    self.num_classes).transpose(2, 1)

        # Decode predictions into bboxes.

        for i in range(num):
            decoded_boxes = decode(loc_data[i], prior_data, self.variance)
            # For each class, perform nms
            conf_scores = conf_preds[i].clone()
            for cl in range(1, self.num_classes):
                c_mask = conf_scores[cl].gt(self.conf_thresh)
                scores = conf_scores[cl][c_mask]
                if scores.size(0) == 0:
                    continue
                l_mask = c_mask.unsqueeze(1).expand_as(decoded_boxes)
                boxes = decoded_boxes[l_mask].view(-1, 4)
                # idx of highest scoring and non-overlapping boxes per class
                ids = nms(boxes, scores, self.nms_thresh)
                if len(ids) < self.top_k:
                    output[i, cl, :len(ids)] = torch.cat((scores[ids].unsqueeze(1), boxes[ids]), 1)
                else:
                    output[i, cl, :self.top_k] = torch.cat((scores[ids[:self.top_k]].unsqueeze(1),
                                                            boxes[ids[:self.top_k]]), 1)
        flt = output.contiguous().view(num, -1, 5)
        _, idx = flt[:, :, 0].sort(1, descending=True)
        _, rank = idx.sort(1)
        flt[(rank < self.top_k).unsqueeze(-1).expand_as(flt)].fill_(0)
        return output


if __name__ == '__main__':
    from options.train import cfg

    detect = Detect(cfg)
    loc_data = torch.randn(16, 8732, 4)
    conf_data = torch.randn(16, 8732, 81)
    prior_data = torch.randn(8732, 4)

    out = detect.forward(loc_data, conf_data, prior_data)
    print('Detect output shape:', out.shape)

import torch
import numpy as np
import torch.nn as nn


# according to stride, grid of each feature map -> original image
def coords_fmap2orig(feature, stride):
    '''
    transfor one fmap coords to orig coords
    Args
    feature map [batch_size,h,w,c]
    stride int

    Returns
    coords [n,2]
    '''
    h, w = feature.shape[1:3]
    shifts_x = torch.arange(0, w * stride, stride, dtype=torch.float32)
    shifts_y = torch.arange(0, h * stride, stride, dtype=torch.float32)

    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
    shift_x = torch.reshape(shift_x, [-1])
    shift_y = torch.reshape(shift_y, [-1])
    coords = torch.stack([shift_x, shift_y], -1) + stride // 2
    return coords


# fcos_bobdy pred and gt match
class GenTargets(nn.Module):
    def __init__(self, strides, limit_range):
        super(GenTargets, self).__init__()
        self.strides = strides
        self.limit_range = limit_range
        assert len(strides) == len(limit_range)

    def forward(self, inputs):
        '''
        inputs  
        [0] list [cls_logits, cnt_logits, reg_preds]
        cls_logits  list contains five [batch_size, class_num, h, w]
        cnt_logits  list contains five [batch_size, 1, h, w]
        reg_preds   list contains five [batch_size, 4, h, w]
        [1] gt_boxes [batch_size, m, 4]  FloatTensor
        [2] classes [batch_size, m]  LongTensor

        Returns
        cls_targets:[batch_size,sum(_h*_w), 1]
        cnt_targets:[batch_size,sum(_h*_w), 1]
        reg_targets:[batch_size,sum(_h*_w), 4]
        '''
        cls_logits, cnt_logits, reg_preds = inputs[0]
        gt_boxes = inputs[1]
        classes = inputs[2]
        cls_targets_all_level = []
        cnt_targets_all_level = []
        reg_targets_all_level = []
        assert len(self.strides) == len(cls_logits)
        for level in range(len(cls_logits)):
            level_out = [cls_logits[level], cnt_logits[level], reg_preds[level]]
            level_targets = self._gen_level_targets(level_out, gt_boxes, classes, self.strides[level],
                                                    self.limit_range[level])
            cls_targets_all_level.append(level_targets[0])
            cnt_targets_all_level.append(level_targets[1])
            reg_targets_all_level.append(level_targets[2])

        return torch.cat(cls_targets_all_level, dim=1), torch.cat(cnt_targets_all_level, dim=1), torch.cat(
            reg_targets_all_level, dim=1)

    def _gen_level_targets(self, out, gt_boxes, classes, stride, limit_range, sample_radiu_ratio=1.5):
        '''
        Args  
        out list contains [[batch_size, class_num, h, w],[batch_size, 1, h, w],[batch_size, 4, h, w]]
        gt_boxes [batch_size, m, 4]
        classes [batch_size, m]
        stride int  
        limit_range list [min, max]

        Returns  
        cls_targets, cnt_targets, reg_targets
        '''
        cls_logits, cnt_logits, reg_preds = out
        batch_size = cls_logits.shape[0]
        class_num = cls_logits.shape[1]
        m = gt_boxes.shape[1]

        # [batch_size, h, w, class_num]
        cls_logits = cls_logits.permute(0, 2, 3, 1)
        # [h*w, 2]
        coords = coords_fmap2orig(cls_logits, stride).to(device=gt_boxes.device)
        # [batch_size, h*w, class_num]
        cls_logits = cls_logits.reshape((batch_size, -1, class_num))

        cnt_logits = cnt_logits.permute(0, 2, 3, 1)
        cnt_logits = cnt_logits.reshape((batch_size, -1, 1))
        reg_preds = reg_preds.permute(0, 2, 3, 1)
        reg_preds = reg_preds.reshape((batch_size, -1, 4))

        h_mul_w = cls_logits.shape[1]

        x = coords[:, 0]
        y = coords[:, 1]
        # [1, h*w, 1] - [batch_size, 1, m] --> [batch_size, h*w, m]
        l_off = x[None, :, None] - gt_boxes[..., 0][:, None, :]
        t_off = y[None, :, None] - gt_boxes[..., 1][:, None, :]
        r_off = gt_boxes[..., 2][:, None, :] - x[None, :, None]
        b_off = gt_boxes[..., 3][:, None, :] - y[None, :, None]
        # [batch_size, h*w, m, 4]
        ltrb_off = torch.stack([l_off, t_off, r_off, b_off], dim=-1)

        # [batch_size, h*w, m]
        areas = (ltrb_off[..., 0] + ltrb_off[..., 2]) * (ltrb_off[..., 1] + ltrb_off[..., 3])

        # [batch_size, h*w, m]
        off_min = torch.min(ltrb_off, dim=-1)[0]
        # [batch_size, h*w, m]
        off_max = torch.max(ltrb_off, dim=-1)[0]

        mask_in_gtboxes = off_min > 0
        mask_in_level = (off_max > limit_range[0]) & (off_max <= limit_range[1])

        radiu = stride * sample_radiu_ratio
        gt_center_x = (gt_boxes[..., 0] + gt_boxes[..., 2]) / 2
        gt_center_y = (gt_boxes[..., 1] + gt_boxes[..., 3]) / 2

        # [1, h*w, 1] - [batch_size, 1, m] --> [batch_size, h*w, m]
        c_l_off = x[None, :, None] - gt_center_x[:, None, :]
        c_t_off = y[None, :, None] - gt_center_y[:, None, :]
        c_r_off = gt_center_x[:, None, :] - x[None, :, None]
        c_b_off = gt_center_y[:, None, :] - y[None, :, None]

        # [batch_size, h*w, m, 4]
        c_ltrb_off = torch.stack([c_l_off, c_t_off, c_r_off, c_b_off], dim=-1)
        c_off_max = torch.max(c_ltrb_off, dim=-1)[0]
        mask_center = c_off_max < radiu

        mask_pos = mask_in_gtboxes & mask_in_level & mask_center  # [batch_size, h*w, m]

        areas[~mask_pos] = 99999999

        # [batch_size, h*w]
        areas_min_ind = torch.min(areas, dim=-1)[1]

        # [batch_size*h*w, 4]
        reg_targets = ltrb_off[
            torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]
        # [batch_size, h*w, 4]
        reg_targets = torch.reshape(reg_targets, (batch_size, -1, 4))

        # [batch_size, h*w, m]
        classes = torch.broadcast_tensors(classes[:, None, :], areas.long())[0]
        cls_targets = classes[
            torch.zeros_like(areas, dtype=torch.bool).scatter_(-1, areas_min_ind.unsqueeze(dim=-1), 1)]
        # [batch_size, h*w, 1]
        cls_targets = torch.reshape(cls_targets, (batch_size, -1, 1))

        # [batch_size, h*w]
        left_right_min = torch.min(reg_targets[..., 0], reg_targets[..., 2])
        left_right_max = torch.max(reg_targets[..., 0], reg_targets[..., 2])
        top_bottom_min = torch.min(reg_targets[..., 1], reg_targets[..., 3])
        top_bottom_max = torch.max(reg_targets[..., 1], reg_targets[..., 3])
        # [batch_size, h*w, 1]
        cnt_targets = ((left_right_min * top_bottom_min) / (left_right_max * top_bottom_max + 1e-10)).sqrt().unsqueeze(
            dim=-1)

        assert reg_targets.shape == (batch_size, h_mul_w, 4)
        assert cls_targets.shape == (batch_size, h_mul_w, 1)
        assert cnt_targets.shape == (batch_size, h_mul_w, 1)

        # process neg coords
        # [batch_size, h*w]
        mask_pos_2 = mask_pos.long().sum(dim=-1)
        # num_pos=mask_pos_2.sum(dim=-1)
        # assert num_pos.shape==(batch_size,)
        mask_pos_2 = mask_pos_2 >= 1
        assert mask_pos_2.shape == (batch_size, h_mul_w)
        # [batch_size, h*w, 1]
        cls_targets[~mask_pos_2] = 0
        cnt_targets[~mask_pos_2] = -1
        reg_targets[~mask_pos_2] = -1

        return cls_targets, cnt_targets, reg_targets


class ClipBoxes(nn.Module):
    def __init__(self):
        super(ClipBoxes, self).__init__()

    def forward(self, batch_imgs, batch_boxes):
        batch_boxes = batch_boxes.clamp_(min=0)
        h, w = batch_imgs.shape[2:]
        batch_boxes[..., [0, 2]] = batch_boxes[..., [0, 2]].clamp_(max=w - 1)
        batch_boxes[..., [1, 3]] = batch_boxes[..., [1, 3]].clamp_(max=h - 1)
        return batch_boxes



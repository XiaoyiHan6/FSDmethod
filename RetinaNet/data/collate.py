import torch
import numpy as np

def collater(data):
    imgs = [s['img'] for s in data]
    annots = [s['annot'] for s in data]
    scales = [s['scale'] for s in data]

    # skimage shape[0]=h,shape[1]=w
    # B,H,W,3 -> B,3,H,W
    heights = [int(s.shape[0]) for s in imgs]
    widths = [int(s.shape[1]) for s in imgs]
    batch_size = len(imgs)

    max_height = np.array(heights).max()
    max_width = np.array(widths).max()

    padded_imgs = np.zeros((batch_size, max_height, max_width, 3), dtype=np.float32)

    for i, img in enumerate(imgs):
        padded_imgs[i, :int(img.shape[0]), :int(img.shape[1]), :] = img
    padded_imgs = torch.from_numpy(padded_imgs)
    padded_imgs = padded_imgs.permute(0, 3, 1, 2)

    max_num_annots = max(annot.shape[0] for annot in annots)

    if max_num_annots > 0:
        annot_padded = np.ones((len(annots), max_num_annots, 5), dtype=np.float32) * -1
        for idx, annot in enumerate(annots):
            if annot.shape[0] > 0:
                annot_padded[idx, :annot.shape[0], :] = annot
    else:
        annot_padded = np.ones((len(annots), 1, 5), dtype=np.float32) * -1
    annot_padded = torch.from_numpy(annot_padded)

    scales = np.array(scales, dtype=np.float32)
    scales = torch.from_numpy(scales)
    return {'img': padded_imgs, 'annot': annot_padded, 'scale': scales}

import skimage
import numpy as np


class Resizer(object):
    """Convert ndarrays in sample to Tensors."""

    def __init__(self, min_side=360):
        if min_side == 360:
            self.min_side = min_side
            self.max_side = 608
        elif min_side == 608:
            self.min_side = min_side
            self.max_side = 1024

    def __call__(self, sample):
        # def __call__(self, sample, min_side=608, max_side=1024):

        image, annots = sample['img'], sample['annot']

        rows, cols, cns = image.shape

        smallest_side = min(rows, cols)

        # rescale the image so the smallest side is min_side
        scale = self.min_side / smallest_side

        # check if the largest side is now greater than max_side, which can happen
        # when images have a large aspect ratio
        largest_side = max(rows, cols)

        if largest_side * scale > self.max_side:
            scale = self.max_side / largest_side

        # resize the image with the computed scale
        image = skimage.transform.resize(image, (int(round(rows * scale)), int(round((cols * scale)))))
        rows, cols, cns = image.shape

        pad_w = 32 - rows % 32
        pad_h = 32 - cols % 32

        new_image = np.zeros((rows + pad_w, cols + pad_h, cns)).astype(np.float32)
        new_image[:rows, :cols, :] = image.astype(np.float32)

        annots[:, :4] *= scale

        return {'img': new_image, 'annot': annots, 'scale': scale}


class Augmenter(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample, flip_x=0.5):
        if np.random.rand() < flip_x:
            image, annots = sample['img'], sample['annot']
            image = image[:, ::-1, :]

            rows, cols, channels = image.shape

            x1 = annots[:, 0].copy()
            x2 = annots[:, 2].copy()

            x_tmp = x1.copy()

            annots[:, 0] = cols - x2
            annots[:, 2] = cols - x_tmp

            sample = {'img': image, 'annot': annots}

        return sample


class Normalizer(object):

    def __init__(self):
        self.mean = np.array([[[0.485, 0.456, 0.406]]])
        self.std = np.array([[[0.229, 0.224, 0.225]]])

    def __call__(self, sample):
        image, annots = sample['img'], sample['annot']

        return {'img': ((image.astype(np.float32) - self.mean) / self.std), 'annot': annots}


class UnNormalizer(object):
    def __init__(self, mean=None, std=None):
        if mean == None:
            self.mean = [0.485, 0.456, 0.406]
        else:
            self.mean = mean
        if std == None:
            self.std = [0.229, 0.224, 0.225]
        else:
            self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    from skimage import io

    img_1 = io.imread('/data/FireDataset/images/000001.jpg')
    annots_1 = np.array([[305., 360., 984., 723., 1.],
                         [90., 2., 1992., 812., 2.]])

    img_2 = io.imread('/data/FireDataset/images/000005.jpg')
    annots_2 = np.array([[105., 264., 140., 307., 1.],
                         [266., 39., 563., 343., 1.],
                         [2., 2., 600., 344., 2.]])

    plt.figure('org')
    io.imshow(img_1)

    for i, coord in enumerate(annots_1):
        a = plt.Rectangle((coord[0], coord[1]),
                          coord[2] - coord[0],
                          coord[3] - coord[1],
                          fill=False,
                          edgecolor='r', linewidth=2)
        plt.gca().add_patch(a)
    plt.show()

    sample = {'img': img_1, 'annot': annots_1}
    resizer = Resizer()
    sample = resizer(sample)
    dst_img, dst_annots = sample['img'], sample['annot']
    plt.figure('resize')
    plt.imshow(dst_img)
    for i, coord in enumerate(dst_annots):
        a = plt.Rectangle((coord[0], coord[1]),
                          coord[2] - coord[0],
                          coord[3] - coord[1],
                          fill=False,
                          edgecolor='g', linewidth=2)
        plt.gca().add_patch(a)
    plt.show()
    print(dst_img.shape)

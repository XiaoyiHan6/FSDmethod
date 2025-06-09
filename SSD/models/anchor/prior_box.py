from __future__ import division
from math import sqrt as sqrt
from itertools import product as product
import torch


class PriorBox(object):
    """Compute priorbox coordinates in center-offset form for each source
    feature map.
    """

    def __init__(self, cfg):
        super(PriorBox, self).__init__()
        self.image_size = cfg['Data']['size']
        # number of priors for feature map location (either 4 or 6)
        self.num_priors = len(cfg['Prior_box']['aspect_ratios'])
        self.variance = cfg['Prior_box']['variance'] or [0.1]
        self.feature_maps = cfg['Prior_box']['feature_maps']
        self.min_sizes = cfg['Prior_box']['min_sizes']
        self.max_sizes = cfg['Prior_box']['max_sizes']
        self.steps = cfg['Prior_box']['steps']
        self.aspect_ratios = cfg['Prior_box']['aspect_ratios']
        self.clip = cfg['Prior_box']['clip']
        self.version = cfg['Data']['name']
        for v in self.variance:
            if v <= 0:
                raise ValueError('Variances must be greater than 0')

    def forward(self):
        mean = []
        for k, f in enumerate(self.feature_maps):
            for i, j in product(range(f), repeat=2):
                f_k = self.image_size / self.steps[k]
                # unit center x,y
                cx = (j + 0.5) / f_k
                cy = (i + 0.5) / f_k

                # aspect_ratio: 1
                # rel size: min_size
                s_k = self.min_sizes[k] / self.image_size
                mean += [cx, cy, s_k, s_k]

                # aspect_ratio: 1
                # rel size: sqrt(s_k * s_(k+1))
                s_k_prime = sqrt(s_k * (self.max_sizes[k] / self.image_size))
                mean += [cx, cy, s_k_prime, s_k_prime]

                # rest of aspect ratios
                for ar in self.aspect_ratios[k]:
                    mean += [cx, cy, s_k * sqrt(ar), s_k / sqrt(ar)]
                    mean += [cx, cy, s_k / sqrt(ar), s_k * sqrt(ar)]
        # back to torch land
        output = torch.Tensor(mean).view(-1, 4)
        if self.clip:
            output.clamp_(max=1, min=0)
        return output


if __name__ == '__main__':
    from options.train import cfg

    box = PriorBox(cfg)
    print('Priors box shape: ', box.forward().shape)
    print('Priors box:\n', box.forward())

import os
import yaml
import argparse
import numpy as np
from torch.utils.data import DataLoader
from data.augmentations import Transforms
from data.voc import VOC_ROOT, VocDetection
from data.Fire import VISIFIRE_VOC_ROOT, FIRESENSE_VOC_ROOT, FURG_FIRE_DATASET_VOC_ROOT, BOWFIREDATASET_VOC_ROOT, \
    FIRE_SMOKE_DATASET_VOC_ROOT, MY_FIRE_SMOKE_DATASET_ROOT, FireDetection

base_path = '/data/PycharmProject/pytorch-MyFireNet-master'

train_log = os.path.join(base_path, 'log')
tensorboard_log = os.path.join(base_path, 'tensorboard')
checkpoints = os.path.join(base_path, 'checkpoints')


# argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for training a FireNet network')
    parser.add_argument('--tensorboard',
                        type=str,
                        default=False,
                        help='Use tensorboard for loss visualization')
    parser.add_argument('--tensorboard_log',
                        type=str,
                        default=tensorboard_log,
                        help='Use tensorboard for loss visualization')
    parser.add_argument('--cuda',
                        type=str,
                        default=True,
                        help='Use CUDA to train model')
    parser.add_argument('--log_folder',
                        type=str,
                        default=train_log)
    parser.add_argument('--log_name',
                        type=str,
                        default="train")
    parser.add_argument('--checkpoints',
                        type=str,
                        default=checkpoints,
                        help='Checkpoints state_dict folder to resume training from')
    parser.add_argument('--resume',
                        type=str,
                        default='MyFireNet_FCOS_resnet50_myfire_trainval400_sknet_best_9711.pth',
                        help='Checkpoints state_dict file to resume training from')
    parser.add_argument('--config',
                        type=str,
                        default='{}/configs/MyFire_trainval_voc416.yaml'.format(base_path),
                        help='configuration file *.yaml')

    return parser.parse_args()


# Load yaml
def parse_config(yaml_path):
    if not os.path.isfile(yaml_path):
        raise ValueError(f'{yaml_path} not exists.')
    with open(yaml_path, 'r', encoding='utf-8') as f:
        try:
            dict = yaml.safe_load(f.read())
        except yaml.YAMLError:
            raise ValueError('Error parsing YAML file:' + yaml_path)
    return dict


class set_config(object):
    # Load yaml
    args = parse_args()
    if args.config:
        cfg = parse_config(args.config)
    else:
        raise ValueError('--config must be specified.')

    transforms = Transforms()
    # Create the data loaders
    if cfg['Data']['name'] == 'VOC':
        if cfg['Data']['root'] != VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset VOC')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default VOC dataset, but " +
                             "--dataset_root was not specified.")

        dataset_train = VocDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                     split='trainval', is_train=True, augment=transforms)
        dataset_val = VocDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                   split='test', is_train=False, augment=None)

    elif cfg['Data']['name'] == 'visifire':
        if cfg['Data']['root'] != VISIFIRE_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset VISIFIRE')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default VISIFRE dataset, but " +
                             "--dataset_root was not specified.")
        # augment=transforms
        dataset_train = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                      split='train', is_train=True, augment=transforms)
        dataset_val = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                    split='test', is_train=False, augment=None)

    elif cfg['Data']['name'] == 'firesense':
        if cfg['Data']['root'] != FIRESENSE_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset FIRESENSE')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default FIRESENSE dataset, but " +
                             "--dataset_root was not specified.")

        dataset_train = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                      split='train', is_train=True, augment=transforms)
        dataset_val = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                    split='test', is_train=False, augment=None)

    elif cfg['Data']['name'] == 'furg-fire-dataset':
        if cfg['Data']['root'] != FURG_FIRE_DATASET_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset FURG-FIRE-Dataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default FURG-FIRE-Dataset, but " +
                             "--dataset_root was not specified.")
        dataset_train = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                      split='train', is_train=True, augment=transforms)
        dataset_val = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                    split='test', is_train=False, augment=None)

    elif cfg['Data']['name'] == 'bowfiredataset':
        if cfg['Data']['root'] != BOWFIREDATASET_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset BoWFireDataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default BoWFireDataset, but " +
                             "--dataset_root was not specified.")
        dataset_train = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                      split='train', is_train=True, augment=transforms)
        dataset_val = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                    split='test', is_train=False, augment=None)

    elif cfg['Data']['name'] == 'fire-smoke-dataset':
        if cfg['Data']['root'] != FIRE_SMOKE_DATASET_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset Fire-Smoke-Dataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default Fire-Smoke-Dataset, but " +
                             "--dataset_root was not specified.")
        dataset_train = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                      split='train', is_train=True, augment=transforms)
        dataset_val = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                    split='test', is_train=False, augment=None)

    elif cfg['Data']['name'] == 'myfire':
        if cfg['Data']['root'] != MY_FIRE_SMOKE_DATASET_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset MyFireDataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default MyFireDataset, but " +
                             "--dataset_root was not specified.")
        dataset_train = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                      split='train', is_train=True, augment=transforms)
        dataset_val = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                    split='test', is_train=False, augment=None)

    elif cfg['Data']['name'] == 'minimyfire':
        if cfg['Data']['root'] != MY_FIRE_SMOKE_DATASET_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset mini MyFireDataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default mini MyFireDataset, but " +
                             "--dataset_root was not specified.")
        dataset_train = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                      split='minitrain', is_train=True, augment=transforms)
        dataset_val = FireDetection(cfg['Data']['root'], resize=cfg['Data']['size'],
                                    split='minitest', is_train=False, augment=None)

    elif cfg['Data']['name'] == 'myfire_trainval':
        if cfg['Data']['root'] != MY_FIRE_SMOKE_DATASET_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset MyFireDataset')

        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default MyFireDataset, but " +
                             "--dataset_root was not specified.")

        dataset_train = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                      split='trainval', is_train=True, augment=transforms)
        dataset_val = FireDetection(cfg['Data']['root'], resize=cfg['Data']['size'],
                                    split='test', is_train=False, augment=None)

    else:
        raise ValueError('Dataset type not understood (must be csv or coco), exiting.')

    dataloader_train = DataLoader(dataset_train, batch_size=cfg['Optimize']['batch_size'],
                                  collate_fn=dataset_train.collate_fn, shuffle=True, num_workers=4,
                                  worker_init_fn=np.random.seed(0), drop_last=True)
    dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False,
                                collate_fn=dataset_val.collate_fn)

    iter_size = len(dataset_train) // cfg['Optimize']['batch_size']


set_config = set_config()

# args
args = set_config.args

# cfg
cfg = set_config.cfg
args.log_name = cfg['Models']['name'] + "_" + cfg['Data']['name'] + str(cfg['Data']['size'][0]) + "_train"

# dataset_train
dataset_train = set_config.dataset_train

# dataloader_train
dataloader_train = set_config.dataloader_train

# dataset_val
dataset_val = set_config.dataset_val

dataloader_val = set_config.dataloader_val

# iter_size
iter_size = set_config.iter_size

if __name__ == '__main__':

    print("args:{}\nconfig:{}.".format(args, cfg))
    for data in dataloader_train:
        imgs, boxes, classes = data
        print("imgs.shape:", imgs.shape)
        print("boxes:", boxes)
        print("boxes.shape", boxes.shape)
        print("classes:", classes)
        print("classes.shape:", classes.shape)

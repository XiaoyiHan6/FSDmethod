import os
import yaml
import torch
import argparse
from models.ssd import SSD
from data.collate import collate
from torch.utils.data import DataLoader
from data.augmentations import SSDAugmentation
from data.voc0712 import VOC_ROOT, VOCDetection
from data.Fire import VISIFIRE_VOC_ROOT, FIRESENSE_VOC_ROOT, FURG_FIRE_DATASET_VOC_ROOT, BOWFIREDATASET_VOC_ROOT, \
    FIRE_SMOKE_DATASET_VOC_ROOT, MY_FIRE_SMOKE_DATASET_ROOT, FireDetection

base_path = '/data/PycharmProject/pytorch-ssd-master'

train_log = os.path.join(base_path, 'log')
tensorboard_log = os.path.join(base_path, 'tensorboard')
checkpoints = os.path.join(base_path, 'checkpoints')

MEANS = (104, 117, 123)


# argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Simple training script for training a SSD network')
    parser.add_argument('--training',
                        type=str,
                        default=True,
                        help='Model is training or testing')
    parser.add_argument('--pretrained',
                        type=str,
                        default=False,
                        help='Pretrained base model')
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
    parser.add_argument('--resume',
                        type=str,
                        default=None,
                        help='Checkpoint state_dict file to resume training from')
    parser.add_argument('--checkpoints',
                        type=str,
                        default=checkpoints,
                        help='Directory for saving checkpoint models')
    parser.add_argument('--config',
                        type=str,
                        default='{}/configs/MyFire_trainval_voc300.yaml'.format(base_path),
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

    # Create the data loaders
    if cfg['Data']['name'] == 'VOC':
        if cfg['Data']['root'] != VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset VOC')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default VOC dataset, but " +
                             "--dataset_root was not specified.")

        dataset_train = VOCDetection(cfg['Data']['root'],
                                     transform=SSDAugmentation(size=cfg['Data']['size'], mean=MEANS))
        dataset_val = VOCDetection(cfg['Data']['root'], image_set=[('2007', 'test')])

    elif cfg['Data']['name'] == 'visifire':
        if cfg['Data']['root'] != VISIFIRE_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset VISIFIRE')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default VISIFRE dataset, but " +
                             "--dataset_root was not specified.")
        dataset_train = FireDetection(cfg['Data']['root'],
                                      transform=SSDAugmentation(size=cfg['Data']['size'], mean=MEANS))
        dataset_val = FireDetection(cfg['Data']['root'], image_set='test')

    elif cfg['Data']['name'] == 'firesense':
        if cfg['Data']['root'] != FIRESENSE_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset FIRESENSE')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default FIRESENSE dataset, but " +
                             "--dataset_root was not specified.")
        dataset_train = FireDetection(cfg['Data']['root'],
                                      transform=SSDAugmentation(size=cfg['Data']['size'], mean=MEANS))
        dataset_val = FireDetection(cfg['Data']['root'], image_set='test')

    elif cfg['Data']['name'] == 'furg-fire-dataset':
        if cfg['Data']['root'] != FURG_FIRE_DATASET_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset FURG-FIRE-Dataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default FURG-FIRE-Dataset, but " +
                             "--dataset_root was not specified.")
        dataset_train = FireDetection(cfg['Data']['root'],
                                      transform=SSDAugmentation(size=cfg['Data']['size'], mean=MEANS))
        dataset_val = FireDetection(cfg['Data']['root'], image_set='test')

    elif cfg['Data']['name'] == 'bowfiredataset':
        if cfg['Data']['root'] != BOWFIREDATASET_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset BoWFireDataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default BoWFireDataset, but " +
                             "--dataset_root was not specified.")
        dataset_train = FireDetection(cfg['Data']['root'],
                                      transform=SSDAugmentation(size=cfg['Data']['size'], mean=MEANS))
        dataset_val = FireDetection(cfg['Data']['root'], image_set='test')

    elif cfg['Data']['name'] == 'fire-smoke-dataset':
        if cfg['Data']['root'] != FIRE_SMOKE_DATASET_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset Fire-Smoke-Dataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default Fire-Smoke-Dataset, but " +
                             "--dataset_root was not specified.")
        dataset_train = FireDetection(cfg['Data']['root'],
                                      transform=SSDAugmentation(size=cfg['Data']['size'], mean=MEANS))
        dataset_val = FireDetection(cfg['Data']['root'], image_set='test')

    elif cfg['Data']['name'] == 'myfire':
        if cfg['Data']['root'] != MY_FIRE_SMOKE_DATASET_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset My-Fire-Smoke-Dataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default My-Fire-Smoke-Dataset, but " +
                             "--dataset_root was not specified.")
        dataset_train = FireDetection(cfg['Data']['root'],
                                      transform=SSDAugmentation(size=cfg['Data']['size'], mean=MEANS))
        dataset_val = FireDetection(cfg['Data']['root'], image_set='test')

    elif cfg['Data']['name'] == 'minimyfire':
        if cfg['Data']['root'] != MY_FIRE_SMOKE_DATASET_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset mini MyFireDataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default mini MyFireDataset, but " +
                             "--dataset_root was not specified.")
        dataset_train = FireDetection(cfg['Data']['root'],
                                      transform=SSDAugmentation(size=cfg['Data']['size'], mean=MEANS),
                                      image_set='minitrain')
        dataset_val = FireDetection(cfg['Data']['root'], image_set='minitest')

    elif cfg['Data']['name'] == 'myfire_trainval':
        if cfg['Data']['root'] != MY_FIRE_SMOKE_DATASET_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset mini MyFireDataset-trainval')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default mini MyFireDataset-trainval, but " +
                             "--dataset_root was not specified.")
        dataset_train = FireDetection(cfg['Data']['root'],
                                      transform=SSDAugmentation(size=cfg['Data']['size'], mean=MEANS),
                                      image_set='trainval')
        dataset_val = FireDetection(cfg['Data']['root'], image_set='test')


    else:
        raise ValueError('Dataset type not understood (must be VOC), exiting.')
    dataloader_train = DataLoader(dataset_train, num_workers=2, batch_size=cfg['Optimize']['batch_size'],
                                  collate_fn=collate, shuffle=True, drop_last=True,
                                  generator=torch.Generator(device='cuda'))

    iter_size = len(dataset_train) // cfg['Optimize']['batch_size']

    model = SSD(cfg, training=args.training, pretrained=args.pretrained)

    model_eval = SSD(cfg)


set_config = set_config()

# args
args = set_config.args

# cfg
cfg = set_config.cfg
args.log_name = cfg['Models']['name'] + "_" + cfg['Data']['name'] + str(cfg['Data']['size']) + "_train"
# dataset_train
dataset_train = set_config.dataset_train

# dataloader_train
dataloader_train = set_config.dataloader_train

# dataset_val
dataset_val = set_config.dataset_val

# iter_size
iter_size = set_config.iter_size

# model
model = set_config.model

model_eval = set_config.model_eval

if __name__ == '__main__':
    print("args:{}\nconfig:{}.".format(args, cfg))
    for data in dataloader_train:
        imgs, targets = data
        print("imgs:", imgs.shape)
        print("targets:", targets)

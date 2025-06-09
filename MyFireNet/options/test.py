import os
import yaml
import torch
import argparse
from models.MyFireNet import FCOSDetector
from data.voc import VOC_ROOT, VocDetection
from data.Fire import VISIFIRE_VOC_ROOT, FIRESENSE_VOC_ROOT, FURG_FIRE_DATASET_VOC_ROOT, BOWFIREDATASET_VOC_ROOT, \
    FIRE_SMOKE_DATASET_VOC_ROOT, MY_FIRE_SMOKE_DATASET_ROOT, FireDetection

base_path = '/data/PycharmProject/pytorch-MyFireNet-master'

test_log = os.path.join(base_path, 'log')
checkpoints = os.path.join(base_path, 'checkpoints')
Results = os.path.join(base_path, 'results')


# argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Simple testing script for testing a FireNet network')
    parser.add_argument('--cuda',
                        type=str,
                        default=True,
                        help='Use CUDA to test model')
    parser.add_argument('--log_folder',
                        type=str,
                        default=test_log)
    parser.add_argument('--log_name',
                        type=str,
                        default="test")
    parser.add_argument('--results',
                        type=str,
                        default=Results,
                        help='Save the result files.')
    parser.add_argument('--checkpoints',
                        type=str,
                        default=checkpoints,
                        help='Checkpoints state_dict folder to resume testing from')
    parser.add_argument('--evaluate',
                        type=str,
                        default='MyFireNet_FCOS_resnet50_minifire400_augment_no_best.pth',
                        help='Checkpoints state_dict file to evaluating model from')
    parser.add_argument("--visual_threshold",
                        type=float,
                        default=0.4,
                        help="visual threshold")
    parser.add_argument('--config',
                        type=str,
                        default='{}/configs/MyFire_trainval_voc416_test.yaml'.format(base_path),
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

        dataset_test = VocDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                    split='test', is_train=False, augment=None)
    elif cfg['Data']['name'] == 'visifire':
        if cfg['Data']['root'] != VISIFIRE_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset VISIFIRE')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default VISIFRE dataset, but " +
                             "--dataset_root was not specified.")
        dataset_test = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                     split='test', is_train=False, augment=None)

    elif cfg['Data']['name'] == 'firesense':
        if cfg['Data']['root'] != FIRESENSE_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset FIRESENSE')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default FIRESENSE dataset, but " +
                             "--dataset_root was not specified.")
        dataset_test = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                     split='test', is_train=False, augment=None)

    elif cfg['Data']['name'] == 'furg-fire-dataset':
        if cfg['Data']['root'] != FURG_FIRE_DATASET_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset FURG-FIRE-Dataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default FURG-FIRE-Dataset, but " +
                             "--dataset_root was not specified.")
        dataset_test = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                     split='test', is_train=False, augment=None)

    elif cfg['Data']['name'] == 'bowfiredataset':
        if cfg['Data']['root'] != BOWFIREDATASET_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset BoWFireDataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default BoWFireDataset, but " +
                             "--dataset_root was not specified.")
        dataset_test = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                     split='test', is_train=False, augment=None)

    elif cfg['Data']['name'] == 'fire-smoke-dataset':
        if cfg['Data']['root'] != FIRE_SMOKE_DATASET_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset Fire-Smoke-Dataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default Fire-Smoke-Dataset, but " +
                             "--dataset_root was not specified.")
        dataset_test = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                     split='test', is_train=False, augment=None)
    elif cfg['Data']['name'] == 'myfire':
        if cfg['Data']['root'] != MY_FIRE_SMOKE_DATASET_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset MyFireDataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default MyFireDataset, but " +
                             "--dataset_root was not specified.")
        dataset_test = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                     split='test', is_train=False, augment=None)
    elif cfg['Data']['name'] == 'minimyfire':
        if cfg['Data']['root'] != MY_FIRE_SMOKE_DATASET_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset mini MyFireDataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default mini MyFireDataset, but " +
                             "--dataset_root was not specified.")
        dataset_test = FireDetection(cfg['Data']['root'], resize=cfg['Data']['size'],
                                     split='minitest', is_train=False, augment=None)
    elif cfg['Data']['name'] == 'myfire_trainval':
        if cfg['Data']['root'] != MY_FIRE_SMOKE_DATASET_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset MyFireDataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default MyFireDataset, but " +
                             "--dataset_root was not specified.")
        dataset_test = FireDetection(root_dir=cfg['Data']['root'], resize=cfg['Data']['size'],
                                     split='test', is_train=False, augment=None)
    else:
        raise ValueError('Dataset type not understood (must be VOC), exiting.')


set_config = set_config()

# args
args = set_config.args

# cfg
cfg = set_config.cfg
args.log_name = cfg['Models']['name'] + "_" + cfg['Data']['name'] + str(cfg['Data']['size'][0]) + "_test"

# dataset_val
dataset_test = set_config.dataset_test


if __name__ == '__main__':
    print("args:{}\nconfig:{}.".format(args, cfg))
    for data in dataset_test:
        imgs, boxes, classes = data
        print("imgs.shape:", imgs.shape)
        print("boxes.shape:", boxes.shape)
        print("classes.shape:", classes.shape)

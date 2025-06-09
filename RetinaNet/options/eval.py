import os
import yaml
import argparse
from torchvision import transforms
from models.RetinaNet import RetinaNet
from data.voc import VOC_ROOT, VocDetection
from data.augmentations import Normalizer, Resizer
from data.Fire import VISIFIRE_VOC_ROOT, FIRESENSE_VOC_ROOT, FURG_FIRE_DATASET_VOC_ROOT, BOWFIREDATASET_VOC_ROOT, \
    FIRE_SMOKE_DATASET_VOC_ROOT, MY_FIRE_SMOKE_DATASET_ROOT, FireDetection, TEST_ROOT

base_path = '/data/PycharmProject/pytorch-retinanet-master'

eval_log = os.path.join(base_path, 'log')
checkpoints = os.path.join(base_path, 'checkpoints')
Results = os.path.join(base_path, 'results')


# argparse
def parse_args():
    parser = argparse.ArgumentParser(description='Simple evaluation script for evaluating a RetinaNet network')
    parser.add_argument('--training',
                        type=str,
                        default=False,
                        help='Model is training or testing')
    parser.add_argument('--cuda',
                        type=str,
                        default=True,
                        help='Use CUDA to train model')
    parser.add_argument('--log_folder',
                        type=str,
                        default=eval_log)
    parser.add_argument('--log_name',
                        type=str,
                        default="eval")
    parser.add_argument('--results',
                        type=str,
                        default=Results,
                        help='Save the result files.')
    parser.add_argument('--checkpoints',
                        type=str,
                        default=checkpoints,
                        help='Checkpoints state_dict folder to resume training from')
    parser.add_argument('--evaluate',
                        type=str,
                        default='RetinaNet_resnet50_bowfiredataset360_baseline_best_8027.pth',
                        help='Checkpoints state_dict file to evaluating model from')
    parser.add_argument('--config',
                        type=str,
                        default='{}/configs/D_BoWFireDataset_voc360.yaml'.format(base_path),
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

        dataset_eval = VocDetection(cfg['Data']['root'], set_name=[('2007', 'test')],
                                    transform=transforms.Compose([Normalizer(), Resizer()]))
    elif cfg['Data']['name'] == 'visifire':
        if cfg['Data']['root'] != VISIFIRE_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset VISIFIRE')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default VISIFRE dataset, but " +
                             "--dataset_root was not specified.")
        dataset_eval = FireDetection(cfg['Data']['root'], transform=transforms.Compose([Normalizer(), Resizer()]),
                                     set_name='test')

    elif cfg['Data']['name'] == 'firesense':
        if cfg['Data']['root'] != FIRESENSE_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset FIRESENSE')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default FIRESENSE dataset, but " +
                             "--dataset_root was not specified.")
        dataset_eval = FireDetection(cfg['Data']['root'], transform=transforms.Compose([Normalizer(), Resizer()]),
                                     set_name='test')

    elif cfg['Data']['name'] == 'furg-fire-dataset':
        if cfg['Data']['root'] != FURG_FIRE_DATASET_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset FURG-FIRE-Dataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default FURG-FIRE-Dataset, but " +
                             "--dataset_root was not specified.")
        dataset_eval = FireDetection(cfg['Data']['root'], transform=transforms.Compose([Normalizer(), Resizer()]),
                                     set_name='test')

    elif cfg['Data']['name'] == 'bowfiredataset':
        if cfg['Data']['root'] != BOWFIREDATASET_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset BoWFireDataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default BoWFireDataset, but " +
                             "--dataset_root was not specified.")
        dataset_eval = FireDetection(cfg['Data']['root'], transform=transforms.Compose([Normalizer(), Resizer()]),
                                     set_name='test')

    elif cfg['Data']['name'] == 'fire-smoke-dataset':
        if cfg['Data']['root'] != FIRE_SMOKE_DATASET_VOC_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset Fire-Smoke-Dataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default Fire-Smoke-Dataset, but " +
                             "--dataset_root was not specified.")
        dataset_eval = FireDetection(cfg['Data']['root'], transform=transforms.Compose([Normalizer(), Resizer()]),
                                     set_name='test')
    elif cfg['Data']['name'] == 'myfire':
        if cfg['Data']['root'] != MY_FIRE_SMOKE_DATASET_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset MyFireDataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default MyFireDataset, but " +
                             "--dataset_root was not specified.")
        dataset_eval = FireDetection(cfg['Data']['root'], transform=transforms.Compose(
            [Normalizer(), Resizer(min_side=cfg['Data']['size'])]),
                                     set_name='test')
    elif cfg['Data']['name'] == 'minimyfire':
        if cfg['Data']['root'] != MY_FIRE_SMOKE_DATASET_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset mini MyFireDataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default mini MyFireDataset, but " +
                             "--dataset_root was/data/PycharmProject/Yolov8 not specified.")
        dataset_eval = FireDetection(cfg['Data']['root'],
                                     transform=transforms.Compose(
                                         [Normalizer(), Resizer(min_side=cfg['Data']['size'])]), set_name='minitest')
    elif cfg['Data']['name'] == 'myfire_trainval':
        if cfg['Data']['root'] != MY_FIRE_SMOKE_DATASET_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset MyFireDataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default MyFireDataset, but " +
                             "--dataset_root was not specified.")
        dataset_eval = FireDetection(cfg['Data']['root'],
                                     transform=transforms.Compose(
                                         [Normalizer(), Resizer(min_side=cfg['Data']['size'])]), set_name='test')
    elif cfg['Data']['name'] == 'test':
        if cfg['Data']['root'] != TEST_ROOT:
            raise ValueError('Must specify dataset_root if specifying dataset TestDataset')
        elif cfg['Data']['root'] is None:
            raise ValueError("WARNING: Using default TestDataset, but " +
                             "--dataset_root was not specified.")
        dataset_eval = FireDetection(cfg['Data']['root'],
                                     transform=transforms.Compose([Normalizer(), Resizer()]), set_name='test')

    else:
        raise ValueError('Dataset type not understood (must be VOC), exiting.')

    model_eval = RetinaNet(backbones_type=cfg['Backbones']['name'] + str(cfg['Backbones']['depth']),
                           num_classes=dataset_eval.num_classes(),
                           attention=cfg['Head']['name'],
                           pretrained=False,
                           training=args.training)


set_config = set_config()

# args
args = set_config.args

# cfg
cfg = set_config.cfg
args.log_name = cfg['Models']['name'] + "_" + cfg['Data']['name'] + str(cfg['Data']['size']) + "_eval"

# dataset_val
dataset_eval = set_config.dataset_eval

retinanet_eval = set_config.model_eval

if __name__ == '__main__':
    print("args:{}\nconfig:{}.".format(args, cfg))
    for data in dataset_eval:
        imgs, targets = data['img'], data['annot']
        print("imgs.shape:", imgs.shape)
        print("targets.shape:", targets.shape)
        print("targets:", targets)

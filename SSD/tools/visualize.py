import os
import cv2
import time
import torch
import random
import logging
from utils.get_logger import get_logger
from data.augmentations import BaseTransform
from data.Fire import VOC_CLASSES as Firelabelmap
from data.voc0712 import VOC_CLASSES as VOClabelmap
from options.test import args, cfg, model_test, dataset_test, MEANS

assert torch.__version__.split('.')[0] == '1'
print('SSD visualize.py CUDA available: {}'.format(torch.cuda.is_available()))

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if cfg['Data']['name'] != 'VOC':
    labelmap = Firelabelmap
else:
    labelmap = VOClabelmap

if not os.path.exists(args.checkpoints):
    os.mkdir(args.checkpoints)

if not os.path.exists(args.results):
    os.mkdir(args.results)

get_logger(args.log_folder, args.log_name)
logger = logging.getLogger(args.log_name)

COLORS = ((255, 255, 0), (201, 207, 142), (0, 0, 255), (122, 190, 255))


def visualized(dataset, model, transform):
    # shuffle
    random.shuffle(dataset.ids)
    img_id = dataset.ids[1]
    img = dataset.pull_image(1)
    height, width, _ = img.shape

    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])
    annots = dataset.pull_item(1)[1]
    for annot in annots:
        xmin = int(annot[0] * img.shape[1])
        ymin = int(annot[1] * img.shape[0])
        xmax = int(annot[2] * img.shape[1])
        ymax = int(annot[3] * img.shape[0])
        # label = int(annot[4])
        cv2.rectangle(img, (xmin, ymin), (xmax, ymax), COLORS[0], 1)
        # font = cv2.FONT_HERSHEY_TRIPLEX
        # cv2.putText(img, labelmap[label], (xmin, int(ymin - 5)), font, 1, COLORS[1], 1)
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1).unsqueeze(dim=0)
    if args.cuda:
        x = x.cuda()
    output = model(x)
    detections = output.data
    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= args.visual_threshold:
            # score = detections[0, i, j, 0].cpu().numpy()
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
            # coords = (pt[0], pt[1], pt[2], pt[3])
            # text = str(i - 1) + ' | ' + str(np.round(score, 2))
            cv2.rectangle(img, (int(pt[0]), int(pt[1])), (int(pt[2]), int(pt[3])), COLORS[2], 1)
            # font = cv2.FONT_HERSHEY_TRIPLEX
            # cv2.putText(img, labelmap[i - 1], (int(pt[2] - 30), int(pt[3] - 15)), font, 1, COLORS[3], 1)
            j += 1
    filename = os.path.join(args.results, cfg['Data']['name'] + '_' + str(img_id[1]) + '.jpg')
    cv2.imwrite(filename, img)



if __name__ == '__main__':
    logger.info("Visualization Program started")
    if args.cuda and torch.cuda.is_available():
        model_test = model_test.cuda()
    if args.evaluate:
        other, ext = os.path.splitext(args.evaluate)
        if ext == '.pkl' or '.pth':
            model_load = os.path.join(args.checkpoints, args.evaluate)
            model_test.load_state_dict(torch.load(model_load))
            print("Loading weights into state dict...")
        else:
            print("Sorry only .pth and .pkl files supported.")
    elif args.evaluate is None:
        print("Sorry, you should load weights!")
    if args.cuda and torch.cuda.is_available():
        model_test = torch.nn.DataParallel(model_test).cuda()
    else:
        model_test = torch.nn.DataParallel(model_test)
    model_test.eval()

    with torch.no_grad():
        t_start = time.time()
        visualized(dataset_test, model_test, BaseTransform(size=cfg['Data']['size'], mean=MEANS))
        t_end = time.time()
        print("FPS:{}".format(str(int(1 / (t_end - t_start)))))

    logger.info("Done!")

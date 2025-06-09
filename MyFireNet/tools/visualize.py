import os
import cv2
import time
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from data.voc import VocDetection
from torchvision import transforms
from data.Fire import FireDetection
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
from models.MyFireNet import FCOSDetector
from options.test import args, cfg, dataset_test


def preprocess_img(image, input_ksize):
    '''
    resize image and bboxes
    Returns
    image_paded: input_ksize
    bboxes: [None,4]
    '''
    min_side, max_side = input_ksize
    h, w, _ = image.shape

    smallest_side = min(w, h)
    largest_side = max(w, h)
    scale = min_side / smallest_side
    if largest_side * scale > max_side:
        scale = max_side / largest_side
    nw, nh = int(scale * w), int(scale * h)
    image_resized = cv2.resize(image, (nw, nh))

    pad_w = 32 - nw % 32
    pad_h = 32 - nh % 32

    image_paded = np.zeros(shape=[nh + pad_h, nw + pad_w, 3], dtype=np.uint8)
    image_paded[:nh, :nw, :] = image_resized
    return image_paded


if __name__ == "__main__":
    cmap = plt.get_cmap('tab20b')
    colors = [cmap(i) for i in np.linspace(0, 1, 20)]

    model = FCOSDetector(mode="inference", cfg=cfg)
    if args.cuda and torch.cuda.is_available():
        model = model.cuda()

    model_load = os.path.join(args.checkpoints, args.evaluate)

    model.load_state_dict(torch.load(model_load))

    if args.cuda and torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)

    model = model.eval()
    print("Loading model...")

    num = random.randint(0, len(dataset_test))

    info = dataset_test.img_ids[num]

    if cfg['Data']['name'] == 'VOC':
        img_path = os.path.join(cfg['Data']['root'], 'JPEGImages', info + ".jpg")
        detection = VocDetection
    else:
        img_path = os.path.join(cfg['Data']['root'], 'images', info + ".jpg")
        detection = FireDetection

    img_bgr = cv2.imread(img_path)
    img_pad = preprocess_img(img_bgr, cfg['Data']['size'])
    img = cv2.cvtColor(img_pad.copy(), cv2.COLOR_BGR2RGB)
    img1 = transforms.ToTensor()(img)
    img1 = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225], inplace=True)(img1)
    img1 = img1

    start_t = time.time()
    with torch.no_grad():
        out = model(img1.unsqueeze_(dim=0))
    end_t = time.time()
    cost_t = 1000 * (end_t - start_t)
    print("===>success processing img, cost time %.2f ms" % cost_t)
    # print(out)
    scores, classes, boxes = out

    boxes = boxes[0].cpu().numpy().tolist()
    classes = classes[0].cpu().numpy().tolist()
    scores = scores[0].cpu().numpy().tolist()
    plt.figure()
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    for i, box in enumerate(boxes):
        pt1 = (int(box[0]), int(box[1]))
        pt2 = (int(box[2]), int(box[3]))
        img_pad = cv2.rectangle(img_pad, pt1, pt2, (0, 255, 0))
        b_color = colors[int(classes[i]) - 1]
        bbox = patches.Rectangle((box[0], box[1]), width=box[2] - box[0], height=box[3] - box[1], linewidth=1,
                                 facecolor='none', edgecolor=b_color)
        ax.add_patch(bbox)
        plt.text(box[0], box[1], s="%s %.3f" % (detection.CLASSES_NAME[int(classes[i])], scores[i]), color='white',
                 verticalalignment='top',
                 bbox={'color': b_color, 'pad': 0})
    plt.axis('off')
    plt.gca().xaxis.set_major_locator(NullLocator())
    plt.gca().yaxis.set_major_locator(NullLocator())
    filename = os.path.join(args.results, cfg['Data']['name'] + "_" + info + ".jpg")
    plt.savefig('{}'.format(filename), bbox_inches='tight', pad_inches=0.0)
    plt.close()

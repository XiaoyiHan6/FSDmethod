import os
import time
import torch
import logging
import numpy as np
from torch import nn
from tools.eval_voc import eval_voc
from utils.get_logger import get_logger
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler
from data.augmentations import BaseTransform
from options.train import args, cfg, model, model_eval, dataloader_train, dataset_train, dataset_val, iter_size, MEANS

assert torch.__version__.split('.')[0] == '1'
print('SSD train.py CUDA available: {}'.format(torch.cuda.is_available()))

# Log
get_logger(args.log_folder, args.log_name)
logger = logging.getLogger(args.log_name)

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.checkpoints):
    os.mkdir(args.checkpoints)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.xavier_uniform_(m.weight.data)
        m.bias.data.zero_()


if __name__ == "__main__":
    logger.info("SSD training started!")
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        # tensorboard
        writer = SummaryWriter(args.tensorboard_log)

    if args.cuda and torch.cuda.is_available():
        model = model.cuda()
    if args.resume:
        other, ext = os.path.splitext(args.resume)
        if ext == '.pkl' or '.pth':
            model_load = os.path.join(args.checkpoints, args.resume)
            model.load_state_dict(torch.load(model_load))
            print("Loading weights into state dict...")
    elif args.pretrained:
        print('Loading pretrained network...')
    else:
        print('Initializing weights...')
        model.apply(weights_init)
    if args.cuda and torch.cuda.is_available():
        model = torch.nn.DataParallel(model).cuda()
    else:
        model = torch.nn.DataParallel(model)
    model.training = True
    model.train()
    optimizer = get_optimizer(cfg=cfg, model=model)
    scheduler = get_scheduler(cfg=cfg, optimizer=optimizer)

    loc_loss = 0
    conf_loss = 0

    logger.info(f'Loading the length of train dataset:{len(dataset_train)}, iter_size:{iter_size}.')
    logger.info(f"args-{args}")

    iter = 0
    index = 0
    best_map = 0.0
    t_start = time.time()
    for epoch_num in range(cfg['Optimize']['epoch']):
        t_epoch_start = time.time()
        epoch_loss = []
        for data in dataloader_train:
            try:
                iter += 1
                optimizer.zero_grad()
                imgs, annots = data

                if args.cuda and torch.cuda.is_available():
                    imgs = imgs.cuda()
                    annots = [ann.cuda() for ann in annots]
                else:
                    annots = [ann for ann in annots]
                loss_l, loss_c = model([imgs, annots])
                loss = loss_l + loss_c
                epoch_loss.append(float(loss))
                loss.backward()
                optimizer.step()
                if args.tensorboard:
                    writer.add_scalar('loss', loss.item(), iter)
                if iter % cfg['Optimize']['display_iter'] == 0:
                    logger.info(f"Epoch:{str(epoch_num + 1)} | Iter:{iter} | "
                                f"lr:{optimizer.param_groups[0]['lr']} | "
                                f"Classification loss:{float(loss_c.item()):.3f} | "
                                f"Regression loss:{float(loss_l.item()):.3f} | "
                                f"Loss:{loss.item():.3f}.")
                del loss_l
                del loss_c
            except Exception as e:
                print(e)
                continue
        # scheduler.step(np.mean(epoch_loss))
        scheduler.step()
        t_epoch_end = time.time()
        h_epoch = (t_epoch_end - t_epoch_start) // 3600
        m_epoch = ((t_epoch_end - t_epoch_start) % 3600) // 60
        s_epoch = ((t_epoch_end - t_epoch_start) % 3600) % 60

        logger.info(
            f"epoch {str(epoch_num + 1)} is finished, and the time is {str(int(h_epoch))}.{str(int(m_epoch))}.{str(int(s_epoch))}")

        torch.save(model.module.state_dict(),
                   "{}/ssd{}_{}{}_{}_{}.pth"
                   .format(args.checkpoints, str(cfg['Data']['size']), cfg['Backbones']['name'].lower(),
                           str(cfg['Backbones']['depth']), cfg['Data']['name'].lower(), cfg['Head']['name']))
        if ((epoch_num + 1) > cfg['Optimize']['display_eval'] and (epoch_num + 1) % 5 == 0) or epoch_num > \
                cfg['Optimize']['epoch'] - 5:
            print("Evaluating...")
            model_eval.load_state_dict(
                torch.load('{}/ssd{}_{}{}_{}_{}.pth'.format(args.checkpoints, str(cfg['Data']['size']),
                                                            cfg['Backbones']['name'].lower(),
                                                            str(cfg['Backbones']['depth']),
                                                            cfg['Data']['name'].lower(), cfg['Head']['name'])))
            model_eval.eval()
            if args.cuda and torch.cuda.is_available():
                model_eval = model_eval.cuda()
            with torch.no_grad():
                t_eval_start = time.time()
                aps = eval_voc(dataset_val, model_eval, BaseTransform(cfg['Data']['size'], MEANS), args.cuda)
                maps = np.mean(aps)
            logger.info(f"Fire mAP={maps:.3f}")
            if maps > best_map:
                logger.info(f"Saving best mAP={maps:.3f} state, epoch:{str(epoch_num + 1)} | iter:{iter}")
                torch.save(model.module.state_dict(),
                           "{}/ssd{}_{}{}_{}_{}_best.pth".format(args.checkpoints, str(cfg['Data']['size']),
                                                                 cfg['Backbones']['name'].lower(),
                                                                 str(cfg['Backbones']['depth']),
                                                                 cfg['Data']['name'].lower(), cfg['Head']['name']))
                best_map = maps
            t_eval_end = time.time()
            h_eval = (t_eval_end - t_eval_start) // 3600
            m_eval = ((t_eval_end - t_eval_start) % 3600) // 60
            s_eval = ((t_eval_end - t_eval_start) % 3600) % 60

            logger.info(
                f"Evaluation is finished, and the time is {str(int(h_eval))}.{str(int(m_eval))}.{str(int(s_eval))}")
    if args.tensorboard:
        writer.close()
    t_end = time.time()
    h = (t_end - t_start) // 3600
    m = ((t_end - t_start) % 3600) // 60
    s = ((t_end - t_start) % 3600) % 60
    logger.info(f"The Program Finished Time is {str(int(h))}.{str(int(m))}.{str(int(s))}")
    logger.info("Done!")

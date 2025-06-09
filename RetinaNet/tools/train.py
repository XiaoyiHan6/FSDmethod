import os
import time
import torch
import logging
import numpy as np
import collections
from tools.eval_voc import eval_voc
from utils.get_logger import get_logger
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler
from options.train import args, cfg, dataset_train, dataloader_train, dataset_val, iter_size, retinanet, \
    retinanet_eval

# assert torch.__version__.split('.')[0] == '1'
print('CUDA available: {}'.format(torch.cuda.is_available()))

# log
get_logger(args.log_folder, args.log_name)
logger = logging.getLogger(args.log_name)

if __name__ == '__main__':
    logger.info("RetinaNet training started!")

    # create SummaryWriter
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        # tensorboard loss
        writer = SummaryWriter(args.tensorboard_log)

    if args.cuda and torch.cuda.is_available():
        retinanet = retinanet.cuda()
    if args.resume:
        other, ext = os.path.splitext(args.resume)
        if ext == '.pkl' or '.pth':
            model_load = os.path.join(args.checkpoints, args.resume)
            retinanet.load_state_dict(torch.load(model_load))
            print("Loading weights into state dict...")
        elif args.pretrained:
            print('Loading pretrained network...')
        else:
            print("Sorry only .pth and .pkl files supported.")
    if args.cuda and torch.cuda.is_available():
        retinanet = torch.nn.DataParallel(retinanet).cuda()
    else:
        retinanet = torch.nn.DataParallel(retinanet)


    retinanet.training = True

    optimizer = get_optimizer(cfg=cfg, model=retinanet)
    scheduler = get_scheduler(cfg=cfg, optimizer=optimizer)

    loss_hist = collections.deque(maxlen=500)

    logger.info(f'Loading the length of train dataset:{len(dataset_train)}, iter_size:{iter_size}.')
    logger.info(f"args - {args}")

    iter = 0
    index = 0
    best_map = 0.0
    t_start = time.time()

    for epoch_num in range(cfg['Optimize']['epoch']):
        t_epoch_start = time.time()
        retinanet.train()
        retinanet.module.freeze_bn()
        epoch_loss = []
        for data in dataloader_train:
            # if iter in cfg['Optimize']['lr_step']:
            #     index += 1
            #     scheduler.forward(optimizer, index)

            try:
                iter += 1
                optimizer.zero_grad()

                if args.cuda and torch.cuda.is_available():
                    classification_loss, regression_loss = retinanet([data['img'].cuda().float(), data['annot']])
                else:
                    classification_loss, regression_loss = retinanet([data['img'].float(), data['annot']])

                classification_loss = classification_loss.mean()
                regression_loss = regression_loss.mean()

                loss = classification_loss + regression_loss

                if bool(loss == 0):
                    continue

                loss.backward()

                torch.nn.utils.clip_grad_norm_(retinanet.parameters(), max_norm=0.1)

                optimizer.step()

                loss_hist.append(float(loss))

                epoch_loss.append(float(loss))

                if args.tensorboard:
                    writer.add_scalar("total_loss", np.mean(epoch_loss), iter)

                if iter % cfg['Optimize']['display_iter'] == 0:
                    logger.info(
                        f"Epoch: {str(epoch_num + 1)} | Iteration: {iter} | Lr: {optimizer.param_groups[0]['lr']} | "
                        f"Classification loss: {float(classification_loss):.3f} | Regression loss: {float(regression_loss):.3f} | Current loss: {float(loss):.3f} | Running loss: {np.mean(loss_hist):.3f}")

                del classification_loss
                del regression_loss
            except Exception as e:
                print(e)
                continue
        # ReduceLROnPlateau
        # scheduler.step(np.mean(epoch_loss))
        scheduler.step()
        t_epoch_end = time.time()

        h_epoch = (t_epoch_end - t_epoch_start) // 3600
        m_epoch = ((t_epoch_end - t_epoch_start) % 3600) // 60
        s_epoch = ((t_epoch_end - t_epoch_start) % 3600) % 60

        logger.info(
            f"epoch {str(epoch_num + 1)} is finished, and the time is {str(int(h_epoch))}.{str(int(m_epoch))}.{str(int(s_epoch))}")

        torch.save(retinanet.module.state_dict(),
                   '{}/RetinaNet_{}{}_{}{}_{}.pth'.format(args.checkpoints, cfg['Backbones']['name'].lower(),
                                                          str(cfg['Backbones']['depth']),
                                                          cfg['Data']['name'].lower(), str(cfg['Data']['size']),
                                                          cfg['Head']['name']))

        if ((epoch_num + 1) > cfg['Optimize']['display_eval'] and (epoch_num + 1) % 5 == 0) or epoch_num > \
                cfg['Optimize']['epoch'] - 5:
            print('Evaluating dataset')
            retinanet_eval.load_state_dict(
                torch.load('{}/RetinaNet_{}{}_{}{}_{}.pth'.format(args.checkpoints, cfg['Backbones']['name'].lower(),
                                                                  str(cfg['Backbones']['depth']),
                                                                  cfg['Data']['name'].lower(),
                                                                  str(cfg['Data']['size']), cfg['Head']['name'])))
            retinanet_eval.eval()
            if args.cuda and torch.cuda.is_available():
                retinanet_eval = retinanet_eval.cuda()
            with torch.no_grad():
                t_eval_start = time.time()
                aps = eval_voc(dataset_val, retinanet_eval)
            maps = np.mean(aps)
            logger.info(f"Fire mAP={maps:.3f}")
            if maps > best_map:
                logger.info(f"Saving best mAP={maps:.3f} state, epoch: {str(epoch_num + 1)} | iter: {iter}")
                torch.save(retinanet.module.state_dict(),
                           '{}/RetinaNet_{}{}_{}{}_{}_best.pth'.format(args.checkpoints,
                                                                       cfg['Backbones']['name'].lower(),
                                                                       str(cfg['Backbones']['depth']),
                                                                       cfg['Data']['name'].lower(),
                                                                       str(cfg['Data']['size']), cfg['Head']['name']))
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

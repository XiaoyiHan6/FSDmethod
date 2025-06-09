import os
import time
import torch
import logging
from tools.eval_voc import eval_voc
from utils.get_logger import get_logger
from utils.optimizer import get_optimizer
from utils.scheduler import get_scheduler
from models.MyFireNet import FCOSDetector
from options.train import args, cfg, dataset_train, dataloader_train, dataloader_val, dataset_val, iter_size

# assert torch.__version__.split('.')[0] == '1'
print('MyFireNet train.py CUDA available: {}'.format(torch.cuda.is_available()))


# log
get_logger(args.log_folder, args.log_name)
logger = logging.getLogger(args.log_name)

if __name__ == '__main__':
    logger.info("MyFireNet training started!")
    myfirenet = FCOSDetector(mode="training", cfg=cfg)
    myfirenet_eval = FCOSDetector(mode="inference", cfg=cfg)
    # create SummaryWriter
    if args.tensorboard:
        from torch.utils.tensorboard import SummaryWriter

        # tensorboard loss
        writer = SummaryWriter(args.tensorboard_log)

    if args.cuda and torch.cuda.is_available():
        myfirenet = myfirenet.cuda()

    if args.resume:
        other, ext = os.path.splitext(args.resume)
        if ext == '.pkl' or '.pth':
            model_load = os.path.join(args.checkpoints, args.resume)
            myfirenet.load_state_dict(torch.load(model_load))
            print("Loading weights into state dict...")
        elif args.pretrained:
            print('Loading pretrained network...')
        else:
            print("Sorry only .pth and .pkl files supported.")
    if args.cuda and torch.cuda.is_available():
        myfirenet = torch.nn.DataParallel(myfirenet).cuda()
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.enabled = True
    else:
        myfirenet = torch.nn.DataParallel(myfirenet)

    myfirenet.training = True

    optimizer = get_optimizer(cfg=cfg, model=myfirenet)
    scheduler = get_scheduler(cfg=cfg, optimizer=optimizer)

    logger.info(f'Loading the length of train dataset:{len(dataset_train)}, iter_size:{iter_size}.')
    logger.info(f"args - {args}")

    iter = 0
    index = 0
    best_map = 0.0
    t_start = time.time()

    for epoch_num in range(cfg['Optimize']['epoch']):
        t_epoch_start = time.time()
        myfirenet.train()

        for data in dataloader_train:
            batch_imgs, batch_boxes, batch_classes = data
            try:
                iter += 1
                optimizer.zero_grad()

                if args.cuda and torch.cuda.is_available():
                    losses = myfirenet(
                        [batch_imgs.cuda(), batch_boxes.cuda(), batch_classes.cuda()])
                else:
                    losses = myfirenet(
                        [batch_imgs, batch_boxes, batch_classes])

                loss = losses[-1]

                # if bool(loss == 0):
                if (loss == 0).any():
                    continue
                loss.mean().backward()

                # torch.nn.utils.clip_grad_norm_(myfirenet.parameters(), 0.1)

                optimizer.step()

                if args.tensorboard:
                    writer.add_scalar("total_loss", loss.mean(), iter)

                if iter % cfg['Optimize']['display_iter'] == 0:
                    logger.info(
                        f"Epoch: {str(epoch_num + 1)} | Iteration: {iter} | Lr: {optimizer.param_groups[0]['lr']} | "
                        f"Class loss: {float(losses[0].mean()):.3f} | Cnt loss: {float(losses[1].mean()):.3f} | "
                        f"Reg loss: {float(losses[2].mean()):.3f} | Current loss: {float(loss.mean()):.3f} | Running loss: {losses[-1].mean():.3f}")

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

        torch.save(myfirenet.module.state_dict(),
                   '{}/{}_{}{}_{}{}_{}.pth'.format(args.checkpoints, cfg['Models']['name'],
                                                cfg['Backbones']['name'].lower(),
                                                str(cfg['Backbones']['depth']),
                                                cfg['Data']['name'].lower(), str(cfg['Data']['size'][0]),
                                                cfg['Head']['name']))

        if ((epoch_num + 1) > cfg['Optimize']['display_eval'] and (epoch_num + 1) % 5 == 0) or epoch_num > \
                cfg['Optimize']['epoch'] - 5:
            print('Evaluating dataset')

            myfirenet_eval.load_state_dict(
                torch.load('{}/{}_{}{}_{}{}_{}.pth'.format(args.checkpoints, cfg['Models']['name'],
                                                           cfg['Backbones']['name'].lower(),
                                                           str(cfg['Backbones']['depth']),
                                                           cfg['Data']['name'].lower(), str(cfg['Data']['size'][0]),
                                                           cfg['Head']['name'])))

            myfirenet_eval.eval()
            if args.cuda and torch.cuda.is_available():
                myfirenet_eval = myfirenet_eval.cuda()

            t_eval_start = time.time()
            with torch.no_grad():
                all_AP = eval_voc(dataloader_val, myfirenet_eval)
            for key, value in all_AP.items():
                logger.info(f"AP for {dataset_val.id2name[int(key)]} is {float(value):.4f}")
            mAP = 0.
            for class_id, class_mAP in all_AP.items():
                mAP += float(class_mAP)
            mAP /= (len(dataset_val.CLASSES_NAME) - 1)
            logger.info(f"Fire mAP={mAP:.3f}")
            if mAP > best_map:
                logger.info(f"Saving best mAP={mAP:.3f} state, epoch: {str(epoch_num + 1)} | iter: {iter}")
                torch.save(myfirenet.module.state_dict(),
                           '{}/{}_{}{}_{}{}_{}_best.pth'.format(args.checkpoints, cfg['Models']['name'],
                                                                cfg['Backbones']['name'].lower(),
                                                                str(cfg['Backbones']['depth']),
                                                                cfg['Data']['name'].lower(),
                                                                str(cfg['Data']['size'][0]), cfg['Head']['name']))
                best_map = mAP
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

import os.path
import warnings
import torch
import torch.nn as nn

import time
import networks
import logging
import numpy as np
from tensorboardX import SummaryWriter

from utils import get_config,set_env,set_logger,set_outdir,get_dataloader
from utils import get_train_setting,load_checkpoint,get_proc,save_checkpoint

from coding_functions.target_criterion_2d import HadamardTargetCoding, LearnableTargetCoding

def main(conf):
    # logger
    # tf_writer = SummaryWriter(log_dir=os.path.join('log', conf['outdir']))
    tf_writer = SummaryWriter(log_dir=conf['outdir'])

    warnings.filterwarnings("ignore")
    best_score = 0.
    epoch_start = 0

    # dataloader
    train_loader, val_loader, ds_train = get_dataloader(conf)

    # device setting
    device = (torch.device('cuda')
              if torch.cuda.is_available()
              else torch.device('cpu'))
    print("Cuda is available!")

    # model
    model = networks.get_model(conf)
    model = nn.DataParallel(model).cuda()
    if conf.weightfile is not None:
        wmodel = networks.get_model(conf)
        wmodel = nn.DataParallel(wmodel).cuda()
        checkpoint_dict = load_checkpoint(wmodel, conf.weightfile)

        if 'best_score' in checkpoint_dict:
            print('best score: {}'.format(best_score))
    else:
        wmodel = model

    # training setting
    criterion, optimizer, scheduler = get_train_setting(model,conf)
    criterion_test = criterion

    if conf.HTC is True:
        train_reg = HadamardTargetCoding(gamma_=conf.gamma_,
                                            code_length=conf.code_length,
                                            classes_num=conf.num_class,
                                  ).cuda()
    elif conf.LTC is True:
        train_reg = LearnableTargetCoding(gamma_=conf.gamma_,
                                            lambda_=conf.lambda_,
                                            beta_=conf.beta_,
                                            code_length=conf.code_length,
                                            classes_num=conf.num_class,
                                            active_type=conf.active_type,
                                            margin_ratio=conf.margin_ratio
                                  ).cuda()
        optimizer.add_param_group({'params': train_reg.target_labels, 'lr': 0.1})
    else:
        train_reg = None

    # training and evaluate process for each epoch
    train, validate = get_proc(conf)

    if conf.resume:
        checkpoint_dict = load_checkpoint(model, conf.resume)
        epoch_start = checkpoint_dict['epoch']

        if 'best_score' in checkpoint_dict:
            best_score = checkpoint_dict['best_score']
            print('best score: {}'.format(best_score))
        print('Resuming training process from epoch {}...'.format(epoch_start))
        optimizer.load_state_dict(checkpoint_dict['optimizer'])
        scheduler.load_state_dict(checkpoint_dict['scheduler'])
        print('Resuming lr scheduler')
        print(checkpoint_dict['scheduler'])
        if conf.HTC or conf.LTC:
            print('Resuming target_labels')
            train_reg.target_labels.data = checkpoint_dict['target_codes']

    if conf.evaluate:
        print(validate(val_loader, model, criterion_test, conf))
        return

    detach_epoch = conf.epochs + 1
    if 'detach_epoch' in conf:
        detach_epoch = conf.detach_epoch

    start_eval = 0
    if 'start_eval' in conf:
        start_eval = conf.start_eval

    ## ------main loop-----
    for epoch in range(epoch_start, conf.epochs):
        # setting just for imbalanced data learning
        if conf.dataset == 'iNaturalist18' or conf.dataset == 'Imagenet-LT':
            cls_num_list = ds_train.get_cls_num_list()
            if conf.train_rule == 'None':
                per_cls_weights = None
            elif conf.train_rule == 'DRW':
                if conf.dataset == 'iNaturalist18':
                    idx = epoch // 160 # for total 200 epochs
                else:
                    idx = epoch // 80 # for total 100 epochs
                betas = [0, 0.9999]
                effective_num = 1.0 - np.power(betas[idx], cls_num_list)
                per_cls_weights = (1.0 - betas[idx]) / np.array(effective_num)
                per_cls_weights = per_cls_weights / np.sum(per_cls_weights) * len(cls_num_list)
                per_cls_weights = torch.FloatTensor(per_cls_weights).cuda()
            else:
                warnings.warn('Sample rule is not listed')
            criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda()

        start_time = time.time()
        lr0 = optimizer.param_groups[0]['lr']
        lr1 = optimizer.param_groups[1]['lr']
        lr2 = optimizer.param_groups[-1]['lr']

        logging.info("Epoch: [{} | {} LR: {} {} {}".format(epoch+1,conf.epochs,lr0, lr1, lr2))

        if epoch == detach_epoch:
            model.module.set_detach(False)

        tmp_loss = train(train_loader, model, criterion, optimizer, train_reg, conf, wmodel)
        infostr = {'Epoch:  {}   train_loss: {}'.format(epoch+1,tmp_loss)}
        logging.info(infostr)
        scheduler.step()

        if epoch > start_eval and (epoch+1) % 1 == 0:
            with torch.no_grad():
                val_score, val_score_top5,val_loss,mscore, ascore = validate(val_loader, model, criterion_test, conf)
                comscore = val_score
                if 'midlevel' in conf:
                    if conf.midlevel:
                        comscore = ascore

                is_best = comscore > best_score
                best_score = max(comscore, best_score)
                print('Epoch:  {:.4f}   loss: {:.4f},gs: {:.4f},gs_acc5: {:.4f},ms:{:.4f},as:{:.4f},bs:{:.4f}'.format(
                    epoch+1,val_loss,val_score,val_score_top5,mscore,ascore,best_score))
                infostr = {'Epoch:  {:.4f}   loss: {:.4f},gs: {:.4f},gs_acc5: {:.4f},ms:{:.4f},as:{:.4f},bs:{:.4f}'.format(
                    epoch+1,val_loss,val_score,val_score_top5,mscore,ascore,best_score)}
                logging.info(infostr)

                tf_writer.add_scalar('acc/test_top1', comscore, epoch)
                tf_writer.add_scalar('acc/test_top1_best', best_score, epoch)
                tf_writer.add_scalar('acc/test_top5', val_score_top5, epoch)

                state_dict = {'epoch': epoch + 1,
                        'state_dict': model.module.state_dict(),
                        'optimizer' : optimizer.state_dict(),
                        'scheduler' : scheduler.state_dict(),
                        'best_score': best_score
                        }
                if conf.HTC or conf.LTC:
                    state_dict['target_codes'] = train_reg.target_labels.data
                save_checkpoint(state_dict, is_best, outdir=conf['outdir'], iteral=conf.iteral)

        end_time = time.time()
        seconds = end_time - start_time
        m, s = divmod(seconds, 60)
        h, m = divmod(m, 60)
        infostr = {"Epoch Time %02d:%02d:%02d" % (h, m, s)}
        logging.info(infostr)
    logging.info({'Best val acc: {}'.format(best_score)})
    print('Best val acc: {}'.format(best_score))
    # return 0

if __name__ == '__main__':
    start_time = time.time()
    # get configs and set envs
    conf = get_config()
    set_env(conf)
    # generate outdir name
    set_outdir(conf)
    # Set the logger
    set_logger(conf)
    main(conf)

    end_time = time.time()
    seconds = end_time - start_time
    m, s = divmod(seconds, 60)
    h, m = divmod(m, 60)
    print("During Time %02d:%02d:%02d" % (h, m, s))




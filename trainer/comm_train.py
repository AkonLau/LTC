import os
import sys
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import cv2
import time
import logging
from utils import *
from colorama import Fore

def train(train_loader, model, criterion, optimizer, train_reg, conf, wmodel=None):

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    reg_losses = AverageMeter('Loss', ':.4e')

    end = time.time()
    model.train()

    pbar = tqdm(train_loader, dynamic_ncols=True, total=len(train_loader),
                ascii=True, bar_format='{l_bar}%s{bar}%s{r_bar}' % (Fore.BLUE, Fore.RESET))
    mixmethod = None

    if 'mixmethod' in conf:
        if 'baseline' not in conf.mixmethod:
            mixmethod = conf.mixmethod
            if wmodel is None:
                wmodel = model

    for idx, (input, target) in enumerate(pbar):
        # measure data loading time
        data_time.update(time.time() - end)
        input = input.cuda()
        target = target.cuda()

        if 'baseline' not in conf.mixmethod:
            input,target_a,target_b,lam_a,lam_b = eval(mixmethod)(input,target,conf,wmodel)
            output,[conv5, conv4_1],moutput,[xf, pool4_1],xlocal_attr = model(input)

            loss_a = criterion(output, target_a)
            loss_b = criterion(output, target_b)
            loss = torch.mean(loss_a* lam_a + loss_b* lam_b)

            if 'midlevel' in conf:
                if conf.midlevel:
                    loss_ma = criterion(moutput, target_a)
                    loss_mb = criterion(moutput, target_b)
                    loss += torch.mean(loss_ma* lam_a + loss_mb* lam_b)

        else:
            output,[conv5, conv4_1],moutput,[xf, pool4_1],xlocal_attr = model(input)
            loss = torch.mean(criterion(output, target))

            if 'midlevel' in conf and conf.midlevel is True:
                loss += torch.mean(criterion(moutput, target))

            # target coding regularization loss
            if conf.HTC is True or conf.LTC is True:
                reg_loss = train_reg(xlocal_attr, target)
                loss += reg_loss
            else:
                reg_loss = torch.tensor([0])

            reg_losses.update(reg_loss.item(), input.size(0))

        # measure accuracy and record loss
        losses.update(loss.item(), input.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        pbar.set_postfix(batch_time=batch_time.avg, data_time=data_time.avg, loss=losses.avg, reg_loss=reg_losses.avg)

    return losses.avg
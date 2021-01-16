#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: trainin_loop_lie.py
# --- Creation Date: 09-01-2021
# --- Last Modified: Sun 10 Jan 2021 03:05:49 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Lie Group VAE Training Loop.
"""

from trainin_loop import parse_val_logs, to_cuda
from logger.utils import mean_log_list, save_model
from tqdm import tqdm
import torch


def train_lie(args, epochs, trainloader, valloader, model, optimiser_ls, loss_fn, logger=None, metric_list=None, cuda=True):
    pb = tqdm(total=epochs, unit_scale=True, smoothing=0.1, ncols=70)
    update_frac = 1./float(len(trainloader) + len(valloader))
    global_step = 0 if not hasattr(args, 'global_step') or args.global_step is None else args.global_step
    loss, val_loss = torch.tensor(0), torch.tensor(0)
    loss_ls = [loss, loss, loss]
    val_loss_ls = [val_loss, val_loss, val_loss]
    mean_logs = {}
    torch.autograd.set_detect_anomaly(True)

    for i in range(epochs):
        for t, data in enumerate(trainloader):
            # for optimiser in optimiser_ls:
                # optimiser.zero_grad()

            for j, optimiser in enumerate(optimiser_ls):
                optimiser.zero_grad()
                model.train()
                # print('model.act_params:', model.act_params)
                data = to_cuda(data) if cuda else data
                out = model.train_step(data, t, loss_fn)
                loss_ls = out['loss_ls']
                loss_ls[j].backward()
                optimiser.step()

            # for j, optimiser in enumerate(optimiser_ls):
                # loss_ls[j].backward()
                # optimiser.step()
                # optimiser.zero_grad()
            pb.update(update_frac)
            pgs = [pg['lr'] for pg in optimiser.param_groups]
            pb.set_postfix_str('ver:{}, loss:{:.3f}, val_loss:{:.3f}, lr:{}'.format(logger.get_version(), loss_ls[0].item(), val_loss_ls[0].item(), pgs))
            global_step += 1

        log_list = []
        with torch.no_grad():
            for t, data in enumerate(valloader):
                model.eval()
                to_cuda(data) if cuda else None
                out = model.val_step(data, t, loss_fn)
                logs = out['out']
                val_loss_ls = out['loss_ls']
                log_list.append(parse_val_logs(t, args, model, data, logger, metric_list, logs, out['state'], global_step))
                pb.update(update_frac)
                pb.set_postfix_str(
                    'ver:{}, loss:{:.3f}, val_loss:{:.3f}'.format(logger.get_version(), loss_ls[0].item(), val_loss_ls[0].item()))
                global_step += 1

        mean_logs = mean_log_list(log_list)
        logger.write_dict(mean_logs, global_step) if logger is not None else None
        save_model(logger, model, args)
    return mean_logs

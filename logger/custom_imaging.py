#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: custom_imaging.py
# --- Creation Date: 31-12-2020
# --- Last Modified: Fri 15 Jan 2021 22:31:25 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Imaging.
"""
from torchvision.utils import save_image, make_grid
import torch
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image, ImageDraw
import io
import numpy as np
from torchvision.transforms import ToTensor
import collections
import warnings
import os
from logger.imaging import Imager


class ShowReconX(Imager):
    def __init__(self,
                 logger,
                 n_per_row=8,
                 to_tb=True,
                 name_1='x1',
                 name_2='x_g_rec'):
        self.logger = logger
        self.n_per_row = n_per_row
        self.to_tb = to_tb
        self.name_1 = name_1
        self.name_2 = name_2

    def __call__(self, model, state, global_step=0):
        x, x_rec = state[self.name_1], state[self.name_2]
        imgs = torch.cat(
            [x[:self.n_per_row], x_rec[:self.n_per_row].sigmoid()], dim=0)
        if not self.to_tb:
            save_image(imgs,
                       './images/%s.png' % self.name_2,
                       nrow=self.n_per_row,
                       normalize=False,
                       pad_value=1)
        else:
            imgs = make_grid(imgs, nrow=self.n_per_row, pad_value=1)
            self.logger.writer.add_image('group_recons/%s' % self.name_2,
                                         imgs.cpu().numpy(), global_step)


class ShowReconXY(Imager):
    def __init__(self, logger, n_per_row=8, to_tb=True):
        self.logger = logger
        self.n_per_row = n_per_row
        self.to_tb = to_tb

    def __call__(self, model, state, global_step=0):
        x, y, x_rec, y_rec = state['x1'], state['x2'], state['x_g_rec'], state[
            'y_g_rec']
        imgs = torch.cat([
            x[:self.n_per_row], y[:self.n_per_row],
            x_rec[:self.n_per_row].sigmoid(), y_rec[:self.n_per_row].sigmoid()
        ],
                         dim=0)
        if not self.to_tb:
            save_image(imgs,
                       './images/g_recons.png',
                       nrow=self.n_per_row,
                       normalize=False,
                       pad_value=1)
        else:
            imgs = make_grid(imgs, nrow=self.n_per_row, pad_value=1)
            self.logger.writer.add_image('g_recons/g_recons',
                                         imgs.cpu().numpy(), global_step)


class PredYHat(Imager):
    def __init__(self, logger, n_per_row=8, to_tb=True):
        self.logger = logger
        self.n_per_row = n_per_row
        self.to_tb = to_tb

    def __call__(self, model, state, global_step=0):
        x, y, y_hat = state['x1'], state['x2'], state['x2_hat']
        imgs = torch.cat([
            x[:self.n_per_row], y[:self.n_per_row],
            y_hat[:self.n_per_row].sigmoid()
        ],
                         dim=0)
        if not self.to_tb:
            save_image(imgs,
                       './images/pred_y_hat.png',
                       nrow=self.n_per_row,
                       normalize=False,
                       pad_value=1)
        else:
            imgs = make_grid(imgs, nrow=self.n_per_row, pad_value=1)
            self.logger.writer.add_image('pred_y_hat/pred_y_hat',
                                         imgs.cpu().numpy(), global_step)


class LatentWalkLie(Imager):
    def __init__(self,
                 logger,
                 latents,
                 dims_to_walk,
                 subgroup_sizes_ls,
                 limits=[-2, 2],
                 steps=8,
                 input_batch=None,
                 to_tb=False):
        self.input_batch = input_batch
        self.logger = logger
        self.latents = latents
        self.dims_to_walk = dims_to_walk
        self.subgroup_sizes_ls = subgroup_sizes_ls
        self.limits = limits
        self.steps = steps
        self.to_tb = to_tb

    def gfeats_to_text(self, gfeats):
        out_str = ''
        gf_idx = 0
        for k in range(self.latents):
            per_g_str_ls = []
            for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
                per_g_str_ls.append('')
            for j in range(self.steps):
                b_idx = 0
                for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
                    e_idx = b_idx + subgroup_size_i
                    # mat_dim = int(math.sqrt(subgroup_size_i))
                    per_g_str_ls[i] += (str(
                        gfeats[gf_idx][b_idx:e_idx].cpu().numpy().round(3)) +
                                        ', ')
                    b_idx = e_idx
                gf_idx += 1
            for str_i in per_g_str_ls:
                out_str += (str_i + '\n\n')
            out_str += ('\n\n' + '=' * 10 + '\n\n')
        return out_str

    def __call__(self, model, state, global_step=0):
        limits, steps, latents, dims_to_walk = self.limits, self.steps, self.latents, self.dims_to_walk
        linspace = torch.linspace(*limits, steps=steps)

        if self.input_batch is None:
            x = torch.zeros(len(dims_to_walk), steps, latents)
        else:
            x = model.rep_fn(self.input_batch)[0]
            x = x.view(1, 1, latents).repeat(len(dims_to_walk), steps, 1)

        x = x.view(len(dims_to_walk), steps, latents)
        ind = 0
        for i in dims_to_walk:
            x[ind, :, i] = linspace
            ind += 1

        x = x.flatten(0, 1)
        imgs, group_feats_G = model.decode_full(x)
        imgs = imgs.sigmoid()
        group_feats_G_text = self.gfeats_to_text(group_feats_G)
        if not self.to_tb:
            save_image(imgs,
                       './images/linspace.png',
                       steps,
                       normalize=False,
                       pad_value=1)
        else:
            img = make_grid(imgs, self.steps, pad_value=1)
            self.logger.writer.add_image('linspaces/linspace',
                                         img.cpu().numpy(), global_step)
            self.logger.writer.add_text('linspaces/group_feats_G',
                                        group_feats_G_text, global_step)


class GroupWalk(Imager):
    def __init__(self, logger, nactions=4, n_to_show=15, to_tb=True):
        self.logger = logger
        self.nactions = nactions
        self.n_to_show = n_to_show
        self.to_tb = to_tb

    def __call__(self, model, state, global_step=0):
        imgs = []
        for ac in range(self.nactions):
            row_act = []
            gz = state['gz2'][:1]
            h = state['x1'].size(-1)
            for i, action in enumerate(range(self.n_to_show)):
                img = model.vae.decode_gfeat(gz).sigmoid().detach().view(
                    -1, state['x1'].shape[1], h, h).cpu().numpy()
                row_act.append(img)
                next_gz = model.apply_act(
                    gz,
                    torch.tensor(ac, device=gz.device).view(1, 1))
                gz = next_gz
            imgs.append(row_act)

        if not self.to_tb:
            plt.close()
            fig, ax = plt.subplots(nrows=self.nactions,
                                   ncols=self.n_to_show,
                                   figsize=(self.n_to_show, self.nactions))
            # fig.subplots_adjust(left=0.125, right=0.9, bottom=0.25, top=0.75, wspace=0.1, hspace=0.1)
            for k, i in enumerate(ax):
                for j, axis in enumerate(i):
                    axis.axis('off')
                    axis.imshow(imgs[k][j])
                    axis.set_xticklabels([])
                    axis.set_yticklabels([])
                    # axis.set_aspect(1)
            plt.tight_layout()
            plt.savefig('./images/lie_action_group_walk.png')
        else:
            img = make_grid(torch.tensor(np.array(imgs)).view(
                -1, state['x1'].shape[1], h, h),
                            self.n_to_show,
                            pad_value=1)
            self.logger.writer.add_image('lie_group_action/group_walk', img,
                                         global_step)


class GroupWalkRL(Imager):
    def __init__(self, logger, nactions=4, n_to_show=15, to_tb=True):
        self.logger = logger
        self.nactions = nactions
        self.n_to_show = n_to_show
        self.to_tb = to_tb

    def __call__(self, model, state, global_step=0):
        imgs = []
        for ac in range(self.nactions):
            row_act = []
            gz = state['x_eg'][:1]
            h = state['x1'].size(-1)
            for i, action in enumerate(range(self.n_to_show)):
                img = model.vae.decode_gfeat(gz).sigmoid().detach()
                row_act.append(
                    img.view(-1, state['x1'].shape[1], h, h).cpu().numpy())
                # next_gz = model.groups.apply_action(
                    # gz,
                    # torch.tensor(ac, device=gz.device).view(1, 1))['new_z']
                next_gz = model.groups.apply_action(
                    model.vae.encode_gfeat(img),
                    torch.tensor(ac,
                                 device=gz.device).view(1,
                                                        1))['new_z'].detach()
                # next_gz = model.groups.apply_action_with_x(
                    # model.vae.encode_gfeat(img),
                    # torch.tensor(ac,
                                 # device=gz.device).view(1,
                                                        # 1), img)['new_z'].detach()
                gz = next_gz
            imgs.append(row_act)

        if not self.to_tb:
            plt.close()
            fig, ax = plt.subplots(nrows=self.nactions,
                                   ncols=self.n_to_show,
                                   figsize=(self.n_to_show, self.nactions))
            # fig.subplots_adjust(left=0.125, right=0.9, bottom=0.25, top=0.75, wspace=0.1, hspace=0.1)
            for k, i in enumerate(ax):
                for j, axis in enumerate(i):
                    axis.axis('off')
                    axis.imshow(imgs[k][j][0].transpose(1, 2, 0))
                    axis.set_xticklabels([])
                    axis.set_yticklabels([])
                    # axis.set_aspect(1)
            plt.tight_layout()
            plt.savefig('./images/lie_action_group_walk.png')
        else:
            img = make_grid(torch.tensor(np.array(imgs)).view(
                -1, state['x1'].shape[1], h, h),
                            self.n_to_show,
                            pad_value=1)
            self.logger.writer.add_image('lie_group_action/group_walk', img,
                                         global_step)

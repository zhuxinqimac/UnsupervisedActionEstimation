#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: lie_vae_action_simple.py
# --- Creation Date: 03-01-2021
# --- Last Modified: Sun 03 Jan 2021 16:33:50 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Lie Group Vae Action Simple Version.
"""
import math
import torch
import numpy as np
from torch import nn
from models.vae import VAE
from models.beta import Flatten, View
from models.lie_vae import LieCeleb
from models.lie_vae_action import LieAction
from logger.custom_imaging import PredYHat, GroupWalk, ShowReconXY


class LieActionSimple(LieAction):
    def __init__(self, args):
        """
        Lie Group Vae Action Simple Class.
        """
        super().__init__(args)

    def get_act_repr_simple(self, act_param):
        b, n_lats = list(act_param.size())
        alg_tmp = torch.triu(
            torch.ones(b, n_lats, 2, 2, device=act_param.device))
        alg_tmp = alg_tmp - alg_tmp.transpose(-2, -1)
        act_alg = act_param.view(b, n_lats, 1, 1) * alg_tmp
        act_repr = torch.matrix_exp(act_alg.view(-1, 2, 2))  # [b*n_lats, 2, 2]
        act_repr = act_repr.view(b, n_lats, 2, 2)
        return act_repr

    def apply_act(self, group_feats, act):
        b, _ = list(act.size())
        act_repr = self.get_act_repr_simple(
            torch.index_select(self.act_params, 0,
                               act.view(b)))  # [b, n_lats, 2, 2]
        group_feats_mat = group_feats.view(b, len(self.subgroup_sizes_ls), 2,
                                           2)  # [b, n_lats, 2, 2]
        group_feats_mat_post = torch.matmul(act_repr, group_feats_mat)
        group_feats_post = group_feats_mat_post.view(b, -1)
        return group_feats_post

    def main_step(self, batch, batch_nb, loss_fn):
        (x, offset), y = batch
        group_feats = self.vae.encode_gfeat(x)
        group_feats_t = self.vae.encode_gfeat(y)
        group_feats_post = self.apply_act(group_feats, offset)
        y_hat = self.vae.decode_gfeat(group_feats_post)

        x_rec = self.vae.decode_gfeat(group_feats)
        y_rec = self.vae.decode_gfeat(group_feats_t)
        loss = self.latent_level_loss(group_feats_t,
                                      group_feats_post,
                                      mean=False)
        state = {
            'x1': x,
            'x2': y,
            'gz1': group_feats,
            'gz2': group_feats_t,
            'gz2_hat': group_feats_post,
            'x2_hat': y_hat,
            'act': offset,
            'x_g_rec': x_rec,
            'y_g_rec': y_rec
        }
        state['loss'] = loss

        self.global_step += 1

        tensorboard_logs = {
            'metric/loss':
            loss,
            'metric/mse_x2':
            self.recon_level_loss(y_hat, y, loss_fn, mean=True),
            'metric/mse_gz2':
            self.latent_level_loss(group_feats_t, group_feats_post, mean=True),
            'metric/latent_diff':
            self.latent_level_loss(group_feats_post, group_feats, mean=True),
            'metric/mse_gz1_mu2':
            self.latent_level_loss(group_feats, group_feats_t, mean=True),
        }

        return {'loss': loss, 'out': tensorboard_logs, 'state': state}

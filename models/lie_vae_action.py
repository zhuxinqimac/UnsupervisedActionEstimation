#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: lie_vae_action.py
# --- Creation Date: 30-12-2020
# --- Last Modified: Sat 02 Jan 2021 17:37:13 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Lie Group Vae Action.
"""
import math
import torch
import numpy as np
from torch import nn
from models.vae import VAE
from models.beta import Flatten, View
from models.lie_vae import LieCeleb
from logger.custom_imaging import PredYHat, GroupWalk, ShowReconXY


class LieAction(nn.Module):
    def __init__(self, args):
        """
        Lie Group Vae Action Class.
        """
        super().__init__()
        self.vae = LieCeleb(args)
        self.subgroup_sizes_ls = args.subgroup_sizes_ls
        self.subspace_sizes_ls = args.subspace_sizes_ls
        self.full_mat_dim = sum(
            map(lambda x: int(math.sqrt(x)), self.subgroup_sizes_ls))
        self.freeze_vae_params()
        self.loss_type = args.loss_type
        self.global_step = 0
        # self.act_params = nn.ParameterList([])
        self.num_actions = args.num_actions
        self.lie_alg_init_scale = args.lie_alg_init_scale
        # for i in range(self.num_actions):
        # self.act_params.append(
        # nn.Parameter(
        # torch.normal(torch.zeros(1), self.lie_alg_init_scale)))
        self.act_params = nn.Parameter(
            torch.normal(
                torch.zeros(self.num_actions, len(self.subgroup_sizes_ls)),
                self.lie_alg_init_scale))

    def freeze_vae_params(self):
        for param in self.vae.parameters():
            param.requires_grad = False

    def load_vae_state(self, state):
        def key_map(s, enc_dec_str):
            idx = s.find(enc_dec_str)
            return s[(idx + len(enc_dec_str) + 1):]

        encoder_state = {
            key_map(k, 'encoder'): v
            for k, v in state.items() if 'encoder' in k
        }
        decoder_state = {
            key_map(k, 'decoder'): v
            for k, v in state.items() if 'decoder' in k
        }
        self.vae.encoder.load_state_dict(encoder_state)
        self.vae.decoder.load_state_dict(decoder_state)
        self.freeze_vae_params()

    def train(self, mode=True):
        super().train(mode)
        self.vae.eval()

    def forward(self, x_act):
        x, act = x_act
        group_feats = self.vae.encode_gfeat(x)
        group_feats_post = self.apply_act(group_feats, act)
        return self.vae.decode_gfeat(group_feats_post)

    def get_act_repr(self, act_param, lie_alg_basis_ls):
        b, n_acts = list(act_param.size())
        assert n_acts == len(
            lie_alg_basis_ls)  # Assume each group has 1 latent dim.
        act_repr = torch.zeros(b, self.full_mat_dim,
                               self.full_mat_dim).to(act_param)
        b_idx = 0
        for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
            mat_dim = int(math.sqrt(subgroup_size_i))
            assert mat_dim * mat_dim == subgroup_size_i
            e_idx = b_idx + mat_dim
            assert list(lie_alg_basis_ls[i].size()) == [1, mat_dim, mat_dim]
            lie_alg = act_param[:, i][..., np.newaxis, np.newaxis] * (
                lie_alg_basis_ls[i] - lie_alg_basis_ls[i].transpose(-2, -1)
            )  # Assume each latent subspace is 1, and oth basis.
            lie_group = torch.matrix_exp(lie_alg)  # [b, mat_dim, mat_dim]
            act_repr[:, b_idx:e_idx, b_idx:e_idx] = lie_group
            b_idx = e_idx
        return act_repr

    def to_group_matrix(self, group_feats):
        b, feat_size = list(group_feats.size())
        assert feat_size == sum(self.subgroup_sizes_ls)
        group_feats_mat = torch.zeros(b, self.full_mat_dim,
                                      self.full_mat_dim).to(group_feats)
        b_idx, b_g_idx = 0, 0
        for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
            mat_dim = int(math.sqrt(subgroup_size_i))
            e_idx = b_idx + mat_dim
            e_g_idx = b_g_idx + subgroup_size_i
            group_feats_mat[:, b_idx:e_idx, b_idx:
                            e_idx] += group_feats[:, b_g_idx:e_g_idx].view(
                                b, mat_dim, mat_dim)
            b_idx, b_g_idx = e_idx, e_g_idx
        return group_feats_mat

    def to_group_feats(self, group_feats_mat):
        b, full_mat_dim, _ = list(group_feats_mat.size())
        assert full_mat_dim == self.full_mat_dim
        group_feats = torch.zeros(b, sum(
            self.subgroup_sizes_ls)).to(group_feats_mat)
        b_idx, b_g_idx = 0, 0
        for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
            mat_dim = int(math.sqrt(subgroup_size_i))
            e_idx = b_idx + mat_dim
            e_g_idx = b_g_idx + subgroup_size_i
            mat = group_feats_mat[:, b_idx:e_idx, b_idx:e_idx]
            group_feats[:, b_g_idx:e_g_idx] += torch.reshape(
                mat, [b, mat_dim * mat_dim])
            b_idx, b_g_idx = e_idx, e_g_idx
        return group_feats

    def apply_act(self, group_feats, act):
        b, _ = list(act.size())
        act_repr = self.get_act_repr(
            torch.index_select(self.act_params, 0, act.view(b)), self.vae.
            decoder.lie_alg_basis_ls)  # [b, full_mat_dim, full_mat_dim]
        group_feats_mat = self.to_group_matrix(
            group_feats)  # [b, full_mat_dim, full_mat_dim]
        group_feats_mat_post = torch.matmul(act_repr, group_feats_mat)
        group_feats_post = self.to_group_feats(group_feats_mat_post)
        return group_feats_post

    def latent_level_loss(self, feats_t, feats_post, mean=False):
        squares = (feats_post - feats_t).pow(2)
        if not mean:
            squares = squares.sum() / feats_post.shape[0]
        else:
            squares = squares.mean()
        return squares

    def recon_level_loss(self, x2_hat, x2, loss_fn, mean=False):
        loss = loss_fn(x2_hat, x2)
        if mean:
            loss = loss / x2[0].numel()
        return loss

    def apply_z_act(self, mu, act):
        b, _ = list(act.size())
        act_repr = torch.index_select(self.act_params, 0, act.view(b))
        assert mu.size() == act_repr.size()
        mu_post = mu + act_repr
        return mu_post

    def main_step(self, batch, batch_nb, loss_fn):
        (x, offset), y = batch
        if self.loss_type == 'on_group':
            group_feats = self.vae.encode_gfeat(x)
            group_feats_t = self.vae.encode_gfeat(y)
            group_feats_post = self.apply_act(group_feats, offset)
            y_hat = self.vae.decode_gfeat(group_feats_post)

            x_rec = self.vae.decode_gfeat(group_feats)
            y_rec = self.vae.decode_gfeat(group_feats_t)
            loss = self.latent_level_loss(group_feats_t,
                                          group_feats_post,
                                          mean=False)
        elif self.loss_type == 'on_alg':
            mulv, group_feats = self.vae.encode_full(x)
            mu, _ = self.vae.unwrap(mulv)
            mulv_t, group_feats_t = self.vae.encode_full(y)
            mu_t, _ = self.vae.unwrap(mulv_t)
            x_rec = self.vae.decode_gfeat(group_feats)
            y_rec = self.vae.decode_gfeat(group_feats_t)

            mu_post = self.apply_z_act(mu, offset)

            y_hat, group_feats_post = self.vae.decode_full(mu_post)
            # loss = self.latent_level_loss(mu_t, mu_post, mean=False)
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
            (group_feats_post - group_feats).pow(2).mean(),
            'metric/mse_gz1_mu2': (group_feats - group_feats_t).pow(2).mean()
        }

        return {'loss': loss, 'out': tensorboard_logs, 'state': state}

    def train_step(self, batch, batch_nb, loss_fn):
        return self.main_step(batch, batch_nb, loss_fn)

    def val_step(self, batch, batch_nb, loss_fn):
        return self.main_step(batch, batch_nb, loss_fn)

    def imaging_cbs(self, args, logger, model, batch=None):
        return [
            ShowReconXY(logger, n_per_row=30, to_tb=True),
            PredYHat(logger, n_per_row=30, to_tb=True),
            GroupWalk(logger,
                      nactions=self.num_actions,
                      n_to_show=60,
                      to_tb=True),
        ]

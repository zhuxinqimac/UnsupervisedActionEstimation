#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: lie_vae.py
# --- Creation Date: 25-12-2020
# --- Last Modified: Thu 14 Jan 2021 19:17:49 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Lie Group Vae.
"""
import math
import torch
import numpy as np
from torch import nn
from models.vae import VAE
from models.beta import Flatten, View
from models.beta import beta_shape_encoder, beta_celeb_encoder
from logger.custom_imaging import ShowReconX, LatentWalkLie
from logger.imaging import ShowRecon, LatentWalk, ReconToTb


class LieCelebEncoder(nn.Module):
    def __init__(self, subgroup_sizes_ls, subspace_sizes_ls, nc, dataset):
        """ Encoder network for lie_group_vae.
        Args:
            subgroup_sizes_ls (list of int): Sizes of subgroup_feats, must be square numbers.
            subspace_sizes_ls (list of int): Dimensions of subspace latents.
            nc (int): Numbr of output channels.
        """
        super().__init__()
        self.subgroup_sizes_ls = subgroup_sizes_ls
        self.subspace_sizes_ls = subspace_sizes_ls
        self.group_feats_size = sum(self.subgroup_sizes_ls)
        self.nc = nc
        if dataset == 'teapot' or dataset == 'teapot_nocolor':
            self.prior_group = nn.Sequential(
                nn.Conv2d(self.nc, 32, 4, 2, 1), nn.ReLU(True), # 64
                nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True), # 32
                nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True), # 16
                nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(True), # 8
                nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(True), # 4
                Flatten(),
                nn.Linear(64 * 4 * 4, 256), nn.ReLU(True),
                nn.Linear(256, self.group_feats_size))
            # self.prior_group = nn.Sequential(
                # nn.Conv2d(self.nc, 16, 8, stride=4), nn.ReLU(True),
                # nn.Conv2d(16, 32, 4, stride=2), nn.ReLU(True),
                # nn.Conv2d(32, 32, 3, stride=1), nn.ReLU(True), Flatten(),
                # nn.Linear(32 * 7 * 7, 256), nn.ReLU(True),
                # nn.Linear(256, self.group_feats_size))
        else:
            self.prior_group = nn.Sequential(
                nn.Conv2d(self.nc, 32, 4, 2, 1), nn.ReLU(True),
                nn.Conv2d(32, 32, 4, 2, 1), nn.ReLU(True),
                nn.Conv2d(32, 64, 4, 2, 1), nn.ReLU(True),
                nn.Conv2d(64, 64, 4, 2, 1), nn.ReLU(True), Flatten(),
                nn.Linear(1024, 256), nn.ReLU(),
                nn.Linear(256, self.group_feats_size))
        self.to_means = nn.ModuleList([])
        self.to_logvar = nn.ModuleList([])
        for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
            self.to_means.append(
                nn.Sequential(
                    nn.Linear(subgroup_size_i, subgroup_size_i * 4),
                    nn.ReLU(True),
                    nn.Linear(subgroup_size_i * 4, subspace_sizes_ls[i]),
                ))
            self.to_logvar.append(
                nn.Sequential(
                    nn.Linear(subgroup_size_i, subgroup_size_i * 4),
                    nn.ReLU(True),
                    nn.Linear(subgroup_size_i * 4, subspace_sizes_ls[i]),
                ))

    def to_gfeat(self, x):
        group_feats = self.prior_group(x)
        return group_feats

    def gfeat_to_lat(self, group_feats):
        b_idx = 0
        means_ls, logvars_ls = [], []
        for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
            e_idx = b_idx + subgroup_size_i
            means_ls.append(self.to_means[i](group_feats[:, b_idx:e_idx]))
            logvars_ls.append(self.to_logvar[i](group_feats[:, b_idx:e_idx]))
            b_idx = e_idx
        outs = torch.cat(means_ls + logvars_ls, dim=-1)
        return outs

    def forward(self, x):
        group_feats = self.prior_group(x)
        b_idx = 0
        means_ls, logvars_ls = [], []
        for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
            e_idx = b_idx + subgroup_size_i
            means_ls.append(self.to_means[i](group_feats[:, b_idx:e_idx]))
            logvars_ls.append(self.to_logvar[i](group_feats[:, b_idx:e_idx]))
            b_idx = e_idx
        outs = torch.cat(means_ls + logvars_ls, dim=-1)
        return outs, group_feats


class LieCelebDecoder(nn.Module):
    def __init__(self,
                 subgroup_sizes_ls,
                 subspace_sizes_ls,
                 lie_alg_init_type_ls,
                 dataset,
                 hy_ncut=0,
                 lie_alg_init_scale=0.001,
                 normalize_alg=False,
                 use_alg_var=False,
                 nc=3):
        """ Decoder network for lie_group_vae.
        Args:
            subgroup_sizes_ls (list of int): Sizes of subgroup_feats, must be square numbers.
            subspace_sizes_ls (list of int): Dimensions of subspace latents.
            lie_alg_init_type_ls (list of str): Init types for each lie algebra.
            hy_ncut (int): Hyper for number of cut to impose on group decomposition.
            lie_alg_init_scale (float): Lie algebra initial scale.
            normalize_alg: If normalize the lie generators.
            use_alg_var (bool): If use a scalar var for each latent dimension.
            nc (int): Number of out channels.
        """
        super().__init__()
        self.subgroup_sizes_ls = subgroup_sizes_ls
        self.subspace_sizes_ls = subspace_sizes_ls
        assert len(self.subgroup_sizes_ls) == len(self.subspace_sizes_ls)
        self.group_feats_size = sum(self.subgroup_sizes_ls)
        self.lie_alg_init_type_ls = lie_alg_init_type_ls
        self.hy_ncut = hy_ncut
        self.lie_alg_init_scale = lie_alg_init_scale
        self.normalize_alg = normalize_alg
        self.use_alg_var = use_alg_var
        self.nc = nc

        # Init lie_alg for each latent dim.
        self.lie_var_ls = nn.ParameterList([])
        self.lie_alg_basis_ls = nn.ParameterList([])
        for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
            mat_dim = int(math.sqrt(subgroup_size_i))
            assert mat_dim * mat_dim == subgroup_size_i
            for j in range(self.subspace_sizes_ls[i]):
                lie_alg_tmp, var_tmp = self.init_alg_basis(
                    i, j, mat_dim, lie_alg_init_type_ls[i], lie_alg_init_scale)
                self.lie_alg_basis_ls.append(lie_alg_tmp)
                self.lie_var_ls.append(var_tmp)

        if dataset == 'teapot' or dataset == 'teapot_nocolor':
            self.post_exp = nn.Sequential(
                nn.Linear(self.group_feats_size, 256),
                nn.Linear(256, 64 * 4 * 4),
                View(64, 4, 4),
                nn.ConvTranspose2d(64, 64, 4, 2, 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 64, 4, 2, 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, 32, 4, 2, 1),
                nn.ReLU(True),
                nn.ConvTranspose2d(32, self.nc, 4, 2, 1),
            )
        else:
            self.post_exp = nn.Sequential(
                nn.Linear(self.group_feats_size, 256), nn.ReLU(True),
                nn.Linear(256, 1024), nn.ReLU(True), View(64, 4, 4),
                nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.ReLU(True),
                nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
                nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(True),
                nn.ConvTranspose2d(32, self.nc, 4, 2, 1))

    def split_latents(self, x):
        # x: [b, dim]
        if self.hy_ncut == 0:
            return [x]
        b, dim = list(x.size())
        split_idx = torch.randint(0, dim + 1, size=[b, self.hy_ncut])
        split_idx, _ = torch.sort(split_idx, dim=-1)

        idx_range = torch.tile(torch.range(dim)[np.newaxis, :], (b, 1))
        masks = []
        mask_last = torch.zeros(b, dim, dtype=x.dtype)
        for i in range(self.hy_ncut):
            mask_tmp = (idx_range < split_idx[:, i:i + 1]).to(
                x)  # change dtype
            masks.append(mask_tmp - mask_last)
            masks_last = mask_tmp
        mask_tmp = (idx_range < split_idx[:, -1:]).to(x)
        masks.append(1. - mask_tmp)
        x_split_ls = [x * mask for mask in masks]
        return x_split_ls

    def val_exp(self, x, lie_alg_basis_ls, lie_var_ls, lie_alg_init_type):
        lie_alg_basis_ls = [p * 1. for p in lie_alg_basis_ls
                            ]  # For torch.cat, convert param to tensor.
        lie_alg_basis = torch.cat(lie_alg_basis_ls,
                                  dim=0)[np.newaxis,
                                         ...]  # [1, lat_dim, mat_dim, mat_dim]
        if lie_alg_init_type == 'oth':
            lie_alg_basis = lie_alg_basis - lie_alg_basis.transpose(-2, -1)
        if self.normalize_alg:
            norm_v = torch.linalg.norm(lie_alg_basis, dim=[-2, -1])
            lie_alg_basis = lie_alg_basis / norm_v
        if self.use_alg_var:
            lie_var = torch.cat(lie_var_ls, dim=1)  # [1, lat_dim]
            lie_alg_basis = lie_alg_basis * lie_var[..., np.newaxis, np.
                                                    newaxis]

        lie_alg_mul = x[
            ..., np.newaxis, np.
            newaxis] * lie_alg_basis  # [b, lat_dim, mat_dim, mat_dim]
        lie_alg = torch.sum(lie_alg_mul, dim=1)  # [b, mat_dim, mat_dim]
        lie_group = torch.matrix_exp(lie_alg)  # [b, mat_dim, mat_dim]
        return lie_group

    def train_exp(self, x, lie_alg_basis_ls, mat_dim, lie_var_ls,
                  lie_alg_init_type):
        lie_alg_basis_ls = [p * 1. for p in lie_alg_basis_ls
                            ]  # For torch.cat, convert param to tensor.
        lie_alg_basis = torch.cat(lie_alg_basis_ls,
                                  dim=0)[np.newaxis,
                                         ...]  # [1, lat_dim, mat_dim, mat_dim]
        if lie_alg_init_type == 'oth':
            lie_alg_basis = lie_alg_basis - lie_alg_basis.transpose(-2, -1)
        if self.normalize_alg:
            norm_v = torch.linalg.norm(lie_alg_basis, dim=[-2, -1])
            lie_alg_basis = lie_alg_basis / norm_v
        if self.use_alg_var:
            lie_var = torch.cat(lie_var_ls, dim=1)  # [1, lat_dim]
            lie_alg_basis = lie_alg_basis * lie_var[..., np.newaxis, np.
                                                    newaxis]

        lie_group = torch.eye(mat_dim, dtype=x.dtype).to(
            x.device)[np.newaxis, ...]  # [1, mat_dim, mat_dim]
        lie_alg = 0.
        latents_in_cut_ls = self.split_latents(x)  # [x0, x1, ...]
        for masked_latent in latents_in_cut_ls:
            lie_alg_sum_tmp = torch.sum(
                masked_latent[..., np.newaxis, np.newaxis] * lie_alg_basis,
                dim=1)
            lie_alg += lie_alg_sum_tmp  # [b, mat_dim, mat_dim]
            lie_group_tmp = torch.matrix_exp(lie_alg_sum_tmp)
            lie_group = torch.matmul(lie_group,
                                     lie_group_tmp)  # [b, mat_dim, mat_dim]
        return lie_group

    def init_alg_basis(self, i, j, mat_dim, lie_alg_init_type,
                       lie_alg_init_scale):
        lie_alg_tmp = nn.Parameter(torch.normal(mean=torch.zeros(
            1, mat_dim, mat_dim),
                                                std=lie_alg_init_scale),
                                   requires_grad=True)
        var_tmp = nn.Parameter(
            torch.normal(torch.zeros(1, 1), lie_alg_init_scale))
        return lie_alg_tmp, var_tmp

    def from_gfeat(self, z):
        output = self.post_exp(z)
        return output

    def forward(self, latents_in):
        latent_dim = list(latents_in.size())[-1]
        assert latent_dim == sum(self.subspace_sizes_ls)

        # Calc exp.
        lie_group_tensor_ls = []
        b_idx = 0
        for i, subgroup_size_i in enumerate(self.subgroup_sizes_ls):
            mat_dim = int(math.sqrt(subgroup_size_i))
            e_idx = b_idx + self.subspace_sizes_ls[i]
            if self.subspace_sizes_ls[i] > 1:
                if not self.training:
                    lie_subgroup = self.val_exp(
                        latents_in[:, b_idx:e_idx],
                        self.lie_alg_basis_ls[b_idx:e_idx],
                        self.lie_var_ls[b_idx:e_idx],
                        self.lie_alg_init_type_ls[i])
                else:
                    lie_subgroup = self.train_exp(
                        latents_in[:, b_idx:e_idx],
                        self.lie_alg_basis_ls[b_idx:e_idx], mat_dim,
                        self.lie_var_ls[b_idx:e_idx],
                        self.lie_alg_init_type_ls[i])
            else:
                lie_subgroup = self.val_exp(latents_in[:, b_idx:e_idx],
                                            self.lie_alg_basis_ls[b_idx:e_idx],
                                            self.lie_var_ls[b_idx:e_idx],
                                            self.lie_alg_init_type_ls[i])
            lie_subgroup_tensor = lie_subgroup.view(-1, mat_dim * mat_dim)
            lie_group_tensor_ls.append(lie_subgroup_tensor)
            b_idx = e_idx
        lie_group_tensor = torch.cat(lie_group_tensor_ls,
                                     dim=1)  # [b, group_feat_size]

        output = self.post_exp(lie_group_tensor)
        return output, lie_group_tensor


class LieCeleb(VAE):
    def __init__(self, args):
        super().__init__(
            # beta_celeb_encoder(args),
            LieCelebEncoder(args.subgroup_sizes_ls, args.subspace_sizes_ls,
                            args.nc, args.dataset),
            LieCelebDecoder(args.subgroup_sizes_ls, args.subspace_sizes_ls,
                            args.lie_alg_init_type_ls, args.dataset,
                            args.hy_ncut, args.lie_alg_init_scale,
                            args.normalize_alg, args.use_alg_var, args.nc),
            args.beta,
            args.capacity,
            args.capacity_leadin)
        self.hy_hes = args.hy_hes
        self.hy_rec = args.hy_rec
        self.hy_commute = args.hy_commute
        self.cycle_latent = args.cycle_latent
        self.cycle_limit = args.cycle_limit
        self.cycle_prob = args.cycle_prob
        self.forward_eg_prob = args.forward_eg_prob
        self.subgroup_sizes_ls = args.subgroup_sizes_ls
        self.subspace_sizes_ls = args.subspace_sizes_ls

    def encode(self, x):
        return self.encoder(x)[0]

    def decode(self, z):
        return self.decoder(z)[0]

    def encode_gfeat(self, x):
        return self.encoder.to_gfeat(x)

    def decode_gfeat(self, z):
        return self.decoder.from_gfeat(z)

    def encode_full(self, x):
        return self.encoder(x)

    def decode_full(self, z):
        return self.decoder(z)

    def calc_basis_mul_ij(self, lie_alg_basis_ls_param):
        lie_alg_basis_ls = [alg_tmp * 1. for alg_tmp in lie_alg_basis_ls_param]
        lie_alg_basis = torch.cat(lie_alg_basis_ls,
                                  dim=0)[np.newaxis,
                                         ...]  # [1, lat_dim, mat_dim, mat_dim]
        _, lat_dim, mat_dim, _ = list(lie_alg_basis.size())
        lie_alg_basis_col = lie_alg_basis.view(lat_dim, 1, mat_dim, mat_dim)
        lie_alg_basis_outer_mul = torch.matmul(
            lie_alg_basis,
            lie_alg_basis_col)  # [lat_dim, lat_dim, mat_dim, mat_dim]
        hessian_mask = 1. - torch.eye(
            lat_dim, dtype=lie_alg_basis_outer_mul.dtype
        )[:, :, np.newaxis, np.newaxis].to(lie_alg_basis_outer_mul.device)
        lie_alg_basis_mul_ij = lie_alg_basis_outer_mul * hessian_mask  # XY
        return lie_alg_basis_mul_ij

    def calc_hessian_loss(self, lie_alg_basis_mul_ij, i):
        hessian_loss = torch.mean(
            torch.sum(torch.square(lie_alg_basis_mul_ij), dim=[2, 3]))
        return hessian_loss

    def calc_commute_loss(self, lie_alg_basis_mul_ij, i):
        lie_alg_commutator = lie_alg_basis_mul_ij - lie_alg_basis_mul_ij.permute(
            0, 1, 3, 2)
        commute_loss = torch.mean(
            torch.sum(torch.square(lie_alg_commutator), dim=[2, 3]))
        return commute_loss

    def group_loss(self, group_feats_E, group_feats_G, lie_alg_basis_ls):
        b_idx = 0
        hessian_loss = 0.
        commute_loss = 0.
        for i, subspace_size in enumerate(self.subspace_sizes_ls):
            e_idx = b_idx + subspace_size
            if subspace_size > 1:
                mat_dim = int(math.sqrt(self.subgroup_sizes_ls[i]))
                assert list(lie_alg_basis_ls[b_idx].size())[-1] == mat_dim
                lie_alg_basis_mul_ij = self.calc_basis_mul_ij(
                    lie_alg_basis_ls[b_idx:e_idx])  # XY
                hessian_loss += self.calc_hessian_loss(lie_alg_basis_mul_ij, i)
                commute_loss += self.calc_commute_loss(lie_alg_basis_mul_ij, i)
            b_idx = e_idx
        rec_loss = torch.mean(
            torch.sum(torch.square(group_feats_E - group_feats_G), dim=1))
        tensorboard_logs = {
            'metric/gfeat_rec_loss': rec_loss,
            'metric/hessian_loss': hessian_loss,
            'metric/commute_loss': commute_loss
        }
        rec_loss *= self.hy_rec
        hessian_loss *= self.hy_hes
        commute_loss *= self.hy_commute
        loss = hessian_loss + commute_loss + rec_loss
        return loss, tensorboard_logs

    def randomly_cycle(self, mu, limit=2.):
        low_mask = mu < -limit
        high_mask = mu > limit
        mu_cycled = mu.clone()
        mu_cycled[low_mask] = mu[low_mask] + 2. * limit
        mu_cycled[high_mask] = mu[high_mask] - 2. * limit
        # mu_cycled = mu.clone()
        # rand_mask = torch.rand(mu.size()) < self.cycle_prob
        # rand_add = np.random.uniform()
        # if rand_add > 0.5:
        # mu_cycled[rand_mask] = mu[rand_mask] + limit
        # else:
        # mu_cycled[rand_mask] = mu[rand_mask] - limit
        return mu_cycled

    def main_step(self, batch, batch_nb, loss_fn):
        x, y = batch

        mulv, group_feats_E = self.encode_full(x)
        mu, lv = self.unwrap(mulv)
        if self.training:
            if self.cycle_latent:
                mu_cycled = self.randomly_cycle(mu, limit=self.cycle_limit)
                z = self.reparametrise(mu_cycled, lv)
            else:
                z = self.reparametrise(mu, lv)
        else:
            z = self.reparametrise(mu, lv)
        x_hat, group_feats_G = self.decode_full(z)

        x_eg_hat = self.decode_gfeat(group_feats_E)
        x_gg_hat = self.decode_gfeat(group_feats_G)

        if self.training:
            rand_n = np.random.uniform()
            # rec_loss = loss_fn(x_hat, x)
            if rand_n < self.forward_eg_prob:
                rec_loss = loss_fn(x_eg_hat, x)
            else:
                rec_loss = loss_fn(x_hat, x)

            # elif 0.25 < rand_n <= 0.5:
            # rec_loss = loss_fn(x_eg_hat, x)
            # else:
            # rec_loss = loss_fn(x_gg_hat, x)
        else:
            rec_loss = loss_fn(x_hat, x)
        group_loss, tensorboard_logs = self.group_loss(
            group_feats_E, group_feats_G, self.decoder.lie_alg_basis_ls)
        total_kl = self.compute_kl(mu, lv, mean=False)
        beta_kl = self.control_capacity(total_kl, self.global_step,
                                        self.anneal)
        # beta_kl = total_kl * self.beta
        loss = rec_loss + beta_kl + group_loss
        state = self.make_state(batch_nb, x_hat, x, y, mu, lv, z)
        state.update({
            'x_eg': group_feats_E,
            'x_gg': group_feats_G,
            'x_eg_rec': x_eg_hat,
            'x_gg_rec': x_gg_hat,
            'x_z_rec': x_hat,
        })
        self.global_step += 1

        tensorboard_logs.update({
            'metric/loss': loss,
            'metric/recon_loss': rec_loss,
            'metric/group_loss': group_loss,
            'metric/beta_kl': beta_kl,
            'metric/total_kl': total_kl
        })
        return {'loss': loss, 'out': tensorboard_logs, 'state': state}

    def imaging_cbs(self, args, logger, model, batch=None):
        cbs = super().imaging_cbs(args, logger, model, batch=batch)
        return [
            ShowRecon(),
            ReconToTb(logger),
            LatentWalkLie(logger,
                          args.latents,
                          list(range(args.latents)),
                          subgroup_sizes_ls=self.subgroup_sizes_ls,
                          limits=[-8, 8],
                          steps=20,
                          input_batch=batch,
                          to_tb=True),
            ShowReconX(logger, to_tb=True, name_1='x1', name_2='x_eg_rec'),
            ShowReconX(logger, to_tb=True, name_1='x1', name_2='x_gg_rec'),
            ShowReconX(logger, to_tb=True, name_1='x1', name_2='x_z_rec'),
        ]
        return cbs

#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: uneven_vae.py
# --- Creation Date: 11-04-2021
# --- Last Modified: Mon 12 Apr 2021 23:30:46 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
UnevenVAE Definition
"""
import math
import torch
import numpy as np
from torch import nn
# import torch.nn.functional as F
from models.vae import VAE
from models.beta import Flatten, View
from models.beta import beta_celeb_encoder
from models.beta import beta_celeb_decoder
from logger.custom_imaging import ShowReconX, LatentWalkLie
from logger.imaging import ShowRecon, LatentWalk, ReconToTb


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True, max_rate=0):
        super().__init__(in_features, out_features, bias)
        # self.weight: (out_features, in_features)
        self.mask = mask  # (out_features, in_features)
        # print('mask:', self.mask)
        self.use_dropout = (max_rate > 0)
        if self.use_dropout:
            dropout_rates = np.linspace(0, max_rate, in_features)
            self.dropout_layers = nn.ModuleList([nn.Dropout(r) for r in dropout_rates])

    def forward(self, x):
        mask = self.mask
        if self.use_dropout:
            mask_ls = []
            for i, dp in enumerate(self.dropout_layers):
                mask_ls.append(dp(mask[:, i:i+1]))
            mask = torch.cat(mask_ls, dim=1)
        weight = self.weight * mask
        return nn.functional.linear(x, weight, self.bias)


def beta_decoder(args):
    if args.uneven_masked_w:
        # mask = torch.flip(torch.triu(torch.ones(256, args.latents)), [1])
        n_segs = 256 // args.latents
        n_resi = 256 % args.latents
        mask_segs = torch.triu(torch.ones(args.latents, args.latents)).repeat([n_segs, 1])
        mask = torch.flip(torch.cat([mask_segs, torch.triu(torch.ones(n_resi, args.latents))], dim=0), [1]).to('cuda')
        first_layer = MaskedLinear(args.latents, 256, mask)
    elif args.uneven_w_max_dropout_rate > 0:
        mask = torch.ones(256, args.latents).to('cuda')
        first_layer = MaskedLinear(args.latents, 256, mask, max_rate=args.uneven_w_max_dropout_rate)
    else:
        first_layer = nn.Linear(args.latents, 256)
    return nn.Sequential(
        first_layer, nn.ReLU(True), nn.Linear(256, 1024),
        nn.ReLU(True), View(64, 4, 4), nn.ConvTranspose2d(64, 64, 4, 2, 1),
        nn.ReLU(True), nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.ReLU(True),
        nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.ReLU(True), nn.ConvTranspose2d(32, args.nc, 4, 2, 1),
        # nn.Sigmoid()
    )


class UnevenVAE(VAE):
    def __init__(self, args):
        super().__init__(beta_celeb_encoder(args), beta_decoder(args), args.beta, args.capacity, args.capacity_leadin)
        for p in self.encoder.modules():
            if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear) or \
                    isinstance(p, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(p.weight)
        for p in self.decoder.modules():
            if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear) or \
                    isinstance(p, nn.ConvTranspose2d):
                torch.nn.init.xavier_uniform_(p.weight)
        # if args.uneven_reg_maxval < 0:
            # val_pre_softplus = nn.Parameter(torch.normal(mean=10 * torch.ones([])), requires_grad=True)
            # self.uneven_reg_maxval = nn.functional.softplus(val_pre_softplus)
        # else:
            # self.uneven_reg_maxval = torch.tensor(args.uneven_reg_maxval, dtype=torch.float32)
        self.uneven_reg_maxval = args.uneven_reg_maxval
        self.exp_uneven_reg = args.exp_uneven_reg
        self.uneven_reg_lambda = args.uneven_reg_lambda
        self.uneven_reg_encoder_lambda = args.uneven_reg_encoder_lambda
        self.use_cumax_adaptive = args.use_cumax_adaptive
        self.orth_lambda = args.orth_lambda
        if self.use_cumax_adaptive:
            self.adap_logits = nn.Parameter(torch.normal(mean=torch.zeros(args.latents)), requires_grad=True)

    def main_step(self, batch, batch_nb, loss_fn):

        x, y = batch

        mu, lv = self.unwrap(self.encode(x))
        z = self.reparametrise(mu, lv)
        x_hat = self.decode(z)

        recon_loss = loss_fn(x_hat, x)

        weight_to_uneven = self.decoder[0].weight  # (out_dim, n_lat)
        uneven_loss, reg = self.uneven_loss(weight_to_uneven, self.uneven_reg_lambda)
        orth_loss = self.orth_loss(weight_to_uneven, self.orth_lambda)
        uneven_enc_loss = 0.
        if self.uneven_reg_encoder_lambda > 0:
            weight_enc_to_uneven = self.encoder[-1].weight  # (n_lat * 2, in_dim)
            weight_enc_to_uneven = torch.transpose(
                weight_enc_to_uneven[:weight_enc_to_uneven.size(0)//2, ...], 0, 1)
            uneven_enc_loss = self.uneven_loss(weight_enc_to_uneven, 
                                               self.uneven_reg_encoder_lambda)
            uneven_loss += uneven_enc_loss

        total_kl = self.compute_kl(mu, lv, mean=False)
        beta_kl = self.control_capacity(total_kl, self.global_step, self.anneal)
        state = self.make_state(batch_nb, x_hat, x, y, mu, lv, z)
        self.global_step += 1

        loss = recon_loss + beta_kl + uneven_loss + orth_loss
        tensorboard_logs = {'metric/loss': loss, 'metric/recon_loss': recon_loss, 'metric/total_kl': total_kl,
                            'metric/beta_kl': beta_kl, 'metric/uneven_loss': uneven_loss,
                            'metric/uneven_enc_loss': uneven_enc_loss, 'metric/uneven_dec_loss': uneven_loss - uneven_enc_loss,
                            'metric/uneven_reg_maxval': self.uneven_reg_maxval, 'metric/orth_loss': orth_loss}
        for i in range(z.size(-1)):
            tensorboard_logs['metric/reg_'+str(i)] = reg[i]
        return {'loss': loss, 'out': tensorboard_logs, 'state': state}

    def uneven_loss(self, weight, loss_lambda):
        '''
        weight: (out_dim, in_dim)
        '''
        if self.use_cumax_adaptive:
            reg_softmax = nn.functional.softmax(self.adap_logits, dim=0)
            reg = torch.cumsum(reg_softmax, dim=0) * self.uneven_reg_maxval
        else:
            reg = torch.linspace(0., self.uneven_reg_maxval, weight.size(1)).to('cuda')
        # print('reg:', reg)
        if self.exp_uneven_reg:
            reg = torch.exp(reg)
        w_in = torch.sum(weight * weight, dim=0)  # (in_dim)
        return torch.sum(w_in * reg, dim=0) * loss_lambda, reg

    def orth_loss(self, weight, loss_lambda):
        '''
        weight: (out_dim, n_lat)
        '''
        w_mul = torch.matmul(weight.transpose(0, 1), weight)  # (n_lat, n_lat)
        ij_mask = 1. - torch.eye(w_mul.size(0), dtype=torch.float32).to('cuda')
        masked_w = w_mul * ij_mask
        return torch.sum(masked_w * masked_w) * loss_lambda

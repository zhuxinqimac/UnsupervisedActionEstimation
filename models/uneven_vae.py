#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: uneven_vae.py
# --- Creation Date: 11-04-2021
# --- Last Modified: Mon 12 Apr 2021 18:05:56 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
UnevenVAE Definition
"""
import math
import torch
import numpy as np
from torch import nn
from models.vae import VAE
from models.beta import Flatten, View
from models.beta import beta_celeb_encoder
from models.beta import beta_celeb_decoder
from logger.custom_imaging import ShowReconX, LatentWalkLie
from logger.imaging import ShowRecon, LatentWalk, ReconToTb


class MaskedLinear(nn.Linear):
    def __init__(self, in_features, out_features, mask, bias=True):
        super().__init__(in_features, out_features, bias)
        # self.weight: (out_features, in_features)
        self.mask = mask  # (out_features, in_features)
        print('mask:', self.mask)

    def forward(self, x):
        weight = self.weight * self.mask
        return F.linear(x, weight, self.bias)


def beta_decoder(args):
    if args.uneven_masked_w:
        # mask = torch.flip(torch.triu(torch.ones(256, args.latents)), [1])
        n_segs = 256 // args.latents
        n_resi = 256 % args.latents
        mask_segs = torch.triu(torch.ones(args.latents, args.latents)).repeat([n_segs, 1])
        mask = torch.flip(torch.cat([mask_segs, torch.triu(torch.ones(n_resi, args.latents))], dim=0), [1]).to('cuda')
        first_layer = MaskedLinear(args.latents, 256, mask)
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

    def main_step(self, batch, batch_nb, loss_fn):

        x, y = batch

        mu, lv = self.unwrap(self.encode(x))
        z = self.reparametrise(mu, lv)
        x_hat = self.decode(z)

        recon_loss = loss_fn(x_hat, x)

        weight_to_uneven = self.decoder[0].weight  # (out_dim, n_lat)
        uneven_loss = self.uneven_loss(weight_to_uneven, self.uneven_reg_lambda)
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

        loss = recon_loss + beta_kl + uneven_loss
        tensorboard_logs = {'metric/loss': loss, 'metric/recon_loss': recon_loss, 'metric/total_kl': total_kl,
                            'metric/beta_kl': beta_kl, 'metric/uneven_loss': uneven_loss,
                            'metric/uneven_enc_loss': uneven_enc_loss, 'metric/uneven_dec_loss': uneven_loss - uneven_enc_loss,
                            'metric/uneven_reg_maxval': self.uneven_reg_maxval}
        return {'loss': loss, 'out': tensorboard_logs, 'state': state}

    def uneven_loss(self, weight, loss_lambda):
        '''
        weight: (out_dim, in_dim)
        '''
        reg = torch.linspace(0., self.uneven_reg_maxval, weight.size(1)).to('cuda')
        # print('reg:', reg)
        if self.exp_uneven_reg:
            reg = torch.exp(reg)
        w_in = torch.sum(weight * weight, dim=0)  # (in_dim)
        return torch.sum(w_in * reg, dim=0) * loss_lambda

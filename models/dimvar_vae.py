#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: dimvar_vae.py
# --- Creation Date: 22-05-2021
# --- Last Modified: Sat 22 May 2021 22:22:13 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
DimVar VAE model.
"""
import math
import torch
import numpy as np
import lpips
from torch import nn, optim
import torch.nn.functional as F
from models.vae import VAE
from models.beta import Flatten, View
from models.beta import beta_shape_encoder, beta_celeb_encoder
from logger.custom_imaging import ShowReconX, LatentWalkLie
from logger.imaging import ShowRecon, LatentWalk, ReconToTb

class DimVarVAE(VAE):
    def __init__(self, encoder, decoder, args):
        super().__init__(encoder, decoder, args.beta, args.capacity, args.capacity_leadin)
        self.latents = args.latents

    def fuse_vars_old(self, mean_v, v, y, b_range):
        v_new = mean_v.clone()
        v_new[b_range, y] = v[b_range, y]
        return v_new

    def fuse_vars(self, mean_v, v, y, b_range):
        y_onehot = F.one_hot(y, self.latents)>0 # (b, latents) bool
        v_new = torch.where(y_onehot, v, mean_v) # (b, latents)
        return v_new

    def rep_fn(self, batch):
        x, _, _ = batch
        mu, lv = self.unwrap(self.encode(x))
        return mu

    def main_step(self, batch, batch_nb, loss_fn):

        x1, x2, y = batch
        batch_size = x1.size(0)

        mu1, lv1 = self.unwrap(self.encode(x1))
        mu2, lv2 = self.unwrap(self.encode(x2))
        mean_mu = (mu1 + mu2) / 2.
        mean_lv = (lv1 + lv2) / 2.

        b_range = torch.arange(batch_size)
        mu1_new = self.fuse_vars(mean_mu, mu1, y, b_range)
        mu2_new = self.fuse_vars(mean_mu, mu2, y, b_range)
        lv1_new = self.fuse_vars(mean_lv, lv1, y, b_range)
        lv2_new = self.fuse_vars(mean_lv, lv2, y, b_range)

        z1 = self.reparametrise(mu1_new, lv1_new)
        z2 = self.reparametrise(mu2_new, lv2_new)

        x1_hat = self.decode(z1)
        x2_hat = self.decode(z2)

        loss1_recon = loss_fn(x1_hat, x1)
        loss2_recon = loss_fn(x2_hat, x2)

        total_kl1 = self.compute_kl(mu1_new, lv1_new, mean=False)
        total_kl2 = self.compute_kl(mu2_new, lv2_new, mean=False)

        beta_kl1 = self.control_capacity(total_kl1, self.global_step, self.anneal)
        beta_kl2 = self.control_capacity(total_kl2, self.global_step, self.anneal)
        state = self.make_state(batch_nb, x1_hat, x1, y, mu1_new, lv1_new, z1)

        loss = loss1_recon + loss2_recon + beta_kl1 + beta_kl2

        self.global_step += 1

        tensorboard_logs = {'metric/loss': loss,
                            'metric/recon_loss1': loss1_recon,
                            'metric/recon_loss2': loss2_recon,
                            'metric/total_kl1': total_kl1, 'metric/total_kl2': total_kl2,
                            'metric/beta_kl1': beta_kl1, 'metric/beta_kl2': beta_kl2}
        return {'loss': loss, 'out': tensorboard_logs, 'state': state}

def dimvar_vae_64(args):
    from models.beta import beta_celeb_encoder, beta_celeb_decoder
    encoder, decoder = beta_celeb_encoder(args), beta_celeb_decoder(args)
    return DimVarVAE(encoder, decoder, args)

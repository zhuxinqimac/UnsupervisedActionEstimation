#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: uneven_facvae.py
# --- Creation Date: 13-04-2021
# --- Last Modified: Sun 18 Apr 2021 17:17:28 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
UnevenFactorVAE Definition
"""
import math
import torch
import numpy as np
from torch import nn
from torch import nn, optim
import torch.nn.functional as F
from models.vae import VAE
from models.beta import Flatten, View
from models.beta import beta_celeb_encoder
from models.beta import beta_celeb_decoder
from models.factor_vae import Discriminator
from models.uneven_vae import UnevenVAE
from logger.custom_imaging import ShowReconX, LatentWalkLie
from logger.imaging import ShowRecon, LatentWalk, ReconToTb


class UnevenFacVAE(UnevenVAE):
    def __init__(self, args):
        super().__init__(args)
        self.discriminator = [Discriminator(args.latents).cuda()]  # Exclude from register.
        self.disc_opt = optim.Adam(self.discriminator[0].parameters(), lr=1e-4, betas=(0.5, 0.9))
        # self.disc_opt = optim.Adam(self.discriminator.parameters(), lr=1e-4)
        self.gamma = float(args.factor_vae_gamma) if args.factor_vae_gamma is not None else 6.4
        if args.xav_init:
            for p in self.encoder.modules():
                if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear) or \
                        isinstance(p, nn.ConvTranspose2d):
                    torch.nn.init.xavier_uniform_(p.weight)
            for p in self.decoder.modules():
                if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear) or \
                        isinstance(p, nn.ConvTranspose2d):
                    torch.nn.init.xavier_uniform_(p.weight)
            for p in self.discriminator[0].modules():
                if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear) or \
                        isinstance(p, nn.ConvTranspose2d):
                    torch.nn.init.xavier_uniform_(p.weight)

    def permute_dims(self, z):
        assert z.dim() == 2

        # z = z.permute(1, 0)
        B, _ = z.size()
        perm_z = []
        for z_j in z.split(1, 1):
            perm = torch.randperm(B).to(z.device)
            perm_z_j = z_j[perm]
            perm_z.append(perm_z_j)

        # return torch.cat(perm_z, 1).permute(1, 0)
        return torch.cat(perm_z, 1)

    def main_step(self, batch, batch_nb, loss_fn):
        out = super().main_step(batch, batch_nb, loss_fn)
        state = out['state']
        x, y, mu, lv, z, x_hat = state['x'], state['y'], state['mu'], state['lv'], state['z'], state['x_hat']

        D_z = self.discriminator[0](z.detach())
        z_perm = self.permute_dims(z)
        D_z_perm = self.discriminator[0](z_perm.detach())
        D_tc_loss = 0.5 * (F.cross_entropy(D_z, torch.zeros(D_z.shape[0], dtype=torch.long).to(D_z.device))
                           + F.cross_entropy(D_z_perm, torch.ones(D_z.shape[0], dtype=torch.long).to(D_z.device)))

        if self.training:
            self.disc_opt.zero_grad()
            D_tc_loss.backward()
            self.disc_opt.step()

        D_z = self.discriminator[0](z)
        vae_loss = out['loss']
        vae_tc_loss = (D_z[:, :1] - D_z[:, 1:]).mean() * self.gamma

        tensorboard_logs = out['out']
        tensorboard_logs.update({'metric/loss': vae_loss+vae_tc_loss, 'metric/tc_loss': vae_tc_loss.detach(), 'metric/disc_tc_loss': D_tc_loss.detach()})

        self.global_step += 1

        return {'loss': vae_loss + vae_tc_loss,
                'out': tensorboard_logs,
                'state': state}

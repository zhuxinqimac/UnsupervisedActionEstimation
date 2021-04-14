#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: uneven_dipvae.py
# --- Creation Date: 13-04-2021
# --- Last Modified: Tue 13 Apr 2021 22:57:45 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
UnevenDIPVAE Definition
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
from models.uneven_vae import UnevenVAE
from models.dip_vae import dip_vae_i_loss, dip_vae_ii_loss
from logger.custom_imaging import ShowReconX, LatentWalkLie
from logger.imaging import ShowRecon, LatentWalk, ReconToTb


class UnevenDIPVAE(UnevenVAE):
    def __init__(self, args):
        super().__init__(args)
        self.type = args.model

        if self.type == 'uneven_dip_vae_i':
            self.dip_loss = lambda mu, lv: dip_vae_i_loss(mu)
        else:
            self.dip_loss = lambda mu, lv: dip_vae_ii_loss(mu, lv)

    def main_step(self, batch, batch_nb, loss_fn):
        out = super().main_step(batch, batch_nb, loss_fn)
        state = out['state']
        x, y, mu, lv, z, x_hat = state['x'], state['y'], state['mu'], state['lv'], state['z'], state['x_hat']

        dip_loss = self.dip_loss(mu, lv)
        vae_loss = out['loss']

        self.global_step += 1

        tensorboard_logs = out['out']
        tensorboard_logs['metric/dip_loss'] = dip_loss.detach()
        return {'loss': vae_loss + dip_loss, 'out': tensorboard_logs,
                'state': state}

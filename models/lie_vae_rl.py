#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: lie_vae_rl.py
# --- Creation Date: 06-01-2021
# --- Last Modified: Sat 16 Jan 2021 00:36:43 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Lie Group Vae with Reinforce.
"""
import math
import torch
import numpy as np
from torch import nn
from models.vae import VAE
from models.beta import Flatten, View
from models.beta import beta_shape_encoder, beta_celeb_encoder
from models.utils import clip_hook
from models.group_vae import GroupVAE
from models.lie_vae_group_actions import ReinforceLieGroupWiseAction
from logger.custom_imaging import ShowReconX, LatentWalkLie
from logger.custom_imaging import PredYHat, GroupWalk, ShowReconXY, GroupWalkRL
from logger.imaging import ShowRecon, LatentWalk, ReconToTb
from logger.imaging import RewardPlot, ActionListTbText, ActionWiseRewardPlot
from logger.imaging import ActionPredictionPlot, ActionDensityPlot, ActionStepsToTb
from logger.imaging import ShowLearntAction, AttentionTb


class ActionConvAttentionEncoderLieGroup(nn.Module):
    def __init__(self, in_latents, in_nc, difference=False, zero_init=True):
        super().__init__()
        self.difference = difference
        actions = in_latents * 2  # Every latent code corresponds to 2 actions (- and +).
        frame_multi = 1 if difference else 2
        self.net = nn.Sequential(  # 64x64
            nn.Conv2d(in_nc * frame_multi, 32, 3, 2), nn.ReLU(True),
            nn.Conv2d(32, 16, 3, 2), nn.ReLU(True), nn.Conv2d(16, 16, 3, 2),
            nn.ReLU(True), View(-1), nn.Linear(784, actions))
        for p in self.net.modules():
            if isinstance(p, nn.Conv2d):
                nn.init.orthogonal_(p.weight)
        if zero_init:
            nn.init.ones_(list(self.net.modules())[-1].bias)
            nn.init.zeros_(list(self.net.modules())[-1].weight)

    def forward(self, x):
        if self.difference:
            x = (x[:, 0] - x[:, 1]).unsqueeze(1)
        out = self.net(x.detach()).abs().softmax(-1)
        return out


class ReinforceLieGroupVAE(GroupVAE):
    def __init__(self,
                 vae,
                 action_encoder,
                 nlatents,
                 nactions,
                 beta,
                 max_capacity=None,
                 capacity_leadin=None,
                 use_regret=False,
                 base_policy_weight=0.9,
                 base_policy_epsilon=1.0005,
                 num_action_steps=1,
                 multi_action_strategy='reward',
                 reinforce_discount=0.99,
                 gamma=None,
                 use_cob=False,
                 lie_alg_init_scale=0.1,
                 learning_rate=None,
                 supervised_train=False,
                 in_nc=1,
                 hy_st=0.,
                 dataset='dsprites',
                 entropy_weight=0.):
        """ RGrVAE model

        Args:
            vae (models.VAE): Backbone VAE module
            action_encoder (nn.Module): torch Module that encodes image pairs into a policy distribution
            nlatents (int): Number of latent dimensions
            nactions (int): Number of actions
            beta (float): Weighting for the KL divergence
            max_capacity (float): Max capacity for capactiy annealing
            capacity_leadin (int): Capacity leadin, linearly scale capacity up to max over leadin steps
            use_regret (bool): Use reinforement regret
            base_policy_weight (float): Base weight to apply policy over random
            base_policy_epsilon (float): Increase policy weight by (1-weight)*epsilon
            num_action_steps (int): Number of actions to allow
            multi_action_strategy (str): One of ['reward', 'returns']
            reinforce_discount (float): Discount rewards factor for calcuating returns
            gamma (float): GrVAE gamma for weighting prediction loss
            use_cob (bool): Allow change of basis for representations
            learning_rate (float): Learning rate
            entropy_weight (float): Exploration entropy weight. Weighted entropy is subtracted from loss
            lie_alg_init_scale (float): Lie algebra initial scale.
        """
        super().__init__(vae.encoder, vae.decoder, action_encoder, nlatents,
                         nactions, beta, max_capacity, capacity_leadin, gamma,
                         learning_rate)
        self.groups = ReinforceLieGroupWiseAction(
            latents=nlatents,
            action_encoder=action_encoder,
            base_policy_weight=base_policy_weight,
            base_policy_epsilon=base_policy_epsilon,
            use_regret=use_regret,
            in_nc=in_nc,
            decoder=vae.decoder,
            encoder=vae.encoder,
            multi_action_strategy=multi_action_strategy,
            reinforce_discount=reinforce_discount,
            use_cob=use_cob,
            lie_alg_init_scale=lie_alg_init_scale,
            supervised_train=supervised_train,
            dataset=dataset,
            entropy_weight=entropy_weight)
        self.num_action_steps = num_action_steps
        self.vae = vae
        self.supervised_train = supervised_train
        self.hy_st = hy_st

        for p in self.parameters():
            p.register_hook(clip_hook) if p.requires_grad else None

    def group_params(self):
        return list(self.groups.g_strength_net.parameters())

    def encode_full(self, x):
        return self.vae.encode_full(x)

    def decode_full(self, z):
        return self.vae.decode_full(z)

    def imaging_cbs(self, args, logger, model, batch=None):
        cbs = [
            ShowRecon(),
            ReconToTb(logger),
            LatentWalkLie(logger,
                          args.latents,
                          list(range(args.latents)),
                          subgroup_sizes_ls=self.vae.subgroup_sizes_ls,
                          limits=[-3, 3],
                          steps=20,
                          input_batch=batch,
                          to_tb=True),
            ShowReconX(logger, to_tb=True, name_1='x1', name_2='x_eg_rec'),
            ShowReconX(logger, to_tb=True, name_1='x1', name_2='x_gg_rec'),
            ShowReconX(logger, to_tb=True, name_1='x1', name_2='x_z_rec'),
            PredYHat(logger, n_per_row=30, to_tb=True),
            GroupWalkRL(logger,
                        nactions=self.nactions,
                        n_to_show=60,
                        to_tb=True),
            ShowLearntAction(logger, to_tb=True),
            AttentionTb(logger),
            RewardPlot(logger),
            ActionListTbText(logger),
            ActionWiseRewardPlot(logger),
            ActionPredictionPlot(logger),
            ActionDensityPlot(logger),
            ActionStepsToTb(logger),
        ]
        return cbs

    def predict_next_eg(self, state):
        img_list = [state['x1']]

        rand_n = np.random.uniform()
        if rand_n < 0.5:
            z1, x1, x2 = state['x_gg_static'], state['x1'], state['x2']
        else:
            z1, x1, x2 = state['x_eg'], state['x1'], state['x2']
        # z1, x1, x2 = state['x_eg'], state['x1'], state['x2']
        # z1, x1, x2 = state['x_gg_static'], state['x1'], state['x2']
        st_ls = []
        for i in range(self.num_action_steps):
            z2_dict, out = self.groups.predict_next_z(z1, x1, x2,
                                                      self.training,
                                                      state['true_actions'])
            z2, st_selected = z2_dict['new_z'], z2_dict['st_selected']
            st_ls.append(st_selected)
            z1 = z2
            # x1_old = x1
            x1 = self.vae.decode_gfeat(z2).detach()
            if not self.supervised_train:
                # self.groups.reinforce_params[-1].x_preact = x1_old
                self.groups.reinforce_params[-1].x = x1
            img_list.append(x1.sigmoid())
        img_list.append(x2)
        out['action_sets'] = img_list
        return z2, out, st_ls

    def policy_loss(self, state, reset=True, loss_fn=None):
        if self.supervised_train:
            return 0, {}, {}
        else:
            return self.groups.loss(
                state['y_eg'],
                x2=state['x2'],
                reset=reset,
                loss_fn=loss_fn,
            )

    def pred_loss(self, state, loss_fn):
        latent_loss = loss_fn(state['x2_hat'], state['x2'])
        latent_loss += self.gamma * self.latent_level_loss(
            state['y_eg_hat'], state['y_eg'], mean=False)
        # latent_loss += self.gamma * self.latent_level_loss(
        # state['y_eg_hat'], state['y_gg'], mean=False)
        return latent_loss

    def loss_fn_predict_next_eg(self, state, loss_fn=None, reset=True):
        policy_loss, logs, outs = self.policy_loss(state, reset, loss_fn)
        latent_loss = self.pred_loss(state, loss_fn)
        # return policy_loss + latent_loss, logs, outs
        return policy_loss, latent_loss, logs, outs

    def main_step(self, batch, batch_nb, loss_fn):
        (x, offset), y = batch
        out = self.vae.main_step((x, y), batch_nb, loss_fn)
        state = out['state']
        x, y, mu, lv, z, x_hat = state['x'], state['y'], state['mu'], state[
            'lv'], state['z'], state['x_hat']
        x_eg, x_gg = state['x_eg'], state['x_gg']
        x_eg_rec = state['x_eg_rec']
        x_gg_rec = state['x_gg_rec']
        x_z_rec = state['x_z_rec']

        mulv2, y_eg = self.vae.encode_full(y)
        mu2, lv2 = self.vae.unwrap(mulv2)
        _, y_gg = self.vae.decode_full(mu2)
        _, x_gg_static = self.vae.decode_full(mu)

        state = {
            'x1': x,
            'x2': y,
            'x_eg': x_eg,
            'x_gg': x_gg,
            'x_gg_static': x_gg_static,
            'y_eg': y_eg,
            'y_gg': y_gg,
            'true_actions': offset
        }
        y_eg_hat, outs, st_ls = self.predict_next_eg(state)

        x2_hat = self.vae.decode_gfeat(y_eg_hat)
        # y_eg_hat_safe = self.vae.encode_gfeat(x2_hat.sigmoid())
        state.update({
            # 'y_eg_hat_not_safe': y_eg_hat,
            # 'y_eg_hat': y_eg_hat_safe,
            'y_eg_hat': y_eg_hat,
            'x2_hat': x2_hat,
            'loss_fn': loss_fn
        })

        vae_loss = out['loss']
        policy_loss, latent_loss, loss_logs, loss_out = self.loss_fn_predict_next_eg(
            state, loss_fn)
        pred_loss = policy_loss + latent_loss
        st_loss = torch.cat(st_ls, dim=-1).pow(2).mean()
        # print('st_loss.size:', st_loss.size())
        # print(out['out'])

        out_state = self.make_state(batch_nb, x_hat, x, y, mu, lv, z)
        out_state['recon_hat'] = x2_hat
        out_state['true_recon'] = self.vae.decode_gfeat(y_eg)
        out_state['true_actions'] = offset
        out_state['x_eg_rec'] = x_eg_rec
        out_state['x_gg_rec'] = x_gg_rec
        out_state['x_z_rec'] = x_z_rec
        out_state['x2_hat'] = x2_hat
        out_state['x_eg'] = x_eg
        out_state.update(outs)
        out_state.update(loss_out)
        self.global_step += 1

        tensorboard_logs = {
            'metric/loss':
            vae_loss + pred_loss + st_loss,
            'metric/policy_loss':
            policy_loss,
            'metric/latent_loss':
            latent_loss,
            'metric/pred_loss':
            pred_loss,
            'metric/ls_loss':
            st_loss,
            'metric/total_kl_meaned':
            self.vae.compute_kl(mu, lv, mean=True),
            'metric/mse_x1':
            self.recon_level_loss(x_hat, x, loss_fn, mean=True),
            'metric/mse_x2':
            self.recon_level_loss(x2_hat, y, loss_fn, mean=True),
            'metric/mse_y_eg':
            self.latent_level_loss(y_eg_hat, y_eg, mean=True),
            'metric/mse_y_eghat_gg':
            self.latent_level_loss(y_eg_hat, y_gg, mean=True),
            'metric/mse_y_gg_eg':
            self.latent_level_loss(y_gg, y_eg, mean=True),
            'metric/gfeat_diff':
            self.latent_level_loss(y_eg_hat, x_eg, mean=True),
            'metric/mse_xeg_yeg':
            self.latent_level_loss(x_eg, y_eg, mean=True)
        }
        tensorboard_logs.update(loss_logs)
        tensorboard_logs.update(out['out'])
        return {
            'loss': vae_loss + pred_loss + self.hy_st * st_loss,
            # 'loss': vae_loss + pred_loss,
            'loss_ls': [vae_loss + latent_loss + st_loss, policy_loss],
            'out': tensorboard_logs,
            'state': out_state
        }
        # optimiser_ls = [torch.optim.Adam(model.vae_params(), lr=args.learning_rate * 1),
        # torch.optim.Adam(model.action_params(), lr=args.policy_learning_rate * 1),
        # torch.optim.Adam(model.group_params(), lr=args.group_learning_rate)]


def lie_rl_group_vae(*ars, **kwargs):
    def _group_vae(args, base_model=None):
        if base_model is not None:
            base_model = base_model
        else:
            from models.models import models
            base_model = models[args.base_model](args)
        action_encoder = ActionConvAttentionEncoderLieGroup(args.latents,
                                                            in_nc=args.nc)
        return ReinforceLieGroupVAE(
            base_model,
            action_encoder,
            nlatents=args.latents,
            nactions=args.latents * 2,
            in_nc=args.nc,
            hy_st=args.hy_st,
            beta=args.beta,
            max_capacity=args.capacity,
            capacity_leadin=args.capacity_leadin,
            use_regret=args.use_regret,
            base_policy_weight=args.base_policy_weight,
            base_policy_epsilon=args.base_policy_epsilon,
            num_action_steps=args.num_action_steps,
            multi_action_strategy=args.multi_action_strategy,
            reinforce_discount=args.reinforce_discount,
            gamma=args.gvae_gamma,
            use_cob=args.use_cob,
            dataset=args.dataset,
            lie_alg_init_scale=args.lie_alg_init_scale,
            learning_rate=args.learning_rate,
            supervised_train=args.supervised_train,
            entropy_weight=args.entropy_weight)

    return _group_vae

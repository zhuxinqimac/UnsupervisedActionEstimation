#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: lie_vae_policy_grads.py
# --- Creation Date: 08-01-2021
# --- Last Modified: Wed 13 Jan 2021 23:20:09 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Docstring
"""
from torch.nn import functional as F
from models.policy_grads import VAEReinforceBase, ReinforceParams


class LieVAEReinforceBase(VAEReinforceBase):
    def __init__(self,
                 base_policy_weight,
                 base_policy_epsilon,
                 normalised_reward,
                 use_regret,
                 decoder,
                 rep_fn,
                 multi_action_strategy='reward',
                 reinforce_discount=0.99,
                 encoder=None,
                 dataset='dsprites',
                 entropy_weight=0.):
        super().__init__(base_policy_weight=base_policy_weight,
                         base_policy_epsilon=base_policy_epsilon,
                         normalised_reward=normalised_reward,
                         use_regret=use_regret,
                         decoder=decoder,
                         encoder=encoder,
                         rep_fn=rep_fn,
                         multi_action_strategy=multi_action_strategy,
                         reinforce_discount=reinforce_discount,
                         entropy_weight=entropy_weight)
        self.dataset = dataset

    # def _img_reward(self, x1, new_x2, true_x2, loss_fn=None):
    # d_pre_action = loss_fn(x1, true_x2)
    # d_post_action = loss_fn(new_x2, true_x2)
    # print('d_post_action.size:', d_post_action.size())
    # return (d_pre_action - d_post_action).float().detach()

    def sample_next_z_with_x(self, attn_dist, z, x, training=True):
        probs, action = self.sample(attn_dist, training)
        z2_dict = self.apply_action_with_x(z, action, x)
        z2 = z2_dict['new_z']
        self.reinforce_params.append(ReinforceParams(probs, z, z2, action, attn_dist))
        self.reinforce_params[-1].x_preact = x
        return z2_dict

    def apply_action_with_x(self, z, action, x):
        return self.rep_fn(z, action, x)

    def _img_reward_bce(self, x1, new_x2, true_x2, loss_fn=None):
        d_pre_action = F.binary_cross_entropy_with_logits(
            x1.view(-1, 64 * 64), true_x2.view(-1, 64 * 64),
            reduction='none').sum(-1)
        d_post_action = F.binary_cross_entropy_with_logits(
            new_x2.view(-1, 64 * 64),
            true_x2.view(-1, 64 * 64),
            reduction='none').sum(-1)
        # print('d_post_action.size:', d_post_action.size())
        return (d_pre_action - d_post_action).float().detach()
        # return (d_post_action - d_pre_action).float().detach()
        # return (-d_post_action).detach()

    def _img_reward_l2(self, x1, new_x2, true_x2, loss_fn=None):
        d_pre_action = (x1.view(-1, 64 * 64).sigmoid() - true_x2.view(-1, 64 * 64)).pow(2).sum(-1)
        d_post_action = (new_x2.view(-1, 64 * 64).sigmoid() - true_x2.view(-1, 64 * 64)).pow(2).sum(-1)
        # print('d_post_action.size:', d_post_action.size())
        return (d_pre_action - d_post_action).float().detach()
        # return (d_post_action - d_pre_action).float().detach()
        # return (-d_post_action).detach()

    def reward(self, old_z, action, dist, target_params, loss_fn=None, x1=None):
        new_z = self.apply_action(old_z, action)['new_z']
        # new_z = self.apply_action_with_x(old_z, action, x1)['new_z']
        true_z2 = target_params['z2']
        true_x2 = target_params['x2']
        new_x2 = self.decoder.from_gfeat(new_z)
        x1 = self.decoder.from_gfeat(old_z)
        new_z2 = self.encoder.to_gfeat(new_x2.sigmoid())
        # reward_img = self._img_reward_l2(x1, new_x2, true_x2, loss_fn=loss_fn)
        reward_latent = self._latent_reward(old_z, new_z2, true_z2)
        # reward = reward_img + reward_latent
        # reward = reward_img
        reward = reward_latent
        return reward.detach()

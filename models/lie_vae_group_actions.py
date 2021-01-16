#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: lie_vae_actions.py
# --- Creation Date: 06-01-2021
# --- Last Modified: Fri 15 Jan 2021 22:15:21 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Lie Group Vae Actions in Reinforce.
"""
import copy
import numpy as np

from models.group_reps import *
from models.beta import Flatten, View
# from models.policy_grads import VAEReinforceBase
from models.lie_vae_policy_grads import LieVAEReinforceBase
from models.utils import attn_stats, sprites_label_to_action
from torch import nn


class LieGroupWiseAction(nn.Module):
    def __init__(self,
                 latents,
                 action_encoder,
                 decoder=None,
                 in_nc=1,
                 lie_alg_init_scale=0.1,
                 supervised_train=False,
                 use_cob=False):
        super().__init__()
        # groups_params = nn.ParameterList([])
        self.latents = latents
        self.lie_alg_init_scale = lie_alg_init_scale
        self.supervised_train = supervised_train
        # for i in range(latents * 2):
        # groups_params.append(
        # nn.Parameter(
        # torch.normal(torch.zeros(1, 1, 1),
        # self.lie_alg_init_scale)))
        self.decoder = decoder
        groups_params = nn.Parameter(
            torch.normal(torch.zeros(self.latents * 2, 1, 1),
                         self.lie_alg_init_scale))

        self.groups = groups_params
        self.action_encoder = action_encoder
        if action_encoder is not None:
            self.old_action_encoder = copy.deepcopy(action_encoder)
            for p in self.old_action_encoder.parameters():
                p.requires_grad = False
        self.g_strength_net = nn.Sequential(
            nn.Conv2d(2 * sum(self.decoder.subgroup_sizes_ls),
                      2 * sum(self.decoder.subspace_sizes_ls) * 4,
                      1,
                      groups=2 * len(self.decoder.subgroup_sizes_ls)),
            nn.ReLU(True),
            nn.Conv2d(
                2 * sum(self.decoder.subspace_sizes_ls) * 4,
                # 2 * len(self.decoder.subgroup_sizes_ls),
                2 * sum(self.decoder.subspace_sizes_ls),
                1,
                groups=2 * len(self.decoder.subgroup_sizes_ls)),)
            # nn.Softplus())
        # nn.Tanh())
        self.g_strength_net_with_x = nn.Sequential(
            nn.Conv2d(in_nc, 32, 3, 2), nn.ReLU(True), nn.Conv2d(32, 16, 3, 2),
            nn.ReLU(True), nn.Conv2d(16, 16, 3, 2), nn.ReLU(True), View(-1),
            nn.Linear(784, 2 * self.latents), nn.Softplus())

    def to_tb(self, writer, epoch):
        tb_str = '\n\n\n\n'.join([repr(g) for g in self.groups])
        writer.add_text('matrices', tb_str, epoch)

        for i, g in enumerate(self.groups):
            writer.add_scalar('cyclic_orders/values_{}'.format(i), g.detach(),
                              epoch)

    def forward(self, state):
        return self.predict_next_z(state['z1'], state['x1'], state['x2'])

    def get_attn(self, x1, x2, prev_step=False):
        img_pair = torch.cat([x1, x2], 1)
        if prev_step:
            return self.old_action_encoder(img_pair)
        else:
            return self.action_encoder(img_pair)

    def update_prev_params(self):
        self.old_action_encoder.load_state_dict(
            self.action_encoder.state_dict())
        map(lambda p: p.detach(), self.old_action_encoder.parameters())
        map(lambda p: p.requires_grad(False),
            self.old_action_encoder.parameters())

    def predict_next_z(self, z1, x1, x2):
        raise NotImplementedError

    def next_rep(self, z, ac):
        raise NotImplementedError

    def loss(self, *args, **kwargs):
        return 0, {}, {}


class ReinforceLieGroupWiseAction(LieGroupWiseAction, LieVAEReinforceBase):
    def __init__(self,
                 latents=4,
                 action_encoder=None,
                 base_policy_weight=0.9,
                 base_policy_epsilon=1.0005,
                 normalised_reward=True,
                 use_regret=False,
                 in_nc=1,
                 decoder=None,
                 encoder=None,
                 multi_action_strategy='reward',
                 reinforce_discount=0.99,
                 use_cob=False,
                 lie_alg_init_scale=0.1,
                 supervised_train=False,
                 dataset='dsprites',
                 entropy_weight=0.):
        LieGroupWiseAction.__init__(self,
                                    latents,
                                    action_encoder=action_encoder,
                                    decoder=decoder,
                                    in_nc=in_nc,
                                    lie_alg_init_scale=lie_alg_init_scale,
                                    supervised_train=supervised_train,
                                    use_cob=use_cob)
        LieVAEReinforceBase.__init__(
            self,
            base_policy_weight=base_policy_weight,
            base_policy_epsilon=base_policy_epsilon,
            normalised_reward=normalised_reward,
            use_regret=use_regret,
            decoder=decoder,
            encoder=encoder,
            dataset=dataset,
            rep_fn=self.next_rep,
            multi_action_strategy=multi_action_strategy,
            reinforce_discount=reinforce_discount,
            entropy_weight=entropy_weight)

    def predict_next_z(self, z1, x1, x2, training=True, true_actions=None):

        if self.supervised_train:
            true_actions = sprites_label_to_action(true_actions).view(z1.shape[0])
            z2_dict = self.apply_action(z1, true_actions)
            # z2_dict = self.apply_action_with_x(z1, true_actions, x1)
            z2 = z2_dict['new_z']
            out = {}
        else:
            attn = self.get_attn(x1, x2)
            z2_dict = self.sample_next_z(attn, z1, training=training)
            # z2_dict = self.sample_next_z_with_x(attn,
            # z1,
            # x1,
            # training=training)
            z2 = z2_dict['new_z']
            try:
                out = attn_stats(attn, true_actions)
            except Exception as e:
                print(e)
                out = {}
        return z2_dict, out

    def unwrap(self, x):
        return torch.split(x, x.shape[1] // 2, dim=1)

    def reparametrise(self, mu, lv):
        if self.training:
            std = torch.exp(0.5 * lv)
            eps = torch.randn_like(std)
            return mu + std * eps
        else:
            return mu

    def next_rep(self, z, ac, x=None):
        if self.dataset == 'teapot' or self.dataset == 'teapot_nocolor':
            return self.next_rep_single_multi_lat_gfeat(z, ac, x)
        else:
            return self.next_rep_multi_one_lat_gfeat(z, ac, x)

    def next_rep_single_multi_lat_gfeat(self, z, ac, x=None):
        # !! Assume only one feat in subgroup_sizes_ls.
        # z = z.detach()
        b = list(ac.size())[0]
        # print('b=', b)
        mat_dim = int(math.sqrt(self.decoder.subgroup_sizes_ls[0]))
        z_mulv = self.encoder.gfeat_to_lat(z)
        mu, lv = self.unwrap(z_mulv)
        z_lat = self.reparametrise(mu, lv)
        # z = z.view(-1, len(self.decoder.subgroup_sizes_ls), mat_dim,
        # mat_dim)  # [b, n_sub=1, mat_dim, mat_dim]
        # z_new = z.clone()

        lie_alg_basis_ls = []
        for i, lie_alg_tmp in enumerate(self.decoder.lie_alg_basis_ls):
            if self.decoder.lie_alg_init_type_ls[0] == 'oth':
                lie_alg_basis_ls.append(lie_alg_tmp * 1. -
                                        lie_alg_tmp.transpose(-2, -1))
            else:
                lie_alg_basis_ls.append(lie_alg_tmp * 1.)
        lie_alg_basis = torch.cat(lie_alg_basis_ls,
                                  dim=0)  # [n_lats, mat_dim, mat_dim]
        alg_decomp = z_lat.view(b, self.latents, 1, 1) * lie_alg_basis.view(
            1, self.latents, mat_dim, mat_dim)  # [b, n_lats, mat_dim, mat_dim]
        gfeats_decomp = torch.matrix_exp(alg_decomp.view(
            -1, mat_dim, mat_dim)).view(b, self.latents, mat_dim, mat_dim)
        gfeats_decomp_new = gfeats_decomp.clone()

        # Get group action from ac_idx.
        alg_idx = (ac % self.latents).view(b)
        act_lie_algs = torch.index_select(lie_alg_basis.detach(), 0,
                                          alg_idx)  # [b, mat_dim, mat_dim]
        # ac_dir = ((ac >= self.latents).float() - 0.5) * 2.
        st = self.g_strength_net(torch.cat([z, z], dim=1).view(b, -1, 1, 1))
        # print('st.size:', st.size())
        st = st.view(b, 2 * self.latents)  # [b, ac]
        act_groups = torch.matrix_exp(
            # ac_dir.view(b, 1, 1) *
            st[torch.arange(b), ac.view(b)].view(b, 1, 1) *
            act_lie_algs)  # [b, mat_dim, mat_dim]
        gfeats_decomp_new[torch.arange(b), alg_idx] = torch.matmul(
            act_groups, gfeats_decomp[torch.arange(b), alg_idx])
        gfeats_new = torch.eye(mat_dim, dtype=gfeats_decomp_new.dtype).to(
            gfeats_decomp_new.device)[np.newaxis, ...]  # [1, mat_dim, mat_dim]
        for i in range(self.latents):
            gfeats_new = torch.matmul(gfeats_decomp_new[:, i],
                                      gfeats_new)  # [b, mat_dim, mat_dim]

        new_z = gfeats_new.view(b, -1)
        return {'new_z': new_z, 'st_selected': st[torch.arange(b), ac.view(b)]}

    def next_rep_multi_one_lat_gfeat(self, z, ac, x=None):
        # !! Assume all sizes in subgroup_sizes_ls are the same.
        # !! Assume all subspace_i are 1.
        # z = z.detach()
        b = list(ac.size())[0]
        # print('ac:', ac)
        # print('ac.size():', ac.size())
        mat_dim = int(math.sqrt(self.decoder.subgroup_sizes_ls[0]))
        z_split = z.view(-1, len(self.decoder.subgroup_sizes_ls), mat_dim,
                         mat_dim)  # [b, n_sub, mat_dim, mat_dim]
        # print('z.size:', z.size())
        z_split_new = z_split.clone()
        lie_alg_basis_ls = []
        for i, lie_alg_tmp in enumerate(self.decoder.lie_alg_basis_ls):
            if self.decoder.lie_alg_init_type_ls[i] == 'oth':
                lie_alg_basis_ls.append(lie_alg_tmp * 1. -
                                        lie_alg_tmp.transpose(-2, -1))
            else:
                lie_alg_basis_ls.append(lie_alg_tmp * 1.)
        lie_alg_basis = torch.cat(lie_alg_basis_ls, dim=0)
        alg_idx = (ac % self.latents).view(b)

        act_lie_algs = torch.index_select(lie_alg_basis.detach(), 0,
                                          alg_idx)  # [b, mat_dim, mat_dim]
        # act_lie_algs = torch.index_select(lie_alg_basis, 0,
        # alg_idx)  # [b, mat_dim, mat_dim]
        # params = torch.index_select(self.groups, 0, ac.view(b))
        # ac_dir = ((ac >= self.latents).float() - 0.5) * 2.
        st = self.g_strength_net(torch.cat([z, z],
                                           dim=1).view(b, -1, 1, 1)).view(
                                               b, 2 * self.latents)  # [b, ac]
        # st = self.g_strength_net_with_x(x).view(b, 2 * self.latents)  # [b, ac]
        act_groups = torch.matrix_exp(
            # ac_dir.view(b, 1, 1) *
            st[torch.arange(b), ac.view(b)].view(b, 1, 1) *
            act_lie_algs)  # [b, mat_dim, mat_dim]

        z_split_new[torch.arange(b), alg_idx] = torch.matmul(
            act_groups, z_split[torch.arange(b), alg_idx])
        new_z = z_split_new.view(b, -1)
        return {'new_z': new_z, 'st_selected': st[torch.arange(b), ac.view(b)]}

    def loss(self, real, x2, reset=True, loss_fn=None):
        self.update_prev_params()
        policy_loss, tb_dict, out = LieVAEReinforceBase.loss(self,
                                                             real,
                                                             x2,
                                                             loss_fn=loss_fn)
        super().reset() if reset else []
        return policy_loss, tb_dict, out

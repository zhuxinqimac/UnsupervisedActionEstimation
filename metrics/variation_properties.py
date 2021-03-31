#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: variation_properties.py
# --- Creation Date: 23-03-2021
# --- Last Modified: Wed 24 Mar 2021 15:00:32 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
To evaluate how the learned group structure fits the variations in data.
"""

import math
import numpy as np
import torch
import torch.nn as nn
import lpips


class VariationProperty:
    def __init__(self, n_linspace=100, bound=2, n_samples=50):
        super().__init__()
        self.n_linspace = n_linspace
        self.bound = bound
        self.n_samples = n_samples

    def __call__(self, pymodel):
        nlatents = sum(pymodel.subspace_sizes_ls)
        lpips_fn = lpips.LPIPS(net='alex').cuda()
        percept_dim_ls = []
        group_var_dim_ls = []
        linear_var_dim_ls = []
        for i in range(nlatents):
            z = torch.normal(torch.zeros(self.n_samples, 1, nlatents), 1).cuda()
            z = z.repeat(1, self.n_linspace, 1)
            z[:, :, i] = torch.linspace(-self.bound, self.bound, self.n_linspace).view(1, -1).repeat(self.n_samples, 1)
            imgs, gfeats = pymodel.decode_full(z.view(-1, nlatents))
            img_shape = imgs.size()[1:]
            imgs = imgs.view(self.n_samples, self.n_linspace, *img_shape)
            print('imgs.shape:', imgs.size())
            imgs_1 = imgs[:, :-1, ...]
            imgs_2 = imgs[:, 1:, ...]
            # percept_dis = lpips_fn.forward(imgs_1.reshape(-1, *img_shape), imgs_2.reshape(-1, *img_shape))
            percept_dis_ls = []
            for j in range(self.n_samples):
                percept_dis_ls.append(lpips_fn.forward(imgs_1[j], imgs_2[j]))
            percept_dis = torch.cat(percept_dis_ls, dim=0)
            percept_dis = percept_dis.view(self.n_samples, self.n_linspace-1)
            percept_dim = percept_dis.sum(-1).mean(0)
            percept_dim_ls.append(percept_dim)

            mat_dim = int(math.sqrt(gfeats.size()[-1]))
            assert mat_dim * mat_dim == gfeats.size()[-1]
            gfeats = gfeats.view(self.n_samples, self.n_linspace, mat_dim, mat_dim)
            linear_vec = torch.matmul(gfeats, torch.ones(mat_dim).cuda()) # [n_samples, n_linspace, mat_dim]
            linear_vec1 = linear_vec[:, :-1, ...]
            linear_vec2 = linear_vec[:, 1:, ...]
            linear_dis = torch.linalg.norm(linear_vec1 - linear_vec2, dim=-1).view(self.n_samples, self.n_linspace-1 )
            linear_var_dim = linear_dis.sum(-1).mean(0)
            linear_var_dim_ls.append(linear_var_dim)

            gfeats1 = gfeats[:, :-1, ...].view(self.n_samples, self.n_linspace-1, mat_dim * mat_dim)
            gfeats2 = gfeats[:, 1:, ...].view(self.n_samples, self.n_linspace-1, mat_dim * mat_dim)
            group_dis = torch.linalg.norm(gfeats1 - gfeats2, dim=-1).view(self.n_samples, self.n_linspace-1 )
            group_var_dim = group_dis.sum(-1).mean(0)
            group_var_dim_ls.append(group_var_dim)
            print({'v_property/percept_dim_ls': percept_dim_ls,
                'v_property/linear_var_dim_ls': linear_var_dim_ls,
                'v_property/group_var_dim_ls': group_var_dim_ls})


        return {'v_property/percept_dim_ls': percept_dim_ls,
                'v_property/linear_var_dim_ls': linear_var_dim_ls,
                'v_property/group_var_dim_ls': group_var_dim_ls}

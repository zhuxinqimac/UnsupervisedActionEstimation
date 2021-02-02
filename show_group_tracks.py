#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: show_group_tracks.py
# --- Creation Date: 31-01-2021
# --- Last Modified: Mon 01 Feb 2021 14:28:34 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Show group tracks with projectors.
"""

import os
import argparse
import ast
import io
import torch
from torchvision.utils import save_image, make_grid
from datasets.datasets import datasets, set_to_loader, dataset_meta
from trainer import run
from utils import _str_to_list_of_int, _str_to_list_of_str, _str_to_bool
from models.models import models
from models.utils import count_parameters, model_loader

def get_traversal_latents(args, model):
    dims_to_walk = args.group_track_dim_to_walk
    limits = args.group_track_trav_limits
    steps = args.group_track_trav_steps
    linspace = torch.linspace(*limits, steps=steps)
    # normspace = (linspace.pow(7)) / (limits[-1]**6)
    normspace = linspace
    # print(linspace.pow(3))
    # print(limits[-1]**3)
    # print(normspace)

    x = torch.zeros(len(dims_to_walk), steps, args.latents).cuda()
    y = torch.zeros(len(dims_to_walk), steps)

    ind = 0
    for i in dims_to_walk:
        x[ind, :, i] = normspace
        y[ind, :] = ind
        ind += 1

    model.eval()
    with torch.no_grad():
        x = x.flatten(0, 1)
        y = y.flatten().cpu().numpy()
        imgs_pre_act, gfeats = model.decoder(x)
        imgs = imgs_pre_act.sigmoid()
        x = x.cpu().numpy()
        gfeats = gfeats.flatten(1, -1).cpu().numpy()

    if not os.path.isdir(args.group_track_output_dir):
        os.makedirs(args.group_track_output_dir)

    save_image(imgs, os.path.join(args.group_track_output_dir, 'traversals.png'), normalize=False, nrow=steps)
    imgs = imgs.flatten(1, -1).cpu().numpy()

    out_latents = io.open(os.path.join(args.group_track_output_dir, 'latents.tsv'), 'w', encoding='utf-8')
    out_gfeats = io.open(os.path.join(args.group_track_output_dir, 'gfeats.tsv'), 'w', encoding='utf-8')
    out_imgs = io.open(os.path.join(args.group_track_output_dir, 'imgs.tsv'), 'w', encoding='utf-8')
    out_meta = io.open(os.path.join(args.group_track_output_dir, 'metadata.tsv'), 'w', encoding='utf-8')

    # for index, word in enumerate(vocab):
        # if  index == 0: continue # skip 0, it's padding.
        # vec = weights[index]
        # out_latents.write('\t'.join([str(x) for x in vec]) + "\n")
        # out_meta.write(word + "\n")
    for i, latent in enumerate(x):
        gfeat = gfeats[i]
        img = imgs[i]
        out_latents.write('\t'.join([str(j) for j in latent]) + '\n')
        out_gfeats.write('\t'.join([str(j) for j in gfeat]) + '\n')
        out_imgs.write('\t'.join([str(j) for j in img]) + '\n')
        out_meta.write(str(y[i]) + '\n')

    out_latents.close()
    out_gfeats.close()
    out_imgs.close()
    out_meta.close()


parser = argparse.ArgumentParser('Group_projector')
# Basic Training Args
parser.add_argument('--epochs', default=50, type=int)
parser.add_argument('--model', default='forward', type=str, choices=['beta_shapes', 'beta_celeb', 'forward', 'rgrvae', 'dip_vae_i', 'dip_vae_ii', 'beta_forward', 'dforward', 'lie_group', 'lie_group_action', 'lie_group_action_simple', 'lie_group_rl'])
parser.add_argument('--dataset', default='flatland', type=str, choices=['flatland', 'dsprites', 'teapot', 'teapot_nocolor', 'shapes3d'])
parser.add_argument('--fixed_shape', default=None, type=int, help='Fixed shape in dsprites. None for not fixed.')
parser.add_argument('--data-path', default=None, type=str, help='Path to dataset root')
parser.add_argument('--latents', default=4, type=int, help='Number of latents')
parser.add_argument('--batch-size', default=128, type=int)
parser.add_argument('--split', default=0.1, type=float, help='Validation split fraction')
parser.add_argument('--shuffle', default=True, type=ast.literal_eval, help='Shuffle dataset')
parser.add_argument('--lr-scheduler', default='none', choices=['exp', 'none'], type=str)
parser.add_argument('--lr-scheduler-gamma', default=0.99, type=float, help='Exponential lr scheduler gamma')

# Model Loading
parser.add_argument('--base-model', default='beta_forward', type=str, help='Base model for rgrvae, dforward')
parser.add_argument('--base-model-path', default=None, help='Path to base model state which is to be loaded')
parser.add_argument('--load-model', default=False, type=ast.literal_eval, help='Continue training by loading model')
parser.add_argument('--log-path', default=None, type=str, help='Path from which to load model')
parser.add_argument('--global-step', default=None, help='Set the initial logging step value', type=int)

# Learning Rates
parser.add_argument('--learning-rate', '-lr', default=1e-3, type=float, help='Learning rate')
parser.add_argument('--group-learning-rate', default=0.1, type=float, help='Learning rate for internal rgrvae matrix representations')
parser.add_argument('--policy-learning-rate', default=None, type=float, help='Learning rate for policy network')

# Hyperparams
parser.add_argument('--beta', default=1., type=float, help='Beta vae beta')
parser.add_argument('--capacity', default=None, type=float, help='KL Capacity')
parser.add_argument('--capacity-leadin', default=100000, type=int, help='KL capacity leadin')
parser.add_argument('--gvae-gamma', default=None, type=float, help='Weighting for prediction loss. =beta if None')

# Metrics And Vis
parser.add_argument('--visualise', default=True, type=ast.literal_eval, help='Do visualisations')
parser.add_argument('--metrics', default=False, type=ast.literal_eval, help='Calculate disentanglement metrics at each step')
parser.add_argument('--end-metrics', default=True, type=ast.literal_eval, help='Calculate disentanglement metrics at end of run')
parser.add_argument('--evaluate', default=False, type=ast.literal_eval, help='Only run evalulation')

# RGrVAE Args
parser.add_argument('--offset', default=1, type=int, help='Generative factor offset for each action')
parser.add_argument('--use-regret', default=False, type=ast.literal_eval, help='Use regret on reinforce models')
parser.add_argument('--base-policy-weight', default=1., type=float, help='Base weight at which to apply policy over random choice')
parser.add_argument('--base-policy-epsilon', default=0.0000, type=float)
parser.add_argument('--group-structure', default=['c+', 'c-'], type=str, nargs='+', help='Group structure per latent pair in group vae. Options in models/actions: GroupWiseAction')
parser.add_argument('--num-action-steps', default=1, type=int, help='Number of action steps to allow Reinforced GroupVAE')
parser.add_argument('--multi-action-strategy', default='reward', choices=['reward', 'returns'], help='Strategy for reinforcing multiple actions.')
parser.add_argument('--reinforce-discount', default=0.99, type=float, help='Discount factor for reinforce rewards/returns')
parser.add_argument('--use-cob', default=False, type=ast.literal_eval, help='Use change of basis for representaitons in gvae')
parser.add_argument('--entropy-weight', default=0.01, type=float, help='Entropy weight for RL exploration')
parser.add_argument('--noise-name', default=None, choices=['BG', 'Salt', 'Gaussian'])

# ForwardVAE Params
parser.add_argument('--pred-z-loss-type', default='latent', type=str, help='Pred loss in ForwardVAE.')

# Lie Group Model
parser.add_argument('--subgroup_sizes_ls', default=[25, 25, 25, 25], type=_str_to_list_of_int, help='Subgroup sizes list for subspace group vae')
parser.add_argument('--subspace_sizes_ls', default=[1, 1, 1, 1], type=_str_to_list_of_int, help='Subspace sizes list for subspace group vae')
parser.add_argument('--lie_alg_init_type_ls', default=['none', 'none', 'none', 'none'], type=_str_to_list_of_str, help='Lie_alg init type list for subspace group vae')
parser.add_argument('--hy_ncut', default=0, type=int, help='Hyper-param for number of cuts in LieGroupVAE.')
parser.add_argument('--normalize_alg', default=False, type=_str_to_bool, help='If normalize alg in norm_vae')
parser.add_argument('--use_alg_var', default=False, type=_str_to_bool, help='If use alg_var in norm_vae')
parser.add_argument('--lie_alg_init_scale', help='Hyper-param for lie_alg_init_scale.', default=0.001, type=float)
parser.add_argument('--hy_hes', default=0, type=float, help='Hyper-param for Hessian in LieVAE.')
parser.add_argument('--hy_rec', default=1, type=float, help='Hyper-param for gfeats reconstruction in GroupVAE and LieVAE.')
parser.add_argument('--hy_commute', default=0, type=float, help='Hyper-param for commutator in GroupVAE-v3.')
parser.add_argument('--cycle_latent', default=False, type=_str_to_bool, help='If use cycle_latent in LieGroupVAE.')
parser.add_argument('--cycle_limit', default=4., type=float, help='The limit of cycle_latent in LieGroupVAE.')
parser.add_argument('--cycle_prob', default=0.1, type=float, help='The limit of cycle_latent in LieGroupVAE.')
parser.add_argument('--forward_eg_prob', default=0.6667, type=float, help='The prob to forward eg in LieGroupVAE.')
parser.add_argument('--recons_loss_type', default='l2', choices=['l2', 'bce'], type=str, help='The reconstruction type for x.')
parser.add_argument('--no_exp', default=False, type=_str_to_bool, help='If deactivate exp_mapping in LieGroupVAE.')

# Lie Group Action Model
parser.add_argument('--num_actions', default=4, type=int, help='Hyper-param for number of actions in LieAction.')
parser.add_argument('--loss_type', default='on_group', type=str, help='Loss type.')

# Lie Group RL Model
parser.add_argument('--supervised_train', default=False, type=_str_to_bool, help='If use action supervision in LieGroupVAERL.')
parser.add_argument('--hy_st', default=0., type=float, help='Hyper-param for action strength LieGroupVAERL.')

# Group Track Args
parser.add_argument('--group_track_checkpoint', default='here/here', type=str)
parser.add_argument('--group_track_output_dir', default='here/here', type=str)
parser.add_argument('--group_track_trav_limits', default=[-2, 2], type=_str_to_list_of_int)
parser.add_argument('--group_track_trav_steps', default=10, type=int)
parser.add_argument('--group_track_dim_to_walk', default=[0, 1, 2, 3], type=_str_to_list_of_int)
args = parser.parse_args()

args.nc, args.factors = dataset_meta[args.dataset]['nc'], dataset_meta[args.dataset]['factors']
model_state, _ = model_loader(args.group_track_checkpoint)

model = models[args.model](args)

model.load_state_dict(model_state)
model.cuda()

get_traversal_latents(args, model)

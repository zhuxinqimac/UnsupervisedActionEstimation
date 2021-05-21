#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: instant_var_analysis.py
# --- Creation Date: 24-04-2021
# --- Last Modified: Fri 21 May 2021 21:57:11 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Docstring
"""

import argparse
import os
import pdb
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from models.models import models
from models.utils import model_loader
from utils import _str_to_list_of_int, _str_to_list, _str_to_bool, EasyDict

import own_lpips

def load_model(args):
    checkpoint_path = os.path.join(args.ckpt_path, 'checkpoints')
    model_state, old_args = model_loader(checkpoint_path)

    model = models[old_args.model](old_args)
    model.load_state_dict(model_state)
    model.z_dim = old_args.latents
    model.cuda()
    return model

loss_fn_alex = own_lpips.OwnLPIPS(net='alex', lpips=False).cuda()
loss_fn_vgg = own_lpips.OwnLPIPS(net='vgg', lpips=False).cuda()

def get_colors(colors_def):
    # by_hsv = sorted((tuple(mcolors.rgb_to_hsv(mcolors.to_rgb(color))),
                     # name)
                    # for name, color in colors_def.items())
    # names = [name for hsv, name in by_hsv]
    names = list(colors_def.keys())
    print('color names:', names)
    return names[::-1]

def l2_var(x1, x2, args):
    '''
    x1, x2: (b, c, h, w) of torch.tensor
    return: (b)
    '''
    return ((x1 - x2)**2).sum(dim=[1,2,3]).sqrt()

def l1_var(x1, x2, args):
    '''
    x1, x2: (b, c, h, w) of torch.tensor
    return: (b)
    '''
    return (x1 - x2).abs().sum(dim=[1,2,3])

def lpips_vgg_var(x1, x2, args):
    with torch.no_grad():
        dist = loss_fn_vgg.forward(x1, x2, distance=not(args.direction), direction=args.direction,
                                   normalize=True, spatial=args.spatial, target_layer=args.target_layer).squeeze() # (b, (...))
    return dist

def lpips_alex_var(x1, x2, args):
    with torch.no_grad():
        dist = loss_fn_alex.forward(x1, x2, distance=not(args.direction), direction=args.direction,
                                    normalize=True, spatial=args.spatial, target_layer=args.target_layer).squeeze() # (b, (...))
    return dist

metrics = {'l2': l2_var, 'l1': l1_var, 'lpips_alex': lpips_alex_var,
           'lpips_vgg': lpips_vgg_var}
colors = get_colors(mcolors.TABLEAU_COLORS)
# colors = get_colors(mcolors.BASE_COLORS)

def compute_inst_x_var_by_metric_for_z_pairs(z_all, z_all_var, var_metric, model, args):
    z_all = z_all.split(args.batch_size, dim=0)
    z_all_var = z_all_var.split(args.batch_size, dim=0)
    x_var_ls = []
    for i, z in enumerate(z_all):
        z_var = z_all_var[i]
        with torch.no_grad():
            x1 = model.decode(z).sigmoid()
            x2 = model.decode(z_var).sigmoid()
        x_var_ls.append(metrics[var_metric](x1, x2, args).detach().cpu())
    x_var = torch.cat(x_var_ls, dim=0).unsqueeze(0).numpy() # (1, n_samples_per_dim, ...)
    return x_var

def compute_inst_x_var_by_metric_for_dim(var_metric, dim_i, model, args):
    z_all = torch.randn(args.n_samples_per_dim, model.z_dim).to('cuda') # (n, z_dim)
    z_all[:, dim_i] = z_all[:, dim_i] * 0.8 - args.epsilon * 0.5
    z_all_var = z_all.clone()
    z_all_var[:, dim_i] += args.epsilon
    return compute_inst_x_var_by_metric_for_z_pairs(z_all, z_all_var, var_metric, model, args)

def compute_inst_x_var_by_metric_for_all_dims(var_metric, model, args):
    result_dict = EasyDict({'var_metric': var_metric})
    assert max(args.shown_dims) < model.z_dim
    assert min(args.shown_dims) >= 0
    result_dict.shown_dims = args.shown_dims
    x_var_ls = []
    for i in args.shown_dims:
        x_var_tmp = compute_inst_x_var_by_metric_for_dim(var_metric, i, model, args)
        # print('x_var_tmp.shape:', x_var_tmp.shape)
        x_var_ls.append(x_var_tmp)
        # print('x_var_tmp:', x_var_tmp[0, :10])
    result_dict.x_var = np.concatenate(x_var_ls, axis=0) # (shown_dims, n_samples_per_dim, ...)
    # print('x_var.shape:', result_dict.x_var.shape)
    assert result_dict.x_var.shape[0] == len(args.shown_dims)
    assert result_dict.x_var.shape[1] == args.n_samples_per_dim
    return result_dict

def compute_inst_x_var_by_metric_for_anchor(var_metric, anchor_i, anchor_point, model, args):
    z_all = anchor_point.repeat(args.n_samples_per_anchor, 1)
    z_all[:, args.shown_dim] = torch.randn(args.n_samples_per_anchor).to('cuda') * 0.8 - args.epsilon * 0.5 # (n, z_dim)
    z_all_var = z_all.clone()
    z_all_var[:, args.shown_dim] += args.epsilon
    return compute_inst_x_var_by_metric_for_z_pairs(z_all, z_all_var, var_metric, model, args)

def compute_inst_x_var_by_metric_for_all_anchors(var_metric, model, args):
    result_dict = EasyDict({'var_metric': var_metric})
    result_dict.shown_dim = args.shown_dim
    x_var_ls = []
    for i in range(args.n_anchors):
        anchor_point = torch.randn(1, model.z_dim).to('cuda')
        x_var_tmp = compute_inst_x_var_by_metric_for_anchor(var_metric, i, anchor_point, model, args)
        # print('x_var_tmp.shape:', x_var_tmp.shape)
        x_var_ls.append(x_var_tmp)
        # print('x_var_tmp:', x_var_tmp[0, :10])
    result_dict.x_var = np.concatenate(x_var_ls, axis=0) # (n_anchors, n_samples_per_dim, ...)
    return result_dict

def create_title(args, var_metric):
    if args.command == 'inst-var-all-dim':
        title = '{}'.format(args.command)
    elif args.command == 'inst-var-one-dim':
        title = '{}:{}'.format(args.command, args.concept)
    return title

def get_rel_cos_diff(x_var):
    '''
    x_var: (n_dims/n_anchors, n_samples, feat)
    return: cos similarity between x_var and its mean.
    '''
    cos_fn = torch.nn.CosineSimilarity(dim=2)
    x_var = torch.tensor(x_var)
    x_var_avg = x_var.mean(dim=1, keepdim=True)
    print('x_var.shape:', x_var.shape)
    print('x_var_avg.shape:', x_var_avg.shape)
    res = cos_fn(x_var, x_var_avg)
    return res.detach().cpu().numpy()

def plot_hist_all_sets(in_dict, args):
    x_var = in_dict.x_var # (n_dims/n_anchors, n_samples, ...)
    if args.direction:
        x_var = get_rel_cos_diff(x_var)
    assert x_var.ndim == 2
    fig, ax = plt.subplots(figsize=(7,5))
    n_bins = args.n_bins
    # print('x_var.max:', x_var.max())
    # print('x_var.min:', x_var.min())
    print('n_bins:', n_bins)
    labels = args.concepts if args.command == 'inst-var-all-dim' else [str(i) for i in range(args.n_anchors)]
    ax.hist([x_var_i for x_var_i in x_var], n_bins,
            density=True, log=True,
            histtype='stepfilled', color=colors[:x_var.shape[0]], alpha=0.75, label=labels)
    ax.legend(prop={'size': 10})
    ax.set_title(create_title(args, in_dict.var_metric))
    fig.tight_layout()
    plt.show()

def get_cos_grid(x_var, args):
    '''
    x_var: (n_dims/n_anchors, n_samples, feat)
    return: cos similarity between x_var and its mean.
    '''
    cos_fn = torch.nn.CosineSimilarity(dim=2)
    x_var = torch.tensor(x_var)
    x_var_avg = x_var.mean(dim=1) # (n_dims/n_anchors, feat, (h, w))
    x_var_avg_1 = x_var_avg[:, np.newaxis, ...] # (n_dims/n_anchors, 1, feat, (h, w))
    x_var_avg_2 = x_var_avg[np.newaxis, ...] # (1, n_dims/n_anchors, feat, (h, w))
    res = cos_fn(x_var_avg_1, x_var_avg_2)
    print('res.shape:', res.shape) # (n_dims, n_dims, (h, w))
    if res.ndim == 4:
        x_feat_norm = torch.linalg.norm(x_var_avg, dim=1) # (n_dims, h, w)
        print('x_feat_norm.shape:', x_feat_norm.shape)

        # == Hard Thresh for Activations
        # # x_mask = (x_feat_norm > 0.01).float() # (n_dims/n_anchors, (h, w)) for epsilon==1e-3
        # print('>1:', (x_feat_norm>1).sum())
        # print('>1.5:', (x_feat_norm>1.5).sum())
        # print('>2:', (x_feat_norm>2).sum())
        # print('>2.5:', (x_feat_norm>2.5).sum())
        # print('>3:', (x_feat_norm>3).sum())
        # print('>3.5:', (x_feat_norm>3.5).sum())
        # print('>4:', (x_feat_norm>4).sum())
        # print('>4.5:', (x_feat_norm>4.5).sum())
        # print('>5:', (x_feat_norm>5).sum())
        # print('>5.5:', (x_feat_norm>5.5).sum())
        # print('>6:', (x_feat_norm>6).sum())
        # print('>6.5:', (x_feat_norm>6.5).sum())
        # print('>7:', (x_feat_norm>7).sum())
        # print('>7.5:', (x_feat_norm>7.5).sum())
        # # x_mask = (x_feat_norm > 4.5).float() # (n_dims/n_anchors, (h, w)) for epsilon==1e-1 for one_dim orient
        # # x_mask = (x_feat_norm > 9).float() # (n_dims/n_anchors, (h, w)) for epsilon==1e-1 for one_dim wall
        # # x_mask = (x_feat_norm > 2.5).float() # (n_dims/n_anchors, (h, w)) for epsilon==1e-1 for all_dim
        # x_mask = (x_feat_norm > 0).float() # (n_dims/n_anchors, (h, w)) for epsilon==1e-1 for one_dim test

        # == Mask by Norm
        n_dims, h, w = x_feat_norm.size()
        x_feat_norm_viewed = x_feat_norm.view(n_dims, h * w)

        # Mask from max and min
        numerator = x_feat_norm_viewed - x_feat_norm_viewed.min(dim=1, keepdim=True)[0]
        denominator = x_feat_norm_viewed.max(dim=1, keepdim=True)[0] - x_feat_norm_viewed.min(dim=1, keepdim=True)[0]
        print('numerator.shape:', numerator.shape)
        print('denominator.shape:', denominator.shape)
        x_mask = (numerator / (denominator)).view(n_dims, h, w)
        assert x_mask.max() == 1
        assert x_mask.min() == 0

        # Mask from softmax
        # x_mask = F.softmax(x_feat_norm * 100, dim=-1).view(n_dims, h, w)

        # Mask from norm
        # x_mask = x_feat_norm.clone().view(n_dims, h, w) - x_feat_norm_viewed.max(dim=1)[0].view(n_dims, 1, 1)

        x_mask1 = x_mask[:, np.newaxis, ...]
        x_mask2 = x_mask[np.newaxis, ...]
        x_outer_mask = x_mask1 * x_mask2 # (n_dims, n_dims, (h, w))
        assert x_outer_mask.size() == res.size()

        if args.use_mask:
            res = res * x_outer_mask
            res = res.sum(dim=[2, 3]) / (x_outer_mask.sum(dim=[2, 3]) + 1e-6)
        else:
            res = res.mean(dim=[2, 3])
    return res.detach().cpu().numpy()

def plot_cos_grid(in_dict, args):
    x_var = in_dict.x_var # (n_dims/n_anchors, n_samples, feat, (h, w))
    x_val = np.absolute(get_cos_grid(x_var, args)) # (n_dims, n_dims)
    print('x_val.shape:', x_val.shape)
    # print('x_val:', x_var) # (n_dims, n_dims)

    if args.command == 'inst-var-all-dim':
        assert x_val.shape[0] == len(args.concepts)
        ticklabels = args.concepts
    else:
        assert x_val.shape[0] == args.n_anchors
        ticklabels = list(range(args.n_anchors))
    fig, ax = plt.subplots(figsize=(8,8))
    im = ax.imshow(x_val, vmin=0, vmax=1)
    ax.set_xticks(np.arange(x_val.shape[0]))
    ax.set_yticks(np.arange(x_val.shape[0]))
    ax.set_xticklabels(ticklabels)
    ax.set_yticklabels(ticklabels)

    # Loop over data dimensions and create text annotations.
    # for i in range(len(ticklabels)):
        # for j in range(len(ticklabels)):
            # text = ax.text(j, i, round(x_val[i, j], 2), ha="center", va="center", color="w")
    # fig.colorbar(im)

    ax.set_title('Heatmap:'+create_title(args, in_dict.var_metric))
    fig.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Plot instant variation distributions of a generator.')
    subparsers = parser.add_subparsers(help='Sub-commands', dest='command')

    parser_all_dim = subparsers.add_parser('inst-var-all-dim', help='Plot along all latent dimensions.')
    parser_all_dim.add_argument('--ckpt_path', help='Model checkpoint_path.', type=str, default='/mnt/hdd/repo_results/test')
    parser_all_dim.add_argument('--result_dir', help='Results directory.', type=str, default='/mnt/hdd/repo_results/test')
    parser_all_dim.add_argument('--shown_dims', help='The latent dimensions to plot.', type=_str_to_list_of_int, default='[0,1,2,3,4,5,6,7,8,9]')
    parser_all_dim.add_argument('--concepts', help='The concepts corresponding to the shown_dims.', type=_str_to_list, default='[0,1,2,3,4,5,6,7,8,9]')
    parser_all_dim.add_argument('--n_samples_per_dim', help='The sampling size for evaluate instant var on each dimension.', type=int, default=1000)
    parser_all_dim.add_argument('--batch_size', help='The batch size for model forwarding.', type=int, default=100)
    parser_all_dim.add_argument('--epsilon', help='The epsilon for z instant var.', type=float, default=1e-4)
    parser_all_dim.add_argument('--x_var_metric', help='The metric for evaluate x instant var.', type=str, default='l2', choices=['l2', 'l1', 'lpips_alex', 'lpips_vgg'])
    parser_all_dim.add_argument('--hist_bin', help='The bin size for histogram plot.', type=float, default=3)
    parser_all_dim.add_argument('--n_bins', help='The number of bins for histogram plot.', type=int, default=30)
    parser_all_dim.add_argument('--direction', help='If compute direction var in lpips.', type=_str_to_bool, default=False)
    parser_all_dim.add_argument('--spatial', help='If compute direction var with spatial in lpips.', type=_str_to_bool, default=False)
    parser_all_dim.add_argument('--cos_grid', help='If plot cosine grid instead of histogram.', type=_str_to_bool, default=False)
    parser_all_dim.add_argument('--use_mask', help='If use mask in cosine grid.', type=_str_to_bool, default=True)
    parser_all_dim.add_argument('--target_layer', help='The target layer in lpips.', type=int, default=None)

    parser_one_dim = subparsers.add_parser('inst-var-one-dim', help='Plot along one latent dimensions.')
    parser_one_dim.add_argument('--ckpt_path', help='Model checkpoint_path.', type=str, default='/mnt/hdd/repo_results/test')
    parser_one_dim.add_argument('--result_dir', help='Results directory.', type=str, default='/mnt/hdd/repo_results/test')
    parser_one_dim.add_argument('--shown_dim', help='The latent dimension to plot.', type=int, default=0)
    parser_one_dim.add_argument('--concept', help='The concept corresponding to the shown_dim.', type=str, default='None')
    parser_one_dim.add_argument('--n_anchors', help='The number of anchor points to show.', type=int, default=5)
    parser_one_dim.add_argument('--n_samples_per_anchor', help='The sampling size for evaluate instant var on each anchor.', type=int, default=1000)
    parser_one_dim.add_argument('--batch_size', help='The batch size for model forwarding.', type=int, default=100)
    parser_one_dim.add_argument('--epsilon', help='The epsilon for z instant var.', type=float, default=1e-4)
    parser_one_dim.add_argument('--x_var_metric', help='The metric for evaluate x instant var.', type=str, default='l2', choices=['l2', 'l1', 'lpips_alex', 'lpips_vgg'])
    # parser_one_dim.add_argument('--hist_bin', help='The bin size for histogram plot.', type=float, default=3)
    parser_one_dim.add_argument('--n_bins', help='The number of bins for histogram plot.', type=int, default=30)
    parser_one_dim.add_argument('--direction', help='If compute direction var in lpips.', type=_str_to_bool, default=False)
    parser_one_dim.add_argument('--spatial', help='If compute direction var with spatial in lpips.', type=_str_to_bool, default=False)
    parser_one_dim.add_argument('--cos_grid', help='If plot cosine grid instead of histogram.', type=_str_to_bool, default=False)
    parser_one_dim.add_argument('--use_mask', help='If use mask in cosine grid.', type=_str_to_bool, default=True)
    parser_one_dim.add_argument('--target_layer', help='The target layer in lpips.', type=int, default=None)

    args = parser.parse_args()

    model = load_model(args)

    if args.command == 'inst-var-all-dim':
        result_dict = compute_inst_x_var_by_metric_for_all_dims(args.x_var_metric, model, args)
    elif args.command == 'inst-var-one-dim':
        result_dict = compute_inst_x_var_by_metric_for_all_anchors(args.x_var_metric, model, args)

    if args.direction and args.cos_grid:
        plot_cos_grid(result_dict, args)
    else:
        plot_hist_all_sets(result_dict, args)


if __name__ == "__main__":
    main()

#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: diffdim_vae.py
# --- Creation Date: 12-05-2021
# --- Last Modified: Thu 13 May 2021 23:58:47 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
DiffDim VAE model.
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

class DiffDimVAE(VAE):
    def __init__(self, encoder, decoder, args):
        super().__init__(encoder, decoder, args.beta, args.capacity, args.capacity_leadin)
        self.latents = args.latents
        self.train_lpips = args.train_lpips
        if self.train_lpips:
            self.lpips = lpips.LPIPS(net=args.lpips_net, lpips=False, eval_mode=False, pnet_tune=True).net
        else:
            self.lpips = lpips.LPIPS(net=args.lpips_net, lpips=False).net
        self.var_sample_scale = args.var_sample_scale
        self.norm_on_depth = args.norm_on_depth
        self.S_L = 7 if self.lpips == 'squeeze' else 5
        self.sensor_used_layers = args.sensor_used_layers
        self.norm_lambda = args.norm_lambda
        self.use_norm_mask = args.use_norm_mask
        self.divide_mask_sum = args.divide_mask_sum
        self.cos_fn = nn.CosineSimilarity(dim=1)
        self.diff_lambda = args.diff_lambda
        self.diff_capacity_leadin = args.diff_capacity_leadin
        self.diff_capacity = args.diff_capacity
        self.use_dynamic_scale = args.use_dynamic_scale
        self.detach_qpn = args.detach_qpn
        if args.xav_init:
            for p in self.encoder.modules():
                if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear) or \
                        isinstance(p, nn.ConvTranspose2d):
                    torch.nn.init.xavier_uniform_(p.weight)
            for p in self.decoder.modules():
                if isinstance(p, nn.Conv2d) or isinstance(p, nn.Linear) or \
                        isinstance(p, nn.ConvTranspose2d):
                    torch.nn.init.xavier_uniform_(p.weight)

        print(list(self.decoder.parameters())[0].shape)
        self.aux_opt_type = args.aux_opt_type
        if self.aux_opt_type == 'decoder_first':
            self.diff_opt = optim.Adam(list(self.decoder.parameters())[:1], lr=1e-4)
        elif self.aux_opt_type == 'encoder_last':
            self.diff_opt = optim.Adam(list(self.encoder.parameters())[-1:], lr=1e-4)
        elif self.aux_opt_type == 'both':
            self.diff_opt = optim.Adam(list(self.encoder.parameters())[-1:]+list(self.decoder.parameters())[:1], lr=1e-4)
        else:
            raise ValueError('Unknown aux_opt_type:', self.aux_opt_type)

    def diff_control_capacity(self, loss_diff, global_step, logs):
        if self.diff_capacity is not None:
            leadin = 1e5 if self.diff_capacity_leadin is None else self.diff_capacity_leadin
            leadin_delta = torch.tensor((self.diff_capacity / leadin) * global_step).clamp(max=self.diff_capacity)
            logs.update({'metric/leadin': leadin_delta})
            return loss_diff * self.diff_lambda * leadin_delta, logs
        else:
            return loss_diff * self.diff_lambda, logs

    def main_step(self, batch, batch_nb, loss_fn):

        x, y = batch
        batch_size = x.size(0)

        logs = {}
        if self.diff_lambda != 0 and self.training:
            mu, lv = self.unwrap(self.encode(x))
            z = self.reparametrise(mu, lv)

            z_q, z_pos, z_neg = self.get_q_pos_neg(z)

            if self.detach_qpn:
                z_all = torch.cat([z, z_q.detach(), z_pos.detach(), z_neg.detach()], dim=0)
            else:
                z_all = torch.cat([z, z_q, z_pos, z_neg], dim=0)
            x_all_hat = self.decode(z_all)

            loss_diff, logs = self.get_diff_loss(x_all_hat, logs)
            loss_diff, logs = self.diff_control_capacity(loss_diff, self.global_step, logs)
            logs.update({'metric/diff_loss': loss_diff})
            loss_diff *= self.diff_lambda

            self.diff_opt.zero_grad()
            loss_diff.backward()
            self.diff_opt.step()

        mu, lv = self.unwrap(self.encode(x))
        z = self.reparametrise(mu, lv)
        x_hat = self.decode(z)

        loss_recons = loss_fn(x_hat, x)
        total_kl = self.compute_kl(mu, lv, mean=False)
        beta_kl = self.control_capacity(total_kl, self.global_step, self.anneal)
        state = self.make_state(batch_nb, x_hat, x, y, mu, lv, z)

        loss = loss_recons + beta_kl

        tensorboard_logs = {'metric/loss': loss, 'metric/recon_loss': loss_recons,
                            'metric/total_kl': total_kl, 'metric/beta_kl': beta_kl}
        tensorboard_logs.update(logs)
        print('tensorboard_logs:', tensorboard_logs)

        self.global_step += 1
        return {'loss': loss, 'out': tensorboard_logs, 'state': state}

    def sample_batch_pos_neg_dirs(self, batch, z_dim):
        rand = torch.rand(batch, z_dim)
        z_dim_perm = rand.argsort(dim=1) # (b, z_dim)
        return z_dim_perm[:, :2]

    def get_q_pos_neg(self, z):
        '''
        z: (b, latents)
        '''
        batch_size = z.size(0)
        pos_neg_idx = self.sample_batch_pos_neg_dirs(batch_size // 2, self.latents).to(z.device) # (b//2, 2)
        pos_onehot = F.one_hot(pos_neg_idx[:, 0], self.latents) # (b//2, latents)
        neg_onehot = F.one_hot(pos_neg_idx[:, 1], self.latents) # (b//2, latents)
        if self.use_dynamic_scale:
            var_sample_scale = torch.rand(pos_onehot.size(), device=z.device) * self.var_sample_scale
        else:
            var_sample_scale = self.var_sample_scale
        z_q = var_sample_scale * pos_onehot + z[:batch_size//2]
        z_pos = var_sample_scale * pos_onehot + z[batch_size//2:]
        z_neg = var_sample_scale * neg_onehot + z[batch_size//2:] # (b//2, latents)
        return z_q, z_pos, z_neg

    def get_diff_loss(self, imgs_all, logs):
        outs_all = self.lpips.forward(imgs_all.sigmoid()*2-1)
        loss_diff, logs = self.extract_diff_loss(outs_all, logs)
        return loss_diff, logs

    def get_norm_mask(self, diff):
        # norm = torch.linalg.norm(diff, dim=1) # (0.5batch, h, w)
        norm = torch.norm(diff, dim=1) # (0.5batch, h, w)
        b_half, h, w = norm.size()
        norm_viewed = norm.view(b_half, h * w)
        numerator = norm_viewed - norm_viewed.min(dim=1, keepdim=True)[0]
        denominator = norm_viewed.max(dim=1, keepdim=True)[0] - norm_viewed.min(dim=1, keepdim=True)[0] + 1e-6
        # print('numerator.shape:', numerator.shape)
        # print('denominator.shape:', denominator.shape)
        mask = (numerator / denominator).view(b_half, h, w)
        return norm, mask

    def extract_diff_L(self, feats_i):
        # (2.5 * batch, c, h, w)
        batch_25 = feats_i.size(0)
        p1_s = feats_i[:batch_25//5]
        p2_s = feats_i[batch_25//5:2*batch_25//5]
        p1_e = feats_i[2*batch_25//5:3*batch_25//5]
        p2_e_pos = feats_i[3*batch_25//5:4*batch_25//5]
        p2_e_neg = feats_i[4*batch_25//5:]
        diff_q = p1_e - p1_s # (0.5batch, c, h, w)
        diff_pos = p2_e_pos - p2_s
        diff_neg = p2_e_neg - p2_s
        return diff_q, diff_pos, diff_neg

    def extract_loss_L_by_maskdiff(self, diff_q, diff_pos, diff_neg, mask_q, mask_pos, mask_neg, idx, logs):
        mask_pos_comb = mask_q * mask_pos
        mask_neg_comb = mask_q * mask_neg

        if self.use_norm_mask:
            cos_sim_pos = self.cos_fn(diff_q, diff_pos) * mask_pos_comb
            cos_sim_neg = self.cos_fn(diff_q, diff_neg) * mask_neg_comb
            if self.divide_mask_sum:
                loss_pos = (-cos_sim_pos**2).sum(dim=[1,2]) / mask_pos_comb.sum(dim=[1,2]) # (0.5batch)
                loss_neg = (cos_sim_neg**2).sum(dim=[1,2]) / mask_neg_comb.sum(dim=[1,2])
            else:
                loss_pos = (-cos_sim_pos**2).sum(dim=[1,2]) # (0.5batch)
                loss_neg = (cos_sim_neg**2).sum(dim=[1,2])
        else:
            cos_sim_pos = self.cos_fn(diff_q, diff_pos)
            cos_sim_neg = self.cos_fn(diff_q, diff_neg)
            loss_pos = (-cos_sim_pos**2).mean(dim=[1,2])
            loss_neg = (cos_sim_neg**2).mean(dim=[1,2])
        loss_pos = loss_pos.mean()
        # training_stats.report('Loss/M/loss_diff_pos_{}'.format(idx), loss_pos)
        logs.update({'metric/M_loss_diff_pos_{}'.format(idx): loss_pos})
        loss_neg = loss_neg.mean()
        # training_stats.report('Loss/M/loss_diff_neg_{}'.format(idx), loss_neg)
        logs.update({'metric/M_loss_diff_neg_{}'.format(idx): loss_neg})
        loss = loss_pos + loss_neg # (0.5batch)
        return loss, logs

    def extract_loss_L(self, feats_i, idx, logs):
        diff_q, diff_pos, diff_neg = self.extract_diff_L(feats_i)

        norm_q, mask_q = self.get_norm_mask(diff_q) # (0.5batch, h, w), (0.5batch, h, w)
        norm_pos, mask_pos = self.get_norm_mask(diff_pos)
        norm_neg, mask_neg = self.get_norm_mask(diff_neg)
        assert mask_q.max() == 1
        assert mask_q.min() == 0

        loss_diff, logs = self.extract_loss_L_by_maskdiff(diff_q, diff_pos, diff_neg, mask_q, mask_pos, mask_neg, idx, logs)
        # training_stats.report('Loss/M/loss_diff_{}'.format(idx), loss_diff)
        logs.update({'metric/M_loss_diff_{}'.format(idx): loss_diff})
        if self.use_norm_mask:
            loss_norm = sum([(norm**2).sum(dim=[1,2]) / (mask.sum(dim=[1,2]) + 1e-6) \
                             for norm, mask in [(norm_q, mask_q), (norm_pos, mask_pos), (norm_neg, mask_neg)]])
        else:
            loss_norm = sum([(norm**2).sum(dim=[1,2]) \
                             for norm, mask in [(norm_q, mask_q), (norm_pos, mask_pos), (norm_neg, mask_neg)]])
        loss_norm = loss_norm.mean()
        # training_stats.report('Loss/M/loss_norm_{}'.format(idx), loss_norm)
        logs.update({'metric/M_loss_norm_{}'.format(idx): loss_norm})
        return loss_diff + self.norm_lambda * loss_norm, logs

    def extract_norm_mask_wdepth(self, diff_ls):
        norm_mask_ls, norm_ls, max_ls, min_ls = [], [], [], []
        for i, diff in enumerate(diff_ls):
            # diff: (0.5batch, ci, hi, wi)
            norm = torch.norm(diff, dim=1)
            b_half, h, w = norm.size()
            norm_viewed = norm.view(b_half, h * w)
            norm_max = norm_viewed.max(dim=1, keepdim=True)[0] # (b_half, 1)
            norm_min = norm_viewed.min(dim=1, keepdim=True)[0]
            norm_ls.append(norm) # (b_half, hi, wi)
            max_ls.append(norm_max)
            min_ls.append(norm_min)
        real_max = torch.cat(max_ls, dim=1).max(dim=1)[0] # (b_half)
        real_min = torch.cat(min_ls, dim=1).min(dim=1)[0]

        for i, norm in enumerate(norm_ls):
            numerator = norm - real_min.view(b_half, 1, 1)
            denominator = (real_max - real_min).view(b_half, 1, 1) + 1e-6
            mask = (numerator / denominator) # (b_half, hi, wi)
            norm_mask_ls.append(mask)
        return norm_ls, norm_mask_ls

    def extract_depth_diff_loss(self, diff_q_ls, diff_pos_ls, diff_neg_ls, mask_q_ls, mask_pos_ls, mask_neg_ls, logs):
        loss = 0
        for i, diff_q_i in enumerate(diff_q_ls):
            loss_i, logs = self.extract_loss_L_by_maskdiff(diff_q_i, diff_pos_ls[i], diff_neg_ls[i],
                                                           mask_q_ls[i], mask_pos_ls[i], mask_neg_ls[i], i, logs)
            loss += loss_i
        return loss, logs

    def extract_depth_norm_loss(self, norm_q_ls, norm_pos_ls, norm_neg_ls, mask_q_ls, mask_pos_ls, mask_neg_ls):
        loss = 0
        for i, norm_q in enumerate(norm_q_ls):
            if self.use_norm_mask:
                loss_norm = sum([(norm**2).sum(dim=[1,2])/(mask.sum(dim=[1,2]) + 1e-6) for norm, mask in \
                                 [(norm_q, mask_q_ls[i]), (norm_pos_ls[i], mask_pos_ls[i]), (norm_neg_ls[i], mask_neg_ls[i])]])
            else:
                loss_norm = sum([(norm**2).sum(dim=[1,2]) for norm, mask in \
                                 [(norm_q, mask_q_ls[i]), (norm_pos_ls[i], mask_pos_ls[i]), (norm_neg_ls[i], mask_neg_ls[i])]])
            loss += loss_norm
        return loss

    def extract_diff_loss(self, outs, logs):
        if not self.norm_on_depth:
            loss = 0
        else:
            diff_q_ls, diff_pos_ls, diff_neg_ls = [], [], []
        for kk in range(self.S_L - self.sensor_used_layers, self.S_L):
            if not self.norm_on_depth:
                loss_kk, logs = self.extract_loss_L(outs[kk], kk, logs)
                loss += loss_kk
            else:
                diff_q_kk, diff_pos_kk, diff_neg_kk = self.extract_diff_L(outs[kk])
                diff_q_ls.append(diff_q_kk)
                diff_pos_ls.append(diff_pos_kk)
                diff_neg_ls.append(diff_neg_kk)
        if self.norm_on_depth:
            norm_q_ls, mask_q_ls = self.extract_norm_mask_wdepth(diff_q_ls)
            norm_pos_ls, mask_pos_ls = self.extract_norm_mask_wdepth(diff_pos_ls)
            norm_neg_ls, mask_neg_ls = self.extract_norm_mask_wdepth(diff_neg_ls)
            loss_diff, logs = self.extract_depth_diff_loss(diff_q_ls, diff_pos_ls, diff_neg_ls,
                                                     mask_q_ls, mask_pos_ls, mask_neg_ls, logs)
            loss_diff = loss_diff.mean()
            # training_stats.report('Loss/M/loss_diff', loss_diff)
            logs.update({'metric/M_loss_diff': loss_diff})
            loss_norm = self.extract_depth_norm_loss(norm_q_ls, norm_pos_ls, norm_neg_ls, mask_q_ls, mask_pos_ls, mask_neg_ls)
            loss_norm = loss_norm.mean()
            # training_stats.report('Loss/M/loss_norm', loss_norm)
            logs.update({'metric/M_loss_norm': loss_norm})
            loss = loss_diff + self.norm_lambda * loss_norm
        return loss, logs

def diffdim_vae_64(args):
    from models.beta import beta_celeb_encoder, beta_celeb_decoder
    encoder, decoder = beta_celeb_encoder(args), beta_celeb_decoder(args)
    return DiffDimVAE(encoder, decoder, args)

#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: own_lpips.py
# --- Creation Date: 25-04-2021
# --- Last Modified: Mon 26 Apr 2021 16:55:07 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Own version of LPIPS.
"""

import torch
import torch.nn as nn
import lpips

def spatial_average(in_tens, keepdim=True):
    return in_tens.mean([2,3],keepdim=keepdim)

def upsample(in_tens, out_HW=(64,64)): # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.Upsample(size=out_HW, mode='bilinear', align_corners=False)(in_tens)

def interpolate(in_tens, out_HW=(64,64)): # assumes scale factor is same for H and W
    in_H, in_W = in_tens.shape[2], in_tens.shape[3]
    return nn.functional.interpolate(in_tens, size=out_HW, mode='bilinear', align_corners=False)

class OwnLPIPS(lpips.LPIPS):
    def __init__(self, pretrained=True, net='alex', version='0.1', lpips=True, spatial=False, pnet_rand=False,
                 pnet_tune=False, use_dropout=True, model_path=None, eval_mode=True, verbose=True):
        super().__init__(pretrained, net, version, lpips, spatial, pnet_rand,
                         pnet_tune, use_dropout, model_path, eval_mode, verbose)

    def forward(self, in0, in1, distance=True, direction=False, retPerLayer=False, normalize=False, spatial=False):
        assert not(distance and direction)
        if normalize: # turn on this flag if input is [0,1] so it can be adjusted to [-1, +1]
            in0 = 2 * in0  - 1
            in1 = 2 * in1  - 1

        # v0.0 - original release had a bug, where input was not scaled
        in0_input, in1_input = (self.scaling_layer(in0), self.scaling_layer(in1)) if self.version=='0.1' else (in0, in1)
        outs0, outs1 = self.net.forward(in0_input), self.net.forward(in1_input)
        feats0, feats1, diffs = {}, {}, {}

        if distance:
            for kk in range(self.L):
                feats0[kk], feats1[kk] = outs0[kk], outs1[kk]
                diffs[kk] = (feats0[kk]-feats1[kk])**2

            if(self.lpips):
                if(self.spatial):
                    res = [interpolate(self.lins[kk](diffs[kk]), out_HW=in0.shape[2:]) for kk in range(self.L)]
                else:
                    res = [spatial_average(self.lins[kk](diffs[kk]), keepdim=True) for kk in range(self.L)]
            else:
                if(self.spatial):
                    res = [interpolate(diffs[kk].sum(dim=1,keepdim=True), out_HW=in0.shape[2:]) for kk in range(self.L)]
                else:
                    res = [spatial_average(diffs[kk].sum(dim=1,keepdim=True), keepdim=True) for kk in range(self.L)]

            val = res[0]
            for l in range(1,self.L):
                val += res[l]
            
            if (retPerLayer):
                return (val, res)
            else:
                return val

        elif direction:
            for kk in range(self.L):
                feats0[kk], feats1[kk] = outs0[kk], outs1[kk]
                diffs[kk] = feats0[kk]-feats1[kk]
            if spatial:
                res = [interpolate(diffs[kk], out_HW=(16, 16)) for kk in range(self.L)]
            else:
                res = [spatial_average(diffs[kk], keepdim=True).squeeze() for kk in range(self.L)]
            res = torch.cat(res, dim=1)
            return res

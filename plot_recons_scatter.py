#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: plot_recons_scatter.py
# --- Creation Date: 05-02-2021
# --- Last Modified: Fri 05 Feb 2021 21:20:41 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Plot scatter recons vs fvm.
"""

from matplotlib import pyplot as plt
import numpy as np

hes0 = np.array([21.78, 18.78, 26.46, 22.58, 21.75, 24.3, 20.44, 20.59, 20.66, 21.35])
hes1 = np.array([16.86, 20.82, 23.63, 21.65, 22.88, 20.59, 24.95, 21.78, 22.63])
hes5 = np.array([20.91, 22.59, 22.5, 17.16, 20.72, 21.47, 19.26, 21.42, 20.81, 59.04])
hes20 = np.array([24.44, 20.96, 20.62, 23.1, 22.87, 20.34, 26.43, 22.81, 25.35])
hes40 = np.array([20.05, 21.67, 19.9, 20.22, 20.64, 22.01, 22.87, 21.3, 19.48])
hes80 = np.array([21.47, 21.9, 23.06, 20.92, 25.55, 20.49, 22.1, 22.76, 22.58])

beta10 = np.array([54.18, 59.42, 71.71, 54.86, 58.33, 56.96, 54.74, 57.23, 65.15, 55.49])
vae = np.array([18.87, 17.76, 17.9, 17.67, 18.35, 19.96, 17.79, 17.92, 15.66, 18.1])
bottle = np.array([19.39, 16.43, 18.58, 19.58, 16.34, 21.22, 18.83, 16.42, 18.82, 25.73])

com1 = np.array([20.12, 21.66, 24.01, 25.6, 23.35, 22.75, 21.28, 20.47, 18.39, 22.34])
com5 = np.array([27.96, 17.98, 19.84, 18.98, 20.87, 21.91, 21.82, 20.57, 19.02, 18.53])
com20 = np.array([20.56, 20.92, 20.51, 22.38, 22.5, 18.01, 24.65, 19.47, 25.66, 28.02])
com40 = np.array([22.29, 19.62, 23.58, 20.98, 20.76, 23.24, 22.91, 21.43, 23.06, 19.43])
com80 = np.array([22.83, 20.33,21.17, 22.16, 25.73, 23.59, 19.45, 22.08, 19.75, 24.74])

# recons_mean_ls = [hes0.mean(), hes1.mean(), hes5.mean(), hes20.mean(),
                  # hes40.mean(), hes80.mean(), beta10.mean(), vae.mean(),
                  # bottle.mean(), com1.mean(), com5.mean(), com20.mean(), com40.mean(), com80.mean()]
recons_mean_ls = [[hes40.mean(), hes80.mean()], [hes0.mean(), com40.mean(), com80.mean()], [beta10.mean(), 31.8, 35.39], [vae.mean()],
                  [bottle.mean()]]
recons_std_ls = [hes0.std(), hes1.std(), hes5.std(), hes20.std(),
                  hes40.std(), hes80.std(), beta10.std(), vae.std(),
                  bottle.std(), com1.std(), com5.std(), com20.std(), com40.std(), com80.std()]
# fvm_mean_ls = [83.6, 84.2, 83.5, 86.1, 86.2, 85.0, 73.1, 69.4, 74.6, 85.1, 84.0, 85.8,
               # 85.5, 85.5]
fvm_mean_ls = [[86.1, 86.2], [83.6, 85.8, 85.5], [73.1, 70.3, 72.0], [69.4], [74.6]]
# text_ls = ['0', '1', '5', '20', '40', '80', 'beta-10', 'vae', 'bottle', '1', '5', '20', '40', '80']
text_ls = [['40', '80'], ['0', '40', '80'], ['beta-10', 'beta-2', 'beta-5'], ['vae'], ['bottle']]
# colors = ['r'] * 6 + ['b', 'g', 'm'] + ['c'] * 5
# colors = [['r'] * 3] + [['b'], ['g'], ['m']] + [['c'] * 2]
colors = ['r', 'b', 'g', 'm', 'c']
labels = ['Hessian', 'Decomp', r'$\beta$-VAE', 'VAE', 'bottleneck-VAE']
assert len(recons_mean_ls) == len(fvm_mean_ls)

# plt.figure(figsize=(7, 4))
plt.grid(True)
for i in range(5):
    # plt.scatter(recons_mean_ls, fvm_mean_ls, s=recons_std_ls, c=colors, alpha=0.5)
    scatter = plt.scatter(recons_mean_ls[i], fvm_mean_ls[i], c=colors[i], alpha=1, label=labels[i])
    for j, text in enumerate(text_ls[i]):
        plt.annotate(text, (recons_mean_ls[i][j] + 0.5, fvm_mean_ls[i][j] + 0.2))
plt.xlabel('Reconstruction Loss')
plt.ylabel('FactorVAE Metric Score')
plt.legend()
# plt.legend(*scatter.legend_elements(),
                    # loc="lower left", title="Classes")
plt.show()

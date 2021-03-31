#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: plot_kl.py
# --- Creation Date: 04-02-2021
# --- Last Modified: Fri 05 Feb 2021 14:32:55 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Plot kl vs iter.
"""

import pandas
from matplotlib import pyplot as plt
import numpy as np

file = pandas.read_csv('/mnt/hdd/repo_results/UnsupervisedActionEstimation/plot_kl_vs_iter/aggregation.csv')
x = np.linspace(0, 70, 70)
plt.figure(figsize=(7, 4))
plt.grid(True)
colors = ['r', 'b', 'g', 'm', 'c', 'y', 'orange', 'gray']
names = ['VAE', 'bottleneck-VAE', 'Lie Group VAE', r'$\beta$-VAE ($\beta$=10)']
# for i, i_name in enumerate(['noexp', 'rec1', '0', 'for0', '1', '5', '20', '40', '80']):
for i, i_name in enumerate(['noexp', 'rec1', '0', 'beta10']):
    mean_i, std_i = file['Mean_'+i_name].values, file['Std_'+i_name].values
    std_i_high = mean_i + std_i / 2.
    std_i_low = mean_i - std_i / 2.
    plt.plot(x, mean_i, '-', color=colors[i], label=names[i])
    plt.fill_between(x, std_i_low, std_i_high, alpha=0.2, color=colors[i])
# plt.title('KL-divergence loss')
# plt.tight_layout()
plt.xlabel('Training Steps')
plt.ylabel('KL-divergence loss')
# plt.legend(loc='lower right')
plt.legend()
plt.show()

# x = np.linspace(0, 30, 30)
# y = np.sin(x/6*np.pi)
# error = np.random.normal(0.1, 0.02, size=y.shape)
# y += np.random.normal(0, 0.1, size=y.shape)

# plt.plot(x, y, 'k-')
# plt.fill_between(x, y-error, y+error)
# plt.show()

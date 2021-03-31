#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: plot_var_percept.py
# --- Creation Date: 24-03-2021
# --- Last Modified: Wed 24 Mar 2021 15:45:44 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Docstring
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


labels = ['dim1', 'dim2', 'dim3', 'dim4', 'dim5', 'dim6', 'dim7', 'dim8', 'dim9', 'dim10']
# percept_mean = [0.0610, 0.0075, 0.0033, 1.5906e-07, 0.1904, 4.7013e-07, 0.0017, 3.2902e-07, 0.1109, 0.0145]
percept_mean = [0.0610, 0.055, 0.0033, 1.5906e-07, 0.1904, 4.7013e-07, 0.0017, 3.2902e-07, 0.1109, 0.032]
group_mean = [2.0103, 2.0246, 0.5248, 0.0189, 2.9005, 0.0164, 0.7932, 0.0147, 2.2611, 1.1741]
# group_mean = [2.0103, 2.0246, 1.5248, 0.0189, 2.9005, 0.0164, 1.7932, 0.0147, 2.2611, 2.1741]
# group_mean = [3.2124, 2.2860, 1.7461, 0.0180, 3.5205, 0.0175, 1.6800, 0.0172, 2.8393, 2.1714]

x = np.arange(len(labels))  # the label locations
width = 0.35  # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(x - width/2, percept_mean, width, label='Percept', color='b')
# rects2 = ax.bar(x + width/2, group_mean, width, label='Women')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Perceptual Significance', color='b')
ax.set_title('Perceptual Significance vs Group Transformation Scale')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.tick_params(axis='y', labelcolor='b')
# ax.legend()

ax2 = ax.twinx()
rects2 = ax2.bar(x + width/2, group_mean, width, label='Group', color='r')
ax2.set_ylabel('Group Transformation Scale', color='r')
ax2.tick_params(axis='y', labelcolor='r')
# ax2.legend()
# plt.grid(True)

# ax3 = ax.twinx()
# rects1_2 = ax3.plot(x, percept_mean, color='b', linestyle='-', marker='o', linewidth=2.0)
# ax4 = ax.twinx()
# rects1_2 = ax4.plot(x, group_mean, color='r', linestyle='-', marker='o', linewidth=2.0)


# def autolabel(rects):
    # """Attach a text label above each bar in *rects*, displaying its height."""
    # for rect in rects:
        # height = rect.get_height()
        # ax.annotate('{}'.format(height),
                    # xy=(rect.get_x() + rect.get_width() / 2, height),
                    # xytext=(0, 3),  # 3 points vertical offset
                    # textcoords="offset points",
                    # ha='center', va='bottom')


# autolabel(rects1)
# autolabel(rects2)

fig.tight_layout()

plt.show()

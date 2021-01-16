#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: generate_teapot_nocolor.py
# --- Creation Date: 14-01-2021
# --- Last Modified: Thu 14 Jan 2021 18:11:34 AEDT
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Create teapot_nocolor dataset. Code from 
https://github.com/IndustAI/learning-group-structure/blob/master/fig4_teapot.ipynb
"""

from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
import random
import numpy as np
from PIL import Image
from tqdm import tqdm
import os
import numpy as np

def read_obj(filename):
    triangles = []
    vertices = []
    with open(filename) as file:
        for line in file:
            components = line.strip(' \n').split(' ')
            if components[0] == "f": # face data
                # e.g. "f 1/1/1/ 2/2/2 3/3/3 4/4/4 ..."
                indices = list(map(lambda c: int(c.split('/')[0]) - 1, components[1:]))
                for i in range(0, len(indices) - 2):
                    triangles.append(indices[i: i+3])
            elif components[0] == "v": # vertex data
                # e.g. "v  30.2180 89.5757 -76.8089"
                vertex = list(map(lambda c: float(c), components[1:]))
                vertices.append(vertex)
    return np.array(vertices), np.array(triangles)

N_DATA = 10000
    
folder = '../learning-group-structure/teapot_nocolor/'

vertices, triangles = read_obj(folder+'teapot.obj')
angle = 2*np.pi / 10
color_index = 0
actions = []

for i in tqdm(range(N_DATA)):
    
    # First, plot 3D image of a teapot and save as image

    x = np.asarray(vertices[:,0]).squeeze()
    y = np.asarray(vertices[:,1]).squeeze()
    z = np.asarray(vertices[:,2]).squeeze()

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.grid(None)
    ax.axis('off')
    ax.set_xlim([-3, 3])
    ax.set_ylim([-3, 3])
    ax.set_zlim([0, 3])
    ax.plot_trisurf(x, z, triangles, y, shade=True, color='white')
    ax.view_init(100, angle)
    plt.savefig(folder+'teapot'+str(i)+'.png')
    plt.close()
    
    # Then load the image, crop, resize it, and change background color

    img = Image.open(folder+'teapot'+str(i)+'.png').convert('RGB')
    w, h = img.size
    # img = img.crop((100,0,350,258))
    img = img.crop((16*8,16*3,w-16*8,h-16*3))
    img = img.resize((128,128))
    arr = np.array(img)
    np.save(folder+'small_teapot/teapot'+str(i),arr)
    img.save(folder+'small_teapot_jpg/teapot'+str(i)+'.jpg')
    
    # Now select an action to perform that changes the scene.

    action = random.randrange(6)

    if action == 0: # y rotation, positive
        m = np.matrix([[np.cos(angle), 0, np.sin(angle)],[0,1,0],[-np.sin(angle), 0, np.cos(angle)]])
    elif action == 1: # y rotation, negative
        m = np.matrix([[np.cos(angle), 0, -np.sin(angle)],[0,1,0],[np.sin(angle), 0, np.cos(angle)]])
    elif action == 2: # z rotation, positive   
        m = np.matrix([[1,0,0],[0, np.cos(angle), np.sin(angle)],[0, -np.sin(angle), np.cos(angle)]])
    elif action == 3: # z rotation, positive
        m = np.matrix([[1,0,0],[0, np.cos(angle), -np.sin(angle)],[0, np.sin(angle), np.cos(angle)]])
    elif action == 4: # x rotation, positive
        m = np.matrix([[np.cos(angle), np.sin(angle), 0],[-np.sin(angle), np.cos(angle), 0],[0,0,1]])
    elif action == 5: # x rotation, positive
        m = np.matrix([[np.cos(angle), -np.sin(angle), 0],[np.sin(angle), np.cos(angle), 0],[0,0,1]])

    actions.append(action)
    
    # Change viewpoint of teapot
    vertices = vertices*m

# Save action sequence
np.save(folder+'actions',actions)

#!/usr/bin/python
#-*- coding: utf-8 -*-

# >.>.>.>.>.>.>.>.>.>.>.>.>.>.>.>.
# Licensed under the Apache License, Version 2.0 (the "License")
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0

# --- File Name: dsprites_paired.py
# --- Creation Date: 25-05-2021
# --- Last Modified: Tue 25 May 2021 20:32:45 AEST
# --- Author: Xinqi Zhu
# .<.<.<.<.<.<.<.<.<.<.<.<.<.<.<.<
"""
Dataset for paired DSprites. Generated from navigator M of stylegan2.
"""

import numpy as np
from torch.utils.data import Dataset
import os
import shutil
import h5py
import zipfile
from PIL import Image
import torch
import random


class PairedDSpritesStylegan(Dataset):
    """
    Args:
        root (str): Root directory of dataset containing paired_dsprites.h5
        transform (``Transform``, optional): A function/transform that takes in an PIL image and returns a transformed version. E.g, ``transforms.RandomCrop``
    """
    def __init__(self, root, transform=None):
        super().__init__()
        self.file = root
        self.transform = transform

        self.dataset_zip = self.load_data()
        self.images_orig = self.dataset_zip['images_orig'][:]  # array shape [240000,64,64,1], uint8 in range(256)
        self.images_edit = self.dataset_zip['images_edit'][:]  # array shape [240000,64,64,1], uint8 in range(256)
        self.labels_ds = self.dataset_zip['labels_np']
        self.labels = self.labels_ds[:]  # int array shape [240000] in M.z_dim
        self.z_dim = self.labels_ds.attrs['z_dim']

    def load_data(self):
        root = os.path.join(self.file, "paired_dsprites.h5")
        dataset_zip = h5py.File(root, 'r')
        return dataset_zip

    def __getitem__(self, index):
        image_orig = self.images_orig[index]
        image_edit = self.images_edit[index]
        label = self.labels[index]

        if self.transform is not None:
            image_orig = self.transform(image_orig) # (h, w, c) of uint8 to (c, h, w) of range(0, 1)
            image_edit = self.transform(image_edit)

        return image_orig, image_edit, label

    def __len__(self):
        return self.labels.shape[0]

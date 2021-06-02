import numpy as np
import torch
from torch.utils.data import random_split, DataLoader
from torchvision.transforms import Lambda, ToTensor

from datasets.flatland_ds import ForwardVAEDS
from datasets.dsprites import PairSprites
from datasets.teapot_ds import TeapotDS
from datasets.shapes3d import PairShapes3D
from datasets.shapes3d_paired import PairedShapes3dStylegan
from datasets.dsprites_paired import PairedDSpritesStylegan


def sprites_transforms(_):
    return ToTensor(), ToTensor()

def forward_ds_transforms(_):
    lam = lambda x: torch.from_numpy(np.array(x)).float()
    return Lambda(lam), Lambda(lam)

def teapot_transforms(_):
    lam = lambda x: torch.from_numpy(np.array(x)).float()
    return Lambda(lam), Lambda(lam)

def shapes3d_transforms(_):
    return ToTensor(), ToTensor()

def stylegan3d_transforms(_):
    return ToTensor(), ToTensor()

def stylegandsp_transforms(_):
    return ToTensor(), ToTensor()

transforms = {
    'flatland': forward_ds_transforms,
    'dsprites': sprites_transforms,
    'teapot': teapot_transforms,
    'teapot_nocolor': teapot_transforms,
    'shapes3d': shapes3d_transforms,
    'stylegan3d': stylegan3d_transforms,
    'stylegandsp': stylegandsp_transforms,
}


def split(func):  # Splits a dataset into a train and val set
    def splitter(args):
        ds = func(args)
        lengths = int(len(ds) * (1 - args.split)), int(len(ds)) - int(len(ds) * (1 - args.split))
        train_ds, val_ds = random_split(ds, lengths) if args.split > 0 else (ds, None)
        return train_ds, val_ds

    return splitter


def fix_data_path(func):  # Sets the datapath to that in _default_paths if it is None
    def fixer(args):
        args.data_path = args.data_path if args.data_path is not None else _default_paths[args.dataset]
        return func(args)

    return fixer


def set_to_loader(trainds, valds, args):
    trainloader = DataLoader(trainds, batch_size=args.batch_size, num_workers=7, shuffle=args.shuffle, drop_last=False,
                             pin_memory=True)
    if valds is not None:
        valloader = DataLoader(valds, batch_size=args.batch_size, num_workers=7, shuffle=False, drop_last=False,
                               pin_memory=True)
    else:
        valloader = None
    return trainloader, valloader


@split
@fix_data_path
def sprites(args):
    train_transform, test_transform = transforms[args.dataset](args)
    output_targets = True if args.model in ['forward', 'rgrvae', 'lie_group_action', 'lie_group_action_simple', 'lie_group_rl'] else False
    ds = PairSprites(args.data_path, download=False, transform=train_transform, wrapping=True, offset=args.offset,
                     noise_name=args.noise_name, output_targets=output_targets, fixed_shape=args.fixed_shape)
    return ds


@split
@fix_data_path
def forward_vae_ds(args):
    import os
    train_transform, test_transform = transforms[args.dataset](args)
    output_targets = True if args.model in ['forward', 'rgrvae', 'lie_group_action', 'lie_group_action_simple', 'lie_group_rl'] else False
    mean_channels = True

    images_path = os.path.join(args.data_path, 'inputs.npy')
    actions_path = os.path.join(args.data_path, 'actions.npy')
    ds = ForwardVAEDS(images_path, actions_path, transforms=train_transform, output_targets=output_targets,
                      mean_channels=mean_channels, num_steps=args.offset, noise_name=args.noise_name)
    return ds

@split
@fix_data_path
def teapot_ds(args):
    import os
    train_transform, test_transform = transforms[args.dataset](args)
    output_targets = True if args.model in ['forward', 'rgrvae', 'lie_group_action', 'lie_group_action_simple', 'lie_group_rl'] else False

    images_path = os.path.join(args.data_path, 'small_teapot')
    actions_path = os.path.join(args.data_path, 'actions.npy')
    ds = TeapotDS(images_path, actions_path, transforms=train_transform, output_targets=output_targets,
                  num_steps=args.offset, noise_name=args.noise_name)
    return ds

@split
@fix_data_path
def shapes3d(args):
    train_transform, test_transform = transforms[args.dataset](args)
    output_targets = True if args.model in ['forward', 'rgrvae', 'lie_group_action', 'lie_group_action_simple', 'lie_group_rl'] else False
    ds = PairShapes3D(args.data_path, transform=train_transform, wrapping=True, offset=args.offset,
                     noise_name=args.noise_name, output_targets=output_targets)
    return ds

@split
@fix_data_path
def stylegan3d(args):
    train_transform, test_transform = transforms[args.dataset](args)
    ds = PairedShapes3dStylegan(args.data_path, transform=train_transform)
    return ds

@split
@fix_data_path
def stylegandsp(args):
    train_transform, test_transform = transforms[args.dataset](args)
    ds = PairedDSpritesStylegan(args.data_path, transform=train_transform)
    return ds


_default_paths = {
    'flatland': '',
    'dsprites': '',
    'teapot': '',
    'teapot_nocolor': '',
    'shapes3d': '',
    'stylegan3d': '',
    'stylegandsp': '',
}

datasets = {
    'flatland': forward_vae_ds,
    'dsprites': sprites,
    'teapot': teapot_ds,
    'teapot_nocolor': teapot_ds,
    'shapes3d': shapes3d,
    'stylegan3d': stylegan3d,
    'stylegandsp': stylegandsp,
}

dataset_meta = {
    'flatland': {'nc': 1, 'factors': 2, 'max_classes': 40},
    'dsprites': {'nc': 1, 'factors': 5, 'max_classes': 40},
    'teapot': {'nc': 3, 'factors': 4, 'max_classes': 40},
    'teapot_nocolor': {'nc': 3, 'factors': 3, 'max_classes': 40},
    'shapes3d': {'nc': 3, 'factors': 6, 'max_classes': 40},
    'stylegan3d': {'nc': 3, 'factors': 6, 'max_classes': 10},
    'stylegandsp': {'nc': 1, 'factors': 5, 'max_classes': 10},
}

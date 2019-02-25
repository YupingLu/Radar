#!/usr/bin/env python3
# Nexrad dataset class
# Author: Yuping Lu <yupinglu89@gmail.com>
# Last Update: 02/25/2019

# load libs
import os
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

__all__ = ["NexradDataset", "RandomHorizontalFlip", "RandomVerticalFlip", "ToTensor", "Normalize", "RandomCrop"]

class NexradDataset(Dataset):
    """ NEXRAD dataset. """
    
    def __init__(self, root, transform=None):
        """
        Args:
            root (string): Directory with all the nexrad data.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root = root
        self.categories = sorted(os.listdir(root))
        self.cat2idx = dict(zip(self.categories, range(len(self.categories))))
        self.idx2cat = dict(zip(self.cat2idx.values(), self.cat2idx.keys()))
        self.files = []
        
        for (dirpath, dirnames, filenames) in os.walk(self.root):
            for f in filenames:
                if f.endswith('.csv'):
                    o = {}
                    o['radar_path'] = dirpath + '/' + f
                    o['category'] = self.cat2idx[dirpath[dirpath.rfind('/')+1:]]
                    self.files.append(o)
                    
        self.transform = transform
    
    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        radar_path = self.files[idx]['radar_path']
        category = self.files[idx]['category']
        radar = np.loadtxt(radar_path, delimiter=',')
        radar = radar.reshape((4, 30, 30))
        sample = {'radar': radar, 'category': category}
        
        if self.transform:
            sample = self.transform(sample)
			
        return sample

class RandomHorizontalFlip(object):
    """Horizontally flip the given dataset randomly with a given probability.
    Args:
        p (float): probability of the dataset being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['radar'] = np.copy(np.flip(sample['radar'], 2))
        return sample

class RandomVerticalFlip(object):
    """Vertically flip the given dataset randomly with a given probability.
    Args:
        p (float): probability of the dataset being flipped. Default value is 0.5
    """

    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, sample):
        if random.random() < self.p:
            sample['radar'] = np.copy(np.flip(sample['radar'], 1))
        return sample

    
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        radar, category = sample['radar'], sample['category']
        return {'radar': torch.from_numpy(radar).to(torch.float),
                'category': torch.from_numpy(np.array(category, dtype=int))}

class Normalize(object):
    """Normalize a tensor radar with mean and standard deviation."""

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std
        
    def __call__(self, sample):
        for t, m, s in zip(sample['radar'], self.mean, self.std):
            t.sub_(m).div_(s)
        return sample

class RandomCrop(object):
    """Crop the given dataset at a random location."""

    def __init__(self, padding=0):
        self.padding = padding

    def __call__(self, sample):
        """
        Args: Dataset to be cropped.
        Returns: Cropped dataset.
        """
        radar = sample['radar']
        if self.padding > 0:
            radar = np.pad(radar, ((0,),(self.padding,),(self.padding,)), 'mean')

        i = random.randint(0, self.padding*2-1)
        j = random.randint(0, self.padding*2-1)

        sample['radar'] = radar[:, i:i+30, j:j+30]

        return sample

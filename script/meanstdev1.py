#!/usr/bin/env python3
# Compute the mean and std of nexrad data
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date: 02/19/2019

# load libs
import numpy as np
import pandas as pd

root = '/home/ylk/dataloader/train/'
datasets = ['30', '40', '60', '80']

total = []

for data in datasets:
    df = pd.read_csv(root + data + '.txt', header = None)
    
    for i in range(len(df.index)):
    #for i in range(2):
        d = np.loadtxt(root + data + '/' + df.iloc[i,0], delimiter=',')
        d = d.reshape((4, 30, 30))
        total.append(d)
    
means = []
stdevs = []
total = np.array(total)

for i in range(4):
    pixels = total[:,i,:,:].ravel()
    print(pixels.shape)
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

print("means: {}".format(means))
print("stdevs: {}".format(stdevs))
print('transforms.Normalize(mean = {}, std = {})'.format(means, stdevs))
#!/usr/bin/env python3
# Compute the mean of nexrad data
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date: 02/19/2019

# load libs
import numpy as np
import pandas as pd

root = '/home/ylk/dataloader/train/'
datasets = ['30', '40', '60', '80']

means = np.zeros([4])

for data in datasets:
    df = pd.read_csv(root + data + '.txt', header = None)
    
    for i in range(len(df.index)):
    #for i in range(2):
        d = np.loadtxt(root + data + '/' + df.iloc[i,0], delimiter=',')
        d = d.reshape((4, 30, 30))
        for j in range(4):
            pixel = d[j,:,:].ravel()
            means[j] += np.sum(pixel)

print("means: {}".format(means/(100000*900*4)))

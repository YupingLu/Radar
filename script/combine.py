#!/usr/bin/env python3
# Combine four variable datasets into one.
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date: 11/06/2018

# load libs
import numpy as np
import pandas as pd

root = '/home/ylk/data/data3/0/'
datasets = ['30', '40', '60', '80']
path = '/home/ylk/workspace/dataloader/data/'

for data in datasets:
    # Read data frame
    df = pd.read_csv(root + data + '.txt', header = None)
    f_data = open(path + data + '.txt', 'w')

    for i in range(0, 60000, 4):
        # load datasets
        cname = root + data + '/' + df.iloc[i,0] + '.csv'
        kname = root + data + '/' + df.iloc[i+1,0] + '.csv'
        rname = root + data + '/' + df.iloc[i+2,0] + '.csv'
        xname = root + data + '/' + df.iloc[i+3,0] + '.csv'
        n0c = np.loadtxt(cname, delimiter=',')
        n0k = np.loadtxt(kname, delimiter=',')
        n0r = np.loadtxt(rname, delimiter=',')
        n0x = np.loadtxt(xname, delimiter=',')

        # stack four datasets into one
        fname = df.iloc[i,0] + '.csv'
        f_data.write(fname + '\n')
        f = open(path + data + '/' + fname, 'wb')

        np.savetxt(f, n0c, delimiter=',')
        np.savetxt(f, n0k, delimiter=',')
        np.savetxt(f, n0r, delimiter=',')
        np.savetxt(f, n0x, delimiter=',')

        f.close()

    f_data.close()

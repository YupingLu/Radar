#!/usr/bin/env python3
# Crop raw data to create 12*3 30*30 matrices.
# Variables with too many missing values will not be included.
# Missing values are filled by mean values.
# Combine four variable datasets into one.
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date: 03/19/2019

# load libs
import os
import pyart
import numpy as np

idx = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330]
idy = [0, 30, 60]

# the directory of the first sweep csapr data
directory = '/home/ylk/arm/data/'
path = '/home/ylk/arm/processed/'
cnt = 0
for fname in os.listdir(directory):
    if fname.endswith(".nc0"):
        cnt += 1
        # read csapr data
        radar = pyart.io.read_cfradial(directory+fname)
        N0R = radar.fields['clutter_filtered_reflectivity']['data']
        N0X = radar.fields['clutter_filtered_differential_reflectivity']['data']
        N0C = radar.fields['clutter_filtered_copolar_correlation_coefficient']['data']
        N0K = radar.fields['clutter_filtered_specific_differential_phase']['data']

        for j in range(len(idx)):
            for k in range(len(idy)):
                r1 = idx[j]
                c1 = idy[k]
                data_n0r = N0R[r1:r1+30, c1:c1+30]
                data_n0x = N0X[r1:r1+30, c1:c1+30]
                data_n0c = N0C[r1:r1+30, c1:c1+30]
                data_n0k = N0K[r1:r1+30, c1:c1+30]
                # get the number of valid entries of each array
                n0r_size = len(data_n0r.compressed())
                n0x_size = len(data_n0x.compressed())
                n0c_size = len(data_n0c.compressed())
                n0k_size = len(data_n0k.compressed())
                # check missing values. Dataset with many missing values will be passed.
                if n0r_size < 45 or n0x_size < 45 or n0c_size < 45 or n0k_size < 45:
                    continue
                # Replace the missing values with mean values
                n0c = data_n0c.filled(data_n0c.mean())
                n0k = data_n0k.filled(data_n0k.mean())
                n0x = data_n0x.filled(data_n0x.mean())
                n0r = data_n0r.filled(data_n0r.mean())
                # stack four datasets into one
                filename = str(cnt) + '_' + str(r1) + '_' + str(c1) + '.csv'
                f = open(path + '/' + filename, 'wb')
                np.savetxt(f, n0c, delimiter=',')
                np.savetxt(f, n0k, delimiter=',')
                np.savetxt(f, n0r, delimiter=',')
                np.savetxt(f, n0x, delimiter=',')
                f.close()
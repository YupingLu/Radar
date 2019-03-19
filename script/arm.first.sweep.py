#!/usr/bin/env python3
# Extract the first sweep from ARM CSAPR data
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date: 03/19/2019

# load libs
import os
import pyart

directory = '/home/ylk/arm/data/'

for fname in os.listdir(directory):
    if fname.endswith(".nc"):
        radar = pyart.io.read_cfradial(directory+fname)
        radar_zero = radar.extract_sweeps([0])
        pyart.io.write_cfradial('/home/ylk/arm/data1/'+fname+'0', radar_zero)
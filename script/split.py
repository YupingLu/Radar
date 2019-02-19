#!/usr/bin/env python3
# Split datasets into training, validation and test set
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date: 02/19/2019

# 30 Ice\ Crystals  
# 40 Dry\ Snow  
# 60 Rain
# 80 Big\ Drops

# load libs
import pandas as pd
df = pd.read_csv('30.txt', header = None)  #
f_script = open("move_30.sh", "w")  #
name = '30'

count = 0

for i in range(0, len(df.index), 23):
    if count % 2 == 0:
        f_script.write('mv train/' + name + '/' + df.iloc[i,0] + ' test/' + name + '/' + df.iloc[i,0] + '\n')  #
        f_script.write('mv train/' + name + '/' + df.iloc[i+1,0] + ' test/' + name + '/' + df.iloc[i+1,0] + '\n')  #
        f_script.write('mv train/' + name + '/' + df.iloc[i+2,0] + ' validation/' + name + '/' + df.iloc[i+2,0] + '\n')  #
    else:
        f_script.write('mv train/' + name + '/' + df.iloc[i,0] + ' test/' + name + '/' + df.iloc[i,0] + '\n')  #
        f_script.write('mv train/' + name + '/' + df.iloc[i+1,0] + ' validation/' + name + '/' + df.iloc[i+1,0] + '\n')  #
        f_script.write('mv train/' + name + '/' + df.iloc[i+2,0] + ' validation/' + name + '/' + df.iloc[i+2,0] + '\n')  #
        
    count += 1

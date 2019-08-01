#!/usr/bin/env python3
# Split datasets into training, validation and test set
# Version 2
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date: 08/01/2019

# 30 Ice\ Crystals  
# 40 Dry\ Snow  
# 60 Rain
# 80 Big\ Drops

# load libs
import pandas as pd

root = '/raid/ylk/'
datasets = ['0', '30', '60', '90', '120', '150'] 
cats = ['30', '40', '60', '80'] #categories

# start with each category
for cat in cats:
    f_script = open('move_' + cat + '.sh', "w")
    # start with each dataset
    for dataset in datasets:
        df = pd.read_csv(root + dataset + '/dataloader/' + cat + '.txt', header = None) 
        count = 0
        for i in range(0, len(df.index)):
            if i % 5 == 0:
                if count % 2 == 0:
                    f_script.write('cp ' + root + dataset + '/dataloader/' + cat + '/' + df.iloc[i,0] + ' ' + root + 'dataloader/test/' + cat + '/' + df.iloc[i,0] + '\n')
                else:
                    f_script.write('cp ' + root + dataset + '/dataloader/' + cat + '/' + df.iloc[i,0] + ' ' + root + 'dataloader/validation/' + cat + '/' + df.iloc[i,0] + '\n')
                count += 1
            else:
                f_script.write('cp ' + root + dataset + '/dataloader/' + cat + '/' + df.iloc[i,0] + ' ' + root + 'dataloader/train/' + cat + '/' + df.iloc[i,0] + '\n') 

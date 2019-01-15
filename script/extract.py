#!/usr/bin/env python3
# Extract datasets from datacrop3
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date: 10/30/2018

# Load libs
import pandas as pd
import shutil as sh

# Number of datasets to extract for each type
cnt = {
    '30': 15000,
    '40': 30000,
    '60': 30000,
    '70': 537,
    '80': 30000,
    '100': 61
}

# Read data frames
df_label = pd.read_csv('name2/label.txt', header = None)
df_n0c = pd.read_csv('name2/n0c.txt', header = None)
df_n0k = pd.read_csv('name2/n0k.txt', header = None)
df_n0r = pd.read_csv('name2/n0r.txt', header = None)
df_n0x = pd.read_csv('name2/n0x.txt', header = None)

# Extract datasets
for key, value in cnt.items():
    count = 0
    f = open('data3/0/'+key+'.txt',"w")
    # Loop the files
    for i in range(len(df_label.index)):
        if count == value:
            break
        if df_label.iloc[i,0] == int(key):
            # cp to data3/0/key
            src = 'data2/' + df_n0c.iloc[i,0] + '.csv'
            dst = 'data3/0/' + key
            sh.copy2(src, dst)
            f.write(df_n0c.iloc[i,0] + '\n')
            
            src = 'data2/' + df_n0k.iloc[i,0] + '.csv'
            dst = 'data3/0/' + key
            sh.copy2(src, dst)
            f.write(df_n0k.iloc[i,0] + '\n')
            
            src = 'data2/' + df_n0r.iloc[i,0] + '.csv'
            dst = 'data3/0/' + key
            sh.copy2(src, dst)
            f.write(df_n0r.iloc[i,0] + '\n')
            
            src = 'data2/' + df_n0x.iloc[i,0] + '.csv'
            dst = 'data3/0/' + key
            sh.copy2(src, dst)
            f.write(df_n0x.iloc[i,0] + '\n')
            
            count += 1
    f.close()
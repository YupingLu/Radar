#!/usr/bin/env python3
'''
Test script for NEXRAD
Different from test.py. This script is meant to test the raw four variable files.
Currently, this script only measures idx = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330] 
idy = [0, 30, 60, 90, 120, 150] for each variable file.
Copyright (c) Yuping Lu <yupinglu89@gmail.com>, 2019
Last Update: 3/25/2019
'''
# load libs
from __future__ import print_function
import sys
import pyart
from scipy.stats import mode
import numpy as np
import numpy.ma as ma
import matplotlib
import matplotlib.pyplot as plt
matplotlib.rcParams['figure.figsize'] = [35.0, 35.0]
import os
import argparse
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.nexradtest import *
import models

def discrete_cmap(N, base_cmap=None):
    """Create an N-bin discrete colormap from the specified input map"""

    # Note that if base_cmap is a string or None, you can simply do
    #    return plt.cm.get_cmap(base_cmap, N)
    # The following works for string, None, or a colormap instance:

    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)

idx = [0, 30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330] 
idy = [0, 30, 60, 90, 120, 150]
cnt = {
    30 : 'Ice Crystals', # Ice Crystals (IC) #
    40 : 'Dry Snow', # Dry Snow (DS) #
    60 : 'Rain', # Light and/or Moderate Rain (RA) #
    80 : 'Big Drops', # Big Drops (rain) (BD) #
}
cat2idx = {'Big Drops': 0, 'Dry Snow': 1, 'Ice Crystals': 2, 'Rain': 3}
idx2cat = {0: 80, 1: 40, 2: 30, 3: 60}

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def test(args, model, device, test_loader):
    model.eval()
    r = 0
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data['radar'].to(device), data['category'].to(device)
            # compute output
            outputs = model(inputs)
            pred = outputs.max(1)[1] # get the index of the max log-probability
            r = pred.item()
    return r

# Call trained model to classify cropped matrices
def classify(path, label, transform, device, kwargs, args):
    testset = NexradDataset(path, label, transform=transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    
    model = models.__dict__[args.arch](num_classes=4).to(device)

    # Load saved models.
    #eprint("==> Loading model '{}'".format(args.arch))
    assert os.path.isfile(args.path), 'Error: no checkpoint found!'
    checkpoint = torch.load(args.path)
    model.load_state_dict(checkpoint['model'])

    return test(args, model, device, test_loader)

# Crop the file into 12 60*60 matrices
def datacrop(n0h, n0c, n0k, n0r, n0x, transform, device, kwargs, args):
    # np matrix to store the classification results
    results = np.zeros((360, 180))

    # read data
    try:
        N0H = pyart.io.read(n0h)
    except:
        eprint("aa! N0H errors!")
        sys.exit(-1)
    try:
        N0C = pyart.io.read(n0c)
    except:
        eprint("aa! N0C errors!")
        sys.exit(-1)
    try:
        N0K = pyart.io.read(n0k)
    except:
        eprint("aa! N0K errors!")
        sys.exit(-1)
    try:
        N0R = pyart.io.read(n0r)
    except:
        eprint("aa! N0R errors!")
        sys.exit(-1)
    try:
        N0X = pyart.io.read(n0x)
    except:
        eprint("aa! N0X errors!")
        sys.exit(-1)
        
    # Check variable dims. If not match, stop.
    data_n0h = N0H.fields['radar_echo_classification']['data']
    data_n0c = N0C.fields['cross_correlation_ratio']['data']
    data_n0k = N0K.fields['specific_differential_phase']['data']
    data_n0r = N0R.fields['reflectivity']['data']
    data_n0x = N0X.fields['differential_reflectivity']['data']

    if data_n0h.shape != (360, 1200):
        eprint('Error dim: ' + n0h + '\n')
        sys.exit(-1)
    if data_n0c.shape != (360, 1200):
        eprint('Error dim: ' + n0c + '\n')
        sys.exit(-1)
    if data_n0k.shape != (360, 1200):
        eprint('Error dim: ' + n0k + '\n')
        sys.exit(-1)
    if data_n0r.shape != (360, 230):
        eprint('Error dim: ' + n0r + '\n')
        sys.exit(-1)
    if data_n0x.shape != (360, 1200):
        eprint('Error dim: ' + n0x + '\n')
        sys.exit(-1)

    # Extend n0r
    # Expand by 5 times
    data_n0r_repeat = np.repeat(data_n0r, 5, axis=1)
    # Insert another 1 every 23
    tres = np.empty((360, 0))
    for idk in range(1150):
        tres = np.append(tres, data_n0r_repeat[:,idk].reshape(360,1), axis=1)
        if (idk+1) % 23 == 0:
            tres = np.append(tres, data_n0r_repeat[:,idk].reshape(360,1), axis=1)
    if tres.shape != (360, 1200):
        eprint('Error dim: ' + n0r + '\n')
        sys.exit(-1)
    data_n0r = tres
    data_n0r = ma.masked_values(data_n0r, -999.0)

    for j in range(len(idx)):
        for k in range(len(idy)):
            r1 = idx[j]
            c1 = idy[k]
            tmp_n0h = data_n0h[r1:r1+30, c1:c1+30]
            # mask 0, 10, 20, 50, 70, 90, 100, 120, 140, 150
            # If the valid values of n0h is less then 90, abadon that entry.
            mx = ma.masked_values(tmp_n0h, 0.0) 
            mx = ma.masked_values(mx, 10.0) 
            mx = ma.masked_values(mx, 20.0)
            mx = ma.masked_values(mx, 50.0)
            mx = ma.masked_values(mx, 70.0)
            mx = ma.masked_values(mx, 90.0)
            mx = ma.masked_values(mx, 100.0)
            mx = ma.masked_values(mx, 120.0)
            mx = ma.masked_values(mx, 140.0) 
            mx = ma.masked_values(mx, 150.0) 
            t_n0h = mx.compressed()
            unmask_size = len(t_n0h)
            if unmask_size < 45:
                eprint('Too few n0h: ' + n0h \
                                + ' ' + str(r1) + ' ' + str(c1) + '\n')
                continue
            # get the most frequent radar_echo_classification
            m = mode(t_n0h)
            mode_value = m[0][0]
            mode_count = m[1][0]
            if mode_count < 22:
                eprint('Mode is small: ' + n0h \
                                + ' ' + str(r1) + ' ' + str(c1) + '\n')
                continue
            
            tmp_n0c = data_n0c[r1:r1+30, c1:c1+30]
            tmp_n0k = data_n0k[r1:r1+30, c1:c1+30]
            tmp_n0r = data_n0r[r1:r1+30, c1:c1+30]
            tmp_n0x = data_n0x[r1:r1+30, c1:c1+30]
            
            # Replace the missing values with mean values
            t_n0c = tmp_n0c.filled(tmp_n0c.mean())
            t_n0k = tmp_n0k.filled(tmp_n0k.mean())
            t_n0x = tmp_n0x.filled(tmp_n0x.mean())
            t_n0r = tmp_n0r.filled(tmp_n0r.mean())
            
            # Combine 4 2d array into 1 3d array
            fname = './tmp_test/'+str(r1)+str(c1)+'.csv'
            f = open(fname, 'wb')
            np.savetxt(f, t_n0c, delimiter=',')
            np.savetxt(f, t_n0k, delimiter=',')
            np.savetxt(f, t_n0r, delimiter=',')
            np.savetxt(f, t_n0x, delimiter=',')
            f.close()
            
            # classify the file
            acc = classify(fname, cat2idx[cnt[mode_value]], transform, device, kwargs, args)
            # Save results
            results[r1:r1+30, c1:c1+30] = acc
    
    return results

# Save the visualization of classification results
def viz_res(n, vname):
    N = pyart.io.read(n)
    display = pyart.graph.RadarMapDisplay(N)
    x = N.fields[vname]['data']
    
    m = np.zeros_like(x)
    m[:,180:] = 1
    y = ma.masked_array(x, m)
    N.fields[vname]['data'] = y

    fig = plt.figure(figsize=(6, 5))
    
    ax = fig.add_subplot(111)
    display.plot(vname, 0, title=vname, colorbar_label='', ax=ax)
    display.set_limits(xlim=(-50, 50), ylim=(-50, 50), ax=ax)
    plt.show();

    fig.savefig("./tmp_test/"+vname+".png", bbox_inches='tight')

def viz_ress(n, vname):
    N = pyart.io.read(n)
    display = pyart.graph.RadarMapDisplay(N)
    x = N.fields[vname]['data']
    
    m = np.zeros_like(x)
    m[:,180:] = 1
    y = ma.masked_array(x, m)
    y = ma.masked_values(y, 0.0) 
    y = ma.masked_values(y, 10.0) 
    y = ma.masked_values(y, 20.0)
    y = ma.masked_values(y, 50.0)
    y = ma.masked_values(y, 70.0)
    y = ma.masked_values(y, 90.0)
    y = ma.masked_values(y, 100.0)
    y = ma.masked_values(y, 140.0) 
    y = ma.masked_values(y, 150.0) 

    y = np.where(y == 80, 0, y)
    y = np.where(y == 40, 1, y)
    y = np.where(y == 30, 2, y)
    y = np.where(y == 60, 3, y)
    y = ma.masked_where(y > 3, y)

    N.fields[vname]['data'] = y

    fig = plt.figure(figsize=(6, 5))
    
    ax = fig.add_subplot(111)
    #display.plot(vname, 0, title=vname, colorbar_label='', ax=ax)
    display.plot(vname, 0, title=vname, colorbar_label='', ticks=range(4), ticklabs=['Big Drops', 'Dry Snow', 'Ice Crystals', 'Rain'], ax=ax, vmin=-0.5, vmax=3.5, cmap=discrete_cmap(4, 'rainbow'))
    display.set_limits(xlim=(-50, 50), ylim=(-50, 50), ax=ax)
    plt.show();

    fig.savefig("./tmp_test/"+vname+".png", bbox_inches='tight')

# Visualize the classification results
def plot_res(n0h, n0c, n0k, n0r, n0x, results):
    viz_res(n0c, 'cross_correlation_ratio')
    viz_res(n0k, 'specific_differential_phase')
    viz_res(n0x, 'differential_reflectivity')
    viz_res(n0r, 'reflectivity')
    viz_ress(n0h, 'radar_echo_classification')
    
    N0H = pyart.io.read(n0h)
    display_h = pyart.graph.RadarMapDisplay(N0H)
    data_n0h = N0H.fields['radar_echo_classification']['data']
    
    m = np.zeros_like(data_n0h)
    m[:,180:] = 1
    y = ma.masked_array(data_n0h, m)
    y = ma.masked_values(y, 0.0) 
    y = ma.masked_values(y, 10.0) 
    y = ma.masked_values(y, 20.0)
    y = ma.masked_values(y, 50.0)
    y = ma.masked_values(y, 70.0)
    y = ma.masked_values(y, 90.0)
    y = ma.masked_values(y, 100.0)
    y = ma.masked_values(y, 120.0)
    y = ma.masked_values(y, 140.0) 
    y = ma.masked_values(y, 150.0)
    results = ma.masked_where(ma.getmask(y[:,:180]), results)
    for j in range(len(idx)):
        for k in range(len(idy)):
            r1 = idx[j]
            c1 = idy[k]
            y[r1:r1+30, c1:c1+30] = results[r1:r1+30, c1:c1+30]

    N0H.fields['radar_echo_classification']['data'] = y

    fig = plt.figure(figsize=(6, 5))
    
    ax = fig.add_subplot(111)
    display_h.plot('radar_echo_classification', 0, title='classification results', colorbar_label='', ticks=range(4), ticklabs=['Big Drops', 'Dry Snow', 'Ice Crystals', 'Rain'], ax=ax, vmin=-0.5, vmax=3.5, cmap=discrete_cmap(4, 'rainbow'))
    #display_h.plot('radar_echo_classification', 0, title='classification results', colorbar_label='', ax=ax, cmap=cMap)
    display_h.set_limits(xlim=(-50, 50), ylim=(-50, 50), ax=ax)
    plt.show();

    fig.savefig("./tmp_test/res.png", bbox_inches='tight')

def main():
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
    
    parser = argparse.ArgumentParser(description='PyTorch NEXRAD Test')
    # Model options
    parser.add_argument('--arch', '-a', metavar='ARCH', default='vgg19_bn',
                        choices=model_names,
                        help='model architecture: ' +
                             ' | '.join(model_names) +
                            ' (default: vgg19_bn)')
    parser.add_argument('--batch-size', type=int, default=256, metavar='N',
                        help='input batch size for test (default: 256)')
    #Device options
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--gpu-id', type=str, default='3', metavar='N',
                        help='id(s) for CUDA_VISIBLE_DEVICES (default: 3)')
    # Miscs
    parser.add_argument('--seed', type=int, default=20190225, metavar='S',
                        help='random seed (default: 20190225)')
    # Path to saved models
    parser.add_argument('--path', type=str, default='checkpoint/vgg19_bn.pth.tar', metavar='PATH',
                        help='path to save models (default: checkpoint/vgg19_bn.pth.tar)')

    args = parser.parse_args()
    
    # path to the raw data
    n0h = '/home/ylk/nexrad/test_nexrad/processed/KOUN_SDUS84_N0HVNX_201801011620'
    n0c = '/home/ylk/nexrad/test_nexrad/processed/KOUN_SDUS84_N0CVNX_201801011620'
    n0k = '/home/ylk/nexrad/test_nexrad/processed/KOUN_SDUS84_N0KVNX_201801011620'
    n0r = '/home/ylk/nexrad/test_nexrad/processed/KOUN_SDUS54_N0RVNX_201801011620'
    n0x = '/home/ylk/nexrad/test_nexrad/processed/KOUN_SDUS84_N0XVNX_201801011620'

    # Use CUDA
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_id
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if use_cuda:
        torch.cuda.manual_seed_all(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")
    kwargs = {'num_workers': 4, 'pin_memory': True} if use_cuda else {}
        
    transform = transforms.Compose([
        ToTensor(),
        Normalize(mean=[0.7324, 0.0816, 4.29, 0.7663],
                  std=[0.1975, 0.4383, 13.1661, 2.118])
    ])
    
    results = datacrop(n0h, n0c, n0k, n0r, n0x, transform, device, kwargs, args)

    # Visualize the classification results
    plot_res(n0h, n0c, n0k, n0r, n0x, results)
    
if __name__ == '__main__':
    main()

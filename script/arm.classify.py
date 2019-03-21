#!/usr/bin/env python3
'''
Radar classification script for ARM CSAPR
Copyright (c) Yuping Lu <yupinglu89@gmail.com>, 2019
Last Update: 03/19/2019
'''
# load libs
from __future__ import print_function
import sys

import os
import argparse
import random

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.armdataset import *
import models

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def classify(args, model, device, test_loader, class_names):
    
    model.eval()

    with torch.no_grad():
        for data in test_loader:
            inputs = data['radar'].to(device)
            
            # compute output
            outputs = model(inputs)
            pred = outputs.max(1)[1] # get the index of the max 
            print(class_names[pred.item()])
    
def main():
    model_names = sorted(name for name in models.__dict__
        if name.islower() and not name.startswith("__")
        and callable(models.__dict__[name]))
    
    parser = argparse.ArgumentParser(description='PyTorch ARM Test')
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
    parser.add_argument('--seed', type=int, default=20181212, metavar='S',
                        help='random seed (default: 20181212)')
    # Path to saved models
    parser.add_argument('--path', type=str, default='checkpoint/vgg19_bn.pth.tar', metavar='PATH',
                        help='path to save models (default: checkpoint/vgg19_bn.pth.tar)')
    # Path to ARM datasets
    parser.add_argument('--root', type=str, default='/home/ylk/dataloader/arm', metavar='ROOT',
                        help='path to ARM datasets (default: /home/ylk/dataloader/arm)')
    
    args = parser.parse_args()
    
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
        Normalize(mean=[0.7207, 0.0029, -1.6154, 0.5690],
                  std=[0.1592, 0.0570, 12.1113, 2.2363])
    ])

    testset = ARMDataset(root=args.root, transform=transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)
    class_names = testset.idx2cat
    model = models.__dict__[args.arch](num_classes=4).to(device)
        
    # Load saved models.
    eprint("==> Loading model '{}'".format(args.arch))
    assert os.path.isfile(args.path), 'Error: no checkpoint found!'
    checkpoint = torch.load(args.path)
    model.load_state_dict(checkpoint['model'])
    
    classify(args, model, device, test_loader, class_names)
            
if __name__ == '__main__':
    main()

#!/usr/bin/env python3
'''
Test script for NEXRAD
Copyright (c) Yuping Lu <yupinglu89@gmail.com>, 2019
Last Update: 03/25/2019
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
from datasets.nexraddataset import *
import models

def eprint(*args, **kwargs):
    print(*args, file=sys.stderr, **kwargs)

def test(args, model, device, test_loader, criterion):
    
    model.eval()
    test_loss = 0
    correct = 0
    acc = 0
    
    with torch.no_grad():
        for data in test_loader:
            inputs, labels = data['radar'].to(device), data['category'].to(device)
            
            # compute output
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            # measure accuracy and record loss   
            test_loss += loss.item() # sum up batch loss
            pred = outputs.max(1)[1] # get the index of the max log-probability
            correct += pred.eq(labels).sum().item()
    
    # print average loss and accuracy
    test_loss /= len(test_loader)
    acc = 100. * correct / len(test_loader.dataset)
    print('Test set: Average loss:\t'
          '{:.3f}\t'
          'Accuracy: {}/{}\t'
          '{:.3f}'.format(test_loss, correct, len(test_loader.dataset), acc))

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

    testset = NexradDataset(root='/home/ylk/dataloader/test/', transform=transform)
    test_loader = DataLoader(testset, batch_size=args.batch_size, shuffle=False, **kwargs)

    model = models.__dict__[args.arch](num_classes=4).to(device)
    
    criterion = nn.CrossEntropyLoss()
    
    # Load saved models.
    eprint("==> Loading model '{}'".format(args.arch))
    assert os.path.isfile(args.path), 'Error: no checkpoint found!'
    checkpoint = torch.load(args.path)
    model.load_state_dict(checkpoint['model'])
    
    test(args, model, device, test_loader, criterion)
            
if __name__ == '__main__':
    main()

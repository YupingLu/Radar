#!/usr/bin/env python3
'''
Test script for ARM CSAPR
Copyright (c) Yuping Lu <yupinglu89@gmail.com>, 2019
Last Update: 03/19/2019
'''
# load libs
import torch
import torch.nn as nn
from torchvision import transforms
from datasets.nexradtest import *
import models
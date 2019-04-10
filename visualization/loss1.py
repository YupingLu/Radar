#!/usr/bin/env python3
# Accuracy and loss visualization
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date: 04/09/2019

# Load the lib
import matplotlib.pyplot as plt
import pandas as pd

# Read outputs
df_resnet18 = pd.read_csv('resnet18.out', delimiter="\t", header = None)
df_vgg19_bn = pd.read_csv('vgg19_bn.out', delimiter="\t", header = None)
df_resnet101 = pd.read_csv('resnet101.out', delimiter="\t", header = None)
df_alexnet = pd.read_csv('alexnet.out', delimiter="\t", header = None)
df_densenet121 = pd.read_csv('densenet121.out', delimiter="\t", header = None)

# Get losses and accuracies
epochs = 0
tl_resnet18 = []
ta_resnet18 = []
tl_vgg19_bn = []
ta_vgg19_bn = []
tl_resnet101 = []
ta_resnet101 = []
tl_alexnet = []
ta_alexnet = []
tl_densenet121 = []
ta_densenet121 = []

for i in range(0, len(df_resnet18.index), 2):
    epochs += 1
    tl_resnet18.append(df_resnet18.iloc[i, 1])
    ta_resnet18.append(df_resnet18.iloc[i, 3])
    tl_vgg19_bn.append(df_vgg19_bn.iloc[i, 1])
    ta_vgg19_bn.append(df_vgg19_bn.iloc[i, 3])
    tl_resnet101.append(df_resnet101.iloc[i, 1])
    ta_resnet101.append(df_resnet101.iloc[i, 3])
    tl_alexnet.append(df_alexnet.iloc[i, 1])
    ta_alexnet.append(df_alexnet.iloc[i, 3])
    tl_densenet121.append(df_densenet121.iloc[i, 1])
    ta_densenet121.append(df_densenet121.iloc[i, 3])

# Create count of the number of epochs
epoch_count = range(1, epochs + 1)

# Visualize loss history
f = plt.figure()
ax1 = plt.axes()

plt.plot(epoch_count, tl_resnet18, 'b-', linewidth=1)
plt.plot(epoch_count, tl_vgg19_bn, 'g-', linewidth=1)
plt.plot(epoch_count, tl_resnet101, 'r-', linewidth=1)
plt.plot(epoch_count, tl_alexnet, 'm-', linewidth=1)
plt.plot(epoch_count, tl_densenet121, 'y-', linewidth=1)
plt.legend(['ResNet-18', 'VGG-19_BN', 'ResNet-101', 'AlexNet', 'DenseNet-121'])
plt.xlabel('epochs')
plt.ylabel('loss')
plt.ylim((0.28, 0.55))   # set the ylim to bottom, top
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
plt.show()

f.savefig('loss.png', bbox_inches='tight')

# Visualize accuracy history
f = plt.figure()
ax1 = plt.axes()

plt.plot(epoch_count, ta_resnet18, 'b-', linewidth=1)
plt.plot(epoch_count, ta_vgg19_bn, 'g-', linewidth=1)
plt.plot(epoch_count, ta_resnet101, 'r-', linewidth=1)
plt.plot(epoch_count, ta_alexnet, 'm-', linewidth=1)
plt.plot(epoch_count, ta_densenet121, 'y-', linewidth=1)
plt.legend(['ResNet-18', 'VGG-19_BN', 'ResNet-101', 'AlexNet', 'DenseNet-121'])
plt.xlabel('epochs')
plt.ylabel('accuracy(%)')
plt.ylim((75, 88))   # set the ylim to bottom, top
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
plt.show();

f.savefig('accuracy.png', bbox_inches='tight')

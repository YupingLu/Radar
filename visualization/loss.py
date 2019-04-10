#!/usr/bin/env python3
# Accuracy and loss visualization
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date: 02/28/2019

# Load the lib
import matplotlib.pyplot as plt
import pandas as pd

fname = 'resnet18'

# Read outputs
df = pd.read_csv(fname+'.out', delimiter="\t", header = None)

# Get losses and accuracies
epochs = 0
train_loss = []
train_accuracy = []
validation_loss = []
validation_accuracy = []

for i in range(0, len(df.index), 2):
    epochs += 1
    train_loss.append(df.iloc[i, 1])
    train_accuracy.append(df.iloc[i, 3])
    validation_loss.append(df.iloc[i+1, 1])
    validation_accuracy.append(df.iloc[i+1, 3])

# Create count of the number of epochs
epoch_count = range(1, epochs + 1)

# Visualize loss history
f = plt.figure()
ax1 = plt.axes()

plt.plot(epoch_count, train_loss, 'r--', linewidth=1)
plt.plot(epoch_count, validation_loss, 'b-', linewidth=1)
plt.legend(['ResNet-18 Train', 'ResNet-18 Val'])
plt.xlabel('epochs')
plt.ylabel('loss')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
plt.show()

f.savefig(fname+'.loss.png', bbox_inches='tight')

# Visualize accuracy history
f = plt.figure()
ax1 = plt.axes()

plt.plot(epoch_count, train_accuracy, 'r--', linewidth=1)
plt.plot(epoch_count, validation_accuracy, 'b-', linewidth=1)
plt.legend(['ResNet-18 Train', 'ResNet-18 Val'])
plt.xlabel('epochs')
plt.ylabel('accuracy(%)')
ax1.spines['right'].set_visible(False)
ax1.spines['top'].set_visible(False)
plt.show();

f.savefig(fname+'.accuracy.png', bbox_inches='tight')

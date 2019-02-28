#!/usr/bin/env python3
# Accuracy and loss visualization
# Author: Yuping Lu <yupinglu89@gmail.com>
# Date: 02/28/2019

%matplotlib inline
# Load the lib
import matplotlib.pyplot as plt
import pandas as pd

fname = 'resnet101'

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

plt.plot(epoch_count, train_loss, 'r--')
plt.plot(epoch_count, validation_loss, 'b-')
plt.legend(['Train Loss', 'Validation Loss'])
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.show()

f.savefig(fname+'.loss.png', bbox_inches='tight')

# Visualize accuracy history
f = plt.figure()

plt.plot(epoch_count, train_accuracy, 'r--')
plt.plot(epoch_count, validation_accuracy, 'b-')
plt.legend(['Train Accuracy', 'Validation Accuracy'])
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.show();

f.savefig(fname+'.accuracy.png', bbox_inches='tight')
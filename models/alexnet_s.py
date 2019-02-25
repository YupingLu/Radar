'''
Alexnet for NEXRAD. 
Alexnet with smaller parameters
'''
import torch.nn as nn


__all__ = ['AlexNet_S','alexnet_s']


class AlexNet_S(nn.Module):

    def __init__(self, num_classes=4):
        super(AlexNet_S, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=4, stride=2, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(32, 48, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(48, 96, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(96, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(64 * 3 * 3, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(256, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), 64 * 3 * 3)
        x = self.classifier(x)
        return x

    
def alexnet_s(**kwargs):
    model = AlexNet_S(**kwargs)
    return model
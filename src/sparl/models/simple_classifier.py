import torch
import torch.nn as nn
import torch.functional as F


"""
Left-Right Classifier
"""
class LRClassifier(nn.Module):
    def __init__(self):
        super(LRClassifier, self).__init__()

        self.features = nn.Sequential(
            ConvBlock(2, 32, 3, 1, 1),
            ConvBlock(32, 64, 4, 2, 1),
            ConvBlock(64, 32, 4, 2, 1),
            ConvBlock(32, 8, 4, 2, 1),
            Flatten()
        )

        self.fc = nn.Sequential(
            nn.Linear(7680, 256),
            nn.Dropout(0.5),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        x = self.features(x)
        return self.fc(x)


"""
Basic block for the feature extraction of our network
"""
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=(kernel_size, 1), 
                stride=(stride, 1), 
                padding=(padding, 1)
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


"""
Helper class to flatten the output of a convolutional model
"""
class Flatten(nn.Module):
    def __init__(self):
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)
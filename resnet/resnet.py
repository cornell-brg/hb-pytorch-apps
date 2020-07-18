"""
ResNet workload
07/17/2020 Bandhav Veluri
"""

import sys
import os
import torch
import torch.nn as nn
import torchvision

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import utils  # noqa: E402

class Block(nn.Module):
   def __init__(self, in_channels, out_channels, residual=False):
       super().__init__()
       self.layers = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                     padding=1, bias=False),
           nn.BatchNorm2d(out_channels),
           nn.ReLU(inplace=True),
       )
       self.skip = None
       if residual:
           self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                 stride=1, padding=0, bias=False)

   def forward(self, xin):
       x = self.layers(xin)
       if self.skip:
           x = x + self.skip(xin)
       return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            Block(16, 32),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            Block(32, 64),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            Block(64, 128),
            nn.AdaptiveMaxPool2d((1, 1)), # global pooling
        )

        self.fc = nn.Linear(128, 2)

        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, data):
        x = self.conv(data)
        x = 0.125 * self.fc(x)
        x = self.softmax()
        return x

if __name__ == "__main__":
    print(ResNet())

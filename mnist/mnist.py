#!/bin/python

import torch.nn as nn

class MNIST(nn.Module):
    def __init__(self):
        super(MNIST, self).__init__()
        self.mnist = nn.Sequential(nn.Linear(784, 128),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 10),
            )
    def forward(self, x):
        return self.mnist(x)

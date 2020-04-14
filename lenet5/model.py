"""
Small 5-layer CNN workload
03/16/2020 Bandhav Veluri
"""

import torch
import torch.nn as nn

class model(nn.Module):
    """
    LeNet-5

    https://cs.nyu.edu/~yann/2010f-G22-2565-001/diglib/lecun-98.pdf
    (Page 7)
    """
    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
        )

        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, data):
        x = self.conv(data)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x

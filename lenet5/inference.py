"""
Test on a small CNN
03/16/2020 Bandhav Veluri
"""
import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import numpy as np
import torch
import torch.nn as nn
import model
import utils

torch.manual_seed(42)

# Test routine
@torch.no_grad()
def inference(net, loader, loss_func, hb=False):
    test_loss = 0.0
    num_correct = 0

    for batch_idx, (data, labels) in enumerate(loader, 0):
        if hb:
            data, labels = data.hammerblade(), labels.hammerblade()
        output = net(data)
        loss = loss_func(output, labels)
        pred = output.max(1)[1]
        num_correct += pred.eq(labels.view_as(pred)).sum().item()

        if batch_idx == 100:
            break

    test_loss /= len(loader.dataset)
    test_accuracy = 100. * (num_correct / len(loader.dataset))

    print('Test set: Average loss={:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, num_correct, len(loader.dataset), test_accuracy
    ))

if __name__ == "__main__":
    net = model.Network()
    print(net)
    print(utils.ATOL)

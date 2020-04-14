"""
Test on a small CNN
03/16/2020 Bandhav Veluri
"""

import numpy as np
import torch
import torch.nn as nn

torch.manual_seed(42)
random.seed(42)

# Train routine
def train(net, loader, optimizer, loss_func, epochs, batches=None, hb=False):
    print('Training {} for {} epoch(s)...\n'.format(type(net).__name__, epochs))
    for epoch in range(epochs):
        losses = []

        for batch_idx, (data, labels) in enumerate(loader, 0):
            if hb:
                data, labels = data.hammerblade(), labels.hammerblade()
            batch_size = len(data)
            optimizer.zero_grad()
            outputs = net(data)
            loss = loss_func(outputs, labels)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()

            if (batches is None and batch_idx % 1000 == 0) or \
                    (batches is not None and batch_idx < batches):
                print('epoch {} : [{}/{} ({:.0f}%)]\tLoss={:.6f}'.format(
                    epoch, (batch_idx + 1) * batch_size, len(loader.dataset),
                    100. * (batch_idx / len(loader)), loss.item()
                ))
            else:
                break

        print('epoch {} : Average Loss={:.6f}\n'.format(
            epoch, np.mean(losses)
        ))

# Test routine
@torch.no_grad()
def test(net, loader, loss_func, hb=False):
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

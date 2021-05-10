#!/usr/bin/env python

"""
    convnet/main.py

    Note to program performers:
        - This is a (very) pared down version of Workflow #5 from the August releast
        We had to reduce the size because testers don't necessarily have access
        to GPUs, and anything other than a toy CNN would require impractically
        long iteration times.
"""

import os
import sys
import json
import argparse
import numpy as np
from time import time
from tqdm import tqdm

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

try:
    torch.hammerblade.init()
except:
    pass

# --
# CLI

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cuda', action="store_true")
    parser.add_argument('--hammerblade', action="store_true")
    parser.add_argument('--num-epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--batch-size', type=int, default=128)

    parser.add_argument('--seed', type=int, default=123)
    return parser.parse_args()

# --
# Define model

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.shape[0], -1)


class Block(nn.Module):
    def __init__(self, in_channels, out_channels, residual=False):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_channels, affine=False, track_running_stats=False),
            nn.ReLU(),
        )

        self.skip = None
        if residual:
            self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False)

    def forward(self, xin):
        x = self.layers(xin)

        if self.skip:
            x = x + self.skip(xin)

        return x


class CIFAR2Net(nn.Module):
    def __init__(self, num_classes=2, in_channels=3, channels=[16, 32, 64, 128], classifier_weight=0.125):
        super().__init__()

        self.layers = nn.Sequential(
            Block(in_channels=in_channels, out_channels=channels[0], residual=False),

            Block(in_channels=channels[0], out_channels=channels[1], residual=True),
            nn.MaxPool2d(kernel_size=2),

            Block(in_channels=channels[1], out_channels=channels[2], residual=True),
            nn.MaxPool2d(kernel_size=2),

            Block(in_channels=channels[2], out_channels=channels[3], residual=True),
            nn.MaxPool2d(kernel_size=8),
            Flatten(),
        )

        self.classifier = nn.Linear(channels[3], num_classes, bias=False)
        self.classifier_weight = classifier_weight

    def forward(self, x):
        x = self.layers(x)
        x = self.classifier(x) * self.classifier_weight
        return x


def set_lr(opt, lr):
    for p in opt.param_groups:
        p['lr'] = lr


def train_one_epoch(model, opt, dataloader, lr, num_epochs, hammerblade):
    _ = model.train()
    for i, (x, y) in enumerate(tqdm(dataloader)):
        if hammerblade: x, y = x.hammerblade(), y.hammerblade()

        progress = (epoch + i / len(dataloader)) / num_epochs
        set_lr(opt, lr * (1 - progress))

        out = model(x)
        opt.zero_grad()
        loss = F.cross_entropy(out, y)
        loss.backward()
        opt.step()
        break # only train one iteration

    return model


def predict(model, dataloader, hammerblade):
    _ = model.eval()
    preds = []
    for x, y in dataloader:
        if hammerblade: x, y = x.hammerblade(), y.hammerblade()

        with open('inference_kernel.json',) as f:
            route = json.load(f)
            torch.hammerblade.profiler.route.set_route_from.json(route)
        torch.hammerblade.profiler.enable()
        out   = model(x)
        preds.append(out.argmax(dim=-1).detach().cpu().numpy())
        torch.hammerblade.profiler.disable()
        exit()  # only predict one iteration

    return np.hstack(preds)


if __name__ == "__main__":

    # --
    # CLI

    args = parse_args()

    _ = np.random.seed(args.seed)
    _ = torch.manual_seed(args.seed + 1)
    if args.cuda:
        _ = torch.cuda.manual_seed(args.seed + 2)

    # --
    # IO

    X_train = torch.FloatTensor(np.load('data/cifar2/X_train.npy'))
    X_test  = torch.FloatTensor(np.load('data/cifar2/X_test.npy'))
    y_train = torch.LongTensor(np.load('data/cifar2/y_train.npy'))
    y_test  = torch.LongTensor(np.load('data/cifar2/y_test.npy'))

    train_dataloader = DataLoader(
        TensorDataset(X_train, y_train),
        batch_size=args.batch_size,
        shuffle=True,
        pin_memory=args.cuda,
    )

    test_dataloader = DataLoader(
        TensorDataset(X_test, y_test),
        batch_size=args.batch_size,
        shuffle=False,
        pin_memory=args.cuda,
    )

    # --
    # Define model

    model = CIFAR2Net()
    if args.hammerblade:
        model = model.hammerblade()
        model.move_buffers_to_cpu(torch.nn.BatchNrom2d, ['num_batches_tracked'])

    print(model, file=sys.stderr)

    opt = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    # --
    # Train

    t = time()
    for epoch in range(args.num_epochs):
        # Train
        model = train_one_epoch(model, opt, train_dataloader,
            lr=args.lr, num_epochs=args.num_epochs, hammerblade=args.hammerblade)

        # Evaluate
        preds = predict(model, test_dataloader, hammerblade=args.hammerblade)

print("convnet Training Cosim is Done!")

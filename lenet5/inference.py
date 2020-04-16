"""
Inference on a small CNN
03/16/2020 Bandhav Veluri
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import copy
import time
from torch.utils.data     import DataLoader
from torchvision          import transforms
from torchvision.datasets import MNIST

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import model
import utils

torch.manual_seed(42)

if __name__ == "__main__":
    # Parse inference arguments
    args = utils.argparse_inference()

    # Data
    transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])

    test_data  = MNIST('./data', train=False, download=True,
                       transform=transforms)

    test_loader  = DataLoader(test_data, batch_size=args.batch_size, num_workers=0)

    model = model.LeNet5()
    print(model)

    model.load_state_dict(torch.load(args.filename))

    if args.hammerblade:
        model.hammerblade()
        print("Model is set to run on HammerBlade")
    else:
        model.cpu()
        print("Model is set to run on CPU")

    # Quit here if dry run
    if args.dry:
      exit(0)

    # Inference
    test(model, test_loader, loss_func, args.nbatch)

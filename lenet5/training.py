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
    # Parse trainign arguments
    args = utils.argparse_training()

    # Data
    transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])

    train_data  = MNIST('./data', train=True, download=True,
                       transform=transforms)

    train_loader  = DataLoader(train_data, batch_size=args.batch_size, num_workers=0)

    # Model & Hyper-parameters
    BATCH_SIZE = 32
    LEARNING_RATE = 0.02
    MOMENTUM = 0.9
    EPOCHS = 1

    model = model.LeNet5()
    print(model)
    
    if args.hammerblade:
        model.hammerblade()
        print("Model is set to run on HammerBlade")
    else:
        model.cpu()
        print("Model is set to run on CPU")

    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    loss_func = nn.CrossEntropyLoss()

    # Quit here if dry run
    if args.dry:
        exit(0)
    
    utils.train(model, train_loader, optimizer, loss_func, args.nepoch, args.nbatch)
    
    # Save model
    if args.save_model:
        print("Saving model to " + args.save_filename)
        model_cpu = copy.deepcopy(model)
        model_cpu.to(torch.device("cpu"))
        torch.save(model_cpu.state_dict(), args.save_filename)

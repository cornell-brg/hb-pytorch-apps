#!/bin/python
#=========================================================================
# pytorch-mnist
# adapted from
# https://github.com/iam-mhaseeb/Multi-Layer-Perceptron-MNIST-with-PyTorch
# https://medium.com/@aungkyawmyint_26195/multi-layer-perceptron-mnist-pytorch-463f795b897a
# https://towardsdatascience.com/multi-layer-perceptron-usingfastai-and-pytorch-9e401dd288b8
#=========================================================================

import sys
import os
import torch
from torch                import nn
from torch.utils.data     import DataLoader
from torchvision          import transforms
from torchvision.datasets import MNIST
import numpy as np
import argparse
import copy
import time
import random
from tqdm import tqdm
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from utils import parse_model_args, PrintLayer

#-------------------------------------------------------------------------
# Parse command line arguments
#-------------------------------------------------------------------------

args = parse_model_args()

#-------------------------------------------------------------------------
# Prepare Dataset
#-------------------------------------------------------------------------

train_data = MNIST( './data', train=True, download=True,
                    transform=transforms.ToTensor() )
test_data  = MNIST( './data', train=False, download=True,
                    transform=transforms.ToTensor() )

train_loader = DataLoader(train_data, batch_size=args.batch_size, num_workers=0)
test_loader  = DataLoader(test_data, batch_size=args.batch_size, num_workers=0)

#-------------------------------------------------------------------------
# Multilayer Preception for MNIST
#-------------------------------------------------------------------------

class MLPModel(nn.Module):

  def __init__(self):
    super(MLPModel, self).__init__()

    self.mnist = nn.Sequential \
    (
      nn.Linear(784, 128),
      nn.ReLU(),
      nn.Dropout(0.2),
      PrintLayer(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Dropout(0.2),
      PrintLayer(),
      nn.Linear(64, 10),
    )

  def forward(self, x):
    return self.mnist(x)

#-------------------------------------------------------------------------
# main
#-------------------------------------------------------------------------

model = MLPModel()

# Load pretrained model if necessary
if args.load_model:
  model.load_state_dict(torch.load(args.model_filename))

# Move model to HammerBlade if using HB
if args.hammerblade:
  model.to(torch.device("hammerblade"))

print(model)

# Quit here if dry run
if args.dry:
  exit(0)

# Training
if args.training:
  print("Training for " + str(args.nepoch) + " epochs")

  optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
  criterion = nn.CrossEntropyLoss()

  # Timer
  training_start = time.time()

  for epoch in range(args.nepoch):
    # Batch counter
    batches = 0

    # Monitor training loss
    train_loss = 0.0

    # Prep model for *training*
    model.train()

    for data, target in tqdm(train_loader, total=len(train_loader)):
      # Advance batch counter
      if batches >= args.nbatch:
        break
      batches += 1

      # Clear the gradients of all optimized variables
      optimizer.zero_grad()
      # Move data to correct device
      if args.hammerblade:
        input_data = data.view(-1, 28*28).hammerblade()
        input_target = target.hammerblade()
      else:
        input_data = data.view(-1, 28*28)
        input_target = target
      # Forward pass: compute predicted outputs by passing inputs to the model
      output = model(input_data)
      if args.verbose > 1:
        print("output")
        print(output)
      # Calculate the loss
      loss = criterion(output, input_target)
      # Backward pass: compute gradient of the loss with respect to model parameters
      loss.backward()
      # Perform a single optimization step (parameter update)
      optimizer.step()
      # Update running training loss
      train_loss += loss.item()*data.size(0)

    # Print training statistics
    # Calculate average loss over an epoch
    train_loss = train_loss/len(train_loader.dataset)

    print('Epoch: {} \tTraining Loss: {:.6f}'.format(
      epoch+1,
      train_loss
    ))

  print("--- %s seconds ---" % (time.time() - training_start))

#-------------------------------------------------------------------------

# Inference
if args.inference:
  print("Inference ...")

  # Batch counter
  batches = 0

  criterion = nn.CrossEntropyLoss()

  # Timer
  inference_start = time.time()

  # Initialize lists to monitor test loss and accuracy
  test_loss = 0.0
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))

  # Prep model for *evaluation*
  model.eval()

  for data, target in tqdm(test_loader, total=len(test_loader)):
    # Advance batch counter
    if batches >= args.nbatch:
      break
    batches += 1

    # Move data to correct device
    if args.hammerblade:
      input_data = data.view(-1, 28*28).hammerblade()
      input_target = target.hammerblade()
    else:
      input_data = data.view(-1, 28*28)
      input_target = target
    # Forward pass: compute predicted outputs by passing inputs to the model
    output = model(input_data)
    if args.verbose > 1:
      print("output")
      print(output)
    # Calculate the loss
    loss = criterion(output, input_target)
    # Ppdate test loss
    test_loss += loss.item()*data.size(0)
    # Convert output probabilities to predicted class
    _, pred = torch.max(output.cpu(), 1)
    # Compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    for i in range(len(target)):
      label = target.data[i]
      class_correct[label] += correct[i].item()
      class_total[label] += 1

  print("--- %s seconds ---" % (time.time() - inference_start))

  # Calculate and print avg test loss
  test_loss = test_loss/len(test_loader.sampler)
  print('Test Loss: {:.6f}\n'.format(test_loss))

  # Calculate and print accuracy
  print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))

#-------------------------------------------------------------------------
# Model saving
#-------------------------------------------------------------------------

if args.save_model:
  print("Saving model to " + args.model_filename)
  model_cpu = copy.deepcopy(model)
  model_cpu.to(torch.device("cpu"))
  torch.save(model_cpu.state_dict(), args.save_filename)

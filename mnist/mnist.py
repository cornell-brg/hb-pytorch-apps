#!/bin/python
#=========================================================================
# pytorch-mnist
#=========================================================================

import torch
from torch                import nn
from torch.utils.data     import DataLoader
from torchvision          import transforms
from torchvision.datasets import MNIST
import numpy as np
import argparse
import copy

#-------------------------------------------------------------------------
# Parse command line arguments
#-------------------------------------------------------------------------

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--nepoch', default=30, type=int,
                    help="number of training epochs")
parser.add_argument('--hammerblade', default=False, action='store_true',
                    help="run MLP MNIST on HammerBlade")
parser.add_argument("--verbosity", default=False, action='store_true',
                    help="increase output verbosity")
parser.add_argument("--print-internal", default=False, action='store_true',
                    help="print internal buffers")
parser.add_argument("--dry", default=False, action='store_true',
                    help="dry run")
parser.add_argument("--save-model", default=False, action='store_true',
                    help="save trained model to file")
parser.add_argument('--save-filename', default="trained_model", type=str,
                    help="filename of the saved model")
args = parser.parse_args()

#-------------------------------------------------------------------------
# Prepare Dataset
#-------------------------------------------------------------------------

train_data = MNIST( './data', train=True, download=True,
                    transform=transforms.ToTensor() )
test_data  = MNIST( './data', train=False, download=True,
                    transform=transforms.ToTensor() )

train_loader = DataLoader(train_data, batch_size=20, num_workers=0)
test_loader  = DataLoader(test_data, batch_size=20, num_workers=0)

#-------------------------------------------------------------------------
# Print Layer
#-------------------------------------------------------------------------
class PrintLayer(nn.Module):

  def __init__(self):
    super(PrintLayer, self).__init__()

  def forward(self, x):
    if args.print_internal:
      print(x)
    return x

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
print(model)

if args.hammerblade:
  model.hammerblade()
  print("model is set to run on HammerBlade")
else:
  model.cpu()
  print("model is set to run on CPU")

optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
criterion = nn.CrossEntropyLoss()

# quit here if dry run
if args.dry:
  exit(0)

#-------------------------------------------------------------------------
# Training
#-------------------------------------------------------------------------

model.train()

print('Training starting ...')

# monitor training progress
counter = 0

for epoch in range(args.nepoch):
  # monitor training loss
  train_loss = 0.0

  ###################
  # train the model #
  ###################
  for data, target in train_loader:
    # clear the gradients of all optimized variables
    optimizer.zero_grad()
    # move data to correct device
    if args.hammerblade:
      input_data = data.view(-1, 28*28).hammerblade()
      input_target = target.hammerblade()
    else:
      input_data = data.view(-1, 28*28)
      input_target = target
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(input_data)
    # calculate the loss
    loss = criterion(output, input_target)
    # backward pass: compute gradient of the loss with respect to model parameters
    loss.backward()
    # perform a single optimization step (parameter update)
    optimizer.step()
    # update running training loss
    train_loss += loss.item()*data.size(0)

    # training progress counter
    counter += 1
    if args.verbosity and counter % 100 == 0:
      print('\t{:.3%}'.format(counter / 3000.0 / args.nepoch))

  # print training statistics
  # calculate average loss over an epoch
  train_loss = train_loss/len(train_loader.dataset)

  print('Epoch: {} \tTraining Loss: {:.6f}'.format(
    epoch+1,
    train_loss
  ))

  #-------------------------------------------------------------------------

  ##############
  # evaluation #
  ##############

  # initialize lists to monitor test loss and accuracy
  test_loss = 0.0
  class_correct = list(0. for i in range(10))
  class_total = list(0. for i in range(10))

  # prep model for *evaluation*
  model.eval()

  for data, target in test_loader:
    # move data to correct device
    if args.hammerblade:
      input_data = data.view(-1, 28*28).hammerblade()
      input_target = target.hammerblade()
    else:
      input_data = data.view(-1, 28*28)
      input_target = target
    # forward pass: compute predicted outputs by passing inputs to the model
    output = model(input_data)
    # calculate the loss
    loss = criterion(output, input_target)
    # update test loss
    test_loss += loss.item()*data.size(0)
    # convert output probabilities to predicted class
    _, pred = torch.max(output.cpu(), 1)
    # compare predictions to true label
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))
    for i in range(len(target)):
      label = target.data[i]
      class_correct[label] += correct[i].item()
      class_total[label] += 1

  # calculate and print avg test loss
  test_loss = test_loss/len(test_loader.sampler)
  print('Test Loss: {:.6f}\n'.format(test_loss))

  # calculate and print accuracy
  print('\nTest Accuracy (Overall): %2d%% (%2d/%2d)' % (
        100. * np.sum(class_correct) / np.sum(class_total),
            np.sum(class_correct), np.sum(class_total)))

#-------------------------------------------------------------------------
# Model saving
#-------------------------------------------------------------------------

if args.save_model:
  print("Saving model to " + args.save_filename)
  model_cpu = copy.deepcopy(model)
  model_cpu.to(torch.device("cpu"))
  torch.save(model_cpu.state_dict(), args.save_filename)

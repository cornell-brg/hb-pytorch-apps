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
parser.add_argument('--nbatch', default=1, type=int,
                    help="number of batches to be tested")
parser.add_argument("--verbosity", default=False, action='store_true',
                    help="increase output verbosity")
parser.add_argument("--print-internal", default=False, action='store_true',
                    help="print internal buffers")
parser.add_argument("--dry", default=False, action='store_true',
                    help="dry run")
parser.add_argument('--filename', default="trained_model", type=str,
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

# create cpu model and load pre-trained parameters
model_cpu = MLPModel()
model_cpu.load_state_dict(torch.load(args.filename))

# create a hammerblade model by deepcopying
model = copy.deepcopy(model_cpu)
model.to(torch.device("hammerblade"))

print("model on CPU:")
print(model_cpu)
print("model on HammerBlade:")
print(model)

# set both models to use eval mode
model_cpu.eval()
model.eval()

# quit here if dry run
if args.dry:
  exit(0)

print("Running inference ...")

batch_counter = 0

for data, target in test_loader:
  if batch_counter >= args.nbatch:
    break
  output_cpu = model_cpu(data.view(-1, 28*28))
  output_hb  = model(data.view(-1, 28*28).hammerblade())
  assert output_hb.device == torch.device("hammerblade")
  assert torch.allclose(output_cpu, output_hb.cpu(), atol=1e-06)
  if args.verbosity:
    print("batch " + str(batch_counter))
    print("output_cpu")
    print(output_cpu)
    print("output_hb")
    print(output_hb)
  batch_counter += 1

print("done!")

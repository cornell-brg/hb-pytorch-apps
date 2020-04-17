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
from utils import *

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

train_loader = DataLoader( train_data, batch_size=args.batch_size, num_workers=0 )
test_loader  = DataLoader( test_data, batch_size=args.batch_size, num_workers=0 )

#-------------------------------------------------------------------------
# Input formatting layer
#-------------------------------------------------------------------------
class FormatInput( nn.Module ):

  def __init__( self ):
    super( FormatInput, self ).__init__()

  def forward( self, x ):
    return x.view( -1, 28 * 28 )

#-------------------------------------------------------------------------
# Multilayer Preception for MNIST
#-------------------------------------------------------------------------

class MLPModel( nn.Module ):

  def __init__( self ):
    super( MLPModel, self ).__init__()

    self.mnist = nn.Sequential \
    (
      FormatInput(),
      nn.Linear( 784, 128 ),
      nn.ReLU(),
      nn.Dropout( 0.2 ),
      nn.Linear( 128, 64 ),
      nn.ReLU(),
      nn.Dropout( 0.2 ),
      nn.Linear( 64, 10 ),
    )

  def forward( self, x ):
    return self.mnist( x )

#-------------------------------------------------------------------------
# main
#-------------------------------------------------------------------------

model = MLPModel()

# Load pretrained model if necessary
if args.load_model:
  model.load_state_dict( torch.load(args.model_filename) )

# Move model to HammerBlade if using HB
if args.hammerblade:
  model.to( torch.device("hammerblade") )

print( model )

# Quit here if dry run
if args.dry:
  exit( 0 )

# Training
if args.training:

  optimizer = torch.optim.SGD( model.parameters(), lr=0.01 )
  criterion = nn.CrossEntropyLoss()

  train \
      (
         model,
         train_loader,
         optimizer,
         criterion,
         args
       )

#-------------------------------------------------------------------------

# Inference
if args.inference:

  criterion = nn.CrossEntropyLoss()

  inference \
          (
            model,
            test_loader,
            criterion,
            args
          )

#-------------------------------------------------------------------------
# Model saving
#-------------------------------------------------------------------------

if args.save_model:
  save_model( model, args.model_filename )

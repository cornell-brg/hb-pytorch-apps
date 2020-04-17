# Helper functions for PyTorch-Apps
# 04/17/2020 Bandhav Veluri, Lin Cheng

import argparse
import numpy as np
import random
import torch
import torch.nn as nn
from tqdm import tqdm

#-------------------------------------------------------------------------
# Parse command line options.
# If a workload has options that are specific to it, it should pass in a
# function which adds those arguments
#-------------------------------------------------------------------------

def parse_model_args( workload_args=None ):

  parser = argparse.ArgumentParser( formatter_class=argparse.ArgumentDefaultsHelpFormatter )

  # Common options
  parser.add_argument( '--nepoch', default=-1, type=int,
                       help="number of training epochs" )
  parser.add_argument( '--nbatch', default=-1, type=int,
                       help="number of training/inference batches" )
  parser.add_argument( '--batch-size', default=32, type=int,
                       help="size of each batch" )
  parser.add_argument( '--hammerblade', default=False, action='store_true',
                       help="run MLP MNIST on HammerBlade" )
  parser.add_argument( '--training', default=False, action='store_true',
                       help="run training phase" )
  parser.add_argument( '--inference', default=False, action='store_true',
                       help="run inference phase" )
  parser.add_argument( "-v", "--verbose", default=0, action='count',
                       help="increase output verbosity" )
  parser.add_argument( "--save-model", default=False, action='store_true',
                       help="save trained model to file" )
  parser.add_argument( "--load-model", default=False, action='store_true',
                       help="load trained model from file" )
  parser.add_argument( '--model-filename', default="trained_model", type=str,
                       help="filename of the saved model" )
  parser.add_argument( '--seed', default=42, type=int,
                       help="manual random seed" )
  parser.add_argument( "--dry", default=False, action='store_true',
                       help="dry run" )

  # Inject workload specific options
  if workload_args is not None:
    workload_args( parser )

  # Parse arguments
  args = parser.parse_args()

  # By default, we do both training and inference
  if ( not args.training ) and ( not args.inference ):
    args.training = True
    args.inference = True

  # If not specified, run 30 epochs
  if args.nepoch == -1:
    args.nepoch = 30

  # If nbatch is set, nepoch is forced to be 1
  if args.nbatch == -1:
    args.nbatch = 65535
  else:
    args.nepoch = 1

  # Set random number seeds
  torch.manual_seed( args.seed )
  np.random.seed( args.seed + 1 )
  random.seed( args.seed + 2 )

  # Dump configs
  if args.verbose > 0:
    print(args)

  return args

#-------------------------------------------------------------------------
# Print Layer
#-------------------------------------------------------------------------

class PrintLayer(nn.Module):

  def __init__(self):
    super(PrintLayer, self).__init__()

  def forward(self, x):
    if args.verbose > 1:
      print(x)
    return x


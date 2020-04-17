#!/bin/python
#=========================================================================
# Key kernels of MLP MNIST in isolation
# 04/17/2020 Lin Cheng (lc873@cornell.edu)
#=========================================================================

import torch
import numpy as np
import argparse
import time
import random

#-------------------------------------------------------------------------
# Parse command line arguments
#-------------------------------------------------------------------------

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--list-kernels', default=False, action='store_true',
                    help="print isolated kernels")
parser.add_argument('--full-data', default=False, action='store_true',
                    help="run kernels with full data size")
parser.add_argument('--reduced-data', default=False, action='store_true',
                    help="run kernels with reduced (1/16) data size")
parser.add_argument('--seed', default=42, type=int,
                    help="manual random seed")
args = parser.parse_args()

# Run full data by default
if (not args.full_data) and (not args.reduced_data):
  args.full_data = True

# Set random seeds
torch.manual_seed(args.seed)
np.random.seed(args.seed + 1)
random.seed(args.seed + 2)

#-------------------------------------------------------------------------
# Key kernel impl
#-------------------------------------------------------------------------

def addmm(inputs):
  # Unpack inputs
  M = inputs[0]
  N = inputs[1]
  P = inputs[2]

  # Create random inputs
  self = torch.randn(M, P)
  mat1 = torch.randn(M, N)
  mat2 = torch.randn(N, P)

  print("--- running addmm (%d, %d, %d) ---" % (M, N, P))

  # Timer
  start = time.time()

  torch.addmm(self, mat1, mat2)

  end = time.time()

  print("--- %s seconds ---" % (end - start))

def add(inputs):
  # Unpack inputs
  N = inputs[0]

  # Create random inputs
  data = troch.randn(N)

  print("--- running add (%d) ---" % (N))

  # Timer
  start = time.time()

  torch.add(data, data)

  end = time.time()

  print("--- %s seconds ---" % (end - start))

#-------------------------------------------------------------------------
# Kernels
#-------------------------------------------------------------------------

"""
Kernel      Input Data Size      Tr  In    KFLOPs
---------------------------------------------------
addmm-1    M=20,  N=784, P=128    *   *    4016.6
addmm-2    M=20,  N=128, P=64     *   *     329.0
addmm-3    M=20,  N=64,  P=10     *   *      25.8
addmm-4    M=20,  N=10,  P=64     *          26.9
addmm-5    M=10,  N=20,  P=64     *          26.2
addmm-6    M=20,  N=64,  P=128    *         330.2
addmm-7    M=64,  N=20,  P=128    *         335.9
addmm-8    M=128, N=20,  P=784    *        4114.4
add-1      N=100352               *           0.1
"""

key_kernels = { "addmm-1" : { "kernel_impl"  : addmm,
                              "full_data"    : (20, 784, 128),
                              "reduced_data" : (10, 784, 16)
                            },
                "addmm-2" : { "kernel_impl"  : addmm,
                              "full_data"    : (20, 128, 64),
                              "reduced_data" : (10, 128, 8)
                            },
                "addmm-3" : { "kernel_impl"  : addmm,
                              "full_data"    : (20, 64, 10),
                              "reduced_data" : (10, 64, 2)
                            },
                "addmm-4" : { "kernel_impl"  : addmm,
                              "full_data"    : (20, 10, 64),
                              "reduced_data" : (10, 10, 8)
                            },
                "addmm-5" : { "kernel_impl"  : addmm,
                              "full_data"    : (10, 20, 64),
                              "reduced_data" : (5, 20, 8)
                            },
                "addmm-6" : { "kernel_impl"  : addmm,
                              "full_data"    : (20, 64, 128),
                              "reduced_data" : (10, 64, 16)
                            },
                "addmm-7" : { "kernel_impl"  : addmm,
                              "full_data"    : (64, 20, 128),
                              "reduced_data" : (32, 20, 16)
                            },
                "addmm-8" : { "kernel_impl"  : addmm,
                              "full_data"    : (128, 20, 784),
                              "reduced_data" : (64, 20, 98)
                            },
                "add-1"   : { "kernel_impl"  : add,
                             "full_data"    : (100352),
                             "reduced_data" : (6272)
                           },
              }

class KeyKernel:

  def __init__(self, name, kernel_impl, full_data, reduced_data):
    self.name = name
    self.kernel_impl = kernel_impl
    self.full_data = full_data
    self.reduced_data = reduced_data

  def run_full_data(self):
    self.kernel_impl(full_data)

  def run_reduced_data(self):
    self.kernel_impl(reduced_data)

  def print(self):
    print("kernel %s:" % (self.name))
    print("  - full data    %s" % (str(self.full_data)))
    print("  - reduced data %s" % (str(self.reduced_data)))
    print()

# Parse kernels
key_kernel_objs = {}

for name in key_kernels.keys():
  kernel_impl  = key_kernels[name]["kernel_impl"]
  full_data    = key_kernels[name]["full_data"]
  reduced_data = key_kernels[name]["reduced_data"]
  key_kernel_objs[name] = KeyKernel(name, kernel_impl, full_data, reduced_data)

#-------------------------------------------------------------------------
# List kernels
#-------------------------------------------------------------------------

if args.list_kernels:
  print()
  for name in key_kernel_objs.keys():
    key_kernel_objs[name].print()
  exit(0)

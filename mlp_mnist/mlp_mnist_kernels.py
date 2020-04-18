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
parser.add_argument('--kernels', default='', type=str,
                    help="which kernels to run")
parser.add_argument('--full-data', default=False, action='store_true',
                    help="run kernels with full data size")
parser.add_argument('--reduced-data', default=False, action='store_true',
                    help="run kernels with reduced (1/16) data size")
parser.add_argument('--hammerblade', default=False, action='store_true',
                    help="run kernels on HammerBlade")
parser.add_argument('--seed', default=42, type=int,
                    help="manual random seed")
args = parser.parse_args()

# Run full data by default
if (not args.full_data) and (not args.reduced_data):
  args.full_data = True

# Parse which kernels to run
args.kernels_to_run = args.kernels.split(",")

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

  # Move to HB if necessary
  if args.hammerblade:
    self = self.hammerblade()
    mat1 = mat1.hammerblade()
    mat2 = mat2.hammerblade()

  print("  --- running addmm (%d, %d, %d) ---" % (M, N, P))

  # Timer
  start = time.time()

  torch.addmm(self, mat1, mat2)

  end = time.time()

  print("  --- %s seconds ---" % (end - start))

def add(inputs):
  # Unpack inputs
  N = inputs[0]

  # Create random inputs
  data = torch.randn(N)

  # Move to HB if necessary
  if args.hammerblade:
    data = data.hammerblade()

  print("  --- running add (%d) ---" % (N))

  # Timer
  start = time.time()

  torch.add(data, data)

  end = time.time()

  print("  --- %s seconds ---" % (end - start))

#-------------------------------------------------------------------------
# Kernels
#-------------------------------------------------------------------------

"""
Kernel      Input Data Size      Tr  In    KFLOPs
---------------------------------------------------
addmm-1    M=32,  N=784, P=128    *   *    4016.6
addmm-2    M=32,  N=128, P=64     *   *     329.0
addmm-3    M=32,  N=64,  P=10     *   *      25.8
addmm-4    M=32,  N=10,  P=64     *          26.9
addmm-5    M=10,  N=32,  P=64     *          26.2
addmm-6    M=32,  N=64,  P=128    *         330.2
addmm-7    M=64,  N=32,  P=128    *         335.9
addmm-8    M=128, N=32,  P=784    *        4114.4
add-1      N=100352               *           0.1
"""

key_kernels = { "addmm-1" : { "kernel_impl"  : addmm,
                              "full_data"    : [32, 784, 128],
                              "reduced_data" : [2, 784, 128]
                            },
                "addmm-2" : { "kernel_impl"  : addmm,
                              "full_data"    : [32, 128, 64],
                              "reduced_data" : [2, 128, 64]
                            },
                "addmm-3" : { "kernel_impl"  : addmm,
                              "full_data"    : [32, 64, 10],
                              "reduced_data" : [2, 64, 10]
                            },
                "addmm-4" : { "kernel_impl"  : addmm,
                              "full_data"    : [32, 10, 64],
                              "reduced_data" : [2, 10, 64]
                            },
                "addmm-5" : { "kernel_impl"  : addmm,
                              "full_data"    : [10, 32, 64],
                              "reduced_data" : [10, 2, 64]
                            },
                "addmm-6" : { "kernel_impl"  : addmm,
                              "full_data"    : [32, 64, 128],
                              "reduced_data" : [2, 64, 128]
                            },
                "addmm-7" : { "kernel_impl"  : addmm,
                              "full_data"    : [64, 32, 128],
                              "reduced_data" : [64, 2, 128]
                            },
                "addmm-8" : { "kernel_impl"  : addmm,
                              "full_data"    : [128, 32, 784],
                              "reduced_data" : [128, 2, 784]
                            },
                "add-1"   : { "kernel_impl"  : add,
                             "full_data"    : [100352],
                             "reduced_data" : [6272]
                           },
              }

class KeyKernel:

  def __init__(self, name, kernel_impl, full_data, reduced_data):
    self.name = name
    self.kernel_impl = kernel_impl
    self.full_data = full_data
    self.reduced_data = reduced_data

  def run_full_data(self):
    self.kernel_impl(self.full_data)

  def run_reduced_data(self):
    self.kernel_impl(self.reduced_data)

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

# By default, run all kernels
if args.kernels == '':
  args.kernels_to_run = key_kernels.keys()

for k in args.kernels_to_run:
  if not k in key_kernel_objs:
    print("ERROR: unrecognized kernel -- %s" % (k))
    exit(1)

#-------------------------------------------------------------------------
# List kernels
#-------------------------------------------------------------------------

if args.list_kernels:
  print()
  for name in key_kernel_objs.keys():
    key_kernel_objs[name].print()
  exit(0)

#-------------------------------------------------------------------------
# Run kernels
#-------------------------------------------------------------------------

print()
for k in args.kernels_to_run:

  print("Kernel %s:" % (k))

  # Full data
  if args.full_data:
    key_kernel_objs[k].run_full_data()

  # Reduced data
  if args.reduced_data:
    key_kernel_objs[k].run_reduced_data()

  print()

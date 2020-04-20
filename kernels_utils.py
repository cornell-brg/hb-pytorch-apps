"""
Helper functions for isolated workload kernels
04/20/2020 Lin Cheng (lc873@cornell.edu)
"""

import argparse
import numpy as np
import random
import time
import torch

# -------------------------------------------------------------------------
# Parse command line arguments
# -------------------------------------------------------------------------


def parse_kernels_in_isolation_args(workload_args=None):
    """
    Parse command line options.
    If a workload has options that are specific to it, it should pass in a
    function which adds those arguments
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Common options
    parser.add_argument('--list-kernels', default=False, action='store_true',
                        help="print isolated kernels")
    parser.add_argument('--kernels', default='', type=str,
                        help="which kernels to run")
    parser.add_argument('--full-data', default=False, action='store_true',
                        help="run kernels with full data size")
    parser.add_argument('--reduced-data', default=False, action='store_true',
                        help="run kernels with reduced (1/16) data size")
    parser.add_argument('--hammerblade', default=False, action='store_true',
                        help="run MLP MNIST on HammerBlade")
    parser.add_argument("-v", "--verbose", default=0, action='count',
                        help="increase output verbosity")
    parser.add_argument('--seed', default=42, type=int,
                        help="manual random seed")

    # Inject workload specific options
    if workload_args is not None:
        workload_args(parser)

    # Parse arguments
    args = parser.parse_args()

    # Run full data by default
    if (not args.full_data) and (not args.reduced_data):
        args.full_data = True

    # Set random number seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed + 1)
    random.seed(args.seed + 2)

    # Dump configs
    if args.verbose > 0:
        print(args)

    return args

# -------------------------------------------------------------------------
# Key kernel wrapper class
# -------------------------------------------------------------------------


class KeyKernel:

    def __init__(self, name, kernel_impl, full_data, reduced_data):
        self.name = name
        self.kernel_impl = kernel_impl
        self.full_data = full_data
        self.reduced_data = reduced_data

    def run_full_data(self, hammerblade):
        self.kernel_impl(self.full_data, hammerblade)

    def run_reduced_data(self, hammerblade):
        self.kernel_impl(self.reduced_data, hammerblade)

    def _print(self):
        print("kernel %s:" % (self.name))
        print("  - full data    %s" % (str(self.full_data)))
        print("  - reduced data %s" % (str(self.reduced_data)))
        print()


# -------------------------------------------------------------------------
# Key kernel impl
# -------------------------------------------------------------------------

# ------- addmm ----------
def addmm(inputs, hammerblade):
    # Unpack inputs
    M = inputs[0]
    N = inputs[1]
    P = inputs[2]

    # Create random inputs
    self = torch.randn(M, P)
    mat1 = torch.randn(M, N)
    mat2 = torch.randn(N, P)

    # Move to HB if necessary
    if hammerblade:
        self = self.hammerblade()
        mat1 = mat1.hammerblade()
        mat2 = mat2.hammerblade()

    print("  --- running addmm (%d, %d, %d) ---" % (M, N, P))

    # Timer
    start = time.time()

    torch.addmm(self, mat1, mat2)

    end = time.time()

    print("  --- %s seconds ---" % (end - start))


# ------- add ----------
def add(inputs, hammerblade):
    # Unpack inputs
    N = inputs[0]

    # Create random inputs
    data = torch.randn(N)

    # Move to HB if necessary
    if hammerblade:
        data = data.hammerblade()

    print("  --- running add (%d) ---" % (N))

    # Timer
    start = time.time()

    torch.add(data, data)

    end = time.time()

    print("  --- %s seconds ---" % (end - start))

# -------------------------------------------------------------------------
# Create app kernel wrapper
# -------------------------------------------------------------------------


def create_kernel_wrapper(key_kernels):
    kernels = {}
    for name in key_kernels.keys():
        kernel_impl = key_kernels[name]["kernel_impl"]
        full_data = key_kernels[name]["full_data"]
        reduced_data = key_kernels[name]["reduced_data"]
        kernels[name] = KeyKernel(name, kernel_impl, full_data, reduced_data)
    return kernels

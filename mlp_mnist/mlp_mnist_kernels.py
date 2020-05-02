"""
Key kernels of MLP MNIST in isolation
04/17/2020 Lin Cheng (lc873@cornell.edu)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import kernels_utils as ku

# -------------------------------------------------------------------------
# Kernels
# -------------------------------------------------------------------------

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

# flake8: noqa
key_kernels = { "addmm-1" : { "kernel_impl"  : ku.addmm,
                              "full_data"    : [32, 784, 128],
                              "reduced_data" : [2, 784, 128]
                            },
                "addmm-2" : { "kernel_impl"  : ku.addmm,
                              "full_data"    : [32, 128, 64],
                              "reduced_data" : [2, 128, 64]
                            },
                "addmm-3" : { "kernel_impl"  : ku.addmm,
                              "full_data"    : [32, 64, 10],
                              "reduced_data" : [2, 64, 10]
                            },
                "addmm-4" : { "kernel_impl"  : ku.addmm,
                              "full_data"    : [32, 10, 64],
                              "reduced_data" : [2, 10, 64]
                            },
                "addmm-5" : { "kernel_impl"  : ku.addmm,
                              "full_data"    : [10, 32, 64],
                              "reduced_data" : [10, 2, 64]
                            },
                "addmm-6" : { "kernel_impl"  : ku.addmm,
                              "full_data"    : [32, 64, 128],
                              "reduced_data" : [2, 64, 128]
                            },
                "addmm-7" : { "kernel_impl"  : ku.addmm,
                              "full_data"    : [64, 32, 128],
                              "reduced_data" : [64, 2, 128]
                            },
                "addmm-8" : { "kernel_impl"  : ku.addmm,
                              "full_data"    : [128, 32, 784],
                              "reduced_data" : [128, 2, 784]
                            },
                "add-1"   : { "kernel_impl"  : ku.add,
                             "full_data"    : [100352],
                             "reduced_data" : [6272]
                           },
              }

# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":

    # ---------------------------------------------------------------------
    # Parse command line arguments
    # ---------------------------------------------------------------------

    args = ku.parse_kernels_in_isolation_args()

    # -------------------------------------------------------------------------
    # Load kernels
    # -------------------------------------------------------------------------

    key_kernels = ku.create_kernel_wrapper(key_kernels)

    # By default, run all kernels
    if args.kernels == '':
        args.kernels = key_kernels.keys()
    else:
        # Parse which kernels to run
        args.kernels = args.kernels.split(",")

    for k in args.kernels:
        if k not in key_kernels:
            print("ERROR: unrecognized kernel -- %s" % (k))
            exit(1)

    # -------------------------------------------------------------------------
    # List kernels
    # -------------------------------------------------------------------------

    if args.list_kernels:
        print()
        for name in key_kernels.keys():
            key_kernels[name]._print()
        exit(0)

    # -------------------------------------------------------------------------
    # Run kernels
    # -------------------------------------------------------------------------

    print()
    for k in args.kernels:

        print("Kernel %s:" % (k))

        # Full data
        if args.full_data:
            key_kernels[k].run_full_data(args.hammerblade)

        # Reduced data
        if args.reduced_data:
            key_kernels[k].run_reduced_data(args.hammerblade)

        print()

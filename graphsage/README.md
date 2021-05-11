# GraphSAGE

## Getting the runtime profiling for CPU and HB Host

  The pytorch-sparse-workload branch of hb-pytoch repo should be used to build the HB PyTorch environment under emulation mode in order to run  this code.

  The graphsage.py is the file to profile the CPU runtime, HB Host time and really-chip test. 
  Run "python graphsage.py" for CPU perfiling
  Run "python graphsage.py --hammerblade" for HB Host time profiling.
  This file is also the candidate for really chip test, but right now, it has not been tested yet.

## Code for cosim profiling

  The pytorch-sparse-workload branch of hb-pytoch repo should be used to build the HB PyTorch environment under cosim mode in order to run  this code.
  The graphsage_cosim.py is used to get the graphsage.json file and perform the automatic dispatching over the PyTorch operator.
  Run "python graphsage_cosim.py" for cosim profiling.

    

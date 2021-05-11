# Sinkhorn wmd

## How to run workload

1. [Install HB PyTorch](https://github.com/cornell-brg/hb-pytorch/tree/pytorch-sparse-workload) for custom backend kernels. Currently, the sinkhorn-app kernels are supported in the `pytorch-sparse-workload` branch. Build Pytorch in cosim mode with the help of instructions.
Note: Redispatch should be turned on (by default) in `hb-pytorch/CMakeLists.txt`: `HB_REDISPATCH "Enable conditional redispatch to HB" ON`
2. Download Sinkhorn data. The data currently resides [here](https://cornell.box.com/s/5m6uowgjn8mr5ofdu6hrj62psbm3ztce). You'll need access to Cornell Box.
3. Run using Makefile:
```
make
```
To use multi-threading for parallel simulation of kernels do: `make -j16` for 16 threads. To avoid running energy evaluations using `pycosim.saif`, remove `$(ALL_SAIF)` from dependency from the first rule.
Note: Only 1/16 of the workload is run on cosim (1 POD) for weak-scaling to 16 pods. This parameter is given by `HB_DATA_FRAC` set to `16` in both `test_sinkhorn.py` and `test_cdist.py`.
4. To collect performance results and calculate rough energy numbers:
```
python collect.py -cdist
```
The `-cdist` argument is used to incorporate the kernel simulations for `cdist` in the final results (which we want!).


## Misc
- Why are there 2 `test_*.py` files? `test_sinkhorn.py` is the entire application BUT only offloads the kernel in the iterative loop - sampled MM, sparse reciprocal, SpMM. `test_cdist.py` offloads the kernel `cdist` only. For the final simulations, we need both files collectively.
- Configuring `cdist` size for cosim: In the Makefile, the argument `--cosim_scale` takes in a factor, `n` which means `1/n` size of the kernel is run and the resulting cosim data needs to be scaled by `n` later. This is done for the `cdist` kernel which has regular access pattern and takes long to finish cosim results. A smaller run can be scaled to the entire kernel. However, for final phase2 eval, we ran the entire kernel since we had time.
# The LGC-ISTA app

This folder contains the LGC-ISTA PyTorch code for HB Manycore and HB Manycore with sparse accelerator integration backend.
The lgc_ista.py code is for HB Manycore backend. Both emulation and co-simulation mode are supported, we can use this code to get the HB Host time result, cosimulation result and also runtime result of CPU backend.
The lgc_ista_hbspmvxcel.py code is for HB Manycore with SpMV accelerator integration, we can use this code to get the HB Host time result, cosimulation result. It should be noticed that since the SpMV kernel code of for the SpMV accelerator cannot be emulated by CPU, it is a fake implementation in emulation mode, the result is wrong. But we still use this to get the HB Host time.

## Runtime and energy profiling for HB Manycore only

The pytorch-sparse-workload branch of hb-pytoch repo should be used to build the HB PyTorch environment in order to run  this code.

For CPU profiling, run:
```python
python lgc_ista.py
```
For HB host time profiling, run:
```python
python lgc_ista.py --hammerblade
```
For cosim runtime profiling, run: 
```python
python run-all-hb.py
```
For cosim saif generation, run:
```python
python run-all-hb-saif.py
```
For collecting the energy, run:
```python
python collect-hb-enery.py
```


## Runtime and energy profiling for HB Manycore with SpMV accelerator integration

The hbpytorch-sparse-cosim branch of hb-pytorch repo should be used to build the HB PyTorch environment in order to run this code.
  
For HB host time profiling, run:
```python
python lgc_ista_hbspmvxcel.py --hammerblade
```
For cosim runtime profiling, run:
```python
python run-all-hbspmvxcel.py
```
For cosim saif generation, run:
```python
python run-all-hbspmvxcel-saif.py
```
For collecting the energy, run:
```python
python collect-hbspmvxcel-enery.py
```

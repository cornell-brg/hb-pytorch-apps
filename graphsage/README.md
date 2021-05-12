# GraphSAGE

## Runtime profiling for CPU and HB Host

The pytorch-sparse-workload branch of hb-pytoch repo should be used to build the HB PyTorch environment under emulation mode in order to run  this code.

For CPU perfiling, run:
```python
python graphsage.py
```
For HB Host time profiling, run:
```python
python graphsage.py --hammerblade
```
This file is also the candidate for really chip test, but right now, it has not been tested yet.

## Code for cosim profiling

The pytorch-sparse-workload branch of hb-pytoch repo should be used to build the HB PyTorch environment under cosim mode in order to run  this code.
The graphsage_cosim.py is used to get the graphsage.json file and perform the automatic dispatching over the PyTorch operator.
For cosim training runtime profiling, run:
```python
python run-all-training.py
```
For cosim training saif file generation, run:
```python
python run-all-training-saif.py
```
Collecting the cosim training runtime profiling data, run:
```python
python collect-training.py
```
Collecting the cosim training runtime energy data, run:
```python
python collect-training-energy.py
```
For cosim inference runtime profiling, run:
```python
python run-all-inference.py
```
For cosim inference saif file generation, run:
```python
python run-all-inference-saif.py
```
Collecting the cosim inference runtime profiling data, run:
```python
python collect-inference.py
```
Collecting the cosim inference runtime energy data, run:
```python
python collect-inference-energy.py

    

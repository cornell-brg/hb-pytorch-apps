#!/bin/bash

# convnet/prep.sh

# --
# Download + prepare data

mkdir -p data

python prep.py

# # If you need to save disk space:
# rm data/cifar-10-python.tar.gz
# rm -r data/cifar-10-batches-py
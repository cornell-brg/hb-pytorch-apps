import torch
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from mnist import MNIST

def test_create_mnist_model():
    model_hb = MNIST()
    model_hb.hammerblade()

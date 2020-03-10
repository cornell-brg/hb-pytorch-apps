import torch
from os import sys, path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))
from mnist import MNIST

def test_create_mnist_forward_1():
    model_cpu = MNIST()
    model_hb = MNIST()
    model_hb.hammerblade()
    x = torch.ones(28*28)
    output_cpu = model_cpu(x)
    output_hb = model_hb(x)
    assert output_hb.device == torch.device("hammerblade")
    assert torch.equal(output_hb.cpu(), output_cpu)

def test_create_mnist_forward_2():
    model_cpu = MNIST()
    model_hb = MNIST()
    model_hb.hammerblade()
    x = torch.randn(28*28)
    output_cpu = model_cpu(x)
    output_hb = model_hb(x)
    assert output_hb.device == torch.device("hammerblade")
    assert torch.equal(output_hb.cpu(), output_cpu)

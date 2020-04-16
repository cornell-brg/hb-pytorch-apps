"""
Test on a small CNN
03/16/2020 Bandhav Veluri
"""
import sys
import os
import numpy as np
import torch
import torch.nn as nn
import numpy as np
import copy
import time
from torch.utils.data     import DataLoader
from torchvision          import transforms
from torchvision.datasets import MNIST

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import model
import utils

torch.manual_seed(42)

# Test routine
@torch.no_grad()
def inference(net, loader, loss_func, hb=False):
    test_loss = 0.0
    num_correct = 0

    for batch_idx, (data, labels) in enumerate(loader, 0):
        if hb:
            data, labels = data.hammerblade(), labels.hammerblade()
        output = net(data)
        loss = loss_func(output, labels)
        pred = output.max(1)[1]
        num_correct += pred.eq(labels.view_as(pred)).sum().item()

        if batch_idx == 100:
            break

    test_loss /= len(loader.dataset)
    test_accuracy = 100. * (num_correct / len(loader.dataset))

    print('Test set: Average loss={:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, num_correct, len(loader.dataset), test_accuracy
    ))

if __name__ == "__main__":
    args = utils.argparse_inference()

    # Data
    transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])

    test_data  = MNIST('./data', train=False, download=True,
                       transform=transforms)

    test_loader  = DataLoader(test_data, batch_size=20, num_workers=0)

    # Create CPU model and load pre-trained parameters
    net = model.Network()
    net.load_state_dict(torch.load(args.filename))

    # Create a HammerBlade model by deepcopying
    net_hb = copy.deepcopy(net)
    net_hb.to(torch.device("hammerblade"))

    print("Model:")
    print(net)

    # Set both models to use eval mode
    net.eval()
    net_hb.eval()

    # Quit here if dry run
    if args.dry:
      exit(0)

    print("Running inference ...")

    start_time = time.time()
    batch_counter = 0

    for data, target in test_loader:
      if batch_counter >= args.nbatch:
        break
      output = net(data)
      output_hb  = net_hb(data.hammerblade())
      assert output_hb.device == torch.device("hammerblade")
      assert torch.allclose(output, output_hb.cpu(), atol=utils.ATOL)
      if args.verbosity:
        print("batch " + str(batch_counter))
        print("output_cpu")
        print(output_cpu)
        print("output_hb")
        print(output_hb)
      batch_counter += 1

    print("done!")
    print("--- %s seconds ---" % (time.time() - start_time))

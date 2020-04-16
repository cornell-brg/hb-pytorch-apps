"""
Inference on a small CNN
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

if __name__ == "__main__":
    args = utils.argparse_inference()

    # Data
    transforms = transforms.Compose([
        transforms.Resize((32,32)),
        transforms.ToTensor(),
    ])

    test_data  = MNIST('./data', train=False, download=True,
                       transform=transforms)

    test_loader  = DataLoader(test_data, batch_size=args.batch_size, num_workers=0)

    # Create CPU model and load pre-trained parameters
    model = model.LeNet5()
    model.load_state_dict(torch.load(args.filename))

    # Create a HammerBlade model by deepcopying
    model_hb = copy.deepcopy(model)
    model_hb.to(torch.device("hammerblade"))

    print("Model:")
    print(model)

    # Set both models to use eval mode
    model.eval()
    model_hb.eval()

    # Quit here if dry run
    if args.dry:
      exit(0)

    print("Running inference ...")

    start_time = time.time()
    batch_counter = 0

    for data, target in test_loader:
      if batch_counter >= args.nbatch:
        break
      output = model(data)
      output_hb  = model_hb(data.hammerblade())
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

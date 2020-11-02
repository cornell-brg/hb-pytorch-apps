"""
11/02/2020 Lin Cheng
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch
from torch import nn
from utils import parse_model_args, train, inference, save_model
from generations import *
from parameters import *

# debug

class Dataset(torch.utils.data.Dataset):

    def __init__(self, generator):
        self.generator = generator(256)

    def __len__(self):
        return 50

    def __getitem__(self, index):
        value = self.generator.__next__()
        X = torch.tensor(value[0]).float()
        y = torch.tensor(value[1]).float()

        return X, y

# -------------------------------------------------------------------------
# Multilayer Preception for MNIST
# -------------------------------------------------------------------------


class ChannelEstModel(nn.Module):

    def __init__(self):
        super(ChannelEstModel, self).__init__()

        self.est = nn.Sequential(nn.Linear(256, 500),
                                   nn.ReLU(),
                                   #nn.Dropout(0.2),
                                   nn.Linear(500, 250),
                                   nn.ReLU(),
                                   #nn.Dropout(0.2),
                                   nn.Linear(250, 120),
                                   nn.ReLU(),
                                   nn.Linear(120, 16),
                                   nn.Sigmoid())

    def forward(self, x):
        return self.est(x.view(-1,256))

# -------------------------------------------------------------------------
# Workload specific command line arguments
# -------------------------------------------------------------------------


def extra_arg_parser(parser):
    parser.add_argument('--lr', default=0.01, type=int,
                        help="learning rate")


# -------------------------------------------------------------------------
# Main
# -------------------------------------------------------------------------

if __name__ == "__main__":

    # ---------------------------------------------------------------------
    # Parse command line arguments
    # ---------------------------------------------------------------------

    args = parse_model_args(extra_arg_parser)

    # ---------------------------------------------------------------------
    # Prepare Dataset
    # ---------------------------------------------------------------------

    training_set = Dataset(training_gen)
    training_generator = torch.utils.data.DataLoader(training_set,
                          batch_size=1, shuffle=True,
                          num_workers=0)

    validation_set = Dataset(validation_gen)
    validation_generator = torch.utils.data.DataLoader(training_set,
                          batch_size=1, shuffle=True,
                          num_workers=0)

    # ---------------------------------------------------------------------
    # Model creation and loading
    # ---------------------------------------------------------------------

    LEARNING_RATE = args.lr
    model = ChannelEstModel()

    optimizer = torch.optim.RMSprop(model.parameters(), lr=0.001)
    # optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.MSELoss()

    # Load pretrained model if necessary
    if args.load_model:
        model.load_state_dict(torch.load(args.model_filename))

    # Move model to HammerBlade if using HB
    if args.hammerblade:
        model.to(torch.device("hammerblade"))

    print(model)

    # Quit here if dry run
    if args.dry:
        exit(0)

    # ---------------------------------------------------------------------
    # Training
    # ---------------------------------------------------------------------

    if args.training:

        train(model,
              training_generator,
              optimizer,
              criterion,
              args)

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------

    if args.inference:

        num_error = [0]

        def collector(outputs, targets):
            outputs = outputs.view(-1,16)
            targets = targets.view(-1,16)
            high = torch.ones_like(targets)
            low  = torch.zeros_like(targets)
            outputs = torch.where( outputs > 0.5, high, low).int()
            targets = targets.int()
            diff = torch.logical_xor(outputs, targets).int()
            error = diff.sum()
            num_error[0] += error

        inference(model,
                  validation_generator,
                  criterion,
                  collector,
                  args)

        num_error = float(num_error[0])
        total_bits = len(validation_generator.dataset) * 256 * 16
        BER = 100. * (num_error / total_bits)

        print('Test set: BER: {}/{} ({:.0f}%)\n'.format(
            num_error,
            total_bits,
            BER
        ))

    # ---------------------------------------------------------------------
    # Model saving
    # ---------------------------------------------------------------------

    if args.save_model:
        save_model(model, args.model_filename)

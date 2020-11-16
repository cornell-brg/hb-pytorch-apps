import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import CIFAR10
from utils import parse_model_args, train, inference, save_model


# -------------------------------------------------------------------------
# ResNet-9 for CIFAR-10
# -------------------------------------------------------------------------

class Resnet9Model(nn.Module):
    """
    A class for the ResNet-9 archicture and training / evaluation helper functions

    ...

    Methods
    -------
    forward(xb)
        Performs forward pass for a provided batch of data
    training_step(self, batch)
        Calculates Loss Function for a given batch
    validation_step(self, batch)
        Peforms validation in terms of evaluating loss / accuracy of model for a given batch
    validation_epoch_end(self, outputs)
        Accumulates losses / accuracies among batches to return the overall epoch loss / accuracy
    epoch_end(self, epoch, result)
        Prints training and validation statistics for an epoch
    """

    def __init__(self, in_channels, num_classes):
        def conv_block(in_channels, out_channels, pool=False):
            """Convolutonal Block involving Conv2D -> Batch Normalization -> ReLU -> MaxPool2D."""
            layers = [nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=False)]
            if pool: layers.append(nn.MaxPool2d(2))
            return nn.Sequential(*layers)

        super().__init__()

        self.conv1 = conv_block(in_channels, 64)
        self.conv2 = conv_block(64, 128, pool=True)
        self.res1 = nn.Sequential(conv_block(128, 128), conv_block(128, 128))

        self.conv3 = conv_block(128, 256, pool=True)
        self.conv4 = conv_block(256, 512, pool=True)
        self.res2 = nn.Sequential(conv_block(512, 512), conv_block(512, 512))

        self.classifier = nn.Sequential(nn.MaxPool2d(4),
                                        nn.Flatten(),
                                        nn.Linear(512, num_classes))

    def forward(self, xb):
        """
        Performs forward pass for a provided batch of data

        Parameters
        ----------
        xb : torch.tensor
            A batch of image data
        """

        out = self.conv1(xb)
        out = self.conv2(out)
        out = self.res1(out) + out
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.res2(out) + out
        out = self.classifier(out)
        return out

# -------------------------------------------------------------------------
# Workload specific command line arguments
# -------------------------------------------------------------------------


def extra_arg_parser(parser):
    parser.add_argument('--lr', default=0.01, type=int,
                        help="learning rate")
    parser.add_argument('--download', action= "store_true", default=False,
                        help='Download CIFAR-10 DataSet locally (stored in ./data folder)')

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

    # normalization params: mean and std for RGB channels in CIFAR10
    stats = ((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    # Normalize, Augment
    train_tfms = transforms.Compose([transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor(),
                            transforms.Normalize(*stats,inplace=True)])
    test_tfms = transforms.Compose([transforms.ToTensor(), transforms.Normalize(*stats)])

    train_data = CIFAR10(root='data/', train=True, download=args.download,
                         transform=train_tfms)
    test_data  = CIFAR10(root='data/', train=False, download=args.download,
                         transform=test_tfms)

    train_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                          num_workers=0)
    test_loader  = DataLoader(test_data, batch_size=args.batch_size,
                          num_workers=0)

    # ---------------------------------------------------------------------
    # Model creation and loading
    # ---------------------------------------------------------------------
    IN_CHANNELS = 3;
    NUM_CLASSES = 16;
    model = Resnet9Model(IN_CHANNELS, NUM_CLASSES);

    LEARNING_RATE = args.lr
    lr = LEARNING_RATE
    EPOCHS = args.nepoch
    GRAD_CLIP = 0.1
    WEIGHT_DECAY = 1e-4

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,
                                 weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=lr, epochs=EPOCHS,
                                                steps_per_epoch=len(train_loader))
    criterion = nn.CrossEntropyLoss()

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
              train_loader,
              optimizer,
              criterion,
              args)

    # ---------------------------------------------------------------------
    # Inference
    # ---------------------------------------------------------------------
    if args.inference:
        num_correct = [0]

        def collector(outputs, targets):
            pred = outputs.cpu().max(1)[1]
            num_correct[0] += pred.eq(targets.cpu().view_as(pred)).sum().item()

        inference(model,
                  test_loader,
                  criterion,
                  collector,
                  args)

        num_correct = num_correct[0]
        test_accuracy = 100. * (num_correct / len(test_loader.dataset))

        print('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            num_correct,
            len(test_loader.dataset),
            test_accuracy
        ))

    # ---------------------------------------------------------------------
    # Model saving
    # ---------------------------------------------------------------------
    if args.save_model:
        save_model(model, args.model_filename)

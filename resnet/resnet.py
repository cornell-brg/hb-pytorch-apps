"""
ResNet workload
07/17/2020 Bandhav Veluri
"""

import sys
import os
import torch
import torch.nn as nn
import torchvision
from Select_CIFAR10_Classes import get_class_i, DatasetMaker

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import utils  # noqa: E402

class Block(nn.Module):
   def __init__(self, in_channels, out_channels, residual=False):
       super().__init__()
       self.layers = nn.Sequential(
           nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1,
                     padding=1, bias=False),
           nn.BatchNorm2d(out_channels),
           nn.ReLU(inplace=True),
       )
       self.skip = None
       if residual:
           self.skip = nn.Conv2d(in_channels, out_channels, kernel_size=1,
                                 stride=1, padding=0, bias=False)

   def forward(self, xin):
       x = self.layers(xin)
       if self.skip:
           x = x + self.skip(xin)
       return x

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1,
                      padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            Block(16, 32),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            Block(32, 64),
            nn.MaxPool2d(kernel_size=(2,2), stride=2),
            Block(64, 128),
            nn.AdaptiveMaxPool2d((1, 1)), # global pooling
        )

        self.fc = nn.Linear(128, 2)

        self.logsoftmax = nn.LogSoftmax(dim=-1)

    def forward(self, data):
        x = self.conv(data)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        x = x * 0.125 # scale layer
        x = self.logsoftmax(x)
        return x

def extra_arg_parser(parser):
    parser.add_argument('--lr', default=0.02, type=int,
                        help="learning rate")
    parser.add_argument('--momentum', default=0.9, type=int,
                        help="momentum")

if __name__ == "__main__":
    # Parse arguments
    args = utils.parse_model_args(extra_arg_parser)

    # Model & hyper-parameters
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr
    MOMENTUM = args.momentum

    model = ResNet()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    loss_func = nn.CrossEntropyLoss()

    # Data
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.ToPILImage(),
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True)
    x_train = trainset.data
    y_train = trainset.targets
    first_two_labels_trainset = \
        DatasetMaker(
            [get_class_i(x_train, y_train, 0), get_class_i(x_train, y_train, 1)],
            transforms
        )
    trainloader = torch.utils.data.DataLoader(
        first_two_labels_trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transforms)
    x_test = testset.data
    y_test = testset.targets
    first_two_labels_testset = \
        DatasetMaker(
            [get_class_i(x_test, y_test, 0), get_class_i(x_test, y_test, 1)],
            transforms
        )
    testloader = torch.utils.data.DataLoader(
        first_two_labels_testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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

    # Training
    if args.training:
        utils.train(model, trainloader, optimizer, loss_func, args)

    # Inference
    if args.inference:

        num_correct = [0]

        def collector(outputs, targets):
            pred = outputs.cpu().max(1)[1]
            num_correct[0] += pred.eq(targets.cpu().view_as(pred)).sum().item()

        utils.inference(model, testloader, loss_func, collector, args)

        num_correct = num_correct[0]
        test_accuracy = 100. * (num_correct / len(testloader.dataset))

        print('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            num_correct,
            len(testloader.dataset),
            test_accuracy
        ))

    # Save model
    if args.save_model:
        utils.save_model(model, args.model_filename)

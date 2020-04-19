"""
5-layer CNN workload
03/16/2020 Bandhav Veluri
"""

import sys
import os
import torch
import torch.nn as nn
import torchvision

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

import utils  # noqa: E402


class LeNet5(nn.Module):
    """
    LeNet-5

    https://cs.nyu.edu/~yann/2010f-G22-2565-001/diglib/lecun-98.pdf
    (Page 7)
    """

    def __init__(self):
        super(LeNet5, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(6, 16, kernel_size=(5, 5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=2),
            nn.Conv2d(16, 120, kernel_size=(5, 5)),
        )

        self.fc = nn.Sequential(
            nn.Linear(120, 84),
            nn.ReLU(),
            nn.Linear(84, 10),
            nn.LogSoftmax(dim=-1),
        )

    def forward(self, data):
        x = self.conv(data)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
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

    model = LeNet5()

    optimizer = torch.optim.SGD(
        model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM)

    loss_func = nn.CrossEntropyLoss()

    # Data
    transforms = torchvision.transforms.Compose([
        torchvision.transforms.Resize((32, 32)),
        torchvision.transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(
        root='./data', train=True, download=True, transform=transforms)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)

    testset = torchvision.datasets.MNIST(
        root='./data', train=False, download=True, transform=transforms)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

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

"""
Resnet-18 Workload on CIFAR10
12/07/2020 Preslav Ivanov (pi57@cornell.edu)
Adapted from: https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST
from utils import parse_model_args, train, inference, save_model
from torchvision.datasets import CIFAR10
from torch import Tensor
try:
    from torch.hub import load_state_dict_from_url
except ImportError:
    from torch.utils.model_zoo import load_url as load_state_dict_from_url
from typing import Type, Any, Callable, Union, List, Optional

# -------------------------------------------------------------------------
# Resnet for CIFAR10
# -------------------------------------------------------------------------
def conv3x3(in_planes: int, out_planes: int, stride: int = 1, groups: int = 1, dilation: int = 1) -> nn.Conv2d:
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes: int, out_planes: int, stride: int = 1) -> nn.Conv2d:
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    # A basic block is comprised of 2 ops (i.e. skip over 2 ops)
    # the conv filter dims for the 2 ops are conv3x3, conv3x3, respectively
    # an op is comrised of Conv -> Batch Norm -> ReLU

    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
    ) -> None:
        super(BasicBlock, self).__init__()
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 =  nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 =  nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(
        self,
        block: BasicBlock,
        layers: List[int],
        num_classes: int = 1000,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilation: Optional[List[bool]] = None,
        norm_layer: Optional[Callable[..., nn.Module]] = None
    ) -> None:
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_layer(self, block: BasicBlock, planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

def _resnet(
    arch: str,
    block: BasicBlock,
    layers: List[int],
    progress: bool,
    **kwargs: Any
) -> ResNet:
    model = ResNet(block, layers, **kwargs)
    return model

def resnet18(progress: bool = True, **kwargs: Any) -> ResNet:
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_.

    Args:
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], progress,
                   **kwargs)

# -------------------------------------------------------------------------
# Workload specific command line arguments
# -------------------------------------------------------------------------


def extra_arg_parser(parser):
    parser.add_argument('--lr', default=0.01, type=int,
                        help="learning rate")


if __name__ == "__main__":
    # -------------------------------------------------------------------------
    # Parse command line arguments
    # -------------------------------------------------------------------------

    args = parse_model_args(extra_arg_parser)

    # -------------------------------------------------------------------------
    # Prepare Dataset
    # -------------------------------------------------------------------------

    train_data = CIFAR10('data/', train=True, download=True, 
                         transform=transforms.ToTensor())
    test_data  = CIFAR10('data/', train=False, download=True, 
                         transform=transforms.ToTensor())

    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                              num_workers=0)
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                             num_workers=0)
    
    # -------------------------------------------------------------------------
    # Model creation and loading
    # -------------------------------------------------------------------------

    model = resnet18(True)
    loss_func = nn.CrossEntropyLoss()
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

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    if args.training:
        WEIGHT_DECAY = 1e-4
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, 
                                weight_decay=WEIGHT_DECAY)
        sched = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, epochs=args.nepoch, 
                                                    steps_per_epoch=len(train_loader))

        train(model,
              train_loader,
              optimizer,
              loss_func,
              args)

    # -------------------------------------------------------------------------
    # Inference
    # -------------------------------------------------------------------------

    if args.inference:
        num_correct = [0]
        def collector(outputs, targets):
            pred = outputs.cpu().max(1)[1]
            num_correct[0] += pred.eq(targets.cpu().view_as(pred)).sum().item()

        inference(model,
                  test_loader,
                  loss_func,
                  collector,
                  args)

        num_correct = num_correct[0]
        test_accuracy = 100. * (num_correct / len(test_loader.dataset))

        print('Test set: Accuracy: {}/{} ({:.0f}%)\n'.format(
            num_correct,
            len(test_loader.dataset),
            test_accuracy
        ))

    # -------------------------------------------------------------------------
    # Model saving
    # -------------------------------------------------------------------------
    if args.save_model:
        save_model(model, args.model_filename)

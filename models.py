from functools import partial

import torch.nn as nn
import torchvision.models


def _change_to_cifar(m):
    m.conv1 = nn.Conv2d(3, m.conv1.out_channels, kernel_size=3, padding=1, bias=False)
    nn.init.kaiming_normal_(m.conv1.weight, mode='fan_out', nonlinearity='relu')
    m.maxpool = nn.Identity()
    return m


def _resnet18(num_classes, is_cifar=False, **kwargs):
    model = torchvision.models.resnet.ResNet(
        torchvision.models.resnet.BasicBlock,
        [2, 2, 2, 2],
        num_classes=num_classes,
        **kwargs)
    if is_cifar:
        model = _change_to_cifar(model)
    return model


def _resnet50(num_classes, is_cifar=False, **kwargs):
    model = torchvision.models.resnet.ResNet(
        torchvision.models.resnet.Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        **kwargs)
    if is_cifar:
        model = _change_to_cifar(model)
    return model


def _wide_resnet_50_2(num_classes, is_cifar=False, **kwargs):
    model = torchvision.models.resnet.ResNet(
        torchvision.models.resnet.Bottleneck,
        [3, 4, 6, 3],
        num_classes=num_classes,
        width_per_group=64 * 2,
        **kwargs)
    if is_cifar:
        model = _change_to_cifar(model)
    return model


models = {
    'resnet18': partial(_resnet18, is_cifar=False),
    'resnet50': partial(_resnet50, is_cifar=False),
    'wide_resnet_50_2': partial(_wide_resnet_50_2, is_cifar=False),
    'resnet18_cifar': partial(_resnet18, is_cifar=True),
    'resnet50_cifar': partial(_resnet50, is_cifar=True),
    'wide_resnet_50_2_cifar': partial(_wide_resnet_50_2, is_cifar=True),
}

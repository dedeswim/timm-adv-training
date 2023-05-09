# Copyright 2020 Deepmind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""WideResNet implementation in PyTorch. From:
https://github.com/deepmind/deepmind-research/blob/master/adversarial_robustness/pytorch/model_zoo.py
"""

from functools import partial
from typing import Type

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model
import torch
import torch.nn as nn
import torch.nn.functional as F

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)
CIFAR100_MEAN = (0.5071, 0.4865, 0.4409)
CIFAR100_STD = (0.2673, 0.2564, 0.2762)


BATCHNORM_MOMENTUM = 0.01
BatchNorm2d = partial(nn.BatchNorm2d, momentum=BATCHNORM_MOMENTUM)


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 10,
        'input_size': (3, 32, 32),
        'pool_size': None,
        'crop_pct': 1.0,
        'interpolation': 'bilinear',
        'fixed_input_size': True,
        'mean': IMAGENET_DEFAULT_MEAN,
        'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'init_conv',
        'classifier': 'logits',
        **kwargs
    }


default_cfgs = {
    'wide_resnet28_10_dm': _cfg(),
    'wide_resnet34_10_dm': _cfg(),
    'wide_resnet34_20_dm': _cfg(),
    'wide_resnet70_16_dm': _cfg(),
    'wide_resnet106_16_dm': _cfg(),
}


class _Swish(torch.autograd.Function):
    """Custom implementation of swish."""

    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class Swish(nn.Module):
    """Module using custom implementation."""

    def forward(self, input_tensor):
        return _Swish.apply(input_tensor)


class _Block(nn.Module):
    """WideResNet Block."""

    def __init__(self, in_planes, out_planes, stride, activation_fn: Type[nn.Module] = nn.ReLU):
        super().__init__()
        self.batchnorm_0 = BatchNorm2d(in_planes)
        self.relu_0 = activation_fn()
        # We manually pad to obtain the same effect as `SAME` (necessary when
        # `stride` is different than 1).
        self.conv_0 = nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=0, bias=False)
        self.batchnorm_1 = BatchNorm2d(out_planes)
        self.relu_1 = activation_fn()
        self.conv_1 = nn.Conv2d(out_planes, out_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.has_shortcut = in_planes != out_planes
        if self.has_shortcut:
            self.shortcut = nn.Conv2d(in_planes,
                                      out_planes,
                                      kernel_size=1,
                                      stride=stride,
                                      padding=0,
                                      bias=False)
        else:
            self.shortcut = None
        self._stride = stride

    def forward(self, x):
        if self.has_shortcut:
            x = self.relu_0(self.batchnorm_0(x))
        else:
            out = self.relu_0(self.batchnorm_0(x))
        v = x if self.has_shortcut else out
        if self._stride == 1:
            v = F.pad(v, (1, 1, 1, 1))
        elif self._stride == 2:
            v = F.pad(v, (0, 1, 0, 1))
        else:
            raise ValueError('Unsupported `stride`.')
        out = self.conv_0(v)
        out = self.relu_1(self.batchnorm_1(out))
        out = self.conv_1(out)
        out = torch.add(self.shortcut(x) if self.has_shortcut else x, out)
        return out


class _BlockGroup(nn.Module):
    """WideResNet block group."""

    def __init__(self, num_blocks, in_planes, out_planes, stride, activation_fn: Type[nn.Module] = nn.ReLU):
        super().__init__()
        block = []
        for i in range(num_blocks):
            block.append(
                _Block(i == 0 and in_planes or out_planes,
                       out_planes,
                       i == 0 and stride or 1,
                       activation_fn=activation_fn))
        self.block = nn.Sequential(*block)

    def forward(self, x):
        return self.block(x)


def init_modules(module: nn.Module) -> None:
    if isinstance(module, nn.Conv2d):
        nn.init.trunc_normal_(module.weight)
        if module.bias is not None:
            nn.init.zeros_(module.bias)
    elif isinstance(module, nn.BatchNorm2d):
        nn.init.ones_(module.weight)
        nn.init.zeros_(module.bias)

class DMWideResNet(nn.Module):
    """WideResNet."""

    def __init__(self,
                 num_classes: int = 10,
                 depth: int = 28,
                 width: int = 10,
                 activation_fn: Type[nn.Module] = nn.ReLU,
                 padding: int = 0,
                 in_chans: int = 3,
                 img_size=32,
                 drop_rate: float = 0.,
                 drop_path_rate=None):
        super().__init__()
        # persistent=False to not put these tensors in the module's state_dict and not try to
        # load it from the checkpoint
        self.padding = padding
        self.num_classes = num_classes
        num_channels = [16, 16 * width, 32 * width, 64 * width]
        assert (depth - 4) % 6 == 0
        num_blocks = (depth - 4) // 6
        self.init_conv = nn.Conv2d(in_chans, num_channels[0], kernel_size=3, stride=1, padding=1, bias=False)
        self.layer = nn.Sequential(
            _BlockGroup(num_blocks, num_channels[0], num_channels[1], 1, activation_fn=activation_fn),
            _BlockGroup(num_blocks, num_channels[1], num_channels[2], 2, activation_fn=activation_fn),
            _BlockGroup(num_blocks, num_channels[2], num_channels[3], 2, activation_fn=activation_fn))
        self.batchnorm = BatchNorm2d(num_channels[3])
        self.relu = activation_fn()
        self.logits = nn.Linear(num_channels[3], self.num_classes)
        self.num_channels = num_channels[3]
        self._init_weights()
    
    def _init_weights(self) -> None:
        nn.init.zeros_(self.logits.weight)
        nn.init.zeros_(self.logits.bias)
        init_modules(self)

    def forward(self, x):
        if self.padding > 0:
            x = F.pad(x, (self.padding, ) * 4)
        out = self.init_conv(x)
        out = self.layer(out)
        out = self.relu(self.batchnorm(out))
        out = F.avg_pool2d(out, 8)
        out = out.view(-1, self.num_channels)
        return self.logits(out)


def _create_wide_resnet(variant, pretrained=False, default_cfg=None, **kwargs):
    model = build_model_with_cfg(DMWideResNet, variant, pretrained, **kwargs)
    return model


@register_model
def wide_resnet28_10_dm(pretrained=False, **kwargs):
    model_args = dict(depth=28, width=10, activation_fn=Swish, **kwargs)
    return _create_wide_resnet('wide_resnet28_10_dm', pretrained, **model_args)


@register_model
def wide_resnet34_10_dm(pretrained=False, **kwargs):
    model_args = dict(depth=34, width=10, activation_fn=Swish, **kwargs)
    return _create_wide_resnet('wide_resnet34_10_dm', pretrained, **model_args)


@register_model
def wide_resnet34_20_dm(pretrained=False, **kwargs):
    model_args = dict(depth=34, width=20, activation_fn=Swish, **kwargs)
    return _create_wide_resnet('wide_resnet34_20_dm', pretrained, **model_args)


@register_model
def wide_resnet70_16_dm(pretrained=False, **kwargs):
    model_args = dict(depth=70, width=16, activation_fn=Swish, **kwargs)
    return _create_wide_resnet('wide_resnet70_16_dm', pretrained, **model_args)


@register_model
def wide_resnet106_16_dm(pretrained=False, **kwargs):
    model_args = dict(depth=106, width=16, activation_fn=Swish, **kwargs)
    return _create_wide_resnet('wide_resnet106_16_dm', pretrained, **model_args)

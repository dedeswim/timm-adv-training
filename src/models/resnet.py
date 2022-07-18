'''ResNet in PyTorch

from https://github.com/MadryLab/robustness/blob/a9541241defd9972e9334bfcdb804f6aefe24dc7/robustness/cifar_models/resnet.py

For Pre-activation ResNet, see 'preact_resnet.py'.
Reference:
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models import register_model
from timm.models.helpers import build_model_with_cfg

from ..data import CIFAR_10_MEAN, CIFAR_10_STD


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 10,
        'input_size': (3, 32, 32),
        'pool_size': (4, 4),
        'crop_pct': 1,
        'interpolation': 'bilinear',
        'mean': CIFAR_10_MEAN,
        'std': CIFAR_10_STD,
        'first_conv': 'conv1',
        'classifier': 'linear',
        **kwargs
    }


default_cfgs = {'resnet50_robustness': _cfg(input_size=(3, 32, 32), interpolation='bicubic')}


class SequentialWithArgs(torch.nn.Sequential):

    def forward(self, input, *args, **kwargs):
        vs = list(self._modules.values())
        l = len(vs)
        for i in range(l):
            if i == l - 1:
                input = vs[i](input, *args, **kwargs)
            else:
                input = vs[i](input)
        return input


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        return F.relu(out)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        return F.relu(out)


class ResNet(nn.Module):
    # feat_scale lets us deal with CelebA, other non-32x32 datasets
    def __init__(self,
                 block,
                 num_blocks,
                 num_classes=10,
                 feat_scale=1,
                 wm=1,
                 in_chans=3,
                 img_size=224,
                 drop_rate=0.0):
        super(ResNet, self).__init__()

        self.num_classes = num_classes

        widths = [64, 128, 256, 512]
        widths = [int(w * wm) for w in widths]

        self.in_planes = widths[0]
        self.conv1 = nn.Conv2d(in_chans, self.in_planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.layer1 = self._make_layer(block, widths[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, widths[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, widths[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, widths[3], num_blocks[3], stride=2)

        self.in_features = feat_scale * widths[3] * block.expansion
        self.linear = nn.Linear(self.in_features, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return SequentialWithArgs(*layers)

    def forward(self, x, with_latent=False, no_relu=False):
        assert (not no_relu),  \
            "no_relu not yet supported for this architecture"
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        pre_out = out.view(out.size(0), -1)
        final = self.linear(pre_out)
        if with_latent:
            return final, pre_out
        return final

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.linear = nn.Linear(self.in_features, num_classes) if num_classes > 0 else nn.Identity()


def _create_resnet(variant, pretrained=False, default_cfg=None, **kwargs):
    model = build_model_with_cfg(ResNet, variant, pretrained, **kwargs)
    return model


@register_model
def resnet50_robustness(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model from the robustness library.
    """
    model_args = dict(block=Bottleneck, num_blocks=[3, 4, 6, 3], **kwargs)
    return _create_resnet('resnet50_robustness', pretrained, **model_args)

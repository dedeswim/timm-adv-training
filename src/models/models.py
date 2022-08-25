from collections import OrderedDict
from timm.models import cait, resnet, vision_transformer, xcit
from timm.models.helpers import build_model_with_cfg
from timm.models.registry import register_model
from torch import nn

from src import utils

default_cfgs = {
    'xcit_small_12_p4_32': xcit._cfg(input_size=(3, 32, 32)),
    'resnet18_32': resnet._cfg(input_size=(3, 32, 32), interpolation='bicubic', crop_pct=1, pool_size=(4, 4)),
    'resnet50_32': resnet._cfg(input_size=(3, 32, 32), interpolation='bicubic', crop_pct=1, pool_size=(4, 4)),
    'vit_tiny_patch4_32': vision_transformer._cfg(input_size=(3, 32, 32), crop_pct=1),
    'vit_tiny_patch16_32_upsample': vision_transformer._cfg(input_size=(3, 32, 32), crop_pct=1),
}


@register_model
def xcit_small_12_p4_32(pretrained=False, **kwargs):
    model_kwargs = dict(
        patch_size=16,  # 16 because the pre-trained model has 16
        embed_dim=384,
        depth=12,
        num_heads=8,
        eta=1.0,
        tokens_norm=True,
        **kwargs)
    model = xcit._create_xcit('xcit_small_12_p4_32', pretrained=pretrained, **model_kwargs)
    assert isinstance(model, xcit.XCiT)
    model = utils.adapt_model_patches(model, 4)
    return model


@register_model
def resnet18_32(pretrained=False, **kwargs):
    """Constructs a ResNet-18 model to work on 32x32 images.
    """
    model_args = dict(block=resnet.BasicBlock, layers=[2, 2, 2, 2], **kwargs)
    model = resnet._create_resnet('resnet18_32', pretrained, **model_args)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


@register_model
def resnet50_32(pretrained=False, **kwargs):
    """Constructs a ResNet-50 model to work on 32x32 images.
    """
    model_args = dict(block=resnet.Bottleneck, layers=[3, 4, 6, 3], **kwargs)
    model = resnet._create_resnet('resnet50_32', pretrained, **model_args)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


@register_model
def vit_tiny_patch4_32(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/4) for small resolution (32x32) images.
    """
    model_kwargs = dict(img_size=32, patch_size=4, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = vision_transformer._create_vision_transformer('vit_tiny_patch4_32',
                                                          pretrained=pretrained,
                                                          **model_kwargs)
    return model


@register_model
def vit_tiny_patch16_224_upsample(pretrained=False, **kwargs):
    """ ViT-Tiny (Vit-Ti/16) for small resolution images, which are upsampled as part of the model.
    """
    model_kwargs = dict(img_size=224, patch_size=16, embed_dim=192, depth=12, num_heads=3, **kwargs)
    model = vision_transformer._create_vision_transformer('vit_tiny_patch16_32_upsample',
                                                          pretrained=pretrained,
                                                          **model_kwargs)
    upsampling_layer = nn.Upsample(size=(224, 224), mode=vision_transformer._cfg()["interpolation"])
    upsampling_model = nn.Sequential(
        OrderedDict([("upsampling", upsampling_layer), ("model", model)]))
    return upsampling_model
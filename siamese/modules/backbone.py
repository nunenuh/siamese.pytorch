import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.hub import load_state_dict_from_url
import torch.utils.model_zoo as model_zoo
from torchvision.models import resnet, mobilenetv2
# from torchwisdom.models import mobilenet
# from torchwisdom.core import nn as layers

from . import classifier


class ResNetBackbone(resnet.ResNet):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNetBackbone, self).__init__(block, layers, num_classes)
        self.block_expansion = block.expansion

class MobileNetV2Backbone(mobilenetv2.MobileNetV2):
    def __init__(self, num_classes: int = 1000):
        super(MobileNetV2Backbone, self).__init__(num_classes=num_classes)
        

def resnet_backbone(pretrained_backbone=True, encoder_digit=64, version=18, in_chan=3, **kwargs):
    if in_chan != 3 and pretrained_backbone:
        raise ValueError("in_chan has to be 3 when you set pretrained=True")

    block = {'18': [2, 2, 2, 2], '34': [3, 4, 6, 3], '50': [3, 4, 6, 3],
             '101': [3, 4, 23, 3], '152': [3, 8, 36, 3]}
    name_ver = 'resnet'+str(version)

    backbone_model = ResNetBackbone(resnet.BasicBlock, block[str(version)], **kwargs)
    if pretrained_backbone:
        state_dict = model_zoo.load_url(resnet.model_urls[name_ver])
        backbone_model.load_state_dict(state_dict)
    expansion = 512 * backbone_model.block_expansion
    backbone_model.fc = classifier.Classfiers(in_features=expansion, n_classes=encoder_digit)
    return backbone_model


def mobilenetv2_backbone(pretrained_backbone=True, encoder_digit=64, progress=True, **kwargs):
    backbone_model = mobilenetv2.MobileNetV2(**kwargs)
    if pretrained_backbone:
        state_dict = load_state_dict_from_url(mobilenetv2.model_urls['mobilenet_v2'], progress=progress)
        backbone_model.load_state_dict(state_dict)
    backbone_model.classifier= classifier.simple_squential_classifier(in_features=backbone_model.last_channel, 
                                                                      n_classes=encoder_digit)

    return backbone_model



if __name__ == '__main__':
    model = resnet_backbone()
    print(model)

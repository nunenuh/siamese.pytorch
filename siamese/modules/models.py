import torch
import torch.nn as nn
import torch.nn.functional as F
from . import backbone

class SiameseNet(nn.Module):
    def __init__(self, feature_extractor: nn.Module):
        super(SiameseNet, self).__init__()
        self.feature_extractor = feature_extractor
        
    def _forward_once(self, x: torch.Tensor):
        return self.feature_extractor(x)
    
    def forward(self, x: torch.Tensor, y: torch.Tensor):
        main = self._forward_once(x)
        comp = self._forward_once(y)
        return main, comp
    
    
def siamese_net(pretrained=True, backbone_name="mobilenetv2", encoder_digit=64, **kwargs):
    if backbone_name.startswith("resnet"):
        version = int(backbone_name.split("resnet")[-1])
        backbone_model = backbone.resnet_backbone(pretrained_backbone=pretrained, 
                                                  encoder_digit=encoder_digit, 
                                                  version=version, **kwargs)
    else:
        backbone_model = backbone.mobilenetv2_backbone(pretrained_backbone=pretrained,
                                                       encoder_digit=encoder_digit, **kwargs)
        
    siamese_net = SiameseNet(feature_extractor=backbone_model)
    return siamese_net


if __name__== "__main__":
    ...
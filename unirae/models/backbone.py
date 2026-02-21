from typing import Tuple

import torch
import torch.nn as nn
from torchvision import models


class ResNet18Backbone(nn.Module):
    def __init__(self, pretrained: bool = False):
        super().__init__()
        weights = models.ResNet18_Weights.DEFAULT if pretrained else None
        net = models.resnet18(weights=weights)
        self.stem = nn.Sequential(
            net.conv1,
            net.bn1,
            net.relu,
            net.maxpool,
            net.layer1,
            net.layer2,
            net.layer3,
            net.layer4,
            net.avgpool,
        )
        self.out_dim = 512

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat = self.stem(x)
        return feat.flatten(1)


class ViTSmallBackbone(nn.Module):
    def __init__(self, image_size: int = 224, pretrained: bool = False):
        super().__init__()
        import timm

        self.model = timm.create_model(
            "vit_small_patch16_224",
            pretrained=pretrained,
            num_classes=0,
            global_pool="avg",
            img_size=image_size,
        )
        self.out_dim = int(self.model.num_features)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


def build_backbone(backbone_name: str, image_size: int, pretrained: bool = False) -> Tuple[nn.Module, int]:
    name = backbone_name.lower()
    if name == "resnet18":
        model = ResNet18Backbone(pretrained=pretrained)
    elif name == "vit_small":
        model = ViTSmallBackbone(image_size=image_size, pretrained=pretrained)
    else:
        raise ValueError(f"Unsupported backbone={backbone_name}. Use resnet18|vit_small")
    return model, int(model.out_dim)

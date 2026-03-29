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


class TimmBackbone(nn.Module):
    def __init__(
        self,
        model_name: str,
        image_size: int = 224,
        pretrained: bool = False,
        global_pool: str = "avg",
    ):
        super().__init__()
        import timm

        self.model = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=global_pool,
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
    elif name in {"swin_tiny_patch4", "patch4", "vit_patch4"}:
        # There is no standard ViT patch4 model in local timm list; use patch4 transformer backbone.
        model = TimmBackbone(
            model_name="swin_tiny_patch4_window7_224",
            image_size=image_size,
            pretrained=pretrained,
        )
    elif name in {"dinov2_vits14", "dino_vits14", "vit_small_dinov2"}:
        # In this environment, DINOv2 checkpoints do not load with global_pool='avg' (fc_norm mismatch).
        # Use token pooling to match checkpoint keys while still returning [B, C] embeddings.
        model = TimmBackbone(
            model_name="vit_small_patch14_dinov2",
            image_size=image_size,
            pretrained=pretrained,
            global_pool="token",
        )
    else:
        raise ValueError(
            f"Unsupported backbone={backbone_name}. "
            "Use resnet18|vit_small|swin_tiny_patch4 (aliases: patch4|vit_patch4)"
            "|dinov2_vits14 (aliases: dino_vits14|vit_small_dinov2)."
        )
    return model, int(model.out_dim)

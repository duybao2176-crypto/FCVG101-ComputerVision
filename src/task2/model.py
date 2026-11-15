"""Models: ResNet18 transfer learning (main) and a tiny CNN from scratch (baseline)."""
import warnings
from typing import Tuple

import torch
from torch import nn
from torchvision import models

from . import config


class SimpleCNN(nn.Module):
    """Baseline trained from scratch: 4 x (Conv -> BatchNorm -> ReLU -> MaxPool) + classifier."""

    def __init__(self, num_classes: int, dropout: float = config.DROPOUT):
        super().__init__()
        channels = [3, 32, 64, 128, 256]
        blocks = []
        for c_in, c_out in zip(channels[:-1], channels[1:]):
            blocks.append(nn.Sequential(
                nn.Conv2d(c_in, c_out, kernel_size=3, padding=1),
                nn.BatchNorm2d(c_out),
                nn.ReLU(),                        # not in-place: keeps Grad-CAM hooks simple
                nn.MaxPool2d(2)))                 # halves height and width
        self.features = nn.Sequential(*blocks)
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),              # any input size -> 1x1
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(channels[-1], num_classes))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))


def build_model(num_classes: int, pretrained: bool = config.PRETRAINED,
                architecture: str = config.ARCHITECTURE,
                dropout: float = config.DROPOUT) -> nn.Module:
    """Create the model. "resnet18" = transfer learning, "simplecnn" = from scratch.

    For ResNet18 the ImageNet-pretrained layers are kept and the last layer (fc,
    1000 ImageNet classes) is replaced by Dropout + Linear(512, num_classes).
    If the weights cannot be downloaded (offline) we warn and use random init.
    """
    if architecture == "simplecnn":
        return SimpleCNN(num_classes, dropout)
    if architecture != "resnet18":
        raise ValueError(f"Unknown architecture '{architecture}' (use 'resnet18' or 'simplecnn')")

    weights = None
    if pretrained:
        try:
            weights = models.ResNet18_Weights.IMAGENET1K_V1
            model = models.resnet18(weights=weights)
        except Exception as error:                # no internet / blocked download
            warnings.warn(f"Could not load pretrained weights ({error}). "
                          f"Falling back to RANDOM initialisation - accuracy will be lower.")
            model = models.resnet18(weights=None)
    else:
        model = models.resnet18(weights=None)
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(model.fc.in_features, num_classes))
    return model


def set_backbone_trainable(model: nn.Module, trainable: bool) -> None:
    """Freeze (False) or unfreeze (True) everything except the classifier head.

    Freezing the backbone for the first epoch lets the new random head learn
    without destroying the good pretrained features.
    """
    head = model.fc if hasattr(model, "fc") else model.classifier
    head_params = {id(p) for p in head.parameters()}
    for param in model.parameters():
        param.requires_grad = trainable or id(param) in head_params


def count_parameters(model: nn.Module) -> Tuple[int, int]:
    """Return (trainable_parameters, total_parameters)."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return trainable, total


def get_target_layer(model: nn.Module) -> nn.Module:
    """Last convolutional stage, used by Grad-CAM (layer4 for ResNet18)."""
    if hasattr(model, "layer4"):
        return model.layer4
    return model.features[-1]


def make_meta(class_names: list, architecture: str = config.ARCHITECTURE,
              pretrained: bool = config.PRETRAINED) -> dict:
    """Everything an app needs to rebuild the model and preprocess images the same way."""
    return {"class_names": class_names, "num_classes": len(class_names),
            "architecture": architecture, "pretrained": pretrained,
            "dropout": config.DROPOUT, "img_size": config.IMG_SIZE,
            "mean": list(config.IMAGENET_MEAN), "std": list(config.IMAGENET_STD)}

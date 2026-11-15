"""Image transforms: augmentation for training, plain preprocessing for evaluation."""
from typing import Sequence

import torch
from torchvision import transforms as T

from . import config


class AddGaussianNoise:
    """Noise injection: add random Gaussian noise to a tensor image in 0..1 range.

    Why: real photos from a warehouse camera are noisy; training on noisy copies
    makes the model less sensitive to sensor noise.
    """

    def __init__(self, std: float = config.NOISE_STD, p: float = config.NOISE_PROB):
        self.std = std
        self.p = p

    def __call__(self, img: torch.Tensor) -> torch.Tensor:
        if torch.rand(1).item() > self.p:          # only noise a fraction of the images
            return img
        noisy = img + torch.randn_like(img) * self.std
        return noisy.clamp(0.0, 1.0)               # keep valid pixel range

    def __repr__(self) -> str:
        return f"AddGaussianNoise(std={self.std}, p={self.p})"


def get_train_transforms(img_size: int = config.IMG_SIZE, noise: bool = True,
                         mean: Sequence[float] = config.IMAGENET_MEAN,
                         std: Sequence[float] = config.IMAGENET_STD) -> T.Compose:
    """Random augmentations (new random result every epoch) + normalization."""
    steps = [
        # Random zoom/crop: simulates the fruit at different distances and positions.
        T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        # Fruit has no "up": mirror image is still a valid fruit.
        T.RandomHorizontalFlip(p=0.5),
        # Small rotation: fruit lies at any angle on the conveyor.
        T.RandomRotation(config.ROTATION_DEGREES),
        # Lighting changes. Hue shift is kept tiny, because colour IS the fresh/rotten signal.
        T.ColorJitter(**config.COLOR_JITTER),
        T.ToTensor(),                              # PIL (0..255) -> tensor (0..1), CxHxW
    ]
    if noise:
        steps.append(AddGaussianNoise())
    steps.append(T.Normalize(mean, std))           # same statistics as ImageNet pretraining
    return T.Compose(steps)


def get_eval_transforms(img_size: int = config.IMG_SIZE,
                        mean: Sequence[float] = config.IMAGENET_MEAN,
                        std: Sequence[float] = config.IMAGENET_STD) -> T.Compose:
    """Deterministic preprocessing for validation, test and the real app."""
    return T.Compose([
        T.Resize(int(img_size * 256 / 224)),       # resize the short side (256 for 224)
        T.CenterCrop(img_size),                    # then cut the centre square
        T.ToTensor(),
        T.Normalize(mean, std),
    ])


def denormalize(tensor: torch.Tensor, mean: Sequence[float] = config.IMAGENET_MEAN,
                std: Sequence[float] = config.IMAGENET_STD) -> torch.Tensor:
    """Undo Normalize so a (C,H,W) or (N,C,H,W) tensor can be shown as an image (0..1)."""
    m = torch.tensor(mean, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    s = torch.tensor(std, dtype=tensor.dtype, device=tensor.device).view(-1, 1, 1)
    return (tensor * s + m).clamp(0.0, 1.0)

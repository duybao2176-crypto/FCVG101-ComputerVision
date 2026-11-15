"""PyTorch Dataset and DataLoaders built from the splits DataFrame."""
from typing import Dict, Optional

import pandas as pd
import torch
from PIL import Image, ImageOps
from torch.utils.data import DataLoader, Dataset

from . import config
from .transforms import get_eval_transforms, get_train_transforms


class FruitDataset(Dataset):
    """One item = (image tensor, integer label). Reads files lazily from disk."""

    def __init__(self, df: pd.DataFrame, transform=None):
        # Prefer the small cached copies (see data_utils.cache_images) when they exist.
        column = "processed_path" if "processed_path" in df.columns else "path"
        self.paths = df[column].tolist()
        self.labels = df["label"].tolist()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int):
        # convert("RGB") makes grayscale / RGBA / palette images 3-channel.
        # exif_transpose: phone photos may be stored rotated; this makes them upright.
        image = ImageOps.exif_transpose(Image.open(self.paths[index])).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)
        return image, self.labels[index]


def compute_class_weights(train_df: pd.DataFrame, num_classes: int) -> torch.Tensor:
    """Inverse-frequency weights: rare classes get a larger loss weight.

    weight_c = N / (num_classes * count_c); a balanced dataset gives all 1.0.
    """
    counts = train_df["label"].value_counts().reindex(range(num_classes), fill_value=0)
    weights = len(train_df) / (num_classes * counts.clip(lower=1))
    return torch.tensor(weights.values, dtype=torch.float32)


def make_dataloaders(splits_df: pd.DataFrame, img_size: int = config.IMG_SIZE,
                     batch_size: int = config.BATCH_SIZE,
                     num_workers: int = config.NUM_WORKERS,
                     augment: bool = True,
                     seed: Optional[int] = config.SEED) -> Dict[str, DataLoader]:
    """Return {"train", "val", "test"} DataLoaders.

    Only the train loader shuffles and uses random augmentation.
    """
    train_tf = get_train_transforms(img_size) if augment else get_eval_transforms(img_size)
    eval_tf = get_eval_transforms(img_size)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    loaders = {}
    for name in ("train", "val", "test"):
        part = splits_df[splits_df["split"] == name]
        if len(part) == 0:
            continue
        loaders[name] = DataLoader(
            FruitDataset(part, train_tf if name == "train" else eval_tf),
            batch_size=batch_size, shuffle=(name == "train"), num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            generator=generator if name == "train" else None)
    return loaders

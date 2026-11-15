"""Central settings for Task 2 (fresh vs rotten fruit classifier).

Every path and hyper-parameter lives here, so notebooks and modules never
contain "magic numbers". Some values can be overridden with environment
variables (handy for quick tests, e.g. TASK2_EPOCHS=1).
"""
import os
import re
from pathlib import Path
from typing import Tuple

# --------------------------------------------------------------------------
# Paths (all relative to the project root, never absolute)
# --------------------------------------------------------------------------
# config.py -> task2 -> src -> project root
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Where the raw images are. Override with env var FRUIT_DATA_DIR.
DATA_DIR = Path(os.environ.get("FRUIT_DATA_DIR", PROJECT_ROOT / "Task2" / "data" / "raw"))

# Resized copies of the images (the "cache"): 256 px short side JPEGs, fast to read.
# Override with env var TASK2_PROCESSED_DIR.
PROCESSED_DIR = Path(os.environ.get("TASK2_PROCESSED_DIR", DATA_DIR.parent / "processed"))

# Where every artifact (model, plots, json files) is written.
OUT_DIR = Path(os.environ.get("TASK2_OUT_DIR", PROJECT_ROOT / "Task2" / "outputs"))

# --------------------------------------------------------------------------
# Data pipeline settings
# --------------------------------------------------------------------------
SEED = 42                      # one seed for numpy, torch and the splits
IMG_SIZE = 224                 # input size expected by ResNet18
MIN_IMAGE_SIZE = 32            # images smaller than this (px) are discarded
IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff")

# Resize & cache step: raw phone photos are huge (up to 4160x3120), so we shrink the
# short side to 256 px once (the evaluation pipeline resizes to 256 then crops 224 anyway).
CACHE_SHORT_SIDE = 256
CACHE_JPEG_QUALITY = 95
# Near-duplicate check: 16x16 average hash (256 bits). Images whose hashes differ in at
# most this many bits are "near duplicates" (e.g. burst photos of the same fruit) and
# are kept together in one split. 10 is deliberately generous: wrongly grouping two
# different photos is harmless, while missing a duplicate would leak.
HASH_SIZE = 16
NEAR_DUP_MAX_DISTANCE = 10

# Split ratios used when the dataset has no ready-made train/test folders.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# ImageNet statistics: the pretrained ResNet18 was trained with these.
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Augmentation strengths (see transforms.py for what each one does).
ROTATION_DEGREES = 20
COLOR_JITTER = dict(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.02)
NOISE_STD = 0.03               # Gaussian noise std on a 0..1 pixel scale
NOISE_PROB = 0.3               # chance that noise is added to an image

# --------------------------------------------------------------------------
# Model / training hyper-parameters
# --------------------------------------------------------------------------
ARCHITECTURE = "resnet18"      # "resnet18" (transfer learning) or "simplecnn"
PRETRAINED = os.environ.get("TASK2_PRETRAINED", "1") not in ("0", "false", "False")
DROPOUT = 0.2
BATCH_SIZE = 32
EPOCHS = int(os.environ.get("TASK2_EPOCHS", "8"))
LEARNING_RATE = 3e-4
WEIGHT_DECAY = 1e-4
FREEZE_EPOCHS = 1              # epochs where only the new classifier head learns
PATIENCE = 4                   # early stopping: stop after N epochs without improvement
USE_CLASS_WEIGHTS = True       # weighted loss to compensate class imbalance
USE_AMP = True                 # mixed precision (only used when a GPU is present)
# 0 workers = no multiprocessing, which avoids many problems on Windows.
NUM_WORKERS = int(os.environ.get("TASK2_NUM_WORKERS", "0"))


def ensure_out_dir() -> Path:
    """Create the output folder if needed and return it."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    return OUT_DIR


def hyperparameters() -> dict:
    """All training hyper-parameters in one dict (used for the report table)."""
    return {
        "architecture": ARCHITECTURE, "pretrained": PRETRAINED, "img_size": IMG_SIZE,
        "batch_size": BATCH_SIZE, "epochs": EPOCHS, "learning_rate": LEARNING_RATE,
        "weight_decay": WEIGHT_DECAY, "dropout": DROPOUT, "freeze_epochs": FREEZE_EPOCHS,
        "patience": PATIENCE, "optimizer": "AdamW", "scheduler": "CosineAnnealingLR",
        "class_weighted_loss": USE_CLASS_WEIGHTS, "mixed_precision": USE_AMP,
        "seed": SEED, "split (train/val/test)": f"{TRAIN_RATIO}/{VAL_RATIO}/{TEST_RATIO}",
    }


# --------------------------------------------------------------------------
# Class-name parsing: "freshapples", "rotten_banana", "Fresh Oranges" ...
# --------------------------------------------------------------------------
def parse_class_name(name: str) -> Tuple[str, str]:
    """Split a folder/class name into (freshness, fruit).

    Examples: "freshapples" -> ("fresh", "apple"),
              "rotten_banana" -> ("rotten", "banana"),
              "Fresh Oranges" -> ("fresh", "orange").
    If no freshness word is found, freshness is "unknown" and fruit is the
    cleaned name, so the code still works with other datasets.
    """
    text = re.sub(r"[^a-z]", "", name.lower())          # keep letters only
    freshness = "unknown"
    for word in ("fresh", "rotten"):
        if word in text:
            freshness = word
            text = text.replace(word, "")
    # Naive plural -> singular ("apples" -> "apple"), good enough for 3 fruits.
    if text.endswith("s") and not text.endswith("ss"):
        text = text[:-1]
    return freshness, text


def canonical_class_name(name: str) -> str:
    """Unified class name such as "fresh_apple" (merges spelling variants)."""
    freshness, fruit = parse_class_name(name)
    if freshness == "unknown":
        return name.strip().lower().replace(" ", "_")
    return f"{freshness}_{fruit}"

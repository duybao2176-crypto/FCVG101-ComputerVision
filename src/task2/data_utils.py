"""Data ingestion, cleaning and splitting for the fruit dataset.

Pipeline order:  discover_dataset -> analyse_images -> clean_dataset -> make_splits
No printing or plotting here: functions return data, notebooks show it.
"""
import hashlib
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from PIL import Image, ImageOps
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from sklearn.model_selection import StratifiedGroupKFold, train_test_split

from . import config

PathLike = Union[str, Path]

# Folder names that mean "this is a ready-made split", mapped to a standard name.
SPLIT_FOLDER_NAMES = {"train": "train", "training": "train",
                      "val": "val", "valid": "val", "validation": "val",
                      "test": "test", "testing": "test"}


# ==========================================================================
# 1. Ingestion
# ==========================================================================
def discover_dataset(root: PathLike = config.DATA_DIR) -> pd.DataFrame:
    """Find every class folder (a folder that directly contains images) under root.

    If a parent folder is called train/val/test it is remembered in
    `source_split`; otherwise source_split is "none" (flat layout).
    Returns one row per image: path, label, class_name, fruit, freshness, source_split.
    """
    root = Path(root)
    if not root.is_dir():
        raise FileNotFoundError(f"Dataset folder not found: {root}. "
                                f"Put the images there (see Task2/README.md).")
    rows = []
    for folder in sorted([root] + [p for p in root.rglob("*") if p.is_dir()]):
        images = sorted(f for f in folder.iterdir()
                        if f.is_file() and f.suffix.lower() in config.IMAGE_EXTENSIONS)
        if not images or folder == root:
            continue  # images directly in root have no class name -> ignore
        # Look at the folders between root and the class folder for "train"/"test".
        parents = [p.lower() for p in folder.relative_to(root).parts[:-1]]
        split = next((SPLIT_FOLDER_NAMES[p] for p in parents if p in SPLIT_FOLDER_NAMES), "none")
        class_name = config.canonical_class_name(folder.name)
        freshness, fruit = config.parse_class_name(folder.name)
        rows += [dict(path=str(img), class_name=class_name, fruit=fruit,
                      freshness=freshness, source_split=split) for img in images]
    if not rows:
        raise FileNotFoundError(f"No images found under {root}")
    df = pd.DataFrame(rows)
    # Integer label = position of the class in the alphabetical class list.
    class_names = sorted(df["class_name"].unique())
    df["label"] = df["class_name"].map({c: i for i, c in enumerate(class_names)})
    return df[["path", "label", "class_name", "fruit", "freshness", "source_split"]]


def class_names_from_df(df: pd.DataFrame) -> List[str]:
    """Class names ordered by their integer label."""
    return df.drop_duplicates("label").sort_values("label")["class_name"].tolist()


def imbalance_ratio(df: pd.DataFrame) -> float:
    """Largest class size divided by smallest class size (1.0 = perfectly balanced)."""
    counts = df["class_name"].value_counts()
    return float(counts.max() / counts.min())


# ==========================================================================
# 2. Image checks and cleaning
# ==========================================================================
def file_md5(path: PathLike) -> str:
    """MD5 hash of the file bytes: identical files give identical hashes."""
    return hashlib.md5(Path(path).read_bytes()).hexdigest()


# EXIF orientation values 5-8 mean "the photo is stored rotated by 90 degrees":
# viewers swap width and height, so we must do the same to measure the real shape.
EXIF_ORIENTATION_TAG = 0x0112
ROTATED_ORIENTATIONS = (5, 6, 7, 8)


def average_hash(gray_thumb: Image.Image, size: int = config.HASH_SIZE) -> str:
    """"Average hash" of an image: size*size bits written as a hex string.

    Shrink to size x size grey pixels, then write 1 where a pixel is brighter than
    the mean. Visually similar photos get hashes that differ in only a few bits.
    """
    pixels = np.asarray(gray_thumb.convert("L").resize((size, size), Image.BILINEAR),
                        dtype=np.float32)
    bits = (pixels > pixels.mean()).flatten().astype(np.uint8)
    return np.packbits(bits).tobytes().hex()


def check_image(path: PathLike) -> dict:
    """Open one image safely and return its basic statistics.

    status is "ok" or "corrupt". Width/height are the real (EXIF-corrected) shape,
    read from the file header, so huge photos are NOT fully decoded. Colour
    statistics use a tiny thumbnail (fast) in HSV space: hue (degrees, circular
    mean), saturation and value (brightness), the last two on a 0..1 scale.
    """
    path = Path(path)
    info = dict(width=0, height=0, mode="", exif_rotated=False,
                file_kb=round(path.stat().st_size / 1024, 1),
                brightness=np.nan, mean_hue=np.nan, mean_sat=np.nan, mean_val=np.nan,
                ahash="", status="ok")
    try:
        with Image.open(path) as img:
            img.verify()                      # cheap integrity check of the file
        with Image.open(path) as img:         # verify() closes the file, so re-open
            width, height = img.size          # header only: no pixel decoding yet
            rotated = img.getexif().get(EXIF_ORIENTATION_TAG, 1) in ROTATED_ORIENTATIONS
            if rotated:
                width, height = height, width
            info.update(width=width, height=height, mode=img.mode, exif_rotated=rotated)
            img.draft("RGB", (128, 128))      # JPEG decodes at 1/2, 1/4 or 1/8 size: much faster
            thumb = ImageOps.exif_transpose(img).convert("RGB")   # upright + 3 channels
            thumb.thumbnail((64, 64))
            gray = np.asarray(thumb.convert("L"), dtype=np.float32)
            hsv = np.asarray(thumb.convert("HSV"), dtype=np.float32)
        info["brightness"] = float(gray.mean())
        info["ahash"] = average_hash(thumb)
        # Hue is an angle, so average it on the circle (otherwise 359 and 1 average to 180).
        # Each pixel is weighted by its saturation: grey/white background pixels have no
        # real hue and would otherwise add noise.
        angle = hsv[..., 0] / 255.0 * 2 * np.pi
        weight = hsv[..., 1] / 255.0 + 1e-6
        info["mean_hue"] = float(np.degrees(np.arctan2((np.sin(angle) * weight).sum(),
                                                       (np.cos(angle) * weight).sum())) % 360)
        info["mean_sat"] = float(hsv[..., 1].mean() / 255.0)
        info["mean_val"] = float(hsv[..., 2].mean() / 255.0)
    except Exception:                          # any read error = corrupt image
        info["status"] = "corrupt"
    return info


def load_thumbnail(path: PathLike, size: int = 160) -> Image.Image:
    """Fast upright RGB square thumbnail (for galleries); safe for huge photos."""
    with Image.open(path) as img:
        img.draft("RGB", (size * 2, size * 2))
        return ImageOps.fit(ImageOps.exif_transpose(img).convert("RGB"), (size, size))


def analyse_images(df: pd.DataFrame) -> pd.DataFrame:
    """Run check_image + md5 on every row and append the results as columns."""
    stats = pd.DataFrame([check_image(p) for p in df["path"]], index=df.index)
    out = pd.concat([df, stats], axis=1)
    out["md5"] = [file_md5(p) for p in df["path"]]
    return out


def clean_dataset(df: pd.DataFrame, min_size: int = config.MIN_IMAGE_SIZE,
                  brightness_range: Tuple[float, float] = (10, 245)
                  ) -> Tuple[pd.DataFrame, dict]:
    """Remove unusable images and return (clean_df, report).

    Removed, in this order:
      1. corrupt        - file cannot be opened
      2. too_small      - width or height below min_size
      3. outlier_brightness - almost black / almost white image
      4. duplicate_conflicting_label - same bytes appear under different classes
      5. duplicate      - same bytes appear more than once in the same class
    Grayscale images are NOT removed; they are converted to RGB when loaded.
    """
    if "status" not in df.columns:
        df = analyse_images(df)
    df = df.copy()
    # Keep "train" rows before "test" rows so duplicates are dropped from test.
    order = {"train": 0, "val": 1, "none": 2, "test": 3}
    df = df.sort_values("source_split", key=lambda s: s.map(order), kind="stable")

    reason = pd.Series("", index=df.index)
    ok = df["status"] == "ok"
    reason[~ok] = "corrupt"
    small = ok & ((df["width"] < min_size) | (df["height"] < min_size))
    reason[small] = "too_small"
    low, high = brightness_range
    outlier = ok & ~small & ((df["brightness"] < low) | (df["brightness"] > high))
    reason[outlier] = "outlier_brightness"

    alive = reason == ""
    # Same file content labelled with different classes: ambiguous -> drop all copies.
    labels_per_hash = df[alive].groupby("md5")["label"].nunique()
    conflicting = df["md5"].isin(labels_per_hash[labels_per_hash > 1].index) & alive
    reason[conflicting] = "duplicate_conflicting_label"
    # Plain duplicates: keep the first copy only.
    alive = reason == ""
    is_dup = df[alive].duplicated(subset="md5", keep="first")
    reason[is_dup[is_dup].index] = "duplicate"

    removed = df[reason != ""].assign(reason=reason[reason != ""])
    clean = df[reason == ""].reset_index(drop=True)
    report = {
        "n_input": int(len(df)),
        "n_output": int(len(clean)),
        "removed_by_reason": removed["reason"].value_counts().to_dict(),
        "n_grayscale_kept": int((clean["mode"] == "L").sum()),
        "min_size": min_size,
        "class_counts_before": df["class_name"].value_counts().sort_index().to_dict(),
        "class_counts_after": clean["class_name"].value_counts().sort_index().to_dict(),
        "imbalance_ratio_after": round(imbalance_ratio(clean), 3),
        "removed_files": removed[["path", "reason"]].head(50).to_dict("records"),
    }
    return clean, report


# ==========================================================================
# 2b. Resize & cache (transformation / scaling)
# ==========================================================================
def _processed_name(row: pd.Series, taken: set) -> str:
    """Unique, file-system-safe name for the cached copy of one image."""
    stem = re.sub(r"[^A-Za-z0-9_-]+", "_", Path(row["path"]).stem).strip("_")
    name = f"{row['source_split']}_{stem}" if row["source_split"] != "none" else stem
    candidate, counter = name, 1
    while (row["class_name"], candidate) in taken:     # two raw files with the same name
        counter += 1
        candidate = f"{name}_{counter}"
    taken.add((row["class_name"], candidate))
    return candidate + ".jpg"


def cache_images(df: pd.DataFrame, out_dir: Optional[PathLike] = None,
                 short_side: int = config.CACHE_SHORT_SIDE,
                 quality: int = config.CACHE_JPEG_QUALITY) -> pd.DataFrame:
    """Write a small upright RGB JPEG copy of every image; add a `processed_path` column.

    Steps per image: (1) fix EXIF rotation, (2) convert to RGB, (3) shrink so the
    SHORT side equals `short_side` (aspect ratio kept, never enlarged),
    (4) save as JPEG into out_dir/<class_name>/. Existing files are skipped, so
    running it twice is cheap. Training then reads these small files instead of
    decoding 10-megapixel phone photos every epoch.
    """
    out_dir = Path(out_dir) if out_dir else config.PROCESSED_DIR
    taken: set = set()
    processed = []
    for _, row in df.iterrows():
        target = out_dir / row["class_name"] / _processed_name(row, taken)
        if not target.exists():
            target.parent.mkdir(parents=True, exist_ok=True)
            with Image.open(row["path"]) as img:
                img.draft("RGB", (short_side, short_side))   # fast decode at reduced scale
                img = ImageOps.exif_transpose(img).convert("RGB")
                scale = short_side / min(img.size)
                if scale < 1:                                  # only shrink, never enlarge
                    img = img.resize((round(img.width * scale), round(img.height * scale)),
                                     Image.LANCZOS)
                img.save(target, "JPEG", quality=quality)
        processed.append(str(target))
    return df.assign(processed_path=processed)


# ==========================================================================
# 2c. Near-duplicate detection (perceptual hash)
# ==========================================================================
def _hash_bits(hashes: pd.Series) -> np.ndarray:
    """Hex hash strings -> (N, bits) array of 0/1 values."""
    return np.stack([np.unpackbits(np.frombuffer(bytes.fromhex(h), dtype=np.uint8))
                     for h in hashes])


def hamming_distances(df: pd.DataFrame, rows_a: np.ndarray, rows_b: np.ndarray) -> np.ndarray:
    """Matrix of hash distances (rows_a x rows_b) = number of differing bits.

    Uses  differing bits = ones(a) + ones(b) - 2 * (a . b),  a fast matrix product.
    """
    bits = _hash_bits(df["ahash"]).astype(np.float32)
    a, b = bits[rows_a], bits[rows_b]
    return np.rint(a.sum(1)[:, None] + b.sum(1)[None, :] - 2 * a @ b.T).astype(int)


def find_near_duplicates(df: pd.DataFrame, max_distance: int = config.NEAR_DUP_MAX_DISTANCE
                         ) -> pd.DataFrame:
    """Pairs of images from DIFFERENT splits whose hashes differ in <= max_distance bits.

    Needs the `ahash` and `split` columns. Returns path_a, split_a, path_b, split_b, distance.
    """
    splits = df["split"].to_numpy()
    rows = []
    for a, b in [("train", "val"), ("train", "test"), ("val", "test")]:
        idx_a, idx_b = np.where(splits == a)[0], np.where(splits == b)[0]
        if len(idx_a) == 0 or len(idx_b) == 0:
            continue
        dist = hamming_distances(df, idx_a, idx_b)
        for i, j in zip(*np.where(dist <= max_distance)):
            rows.append(dict(path_a=df["path"].iloc[idx_a[i]], split_a=a,
                             path_b=df["path"].iloc[idx_b[j]], split_b=b,
                             distance=int(dist[i, j])))
    return pd.DataFrame(rows, columns=["path_a", "split_a", "path_b", "split_b", "distance"])


def near_duplicate_groups(df: pd.DataFrame, max_distance: int = config.NEAR_DUP_MAX_DISTANCE
                          ) -> pd.Series:
    """Group id per image: near-duplicates (hash distance <= max_distance) share an id.

    Used by make_splits so that near-identical photos always land in the SAME split.
    """
    rows = np.arange(len(df))
    close = hamming_distances(df, rows, rows) <= max_distance
    _, group_ids = connected_components(csr_matrix(close), directed=False)
    return pd.Series(group_ids, index=df.index)


def nearest_train_distance(df: pd.DataFrame, split: str = "test") -> pd.Series:
    """For every image of `split`: smallest hash distance (bits) to any TRAIN image.

    Small value = a very similar picture was seen in training. Index = df index.
    """
    idx_split = np.where(df["split"].to_numpy() == split)[0]
    idx_train = np.where(df["split"].to_numpy() == "train")[0]
    nearest = hamming_distances(df, idx_split, idx_train).min(axis=1)
    return pd.Series(nearest, index=df.index[idx_split])


# ==========================================================================
# 3. Train / validation / test split
# ==========================================================================
def _cut(df: pd.DataFrame, fraction: float, seed: int, groups: Optional[pd.Series]):
    """Cut `fraction` of the rows out of df, stratified by class. Returns (rest_idx, part_idx).

    With `groups`, rows of one group (near-duplicates) are never separated; this uses
    StratifiedGroupKFold, so the part size is only approximately `fraction`.
    """
    if groups is None:
        return train_test_split(df.index, test_size=fraction, random_state=seed,
                                stratify=df["label"])
    folds = StratifiedGroupKFold(n_splits=max(2, round(1 / fraction)), shuffle=True,
                                 random_state=seed)
    rest_pos, part_pos = next(folds.split(df, df["label"], groups=groups.loc[df.index]))
    return df.index[rest_pos], df.index[part_pos]


def make_splits(df: pd.DataFrame, seed: int = config.SEED, save: bool = True,
                out_dir: Optional[PathLike] = None,
                near_dup_distance: Optional[int] = config.NEAR_DUP_MAX_DISTANCE) -> pd.DataFrame:
    """Add a `split` column (train/val/test), stratified by class.

    * If the dataset came with a "test" folder, it is used as the test set as it is
      (nothing is moved into or out of it) and the validation set is carved out of the
      training images.
    * Otherwise the whole data is split 70/15/15 (ratios from config).
    * If the df has an `ahash` column and near_dup_distance is not None, near-duplicate
      photos are kept together in one split (no leakage; sizes then only approximately
      70/15/15). Pass near_dup_distance=None for a plain random stratified split.
    Saves splits.csv and class_names.json to out_dir when save=True.
    """
    df = df.copy()
    counts = df["class_name"].value_counts()
    if counts.min() < 3:
        raise ValueError(f"Every class needs at least 3 images to split; got: {counts.to_dict()}")
    groups = None
    if near_dup_distance is not None and "ahash" in df.columns:
        groups = near_duplicate_groups(df, near_dup_distance)
    val_fraction = config.VAL_RATIO / (config.TRAIN_RATIO + config.VAL_RATIO)  # share of train+val

    if (df["source_split"] == "test").any():
        df.loc[df["source_split"] == "test", "split"] = "test"
        df.loc[df["source_split"] == "val", "split"] = "val"
        train_mask = df["source_split"].isin(["train", "none"])
        if (df["source_split"] == "val").any():
            df.loc[train_mask, "split"] = "train"
        else:
            train_idx, val_idx = _cut(df[train_mask], val_fraction, seed, groups)
            df.loc[train_idx, "split"] = "train"
            df.loc[val_idx, "split"] = "val"
    else:
        # First cut off the test set, then cut the validation set from the rest.
        rest_idx, test_idx = _cut(df, config.TEST_RATIO, seed, groups)
        train_idx, val_idx = _cut(df.loc[rest_idx], val_fraction, seed, groups)
        df.loc[train_idx, "split"] = "train"
        df.loc[val_idx, "split"] = "val"
        df.loc[test_idx, "split"] = "test"

    df = df.sort_values(["split", "label"]).reset_index(drop=True)
    if save:
        out = Path(out_dir) if out_dir else config.ensure_out_dir()
        out.mkdir(parents=True, exist_ok=True)
        df.to_csv(out / "splits.csv", index=False)
        (out / "class_names.json").write_text(json.dumps(class_names_from_df(df), indent=2))
    return df


def load_splits(out_dir: Optional[PathLike] = None) -> pd.DataFrame:
    """Read the splits.csv written by make_splits."""
    return pd.read_csv(Path(out_dir or config.OUT_DIR) / "splits.csv")


def check_leakage(df: pd.DataFrame) -> Dict[str, int]:
    """Count images shared between splits (must all be 0).

    Checks the file path and, when available, the MD5 content hash, which also
    catches the same picture stored under two different file names.
    """
    result = {}
    pairs = [("train", "val"), ("train", "test"), ("val", "test")]
    for column in [c for c in ("path", "md5") if c in df.columns]:
        for a, b in pairs:
            shared = set(df.loc[df["split"] == a, column]) & set(df.loc[df["split"] == b, column])
            result[f"{column}: {a} & {b}"] = len(shared)
    return result

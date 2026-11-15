# Task 2 - Deep-Learning Model and Data Pipeline

**Use case:** visual inspection assistant for an organic farm / produce warehouse.
A CNN classifies **fresh vs rotten** for **apple, banana, orange** (6 classes) from a phone photo.

## 1. Dataset

* **Source:** the "Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality" (Sultana, Jahan, Uddin, Mendeley Data V1, 2022, CC BY 4.0, <https://data.mendeley.com/datasets/bdd69gyhv8/1>) with 8 fruits (16 classes: fresh/rotten x 8 fruits), phone photos on plain backgrounds. The downloaded zip held all 16 classes; **only apple, banana and orange were extracted** and used.
* **Used data:** 1,200 images, 200 per class, no ready-made train/test split. The images are large (up to 4160 x 3120 px, median 6.6 MP, about 1 MB each, 1.13 GB in total).
* **Layout expected by the code** (class folder names are discovered automatically; any wrapper folder is fine):

```
Task2/data/raw/
  Original Image/
    FreshApple/   FreshBanana/   FreshOrange/       (200 .jpg each)
    RottenApple/  RottenBanana/  RottenOrange/      (200 .jpg each)
Task2/data/processed/          <- created by notebook 02 (256 px short-side JPEG copies, 24 MB)
```

Rules the code follows (so other layouts work too):
* every folder that directly contains images is one class (folder name = class name);
* names like `FreshApple`, `freshapples`, `rotten_banana`, `Fresh Oranges` are parsed into *freshness* + *fruit* and unified to e.g. `fresh_apple`;
* if `train/` and `test/` (or `val/`) folders exist they are used as the split; otherwise all images are split 70/15/15 (stratified);
* to use another folder set the environment variable `FRUIT_DATA_DIR`; outputs go to `TASK2_OUT_DIR` (default `Task2/outputs`), cached images to `TASK2_PROCESSED_DIR` (default `Task2/data/processed`).

## 2. Run order

Open the notebooks from `Task2/` and run them top to bottom (about 5 minutes in total on an RTX 4060):

| # | Notebook | Produces (in `Task2/outputs/`) |
|---|----------|------------------------------|
| 1 | `01_EDA.ipynb` | `eda_summary.json`, `eda_class_counts.png`, `eda_image_sizes.png`, `eda_color_stats.png`, `eda_samples.png` |
| 2 | `02_preprocess.ipynb` | `splits.csv`, `class_names.json`, `preprocess_summary.json`, `split_distribution.png`, `aug_examples.png`, `aug_noise.png`, `dataloader_batch.png` (+ cached images in `Task2/data/processed/`) |
| 3 | `03_train.ipynb` | `model_best.pt`, `model_meta.json`, `history.json`, `learning_curves.png`; with the baseline on also `baseline_comparison.csv`, `simplecnn_*` files |
| 4 | `04_evaluate.ipynb` | `metrics.json`, `classification_report.csv`, `confusion_matrix.png`, `mistakes_gallery.png`, `near_duplicate_check.png`, `gradcam_examples.png` |

The saved notebooks already contain their outputs (executed on the real data).
Environment switches: `TASK2_EPOCHS=1` (quick test), `TASK2_PRETRAINED=0` (no ImageNet download), `TASK2_RUN_BASELINE=1` (also train the from-scratch CNN in notebook 03; the saved run had it on), `TASK2_NUM_WORKERS`.

## 3. Results (real data, test split = 153 images never used for training or model selection)

| Metric | ResNet18 (transfer learning) | SimpleCNN (from scratch) |
|--------|------------------------------|--------------------------|
| 6-class accuracy (test) | **0.987** (151/153), 95 % bootstrap CI [0.967, 1.000] | 0.889 |
| Macro F1 (test) | **0.985** | - |
| Fruit-type accuracy (3-way) | **1.000** | - |
| Fresh-vs-rotten accuracy (2-way) | **0.987**, CI [0.967, 1.000] | - |
| Rotten recall | 0.974 (2 of 77 rotten fruits passed as fresh) | - |
| Best validation accuracy | 0.994 (epoch 6) | 0.865 (epoch 8) |
| Parameters | 11,179,590 | 390,918 |
| Time per epoch (GPU, 870 training images) | 7.6 s | 6.9 s |
| Inference | 5.3 ms/image (cached 256 px image, GPU); about 62 ms for a raw 12 MP photo | - |

* Split sizes: train 870 / val 170 / test 153 (near-duplicate photos are kept in the same split, so the ratios are not exactly 70/15/15).
* The only 2 test errors are mildly rotten oranges predicted as fresh (confidence 0.77 and 0.53).
* **Caveats:** the test set is small (the 95 % CI is wide); photos come from one source with plain backgrounds; near-duplicate leakage was removed at a hash distance of 10 bits, but several photos of similar fruit remain, so a small optimism cannot be excluded (see `near_duplicate_check.png` and the notebook 04 summary).

## 4. What to explain per notebook

**01 EDA** - understand the data before modelling
* `discover_dataset` builds one table (path, label, class, fruit, freshness, source split).
* Class counts + imbalance ratio (here 1.00: balanced).
* `analyse_images` reads every file: size from the header (EXIF-corrected), colour mode, brightness, HSV statistics from a tiny thumbnail, perceptual hash and MD5 -> finds corrupt/tiny/duplicate files (7 exact duplicates here). Huge images are never fully decoded, so this takes about 15 s.
* Size and aspect-ratio histograms justify "shrink + centre crop" instead of stretching.
* HSV box plots: rotten apple and banana are less saturated, oranges hardly differ -> colour alone is not enough, the CNN must use texture/spots.

**02 Preprocess** - cleaning, resize, split, augmentation
* `clean_dataset`: drops corrupt, <32 px, extreme-brightness outliers and duplicates (MD5); grayscale is kept and converted to RGB; returns a report.
* `cache_images` (transformation/scaling): EXIF fix -> RGB -> short side 256 px -> JPEG q95 into `Task2/data/processed/`; idempotent; 1,134 MB -> 24 MB.
* `make_splits`: stratified, seeded; near-duplicate photos (16x16 average hash, <= 10 bits) stay in the same split.
* `check_leakage` + `find_near_duplicates`: 0 overlaps; a plain random split would have 72 near-duplicate pairs across splits.
* Transforms: train = RandomResizedCrop, flip, rotation, ColorJitter (tiny hue), Gaussian noise, Normalize; eval = Resize 256, CenterCrop 224, Normalize. Augmentation only on the training set.
* Augmentation and noise figures; DataLoader check (batch shape, value range, class weights).

**03 Train** - transfer learning
* `build_model`: ResNet18 pretrained on ImageNet, `fc` replaced by `Dropout -> Linear(512, 6)`; number of classes comes from the data.
* Parameter counts: 11.2 M total, only 3,078 trainable during the warm-up epoch.
* `fit`: AdamW + cosine LR, 1 warm-up epoch with frozen backbone, best checkpoint by validation accuracy, early stopping.
* Class-weighted cross-entropy (weights 0.92-1.10: data is balanced).
* Learning curves: no strong overfitting. Baseline `SimpleCNN` shows the benefit of transfer learning (0.889 vs 0.987 test accuracy).

**04 Evaluate** - on the untouched test split
* Accuracy, bootstrap CI, per-class precision/recall/F1, macro F1, 6x6 confusion matrix.
* Derived metrics from the 6-class output: fruit-type accuracy, fresh-vs-rotten accuracy, rotten recall.
* Mistakes gallery, closest test/train pairs (leakage sanity check), Grad-CAM (hooks on `layer4`), inference speed.

## 5. Hyper-parameters (all in `src/task2/config.py`)

| Parameter | Value |
|-----------|-------|
| Architecture | ResNet18, ImageNet-pretrained (transfer learning) |
| Input size | 224 x 224 (cached at 256 px short side), ImageNet mean/std normalization |
| Optimizer / scheduler | AdamW / CosineAnnealingLR |
| Learning rate, weight decay | 3e-4, 1e-4 |
| Batch size, epochs | 32, 8 (env `TASK2_EPOCHS`), early-stopping patience 4 |
| Warm-up | 1 epoch with frozen backbone |
| Dropout | 0.2 |
| Loss | cross-entropy with inverse-frequency class weights |
| Augmentation | crop scale 0.7-1.0, h-flip 0.5, rotation +-20 deg, jitter (brightness .3, contrast .3, saturation .2, hue .02), Gaussian noise std 0.03 with p=0.3 |
| Split | stratified 70/15/15 with near-duplicate grouping, seed 42 |
| Near-duplicate hash | 16x16 average hash, grouped when <= 10 of 256 bits differ |
| Mixed precision | on (GPU only) |
| DataLoader workers | 0 (4 workers were tested and were slower on Windows) |

## 6. Artifacts and how Task 3 uses them

* `Task2/outputs/model_best.pt` - weights (state dict) of the best epoch.
* `Task2/outputs/model_meta.json` - class names, image size, normalization, architecture.

```python
from src.task2.predict import load_model, predict_image, preprocess_image, gradcam

model = load_model()                        # reads Task2/outputs/
result = predict_image(model, pil_image)    # class_name, fruit, freshness, confidence, probabilities, top3
overlay = gradcam(model, preprocess_image(model, pil_image))   # numpy RGB heatmap overlay
```

`predict_image` accepts a PIL image, a numpy RGB/grayscale array or a file path, and applies the EXIF rotation of phone photos itself.

## 7. Code map (`src/task2/`)

`config.py` settings | `data_utils.py` ingestion, cleaning, resize cache, near-duplicates, splits | `transforms.py` augmentation |
`dataset.py` Dataset + DataLoaders + class weights | `model.py` ResNet18 / SimpleCNN |
`train_utils.py` training loop, bootstrap CI | `predict.py` inference + Grad-CAM | `viz.py` plotting helpers for the notebooks.

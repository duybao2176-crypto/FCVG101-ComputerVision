# FCVG101 Coursework - Visual Inspection Assistant for Produce Quality

A minimum viable computer-vision system for an organic farm / produce warehouse:

* **Task 1 - classical engine:** camera calibration (pinhole model, undistortion), filtering and edges, frequency-domain filtering, Harris / SIFT / ORB features with ratio test and RANSAC.
* **Task 2 - deep learning:** a documented data pipeline and a ResNet18 transfer-learning classifier for **fresh vs rotten** apple / banana / orange (6 classes).
* **Task 3 - assistant:** a Gradio app that understands a text request ("denoise this with a median filter", "is this fruit fresh?"), calls the matching Task 1 / Task 2 function and shows images, numbers and a tool-use trace.

The full report (dataset, pipeline, model, results, application workflow, source code) is `report/FCVG101_Coursework_Report.pdf` (Word source: `report/FCVG101_Coursework_Report.docx`).

## Repository structure

```
FCVG101_Coursework/
├── README.md                  this file
├── requirements.txt           Python dependencies
├── Coursework_FCVG101.docx    assignment text
├── src/                       reusable modules (arrays in, dicts out; no printing / plotting)
│   ├── task1/                 calibration.py, filtering.py, features.py, results.py
│   ├── task2/                 config.py, data_utils.py, transforms.py, dataset.py, model.py,
│   │                          train_utils.py, predict.py, viz.py
│   └── task3/                 router.py (intent recognition), tools.py (six tools)
├── Task1/                     01_calibration / 02_filtering_edges / 03_feature_matching notebooks,
│   │                          OpenCV sample data, outputs/ (JSON, CSV, figures), README.md
│   ├── Chestboard_data/  featureMatch_data/  noise_data/
│   └── outputs/
├── Task2/                     01_EDA / 02_preprocess / 03_train / 04_evaluate notebooks, README.md
│   ├── data/raw/              dataset goes here (not in the repo, see below)
│   ├── data/processed/        256 px cache, created by notebook 02
│   └── outputs/               metrics, figures, splits.csv, model_best.pt, model_meta.json
├── Task3/                     app.py (Gradio), test_tools.py, make_samples.py, samples/,
│                              demo_script.md, README.md
└── report/                    build_report.py, convert_to_pdf.py, capture_ui.py, report (docx + pdf)
```

## Installation

Tested with Python 3.10 on Windows 11 (PyTorch 2.11 + CUDA, NVIDIA RTX 4060 laptop GPU); CPU-only also works but training is slower.

```
python -m venv .venv
.venv\Scripts\activate                 # Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt        # for a CUDA build of torch use the selector on pytorch.org
```

`requirements.txt` contains the packages for the three tasks (numpy, opencv-python, scipy, scikit-image, scikit-learn, matplotlib, pandas, seaborn, Pillow, torch, torchvision, jupyterlab, nbconvert, gradio) and, in a separate block, the packages that are only needed to rebuild the report (python-docx, pywin32, playwright). The Task 3 app uses **Gradio** (the coursework allows Streamlit or Gradio).

## Data

* **Task 1** data is included (OpenCV sample data, <https://github.com/opencv/opencv/tree/4.x/samples/data>): chessboard images `Task1/Chestboard_data`, `lena.jpg` in `Task1/noise_data`, `box.png` / `box_in_scene.png` in `Task1/featureMatch_data`.
* **Task 2** uses a "fresh vs rotten fruits" dataset of phone photographs (8 fruits, 16 classes; only apple, banana and orange were extracted, 200 images per class = 1200 images). The dataset is **not included** (about 1.1 GB). Dataset: N. Sultana, M. Jahan, M. S. Uddin, "Fresh and Rotten Fruits Dataset for Machine-Based Evaluation of Fruit Quality", Mendeley Data, V1, 2022, DOI [10.17632/bdd69gyhv8.1](https://doi.org/10.17632/bdd69gyhv8.1), licence CC BY 4.0 - <https://data.mendeley.com/datasets/bdd69gyhv8/1> (use the `Original Image` folder). Place the images like this (class folder names are discovered automatically, a wrapper folder is fine; another location can be set with the environment variable `FRUIT_DATA_DIR`):

```
Task2/data/raw/Original Image/
    FreshApple/  FreshBanana/  FreshOrange/      (200 .jpg each)
    RottenApple/ RottenBanana/ RottenOrange/     (200 .jpg each)
```

## How to run (run order)

From the project root:

```
# Task 1
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 Task1/01_calibration.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 Task1/02_filtering_edges.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 Task1/03_feature_matching.ipynb

# Task 2 (about 5 minutes in total on the GPU), in this order
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 Task2/01_EDA.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 Task2/02_preprocess.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 Task2/03_train.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=1800 Task2/04_evaluate.ipynb

# Task 3
python Task3/app.py                    # open http://127.0.0.1:7860   (--port 7861 --open)
python src/task3/router.py             # router self-test (29 sentences + 10 parameter checks)
python Task3/test_tools.py [folder]    # headless smoke test of all tools
```

The notebooks can also be opened in JupyterLab and run with "Run all". Task 1 notebooks create `Task1/outputs/calibration.json`, which Task 3 reads; Task 2 notebook 03 creates `model_best.pt` + `model_meta.json`, which the classify tool loads. Every Task 3 tool except `classify` works without Task 2. Optional switches for Task 2: `TASK2_EPOCHS=1` (quick test), `TASK2_PRETRAINED=0` (no ImageNet download), `TASK2_RUN_BASELINE=1` (also train the from-scratch CNN; the saved results used it), `TASK2_NUM_WORKERS`, `TASK2_OUT_DIR`, `TASK2_PROCESSED_DIR`.

Rebuild the report after retraining (numbers are read from the output files): `python report/build_report.py` then `python report/convert_to_pdf.py` (needs Microsoft Word on Windows).

## Key results

| Part | Result |
|---|---|
| Calibration (13 chessboard views, 9 x 6 corners) | fx 536.07, fy 536.02, cx 342.37, cy 235.54 px; k1 -0.265 (barrel); RMS 0.409 px, mean error 0.235 px |
| Denoising, salt & pepper 5 % | noisy 18.21 dB -> median 30.89 dB (SSIM 0.856); box / Gaussian 27.2 dB; bilateral 18.27 dB |
| Denoising, Gaussian noise sigma 25 | noisy 20.36 dB -> bilateral 29.10, Gaussian 28.94, box 28.19, median 28.07 dB |
| Matching box -> scene, ratio 0.75 | SIFT 80 matches, 75 RANSAC inliers (93.8 %); ORB 46 matches, 43 inliers (93.5 %); without ratio test 33.8 % / 13.4 % inliers |
| Fruit classifier (ResNet18), test split of 153 images | accuracy 0.987 (151/153), 95 % bootstrap CI [0.967, 1.000]; macro-F1 0.985; fruit type 1.000; rotten recall 0.974 |
| Baseline SimpleCNN (from scratch) | test accuracy 0.889 |
| Task 3 | router self-test 39/39 checks pass (written by the same author, so optimistic); 15 tool calls OK + 2 intended friendly errors in the smoke test |

Caveats (details in the report): the test set is small (153 images, CI above), all photographs come from one source with plain backgrounds, near-duplicate photographs were grouped with a coarse hash so a small residual optimism cannot be excluded, and the measure tool was only checked on a synthetic picture.

## Trained model artifact

`Task2/outputs/model_best.pt` (ResNet18 state dict, about 42.7 MiB) and `Task2/outputs/model_meta.json`. Shared link: **[shared link - student to add]**.

```python
from src.task2.predict import load_model, predict_image
model = load_model()
print(predict_image(model, "fruit.jpg"))   # class_name, fruit, freshness, confidence, top3
```

## Demo

`Task3/demo_script.md` is a 4-5 minute step-by-step live demo (example prompts and the expected output of each). Example prompts: "Calibrate the camera and undistort this image", "Add salt and pepper noise and denoise it with a median filter size 5", "Match the box with the scene using SIFT, ratio 0.75", "How big is this object? It is 50 cm away", "Is this fruit fresh or rotten?".

## Troubleshooting

* **Windows:** keep `num_workers = 0` for the DataLoaders (the default; workers were slower). If the console shows encoding errors set `PYTHONIOENCODING=utf-8` (PowerShell: `$env:PYTHONIOENCODING='utf-8'`).
* Port 7860 busy: `python Task3/app.py --port 7861`.
* Classify tool says the model is missing: run the Task 2 notebooks 01-04 (or copy `model_best.pt` and `model_meta.json` into `Task2/outputs/`).
* No internet for the ImageNet weights: set `TASK2_PRETRAINED=0` (lower accuracy) or download once.
* Running notebooks overwrites the files in `outputs/`; retraining changes the numbers (RANSAC and training are seeded, GPU results can still differ slightly).

## Credits and attributions

* OpenCV sample images (chessboard, lena, box) from the OpenCV repository, <https://github.com/opencv/opencv/tree/4.x/samples/data>.
* Fruit images: Sultana, Jahan and Uddin, Fresh and Rotten Fruits Dataset (Mendeley Data, 2022, CC BY 4.0), <https://data.mendeley.com/datasets/bdd69gyhv8/1>.
* Libraries: OpenCV, NumPy, SciPy, scikit-image, scikit-learn, PyTorch / torchvision (ResNet18 ImageNet weights), Gradio, matplotlib, pandas, seaborn, python-docx.
* Methods: Zhang (2000) calibration, Harris & Stephens (1988), Canny (1986), Lowe (2004) SIFT, Rublee et al. (2011) ORB, Fischler & Bolles (1981) RANSAC, He et al. (2016) ResNet, Selvaraju et al. (2017) Grad-CAM; full references are in the report.
* License: coursework submission; no open-source licence has been chosen. Third-party data and libraries keep their own licences.

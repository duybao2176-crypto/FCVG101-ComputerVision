# Task 1 - Core Image Analysis Engine

Three notebooks (each about 5-8 minutes to present) built on three small reusable modules.

| Notebook | Topic | Module it uses |
|---|---|---|
| `01_calibration.ipynb` | 1.1 Pinhole model, chessboard calibration, undistortion | `src/task1/calibration.py` |
| `02_filtering_edges.ipynb` | 1.2 Noise, denoising filters, gradients, Canny, FFT filtering, convolution from scratch | `src/task1/filtering.py` |
| `03_feature_matching.ipynb` | 1.3 Harris (from scratch), SIFT/ORB, ratio test, RANSAC homography | `src/task1/features.py` |

`src/task1/results.py` is a tiny helper that writes the key numbers into `outputs/task1_results.json`.
The modules take / return numpy arrays (BGR uint8) and plain dicts, with no printing or plotting, so the Task 3 Gradio app can import them directly.

## How to run
From the project root (`FCVG101_Coursework`):

```
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 Task1/01_calibration.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 Task1/02_filtering_edges.ipynb
python -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=600 Task1/03_feature_matching.ipynb
```
or open them in JupyterLab and "Run all". The first code cell finds the project root by walking up from the current folder until it finds `src`, so the notebooks work from any working directory inside the project.

Outputs in `Task1/outputs/`: `calibration.json`, `task1_results.json` (all key numbers), `filter_metrics.csv`, `feature_metrics.csv` and the figures (`calib_corners.png`, `calib_undistort.png`, `calib_reproj_error.png`, `denoise_gaussian_noise.png`, `denoise_saltpepper.png`, `gradients.png`, `canny.png`, `fft_filtering.png`, `harris_compare.png`, `keypoints_sift_orb.png`, `matches_sift_noratio.png`, `matches_sift_ratio.png`, `matches_orb_ratio.png`, `ratio_sweep.png`, `box_localization.png`).

## Key results
* **Calibration** (13 left images, 9 x 6 inner corners, 640 x 480): `fx = 536.07`, `fy = 536.02`, `cx = 342.37`, `cy = 235.54` px; distortion `k1 = -0.265`, `k2 = -0.047`, `p1 = 0.0018`, `p2 = -0.0003`, `k3 = 0.252` (barrel distortion). RMS = 0.409 px, mean reprojection error = 0.235 px. The chessboard square size is unknown, so 1 unit = 1 square.
* **Pinhole check:** manual `K[R|t]X` equals `cv2.projectPoints` (no distortion) to 2.7e-05 px. After undistortion, rows of corners deviate from a straight line by 0.16 px instead of 3.00 px.
* **Denoising (PSNR dB / SSIM):** Gaussian noise (sigma 25): noisy 20.36 / 0.296, best bilateral 29.10 / 0.703 (Gaussian 28.94, box 28.19 but best SSIM 0.730, median 28.07). Salt & pepper 5 %: noisy 18.21 / 0.338, best **median 30.89 / 0.856**; box and Gaussian 27.2; bilateral fails (18.27).
* **Canny:** on the noisy image, no blur gives 56 373 edge pixels (mostly false); blur sigma 1.5 gives 5 199.
* **From-scratch convolution** equals `cv2.filter2D` (max difference 0.0 in float; at most 1 grey level after uint8 rounding).
* **Frequency domain:** a Gaussian low-pass (D0 = 30) matches a spatial Gaussian blur with sigma 2.72 (PSNR between them 37.8 dB); the ideal low-pass shows ringing.
* **Harris:** 536 corners from scratch vs 620 with `cv2.cornerHarris` (98 % of ours lie within 3 px of an OpenCV corner).
* **Matching** box -> scene: SIFT 604 / 969 keypoints, 80 matches after the 0.75 ratio test, **75 RANSAC inliers (93.8 %)**, about 30-40 ms; ORB 1252 / 1500 keypoints, 46 matches, **43 inliers (93.5 %)**, about 10 ms. Without the ratio test the inlier ratio drops to 33.8 % (SIFT) and 13.4 % (ORB). The box outline is found correctly by both.

## What to explain (cheat sheet)

### 01 Calibration
* **Pinhole model:** `s [u v 1]^T = K [R|t] [X Y Z 1]^T`; `K` holds `fx, fy` (focal length in pixels), `cx, cy` (principal point). Written by hand in `project_points_pinhole(X, K, R, t)`, verified against `cv2.projectPoints`.
* **Corners:** `cv2.findChessboardCorners` + `cv2.cornerSubPix` (sub-pixel refinement) in `find_chessboard_corners(img, (9, 6))`.
* **Calibration:** known 3D board points (Z = 0) + detected 2D corners from 13 views -> `cv2.calibrateCamera` -> `K, dist, rvecs, tvecs`; one call: `calibrate_from_folder(folder, (9, 6), square_size)`.
* **Quality:** reprojection error = distance between detected corners and corners re-projected with the estimated model; RMS 0.409 px (below 1 px is good); the per-image bar chart shows `left02` as the weakest (0.85 px).
* **Distortion:** `k1, k2, k3` radial (negative `k1` = barrel), `p1, p2` tangential; `undistort_image(img, K, dist)` uses `getOptimalNewCameraMatrix` + `cv2.undistort`.
* **Proof it works:** straight rows of corners deviate 3.00 px before and 0.16 px after undistortion.
* **How many images:** at least 3, but 10-20 with varied board poses give stable values (experiment in step 7).

### 02 Filtering and edges
* **Noise:** `add_gaussian_noise` (sensor noise) and `add_salt_pepper_noise` (impulses), fixed seed.
* **Filters:** `apply_filter(img, 'box'|'gaussian'|'median'|'bilateral')`: mean / weighted mean / median / edge-preserving weighted mean. Compare with PSNR and SSIM.
* **Why:** linear filters (box, Gaussian) average Gaussian noise but smear impulses; median discards impulses (wins S&P 30.89 dB); bilateral keeps edges but cannot remove impulses.
* **Gradients:** `sobel_gradients(gray)` -> Gx, Gy, magnitude, direction; `laplacian` (second derivative); `canny_edges(gray, low, high, blur_sigma)` = blur -> gradient -> non-max suppression -> hysteresis. Blur first on noisy images.
* **From scratch:** `convolve2d_from_scratch(gray, kernel)` = for every kernel weight add `weight x shifted padded image`; equals `cv2.filter2D`.
* **Frequency domain:** `fft_spectrum` (log magnitude, centred) and `frequency_filter(gray, 'lowpass'|'highpass', cutoff, 'ideal'|'gaussian')`: multiply the spectrum by a mask; Gaussian low-pass = spatial Gaussian blur (`sigma = N / (2 pi D0)`); the ideal mask gives ringing.

### 03 Feature matching
* **Harris:** `harris_from_scratch(gray)`: Sobel Ix, Iy -> Gaussian-smoothed Ixx, Iyy, Ixy -> `R = det - k trace^2` -> threshold + non-max suppression; compared with `cv2.cornerHarris`.
* **Keypoints:** `detect_keypoints(img, 'sift'|'orb')`: SIFT = scale-space blobs + 128-D float descriptor (L2); ORB = FAST + binary descriptor (Hamming), faster.
* **Matching:** `match_features(des1, des2, method, ratio=0.75)` = `BFMatcher.knnMatch(k=2)` + Lowe ratio test (`d1 < 0.75 d2`).
* **Why the ratio test:** it rejects ambiguous matches; inlier ratio rises from 33.8 % to 93.8 % (SIFT).
* **RANSAC:** `find_homography_ransac(kp1, kp2, good)` fits a homography from random 4-match samples and keeps the model with most inliers (error < 5 px).
* **Localisation:** `draw_object_outline(scene, box_shape, H)` warps the 4 box corners with `H` and draws the quadrilateral.
* **Whole pipeline:** `match_images(img1, img2, 'sift', 0.75)` returns the visualisation, keypoint / match / inlier counts and time. SIFT is more accurate, ORB is about 3-4x faster.

## Assumptions and caveats
* The chessboard square size is not given with the data set, so the board unit is one square (translation vectors are in squares; `K` and distortion are unaffected).
* Filter parameters (kernel size, sigma, thresholds) were fixed by hand, not optimised; the ranking of the filters for Gaussian noise is close (within about 1 dB).
* RANSAC is random; the OpenCV seed is fixed in notebook 03 so the numbers are reproducible. Timings depend on the computer.
* The notebooks are saved with their outputs (executed), so they open ready to present.

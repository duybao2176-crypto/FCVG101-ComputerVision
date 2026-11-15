"""The "tools" of the Task 3 assistant.

Every tool is a thin wrapper around the Task 1 / Task 2 modules and has the same shape:

    tool(images, params) -> ToolResult

    images : list of RGB uint8 numpy arrays (that is what Gradio gives us)
    params : dict with optional settings (missing keys fall back to defaults)

A ToolResult carries everything the UI needs to show: a title, a list of
(label, RGB image) pairs, a markdown text with the numbers and a raw data dict.

Internally Task 1 uses OpenCV, i.e. BGR images, so each tool converts RGB -> BGR on
the way in and BGR -> RGB on the way out (helpers `to_bgr` / `to_rgb`).

To add a new tool: write a function with this signature, add it to TOOLS at the
bottom, add keywords to router.py.  Nothing else changes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import cv2
import numpy as np

from src.task1 import calibration, features, filtering

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_JSON = PROJECT_ROOT / "Task1" / "outputs" / "calibration.json"
DEFAULT_DISTANCE_MM = 500.0     # only used by `measure` when the user gives no distance / scale


# ----------------------------------------------------------------------
# Shared types and helpers
# ----------------------------------------------------------------------
@dataclass
class ToolResult:
    title: str
    images: List[Tuple[str, np.ndarray]] = field(default_factory=list)   # (label, RGB image)
    text: str = ""                                                       # markdown summary
    data: Dict = field(default_factory=dict)                             # raw numbers


class ToolError(Exception):
    """A problem the user can fix (missing file, bad input).  The UI shows the message as is."""


def to_bgr(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_RGB2BGR)


def to_rgb(img: np.ndarray) -> np.ndarray:
    # Grayscale maps (edges, masks) become 3-channel so every gallery image looks the same.
    if img.ndim == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def shrink(img: np.ndarray, max_side: int = 640) -> np.ndarray:
    """Downscale big photos (phones give 4000 px) so the gallery stays fast."""
    scale = max_side / max(img.shape[:2])
    if scale >= 1:
        return img
    return cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)


def need_images(images: List[np.ndarray], n: int, what: str) -> None:
    if len(images) < n:
        raise ToolError(f"This tool needs {n} image(s) ({what}). Please upload them.")


def load_saved_calibration() -> Dict:
    """Task 1 result (K, dist, image_size, rms).  Run Task1/01_calibration.ipynb to create it."""
    if not CALIBRATION_JSON.exists():
        raise ToolError("No saved calibration found. Run Task1/01_calibration.ipynb first, "
                        "or upload 3+ chessboard photos and ask me to recalibrate.")
    return calibration.load_calibration(CALIBRATION_JSON)


def scale_intrinsics(K: np.ndarray, calib_size: Tuple[int, int], img_shape: Tuple[int, ...]) -> np.ndarray:
    """Adapt K to an image that is a resized version of the calibration image.

    fx, cx scale with the width ratio and fy, cy with the height ratio.  If the
    sizes already match this returns K unchanged.
    """
    h, w = img_shape[:2]
    sx, sy = w / calib_size[0], h / calib_size[1]
    K2 = K.copy()
    K2[0, :] *= sx
    K2[1, :] *= sy
    return K2


def undistort_rgb(img_rgb: np.ndarray, calib: Dict) -> Tuple[np.ndarray, np.ndarray]:
    """Undistort an RGB image with a calibration dict -> (undistorted RGB, K used)."""
    K = scale_intrinsics(calib["K"], calib["image_size"], img_rgb.shape)
    out = calibration.undistort_image(to_bgr(img_rgb), K, calib["dist"])
    return to_rgb(out), K


def matrix_markdown(K: np.ndarray, dist: np.ndarray) -> str:
    rows = "\n".join("  [" + "  ".join(f"{v:9.3f}" for v in row) + "]" for row in K)
    d = dict(zip(["k1", "k2", "p1", "p2", "k3"], np.asarray(dist).ravel()))
    dist_txt = ", ".join(f"{k} = {v:.4f}" for k, v in d.items())
    return f"**K** (rows: fx 0 cx / 0 fy cy / 0 0 1)\n\n```\n{rows}\n```\n\n**Distortion**: {dist_txt}"


# ----------------------------------------------------------------------
# Tool 1: calibrate  (Task 1.1)
# ----------------------------------------------------------------------
def recalibrate_from_files(paths: List[str], pattern_size=(9, 6), square_mm: float = 1.0) -> Dict:
    """Detect chessboard corners in the given files and run cv2.calibrateCamera."""
    template = calibration.make_object_points(pattern_size, square_mm)
    obj_pts, img_pts, used, skipped, preview = [], [], [], [], None
    image_size = None
    for path in paths:
        img = cv2.imread(str(path))
        if img is None:
            skipped.append(Path(path).name)
            continue
        image_size = (img.shape[1], img.shape[0])
        found, corners = calibration.find_chessboard_corners(img, pattern_size)
        if not found:
            skipped.append(Path(path).name)
            continue
        obj_pts.append(template)
        img_pts.append(corners)
        used.append(Path(path).name)
        if preview is None:
            preview = calibration.draw_corners(img, pattern_size, corners)
    if len(used) < 3:
        raise ToolError(f"Only {len(used)} image(s) with a detectable {pattern_size[0]}x{pattern_size[1]} "
                        f"chessboard. I need at least 3 (10+ is better).")
    result = calibration.calibrate_camera(obj_pts, img_pts, image_size)
    result.update(image_size=image_size, used=used, skipped=skipped, preview=preview)
    return result


def calibrate(images: List[np.ndarray], params: Dict) -> ToolResult:
    """Undistort the uploaded image with the saved calibration (or a fresh one)."""
    files = params.get("chessboard_files") or []
    recalibrate = bool(params.get("recalibrate")) or len(files) >= 3
    out_images: List[Tuple[str, np.ndarray]] = []

    if recalibrate:
        if len(files) < 3:
            raise ToolError("To recalibrate, upload at least 3 chessboard photos in the "
                            "'Extra images' box (9 x 6 inner corners).")
        square_mm = float(params.get("square_size_mm", 1.0))
        res = recalibrate_from_files(files, (9, 6), square_mm)
        calib = {"K": res["K"], "dist": res["dist"], "image_size": res["image_size"], "rms": res["rms"]}
        source = f"recalibrated from {len(res['used'])} chessboard images (skipped: {len(res['skipped'])})"
        unit = "mm" if square_mm != 1.0 else "squares"
        extra = f"\n\nMean reprojection error: **{res['mean_error']:.3f} px**. Board unit: {unit}."
        out_images.append(("Detected corners (first image)", to_rgb(res["preview"])))
        if not images:   # nothing uploaded in image 1: undistort the first chessboard photo
            images = [to_rgb(cv2.imread(str(files[0])))]
    else:
        calib = load_saved_calibration()
        source = "saved Task 1 calibration (Task1/outputs/calibration.json)"
        extra = ""
    need_images(images, 1, "the image to undistort")

    img = images[0]
    undistorted, K_used = undistort_rgb(img, calib)
    notes = ""
    if (img.shape[1], img.shape[0]) != tuple(calib["image_size"]):
        notes = (f"\n\nNote: the calibration was made at {calib['image_size'][0]}x{calib['image_size'][1]} "
                 f"and your image is {img.shape[1]}x{img.shape[0]}; K was rescaled (valid only if it is "
                 f"a resized photo from the same camera).")
    text = (f"Calibration source: **{source}**\n\nRMS reprojection error: **{calib['rms']:.3f} px**\n\n"
            + matrix_markdown(K_used, calib["dist"]) + extra + notes +
            "\n\nBarrel distortion (negative k1) bends straight lines outwards; "
            "the undistorted image should show them straight again.")
    out_images = [("Original", img), ("Undistorted", undistorted)] + out_images
    return ToolResult("Camera calibration / undistortion", out_images, text,
                      {"K": K_used.tolist(), "dist": list(map(float, calib["dist"])), "rms": calib["rms"]})


# ----------------------------------------------------------------------
# Tool 2: filter  (Task 1.2)
# ----------------------------------------------------------------------
FILTER_NAMES = ["box", "gaussian", "median", "bilateral"]


def add_noise(img_bgr: np.ndarray, kind: str, params: Dict) -> np.ndarray:
    if kind == "gaussian":
        return filtering.add_gaussian_noise(img_bgr, sigma=float(params.get("noise_level", 25)))
    if kind == "saltpepper":
        return filtering.add_salt_pepper_noise(img_bgr, amount=float(params.get("noise_amount", 0.05)))
    return img_bgr


def filter_image(images: List[np.ndarray], params: Dict) -> ToolResult:
    """Optionally add noise, then smooth with one (or all) of the 4 filters."""
    need_images(images, 1, "the image to filter")
    clean = to_bgr(images[0])
    noise = params.get("noise", "none")
    ksize = int(params.get("ksize", 5))
    ksize = ksize if ksize % 2 == 1 else ksize + 1
    # Sensible default: median kills salt & pepper, Gaussian is a good all-rounder.
    chosen = params.get("filter") or ("median" if noise == "saltpepper" else "gaussian")
    names = FILTER_NAMES if chosen == "all" else [chosen]

    noisy = add_noise(clean, noise, params)
    has_reference = noise != "none"      # we only know the "truth" if WE added the noise
    out_images = [("Original", to_rgb(clean))]
    if has_reference:
        out_images.append((f"With {noise} noise", to_rgb(noisy)))

    rows = []
    for name in names:
        result = filtering.apply_filter(noisy, name, ksize=ksize)   # bilateral ignores ksize
        out_images.append((f"{name} filter" + (f" (k={ksize})" if name != "bilateral" else ""), to_rgb(result)))
        if has_reference:
            rows.append(f"| {name} | {filtering.psnr(clean, result):.2f} | {filtering.ssim(clean, result):.3f} |")

    text = f"Filter(s): **{', '.join(names)}**, kernel size **{ksize}**, noise added: **{noise}**.\n\n"
    if has_reference:
        before = f"{filtering.psnr(clean, noisy):.2f} dB / SSIM {filtering.ssim(clean, noisy):.3f}"
        text += (f"Quality against the clean original (higher is better). Noisy image: {before}\n\n"
                 "| filter | PSNR (dB) | SSIM |\n|---|---|---|\n" + "\n".join(rows))
    else:
        text += ("No PSNR/SSIM: without a clean reference they cannot be computed. "
                 "Ask me to *add salt and pepper noise* or *gaussian noise* first to get numbers.")
    text += "\n\nRule of thumb: median for salt & pepper, Gaussian/bilateral for Gaussian noise."
    return ToolResult("Noise and filtering", out_images, text, {"filters": names, "ksize": ksize, "noise": noise})


# ----------------------------------------------------------------------
# Tool 3: edges / frequency  (Task 1.2)
# ----------------------------------------------------------------------
def normalise_to_uint8(arr: np.ndarray) -> np.ndarray:
    return cv2.normalize(arr, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)


def edges(images: List[np.ndarray], params: Dict) -> ToolResult:
    """Sobel / Canny / Laplacian edge maps, or FFT low-/high-pass filtering."""
    need_images(images, 1, "the image to analyse")
    img_bgr = to_bgr(images[0])
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

    if params.get("frequency"):      # "fft", "low-pass", "high-pass" ...
        kind = params["frequency"]
        cutoff = float(params.get("cutoff", 30))
        filtered = filtering.frequency_filter(gray, kind, cutoff, "gaussian")
        spectrum = normalise_to_uint8(filtering.fft_spectrum(gray))
        text = (f"Frequency-domain **{kind}** (Gaussian mask, cut-off D0 = {cutoff:g}).\n\n"
                "FFT -> multiply the spectrum by the mask -> inverse FFT. Low-pass keeps smooth areas "
                "(blur); high-pass keeps fast changes, i.e. edges and noise.")
        return ToolResult("Frequency-domain filtering",
                          [("Grayscale input", to_rgb(gray)), ("FFT log-magnitude spectrum", to_rgb(spectrum)),
                           (f"{kind} result", to_rgb(filtered))], text, {"kind": kind, "cutoff": cutoff})

    method = params.get("edge_method", "canny")
    methods = ["sobel", "laplacian", "canny"] if method == "all" else [method]
    low, high = int(params.get("canny_low", 100)), int(params.get("canny_high", 200))
    out_images = [("Input", to_rgb(img_bgr))]
    lines = []
    for m in methods:
        if m == "canny":
            edge_map = filtering.detect_edges(img_bgr, "canny", low=low, high=high,
                                              blur_sigma=float(params.get("blur_sigma", 1.0)))
            lines.append(f"* **canny** (low {low}, high {high}, Gaussian blur sigma 1.0): "
                         f"{100 * np.count_nonzero(edge_map) / edge_map.size:.1f} % of the pixels are edges")
        else:
            edge_map = filtering.detect_edges(img_bgr, m)
            lines.append(f"* **{m}** gradient magnitude (brighter = stronger edge), normalised to 0-255")
        out_images.append((f"{m} edges", to_rgb(edge_map)))
    text = "Edge detection:\n\n" + "\n".join(lines) + \
        "\n\nSobel = first derivative, Laplacian = second derivative (noise-sensitive), " \
        "Canny = blur + gradient + non-max suppression + hysteresis thresholds."
    return ToolResult("Edge detection", out_images, text, {"methods": methods})


# ----------------------------------------------------------------------
# Tool 4: features  (Task 1.3)
# ----------------------------------------------------------------------
def draw_harris(img_bgr: np.ndarray, k: float = 0.04) -> Tuple[np.ndarray, int]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    corners = features.harris_from_scratch(gray, k=k)["corners"]
    out = img_bgr.copy()
    for x, y in corners:
        cv2.circle(out, (int(x), int(y)), 3, (0, 0, 255), 1)
    return out, len(corners)


def match_two_images(images: List[np.ndarray], method: str, ratio: float) -> ToolResult:
    """Image 1 = object, image 2 = scene: detect, ratio test, RANSAC, outline."""
    res = features.match_images(to_bgr(images[0]), to_bgr(images[1]), method, ratio)
    found = res["homography"] is not None
    text = (f"Method **{method.upper()}**, ratio test **{ratio}**.\n\n"
            f"| keypoints image 1 | keypoints image 2 | matches after ratio test | RANSAC inliers | inlier ratio | time |\n"
            f"|---|---|---|---|---|---|\n"
            f"| {res['n_keypoints1']} | {res['n_keypoints2']} | {res['n_good_matches']} | "
            f"{res['n_inliers']} | {100 * res['inlier_ratio']:.1f} % | {1000 * res['time_s']:.0f} ms |\n\n"
            + ("The object from image 1 was **found** in image 2 (green quadrilateral = homography outline). "
               "Green lines = RANSAC inliers, red = outliers."
               if found else "The object was **not found**: too few consistent matches. Try a larger ratio or the other method."))
    return ToolResult("Feature matching",
                      [("Matches (green = inliers)", to_rgb(res["visualization"])),
                       ("Object located in the scene", to_rgb(res["scene_with_box"]))],
                      text, {k: v for k, v in res.items() if k not in ("visualization", "scene_with_box")})


def features_tool(images: List[np.ndarray], params: Dict) -> ToolResult:
    """Two images -> matching.  One image -> show its keypoints / Harris corners."""
    need_images(images, 1, "one image for keypoints, two for matching")
    method = params.get("method", "sift")
    ratio = min(max(float(params.get("ratio", 0.75)), 0.3), 0.99)

    if len(images) >= 2 and method != "harris":
        return match_two_images(images, method, ratio)

    img_bgr = to_bgr(images[0])
    if method == "harris":
        vis, n = draw_harris(img_bgr)
        text = (f"Harris corner detector written from scratch: **{n}** corners (red circles). "
                "R = det(M) - k trace(M)^2, thresholded at 1 % of max(R) + non-max suppression.")
    else:
        kps, _ = features.detect_keypoints(img_bgr, method)
        vis = cv2.drawKeypoints(img_bgr, kps, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        text = (f"**{method.upper()}** found **{len(kps)}** keypoints (circle size = scale, line = orientation). "
                "Upload a second image to match them.")
    if len(images) >= 2:
        text += "\n\n(Harris has no descriptors, so only image 1 was analysed.)"
    return ToolResult("Keypoints", [("Input", images[0]), (f"{method} result", to_rgb(vis))], text, {"method": method})


# ----------------------------------------------------------------------
# Tool 5: classify  (Task 2 model)
# ----------------------------------------------------------------------
_model_cache: Dict = {}


def load_task2_model():
    """Load the Task 2 model once; reload automatically when the weight file changes.

    Imported lazily so the app starts (and the other tools work) even when torch is
    slow to import or the model has not been trained yet.
    """
    from src.task2 import config, predict            # lazy import
    weights = config.OUT_DIR / "model_best.pt"
    if not weights.exists() or not (config.OUT_DIR / "model_meta.json").exists():
        raise ToolError("The Task 2 model is not available yet. Train it with Task2/03_train.ipynb "
                        f"(it must create `model_best.pt` and `model_meta.json` in `{config.OUT_DIR}`). "
                        "The other tools keep working.")
    key = (str(weights), weights.stat().st_mtime)
    if _model_cache.get("key") != key:
        _model_cache["model"] = predict.load_model()
        _model_cache["key"] = key
    return _model_cache["model"]


def probability_bar_chart(top3: List[Dict]) -> np.ndarray:
    """Horizontal bar chart of the top-3 classes as an RGB image."""
    import matplotlib
    matplotlib.use("Agg")                  # no window: we only want a picture
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(4.5, 2.4), dpi=100)
    names = [t["class_name"] for t in top3][::-1]
    probs = [t["probability"] for t in top3][::-1]
    ax.barh(names, probs, color="#3b7dd8")
    ax.set_xlim(0, 1)
    ax.set_xlabel("probability")
    for y, p in enumerate(probs):
        ax.text(min(p + 0.02, 0.8), y, f"{p:.2f}", va="center")
    fig.tight_layout()
    fig.canvas.draw()
    out = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return out


def classify(images: List[np.ndarray], params: Dict) -> ToolResult:
    """Fresh / rotten fruit classification with Grad-CAM (Task 2)."""
    need_images(images, 1, "a photo of the fruit")
    model = load_task2_model()
    from src.task2 import predict
    pred = predict.predict_image(model, images[0], top_k=3)
    cam = predict.gradcam(model, predict.preprocess_image(model, images[0]))
    unsure = "\n\n**Low confidence**: the model is unsure; the image may differ from the training data." \
        if pred["confidence"] < 0.6 else ""
    top_lines = "\n".join(f"{i + 1}. {t['class_name']}: {100 * t['probability']:.1f} %"
                          for i, t in enumerate(pred["top3"]))
    text = (f"Prediction: **{pred['class_name']}**\n\n"
            f"* Fruit: **{pred['fruit']}**\n* Freshness: **{pred['freshness']}**\n"
            f"* Confidence: **{100 * pred['confidence']:.1f} %**\n\nTop 3:\n\n{top_lines}{unsure}\n\n"
            "Grad-CAM: warm colours mark the image regions that pushed the network towards this class.")
    return ToolResult("Fruit freshness classification",
                      [("Input", shrink(images[0])), ("Grad-CAM", cam), ("Top-3", probability_bar_chart(pred["top3"]))],
                      text, {k: pred[k] for k in ("class_name", "fruit", "freshness", "confidence")})


# ----------------------------------------------------------------------
# Tool 6: measure
# ----------------------------------------------------------------------
def find_largest_object(img_bgr: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Segment the main object with Otsu thresholding -> (contour, binary mask).

    Assumption: one object clearly darker or brighter than a fairly uniform background.
    """
    gray = cv2.GaussianBlur(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY), (5, 5), 0)
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    border = np.concatenate([mask[0], mask[-1], mask[:, 0], mask[:, -1]])
    if border.mean() > 127:              # most border pixels are "on" -> that is the background
        mask = cv2.bitwise_not(mask)     # flip so that the object is white
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)     # remove specks
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)    # fill small holes
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    area_total = mask.shape[0] * mask.shape[1]
    # Ignore tiny blobs and a blob that fills the whole frame (= segmentation failed).
    good = [c for c in contours if 0.005 * area_total < cv2.contourArea(c) < 0.95 * area_total]
    if not good:
        raise ToolError("I could not find a clear object. Use a plain background with one object "
                        "that is clearly darker or brighter than it.")
    return max(good, key=cv2.contourArea), mask


def measure(images: List[np.ndarray], params: Dict) -> ToolResult:
    """Measure the largest object in real-world units.

    Two ways to get the scale (pixels -> mm):
      A) `px_per_mm` given by the user (e.g. from a ruler in the picture), or
      B) the pinhole relation   size_mm = size_px * Z / f   with the focal length f (pixels)
         from the Task 1 calibration and the camera-to-object distance Z (mm) from the user.
    Assumptions (stated in the output): the object is flat, parallel to the image plane and at
    distance Z; lens distortion is removed first.
    """
    need_images(images, 1, "a photo of one object on a plain background")
    img = images[0]
    px_per_mm = params.get("px_per_mm")
    distance = float(params.get("distance_mm", DEFAULT_DISTANCE_MM))
    assumed = "distance_mm" not in params and not px_per_mm

    calib = load_saved_calibration()
    K = scale_intrinsics(calib["K"], calib["image_size"], img.shape)
    if params.get("undistort", True):
        # Straight lines first, then measure.  cv2.undistort keeps K unchanged (newCameraMatrix=K),
        # so the focal length used below is still correct.  (Task 1's undistort_image crops/zooms.)
        work_rgb = to_rgb(cv2.undistort(to_bgr(img), K, calib["dist"]))
    else:
        work_rgb = img
    contour, mask = find_largest_object(to_bgr(work_rgb))
    (cx, cy), (w_px, h_px), angle = cv2.minAreaRect(contour)    # rotated bounding rectangle
    length_px, width_px = max(w_px, h_px), min(w_px, h_px)

    if px_per_mm:
        mm_per_px = 1.0 / float(px_per_mm)
        scale_txt = f"user scale: {float(px_per_mm):g} px/mm"
    else:
        focal = (K[0, 0] + K[1, 1]) / 2.0            # average fx, fy (they are almost equal)
        mm_per_px = distance / focal                 # one pixel covers Z / f millimetres at distance Z
        scale_txt = f"pinhole: size = pixels x Z / f, with f = {focal:.1f} px and Z = {distance:g} mm"
    length_mm, width_mm = length_px * mm_per_px, width_px * mm_per_px

    vis = work_rgb.copy()
    box = cv2.boxPoints(((cx, cy), (w_px, h_px), angle)).astype(np.int32)
    cv2.drawContours(vis, [box], 0, (255, 0, 0), 2)
    label = f"{length_mm:.1f} x {width_mm:.1f} mm"
    x0, y0 = max(int(box[:, 0].min()), 5), max(int(box[:, 1].min()) - 10, 25)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.rectangle(vis, (x0 - 3, y0 - th - 5), (x0 + tw + 3, y0 + 5), (255, 255, 255), -1)   # white label background
    cv2.putText(vis, label, (x0, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 0, 0), 2, cv2.LINE_AA)

    warn = (f"\n\n**Assumed distance:** you did not give one, so Z = {DEFAULT_DISTANCE_MM:g} mm was assumed. "
            "Say e.g. *the object is 30 cm away* for a real measurement.") if assumed else ""
    text = (f"Largest object: **{length_mm:.1f} mm x {width_mm:.1f} mm** "
            f"({length_px:.1f} x {width_px:.1f} px, rotated rectangle, area {cv2.contourArea(contour):.0f} px^2).\n\n"
            f"Scale: {scale_txt}.{warn}\n\n"
            "**Assumptions / accuracy:** the object is flat, facing the camera (parallel to the image plane) and "
            "all at distance Z; the error in mm is as large, in percent, as the error in Z "
            "(10 % wrong distance = 10 % wrong size). "
            + ("The image was undistorted with the Task 1 calibration first. " if params.get("undistort", True)
               else "Lens distortion was NOT removed. ")
            + "Tilted or 3-D objects give wrong values.")
    return ToolResult("Measurement", [("Measured object", vis), ("Segmentation mask", to_rgb(mask))], text,
                      {"length_mm": length_mm, "width_mm": width_mm, "length_px": length_px, "width_px": width_px})


# ----------------------------------------------------------------------
# Registry: the router returns one of these names
# ----------------------------------------------------------------------
TOOLS: Dict[str, Callable[[List[np.ndarray], Dict], ToolResult]] = {
    "calibrate": calibrate,
    "filter": filter_image,
    "edges": edges,
    "features": features_tool,
    "classify": classify,
    "measure": measure,
}


# Which Task 1 / Task 2 code each tool calls (shown in the UI as part of the tool-use trace).
TOOL_BACKENDS: Dict[str, str] = {
    "calibrate": "Task 1.1  src/task1/calibration.py: find_chessboard_corners, calibrate_camera, undistort_image",
    "filter": "Task 1.2  src/task1/filtering.py: add_*_noise, apply_filter, psnr, ssim",
    "edges": "Task 1.2  src/task1/filtering.py: detect_edges (Sobel/Laplacian/Canny), frequency_filter, fft_spectrum",
    "features": "Task 1.3  src/task1/features.py: detect_keypoints, match_images (ratio test + RANSAC), harris_from_scratch",
    "classify": "Task 2    src/task2/predict.py: load_model, predict_image, gradcam",
    "measure": "Task 1.1 calibration (K, distortion) + OpenCV segmentation + pinhole relation size = px * Z / f",
}


# Parameters each tool understands (the app drops everything else, so the trace stays readable).
TOOL_PARAMS: Dict[str, List[str]] = {
    "calibrate": ["recalibrate", "square_size_mm", "chessboard_files"],
    "filter": ["filter", "ksize", "noise", "noise_level", "noise_amount"],
    "edges": ["edge_method", "frequency", "cutoff", "canny_low", "canny_high", "blur_sigma"],
    "features": ["method", "ratio"],
    "classify": [],
    "measure": ["distance_mm", "px_per_mm", "undistort"],
}


def task2_model_status() -> str:
    """One-line status of the Task 2 model (checks the files only, does not import torch)."""
    from src.task2 import config
    ok = (config.OUT_DIR / "model_best.pt").exists() and (config.OUT_DIR / "model_meta.json").exists()
    return "Task 2 model: found, classification is available." if ok else         "Task 2 model: not found yet (run Task2/03_train.ipynb). Classification is disabled, all other tools work."


def run_tool(name: str, images: List[np.ndarray], params: Dict) -> ToolResult:
    if name not in TOOLS:
        raise ToolError(f"Unknown tool '{name}'. Available: {', '.join(TOOLS)}.")
    return TOOLS[name](images, params)

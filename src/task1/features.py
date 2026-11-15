"""Task 1.3 - Feature detection and matching.

Pipeline for a pair of images:
    detect keypoints + descriptors (SIFT / ORB)
    -> k-NN matching (BFMatcher) + Lowe ratio test
    -> RANSAC homography (removes the remaining wrong matches)
    -> draw the matches and the outline of the object in the scene.

All functions take / return numpy arrays or plain dicts (no plotting / printing).
"""

from __future__ import annotations

import time
from typing import Dict, List, Tuple

import cv2
import numpy as np


def _to_gray(img: np.ndarray) -> np.ndarray:
    """BGR -> gray (pass-through if already 2D)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


# ======================================================================
# 1. Harris corner detector from scratch
# ======================================================================
def harris_from_scratch(
    gray: np.ndarray, k: float = 0.04, window: int = 3, thresh_ratio: float = 0.01
) -> Dict:
    """Harris corner detector written step by step.

    Steps:
        1. Image derivatives Ix, Iy (Sobel).
        2. Structure-tensor entries Ixx = Ix*Ix, Iyy = Iy*Iy, Ixy = Ix*Iy,
           each smoothed with a Gaussian window (this is the sum over the neighbourhood).
        3. Response R = det(M) - k * trace(M)^2, with
           det = Ixx*Iyy - Ixy^2 and trace = Ixx + Iyy.
           R is large and positive at corners, negative on edges, ~0 on flat areas.
        4. Keep pixels with R > thresh_ratio * max(R) that are local maxima
           (non-maximum suppression in a 3x3 neighbourhood).

    Args:
        gray: 2D uint8 image.
        k: Harris constant (0.04 - 0.06).
        window: size of the Gaussian smoothing window (odd).
        thresh_ratio: threshold as a fraction of the maximum response.

    Returns:
        dict with 'response' (float32 map R) and 'corners' (N, 2) array of (x, y).
    """
    gray = _to_gray(gray).astype(np.float32)

    # Step 1: derivatives
    Ix = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    Iy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)

    # Step 2: structure tensor entries, smoothed over a window
    Ixx = cv2.GaussianBlur(Ix * Ix, (window, window), 0)
    Iyy = cv2.GaussianBlur(Iy * Iy, (window, window), 0)
    Ixy = cv2.GaussianBlur(Ix * Iy, (window, window), 0)

    # Step 3: corner response
    det = Ixx * Iyy - Ixy * Ixy
    trace = Ixx + Iyy
    R = det - k * trace * trace

    # Step 4: threshold + non-maximum suppression
    corners = _non_max_suppression(R, thresh_ratio * R.max())
    return {"response": R, "corners": corners}


def _non_max_suppression(R: np.ndarray, threshold: float) -> np.ndarray:
    """Return (x, y) of pixels that exceed `threshold` and are the max of their 3x3 neighbourhood."""
    # Dilation puts the local maximum of each 3x3 window into every pixel;
    # a pixel is a local max if it equals that value.
    local_max = cv2.dilate(R, np.ones((3, 3), np.uint8))
    mask = (R > threshold) & (R == local_max)
    ys, xs = np.nonzero(mask)
    return np.stack([xs, ys], axis=1)


def harris_opencv(
    gray: np.ndarray, k: float = 0.04, block_size: int = 3, thresh_ratio: float = 0.01
) -> Dict:
    """Reference Harris using cv2.cornerHarris with the same thresholding + NMS."""
    gray = _to_gray(gray).astype(np.float32)
    R = cv2.cornerHarris(gray, block_size, 3, k)
    return {"response": R, "corners": _non_max_suppression(R, thresh_ratio * R.max())}


# ======================================================================
# 2. Keypoints and descriptors
# ======================================================================
def detect_keypoints(img: np.ndarray, method: str = "sift", **params) -> Tuple[list, np.ndarray]:
    """Detect keypoints and compute their descriptors.

    Args:
        img: BGR or gray uint8 image.
        method: 'sift' (128-D float descriptor) or 'orb' (256-bit binary descriptor).
        **params: orb only: nfeatures (default 1500).

    Returns:
        (keypoints, descriptors). descriptors is None if nothing was found.
    """
    gray = _to_gray(img)
    if method == "sift":
        detector = cv2.SIFT_create()
    elif method == "orb":
        detector = cv2.ORB_create(nfeatures=params.get("nfeatures", 1500))
    else:
        raise ValueError("method must be 'sift' or 'orb'")
    keypoints, descriptors = detector.detectAndCompute(gray, None)
    return list(keypoints), descriptors


# ======================================================================
# 3. Matching
# ======================================================================
def _norm_for(method: str) -> int:
    """SIFT descriptors are float vectors -> L2; ORB descriptors are bits -> Hamming."""
    return cv2.NORM_L2 if method == "sift" else cv2.NORM_HAMMING


def match_features(des1: np.ndarray, des2: np.ndarray, method: str = "sift", ratio: float = 0.75):
    """k-NN matching (k=2) followed by Lowe's ratio test.

    For each descriptor in image 1 we find its 2 nearest neighbours in image 2.
    The match is kept only if  d(best) < ratio * d(second best), i.e. the best
    match is clearly better than the runner-up (so it is distinctive).

    Args:
        des1, des2: descriptor arrays.
        method: 'sift' or 'orb' (selects the distance norm).
        ratio: Lowe ratio threshold (0.75 is a common choice).

    Returns:
        list of cv2.DMatch (queryIdx refers to image 1, trainIdx to image 2).
    """
    if des1 is None or des2 is None or len(des1) < 2 or len(des2) < 2:
        return []
    matcher = cv2.BFMatcher(_norm_for(method))
    pairs = matcher.knnMatch(des1, des2, k=2)
    # Some entries may have fewer than 2 neighbours; keep only complete pairs.
    return [p[0] for p in pairs if len(p) == 2 and p[0].distance < ratio * p[1].distance]


def match_best_only(des1: np.ndarray, des2: np.ndarray, method: str = "sift"):
    """Baseline WITHOUT the ratio test: every descriptor takes its single nearest neighbour,
    with cross-check (the match must be mutual)."""
    if des1 is None or des2 is None:
        return []
    matcher = cv2.BFMatcher(_norm_for(method), crossCheck=True)
    return sorted(matcher.match(des1, des2), key=lambda m: m.distance)


# ======================================================================
# 4. Homography with RANSAC and object outline
# ======================================================================
def find_homography_ransac(kp1, kp2, good, reproj_thresh: float = 5.0):
    """Estimate the homography H (image 1 -> image 2) from matches using RANSAC.

    RANSAC repeatedly fits H from 4 random matches and keeps the model that is
    consistent with the most matches (inliers, reprojection error < threshold).

    Returns:
        (H, inlier_mask). H is 3x3 or None if fewer than 4 matches;
        inlier_mask is a list of 0/1 per match.
    """
    if len(good) < 4:
        return None, []
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, reproj_thresh)
    if H is None:
        return None, []
    return H, mask.ravel().tolist()


def draw_object_outline(
    scene: np.ndarray, object_shape: Tuple[int, int], H: np.ndarray, color=(0, 255, 0)
) -> np.ndarray:
    """Draw the outline of the object (its 4 corners warped by H) on a copy of the scene.

    Args:
        scene: BGR image.
        object_shape: (height, width) of the object image.
        H: homography object -> scene.
    """
    h, w = object_shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    warped = cv2.perspectiveTransform(corners, H)  # where the 4 corners land in the scene
    out = scene.copy()
    cv2.polylines(out, [np.int32(warped)], True, color, 3, cv2.LINE_AA)
    return out


# ======================================================================
# 5. Top-level pipeline
# ======================================================================
def match_images(img1: np.ndarray, img2: np.ndarray, method: str = "sift", ratio: float = 0.75) -> Dict:
    """Detect, match (ratio test), verify with RANSAC and visualise.

    Args:
        img1: the object / first view (BGR).
        img2: the scene / second view (BGR).
        method: 'sift' or 'orb'.
        ratio: Lowe ratio threshold.

    Returns:
        dict with:
            visualization: BGR image, matches drawn (green = RANSAC inliers, red = outliers)
                           and the object outline drawn in the scene if H was found,
            scene_with_box: scene with only the outline drawn (or a copy of the scene),
            n_keypoints1, n_keypoints2, n_raw_matches (one nearest-neighbour match per descriptor of image 1, before the ratio test),
            n_good_matches (after ratio test), n_inliers (after RANSAC),
            inlier_ratio, homography (3x3 list or None), time_s (detect+match+RANSAC).
    """
    t0 = time.perf_counter()
    kp1, des1 = detect_keypoints(img1, method)
    kp2, des2 = detect_keypoints(img2, method)
    good = match_features(des1, des2, method, ratio)
    H, mask = find_homography_ransac(kp1, kp2, good)
    elapsed = time.perf_counter() - t0

    n_inliers = int(sum(mask))
    scene_with_box = img2.copy()
    if H is not None:
        scene_with_box = draw_object_outline(img2, img1.shape[:2], H)

    return {
        "visualization": draw_matches(img1, kp1, scene_with_box, kp2, good, mask),
        "scene_with_box": scene_with_box,
        "n_keypoints1": len(kp1),
        "n_keypoints2": len(kp2),
        "n_raw_matches": len(des1) if des1 is not None else 0,
        "n_good_matches": len(good),
        "n_inliers": n_inliers,
        "inlier_ratio": n_inliers / len(good) if good else 0.0,
        "homography": H.tolist() if H is not None else None,
        "time_s": elapsed,
    }


def draw_matches(img1, kp1, img2, kp2, matches, mask=None, max_draw: int = 100) -> np.ndarray:
    """Draw matches side by side. If `mask` is given, inliers are green and outliers red."""
    if mask:
        # Draw outliers first (red), then inliers (green) on top of the same canvas.
        outliers = [m for m, ok in zip(matches, mask) if not ok]
        inliers = [m for m, ok in zip(matches, mask) if ok]
        flags = cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
        canvas = cv2.drawMatches(img1, kp1, img2, kp2, outliers[:max_draw], None,
                                 matchColor=(0, 0, 255), flags=flags)
        return cv2.drawMatches(img1, kp1, img2, kp2, inliers[:max_draw], canvas,
                               matchColor=(0, 200, 0),
                               flags=flags | cv2.DrawMatchesFlags_DRAW_OVER_OUTIMG)
    return cv2.drawMatches(img1, kp1, img2, kp2, matches[:max_draw], None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

"""Task 1.1 - Image formation and camera calibration.

Pinhole camera model (see notebook 01):

    s * [u, v, 1]^T = K [R | t] [X, Y, Z, 1]^T

    K = [[fx,  0, cx],
         [ 0, fy, cy],
         [ 0,  0,  1]]

All functions take / return numpy arrays (BGR uint8 images) and plain
python types / dicts, so they can be reused by the Gradio app (Task 3).
No plotting and no printing happens in this module.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np

# Stop criteria used for the sub-pixel refinement of the corners.
_SUBPIX_CRITERIA = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)


# ----------------------------------------------------------------------
# 1. Finding the chessboard corners
# ----------------------------------------------------------------------
def make_object_points(pattern_size: Tuple[int, int], square_size: float = 1.0) -> np.ndarray:
    """Build the 3D coordinates of the chessboard corners in the board frame.

    The board lies in the plane Z = 0, the first corner is the origin.

    Args:
        pattern_size: (columns, rows) of INNER corners, e.g. (9, 6).
        square_size: side of one square (any unit; translation vectors
            returned by the calibration will be in the same unit).

    Returns:
        Array of shape (cols*rows, 3), float32.
    """
    cols, rows = pattern_size
    points = np.zeros((rows * cols, 3), np.float32)
    # mgrid gives all (x, y) integer grid positions; we fill X and Y only.
    points[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    return points * square_size


def find_chessboard_corners(
    img: np.ndarray, pattern_size: Tuple[int, int] = (9, 6)
) -> Tuple[bool, Optional[np.ndarray]]:
    """Detect the inner chessboard corners and refine them to sub-pixel accuracy.

    Args:
        img: BGR (or grayscale) uint8 image.
        pattern_size: (columns, rows) of inner corners.

    Returns:
        (found, corners). corners has shape (N, 1, 2) float32, or None if
        the pattern was not found.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    found, corners = cv2.findChessboardCorners(gray, pattern_size)
    if not found:
        return False, None
    # cornerSubPix moves each corner to the exact saddle point of the intensity.
    corners = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), _SUBPIX_CRITERIA)
    return True, corners


def draw_corners(img: np.ndarray, pattern_size: Tuple[int, int], corners: np.ndarray) -> np.ndarray:
    """Return a copy of `img` with the detected corners drawn (rainbow lines)."""
    vis = img.copy()
    cv2.drawChessboardCorners(vis, pattern_size, corners, True)
    return vis


# ----------------------------------------------------------------------
# 2. Calibration
# ----------------------------------------------------------------------
def calibrate_camera(
    object_points: List[np.ndarray],
    image_points: List[np.ndarray],
    image_size: Tuple[int, int],
) -> Dict:
    """Run cv2.calibrateCamera and compute per-image reprojection errors.

    Args:
        object_points: list of (N, 3) float32 arrays (one per image).
        image_points: list of (N, 1, 2) float32 arrays (one per image).
        image_size: (width, height) of the images.

    Returns:
        dict with keys: K (3x3), dist (1D array k1,k2,p1,p2,k3), rvecs,
        tvecs, rms (OpenCV overall RMS), per_image_errors (list of float,
        mean pixel distance per image), mean_error (float).
    """
    rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        object_points, image_points, image_size, None, None
    )
    errors = reprojection_errors(object_points, image_points, K, dist, rvecs, tvecs)
    return {
        "K": K,
        "dist": dist.ravel(),
        "rvecs": rvecs,
        "tvecs": tvecs,
        "rms": float(rms),
        "per_image_errors": errors,
        "mean_error": float(np.mean(errors)),
    }


def reprojection_errors(object_points, image_points, K, dist, rvecs, tvecs) -> List[float]:
    """Mean Euclidean pixel error between detected and re-projected corners, per image."""
    errors = []
    for obj, img_pts, rvec, tvec in zip(object_points, image_points, rvecs, tvecs):
        projected, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
        # Distance between each detected corner and where the model says it should be.
        dist_px = np.linalg.norm(img_pts.reshape(-1, 2) - projected.reshape(-1, 2), axis=1)
        errors.append(float(dist_px.mean()))
    return errors


def calibrate_from_folder(
    folder: str | Path,
    pattern_size: Tuple[int, int] = (9, 6),
    square_size: float = 1.0,
    pattern: str = "left*.jpg",
) -> Dict:
    """Load all images matching `pattern` in `folder` and calibrate the camera.

    Images where the chessboard is not found are skipped (and listed).

    Returns:
        dict from `calibrate_camera` plus: image_size, used_files,
        skipped_files, object_points, image_points, pattern_size, square_size.
    """
    files = sorted(Path(folder).glob(pattern))
    obj_template = make_object_points(pattern_size, square_size)
    obj_pts, img_pts, used, skipped = [], [], [], []
    image_size = None

    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            skipped.append(f.name)
            continue
        image_size = (img.shape[1], img.shape[0])  # (width, height)
        found, corners = find_chessboard_corners(img, pattern_size)
        if found:
            obj_pts.append(obj_template)
            img_pts.append(corners)
            used.append(f.name)
        else:
            skipped.append(f.name)

    if len(used) < 3:
        raise ValueError(f"Need at least 3 usable images, found {len(used)} in {folder}")

    result = calibrate_camera(obj_pts, img_pts, image_size)
    result.update(
        image_size=image_size,
        used_files=used,
        skipped_files=skipped,
        object_points=obj_pts,
        image_points=img_pts,
        pattern_size=pattern_size,
        square_size=square_size,
    )
    return result


# ----------------------------------------------------------------------
# 3. Pinhole model written by hand
# ----------------------------------------------------------------------
def project_points_pinhole(
    X: np.ndarray, K: np.ndarray, R: np.ndarray, t: np.ndarray
) -> np.ndarray:
    """Project 3D world points to pixels with the pinhole model x = K [R|t] X.

    No lens distortion is modelled here (compare with cv2.projectPoints
    using dist = 0 to verify).

    Args:
        X: (N, 3) world points.
        K: (3, 3) intrinsic matrix.
        R: (3, 3) rotation matrix (world -> camera).
        t: (3,) or (3, 1) translation vector.

    Returns:
        (N, 2) pixel coordinates.
    """
    X = np.asarray(X, dtype=np.float64).reshape(-1, 3)
    t = np.asarray(t, dtype=np.float64).reshape(3, 1)
    X_cam = R @ X.T + t              # world -> camera coordinates, shape (3, N)
    x_hom = K @ X_cam                # camera -> homogeneous pixel coords (s*u, s*v, s)
    return (x_hom[:2] / x_hom[2]).T  # perspective divide by the depth s


def check_pinhole_against_opencv(
    object_points: np.ndarray, K: np.ndarray, rvec: np.ndarray, tvec: np.ndarray
) -> float:
    """Max pixel difference between our manual projection and cv2.projectPoints (no distortion)."""
    R, _ = cv2.Rodrigues(rvec)  # rotation vector -> 3x3 matrix
    manual = project_points_pinhole(object_points, K, R, tvec)
    reference, _ = cv2.projectPoints(object_points, rvec, tvec, K, np.zeros(5))
    return float(np.abs(manual - reference.reshape(-1, 2)).max())


# ----------------------------------------------------------------------
# 4. Undistortion
# ----------------------------------------------------------------------
def undistort_image(
    img: np.ndarray, K: np.ndarray, dist: np.ndarray, alpha: float = 0.0
) -> np.ndarray:
    """Remove lens distortion from an image.

    Args:
        img: BGR uint8 image.
        K: 3x3 intrinsic matrix.
        dist: distortion coefficients (k1, k2, p1, p2, k3).
        alpha: 0 = crop to valid pixels only (no black borders),
               1 = keep all source pixels (black borders appear).

    Returns:
        Undistorted image, same size as the input.
    """
    h, w = img.shape[:2]
    new_K, _roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha, (w, h))
    return cv2.undistort(img, K, dist, None, new_K)


def undistort_points(
    points: np.ndarray, K: np.ndarray, dist: np.ndarray, image_size: Tuple[int, int], alpha: float = 0.0
) -> np.ndarray:
    """Move pixel coordinates to where they land in `undistort_image(..., alpha)`.

    Args:
        points: (N, 2) pixel coordinates in the distorted image.
        image_size: (width, height).

    Returns:
        (N, 2) pixel coordinates in the undistorted image.
    """
    new_K, _roi = cv2.getOptimalNewCameraMatrix(K, dist, image_size, alpha, image_size)
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 1, 2)
    # P=new_K re-projects the normalised undistorted points back to pixels.
    return cv2.undistortPoints(pts, K, dist, P=new_K).reshape(-1, 2)


def line_deviation(points: np.ndarray) -> float:
    """Max perpendicular distance (pixels) of points from the line through the first and last point.

    A row of chessboard corners is a straight line in the real world, so after a
    correct undistortion this number should become smaller.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    p0, p1 = pts[0], pts[-1]
    direction = (p1 - p0) / np.linalg.norm(p1 - p0)
    normal = np.array([-direction[1], direction[0]])  # unit vector perpendicular to the line
    return float(np.abs((pts - p0) @ normal).max())


# ----------------------------------------------------------------------
# 5. Save / load (human readable JSON)
# ----------------------------------------------------------------------
def save_calibration(
    path: str | Path,
    K: np.ndarray,
    dist: np.ndarray,
    image_size: Tuple[int, int],
    rms: float,
    extra: Optional[Dict] = None,
) -> None:
    """Save calibration results to a readable JSON file."""
    K = np.asarray(K, dtype=float)
    data = {
        "image_size_wh": [int(image_size[0]), int(image_size[1])],
        "camera_matrix": K.tolist(),
        "fx": K[0, 0], "fy": K[1, 1], "cx": K[0, 2], "cy": K[1, 2],
        "distortion_coefficients": dict(
            zip(["k1", "k2", "p1", "p2", "k3"], np.asarray(dist, float).ravel().tolist())
        ),
        "rms_reprojection_error_px": float(rms),
    }
    if extra:
        data.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def load_calibration(path: str | Path) -> Dict:
    """Load a JSON written by `save_calibration`.

    Returns:
        dict with K (3x3 array), dist (1D array), image_size (w, h), rms, raw (full dict).
    """
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {
        "K": np.array(data["camera_matrix"], dtype=np.float64),
        "dist": np.array(list(data["distortion_coefficients"].values()), dtype=np.float64),
        "image_size": tuple(data["image_size_wh"]),
        "rms": data["rms_reprojection_error_px"],
        "raw": data,
    }

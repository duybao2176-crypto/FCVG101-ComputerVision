"""Create the small synthetic photo used by the `measure` demo.

Run once:  python Task3/make_samples.py
Output:    Task3/samples/measure_demo.png  (640 x 480, same size as the Task 1 calibration)

The picture shows a dark rectangle of exactly 300 x 200 pixels (rotated by 12 degrees) on a
light background, as the Task 1 camera would have photographed it: the outline is passed
through the calibrated lens distortion (cv2.projectPoints), so the app's undistortion step
recovers the true 300 x 200 px.  With size = pixels * Z / f (f = 536 px, Z = 500 mm) the app
should then report roughly 280 x 187 mm (within about 1 %), which is an easy number to check in the demo.
"""
from pathlib import Path
import sys

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.task1.calibration import load_calibration  # noqa: E402

OUT = Path(__file__).resolve().parent / "samples" / "measure_demo.png"


def rectangle_outline(rect, points_per_edge: int = 60) -> np.ndarray:
    """Many points along the rectangle edges (so that straight edges can bend after distortion)."""
    corners = cv2.boxPoints(rect)
    pts = []
    for a, b in zip(corners, np.roll(corners, -1, axis=0)):
        t = np.linspace(0, 1, points_per_edge, endpoint=False)[:, None]
        pts.append(a + t * (b - a))
    return np.vstack(pts)


def distort_pixels(pts: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Ideal pixel positions -> where the real (distorting) lens would put them."""
    normalised = (pts - K[:2, 2]) / np.array([K[0, 0], K[1, 1]])           # pixel -> camera coords at Z = 1
    obj = np.hstack([normalised, np.ones((len(pts), 1))]).astype(np.float64)
    out, _ = cv2.projectPoints(obj, np.zeros(3), np.zeros(3), K, dist)      # apply the lens model
    return out.reshape(-1, 2)


def main() -> None:
    calib = load_calibration(ROOT / "Task1" / "outputs" / "calibration.json")
    img = np.full((480, 640, 3), 235, np.uint8)                 # light grey background
    ideal = rectangle_outline(((320, 240), (300, 200), 12))     # centre, (width, height), angle
    polygon = np.round(distort_pixels(ideal, calib["K"], calib["dist"])).astype(np.int32)
    cv2.fillPoly(img, [polygon], (60, 90, 140))                 # dark blue-ish object
    OUT.parent.mkdir(exist_ok=True)
    cv2.imwrite(str(OUT), img)
    print("saved", OUT)


if __name__ == "__main__":
    main()

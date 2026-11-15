"""Headless smoke test of every Task 3 tool with the sample data.

Run from the project root:   python Task3/test_tools.py [output_folder]
If an output folder is given, the result images are saved there as PNG for a visual check.
The classify tool is reported as "model missing" until Task 2 has produced model_best.pt.
"""
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.task3.tools import ToolError, run_tool, to_rgb   # noqa: E402

OUT_DIR = Path(sys.argv[1]) if len(sys.argv) > 1 else None


def load(path: Path):
    return to_rgb(cv2.imread(str(path)))


def check(name: str, tool: str, images, params) -> None:
    try:
        res = run_tool(tool, images, params)
    except ToolError as e:
        print(f"[friendly error] {name}: {e}")
        return
    print(f"[OK] {name}: {len(res.images)} images, text {len(res.text)} chars")
    if OUT_DIR:
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        for i, (label, img) in enumerate(res.images):
            cv2.imwrite(str(OUT_DIR / f"{name}_{i}.png"), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def main() -> None:
    t1 = ROOT / "Task1"
    chess = load(t1 / "Chestboard_data" / "left01.jpg")
    lena = load(t1 / "noise_data" / "lena.jpg")
    box = load(t1 / "featureMatch_data" / "box.png")
    scene = load(t1 / "featureMatch_data" / "box_in_scene.png")
    measure_img = load(ROOT / "Task3" / "samples" / "measure_demo.png")

    check("calibrate_saved", "calibrate", [chess], {})
    files = sorted(str(p) for p in (t1 / "Chestboard_data").glob("left*.jpg"))
    check("calibrate_new", "calibrate", [], {"chessboard_files": files})
    check("calibrate_no_files", "calibrate", [chess], {"recalibrate": True})
    check("filter_sp_median", "filter", [lena], {"noise": "saltpepper", "filter": "median", "ksize": 5})
    check("filter_all", "filter", [lena], {"noise": "gaussian", "filter": "all"})
    check("filter_no_noise", "filter", [lena], {})
    check("edges_canny", "edges", [lena], {})
    check("edges_all", "edges", [lena], {"edge_method": "all"})
    check("edges_fft", "edges", [lena], {"frequency": "highpass", "cutoff": 20})
    check("features_match", "features", [box, scene], {"method": "sift", "ratio": 0.75})
    check("features_orb", "features", [box, scene], {"method": "orb"})
    check("features_keypoints", "features", [box], {"method": "sift"})
    check("features_harris", "features", [box], {"method": "harris"})
    check("classify", "classify", [lena], {})
    check("measure_distance", "measure", [measure_img], {"distance_mm": 500})
    check("measure_scale", "measure", [measure_img], {"px_per_mm": 2.0})
    check("filter_needs_image", "filter", [], {})


if __name__ == "__main__":
    main()

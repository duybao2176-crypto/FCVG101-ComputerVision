"""Task 3 - Computer-vision assistant (Gradio app).

Run from anywhere:   python Task3/app.py            (then open http://127.0.0.1:7860)
Options:             python Task3/app.py --port 7861 --open

Flow:  user text + images  ->  router (intent)  ->  tool (Task 1 / Task 2 code)  ->  gallery + numbers
The "Tool-use trace" box shows which tool was chosen, why, and which parameters were used.
"""
import argparse
import sys
import traceback
from pathlib import Path

import gradio as gr

# Make `import src...` work no matter where the script is started from.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.task3.router import extract_params, route          # noqa: E402
from src.task3.tools import (TOOL_BACKENDS, TOOL_PARAMS, ToolError, run_tool,   # noqa: E402
                             shrink, task2_model_status)

SAMPLE_DIR = PROJECT_ROOT / "Task3" / "samples"
TASK1 = PROJECT_ROOT / "Task1"

MODES = [
    ("Auto (understand my request)", "auto"),
    ("Calibrate / undistort", "calibrate"),
    ("Noise and filtering", "filter"),
    ("Edges / frequency filtering", "edges"),
    ("Features / matching", "features"),
    ("Classify fruit (Task 2)", "classify"),
    ("Measure an object", "measure"),
]


# ----------------------------------------------------------------------
# 1. Turn the UI values into one params dict
# ----------------------------------------------------------------------
def ui_params(filter_name, noise, ksize, edge_method, feature_method, ratio, distance_mm, px_per_mm, files) -> dict:
    """Values of the 'Advanced settings' panel.  'auto' / empty means: not set."""
    p = {"ksize": int(ksize), "ratio": float(ratio), "noise": noise}
    if filter_name != "auto":
        p["filter"] = filter_name
    if edge_method != "auto":
        p["edge_method"] = edge_method
    if feature_method != "auto":
        p["method"] = feature_method
    if distance_mm:
        p["distance_mm"] = float(distance_mm)
    if px_per_mm:
        p["px_per_mm"] = float(px_per_mm)
    if files:
        p["chessboard_files"] = [str(f) for f in files]
    return p


def format_trace(mode: str, tool: str, confidence, matched, params) -> str:
    """The visible 'tool-use trace' (markdown)."""
    shown = {k: v for k, v in params.items() if k != "chessboard_files"}
    if "chessboard_files" in params:
        shown["chessboard_files"] = f"{len(params['chessboard_files'])} files"
    how = (f"Auto-detected from your text (confidence **{confidence:.2f}**), "
           f"keywords: {', '.join(f'`{m}`' for m in matched) or '-'}"
           if mode == "auto" else "Chosen manually in the mode dropdown")
    return (f"**Tool chosen:** `{tool}`  \n**Why:** {how}  \n"
            f"**Parameters:** `{shown}`  \n**Runs:** {TOOL_BACKENDS[tool]}")


# ----------------------------------------------------------------------
# 2. The function behind the Run button
# ----------------------------------------------------------------------
def handle_request(text, mode, img1, img2, files, *advanced):
    """Returns (gallery, result markdown, trace markdown)."""
    tool = "-"
    try:
        text = text or ""
        images = [shrink(im, 1280) for im in (img1, img2) if im is not None]   # numpy RGB; phone photos are huge
        extra = list(files or [])
        # A set of chessboard files counts as "something to work on" for the router.
        n_inputs = len(images) + (1 if extra else 0)

        # --- step 1: which tool? -------------------------------------
        if mode == "auto":
            decision = route(text, n_inputs)
            if decision.tool is None:
                return [], f"### I need a bit more information\n\n{decision.clarification}", \
                    (f"**No tool chosen** (best confidence {decision.confidence:.2f}, "
                     f"scores {decision.scores}). I ask instead of guessing.")
            tool, confidence, matched = decision.tool, decision.confidence, decision.matched
            text_params = decision.params
        else:
            tool, confidence, matched = mode, 1.0, []
            text_params = extract_params(text, tool)

        # --- step 2: parameters (text beats the advanced panel) ------
        params = {**ui_params(*advanced, extra), **text_params}
        params = {k: v for k, v in params.items() if k in TOOL_PARAMS[tool]}   # keep what this tool uses

        # --- step 3: run the tool ------------------------------------
        result = run_tool(tool, images, params)
        gallery = [(img, label) for label, img in result.images]
        trace = format_trace(mode, tool, confidence, matched, params)
        return gallery, f"### {result.title}\n\n{result.text}", trace

    except ToolError as err:                       # expected problems: show them nicely
        return [], f"### Cannot do that yet\n\n{err}", f"**Tool:** `{tool}` could not run (see message)."
    except Exception as err:                       # unexpected: log for the developer, short message for the user
        traceback.print_exc()
        return [], (f"### Something went wrong\n\nSorry, the tool failed ({type(err).__name__}: {err}). "
                    "Try another image or different settings."), "**Tool run failed.**"


# ----------------------------------------------------------------------
# 3. Examples (only files that exist are offered)
# ----------------------------------------------------------------------
def find_fruit_example():
    """First fruit photo found in the Task 2 data folders (None if the data is not there)."""
    for folder in ("processed", "raw"):
        root = PROJECT_ROOT / "Task2" / "data" / folder
        if root.exists():
            for path in sorted(root.rglob("*")):
                if path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    return str(path)
    return None


def build_examples():
    """Rows: [text, mode, image1, image2, chessboard files]."""
    rows = []
    chess = TASK1 / "Chestboard_data" / "left01.jpg"
    lena = TASK1 / "noise_data" / "lena.jpg"
    box = TASK1 / "featureMatch_data" / "box.png"
    scene = TASK1 / "featureMatch_data" / "box_in_scene.png"
    measure_img = SAMPLE_DIR / "measure_demo.png"
    if chess.exists():
        rows.append(["Calibrate the camera and undistort this image", "auto", str(chess), None, None])
        board_files = [str(p) for p in sorted((TASK1 / "Chestboard_data").glob("left*.jpg"))]
        rows.append(["Recalibrate the camera from my chessboard images", "auto", None, None, board_files])
    if lena.exists():
        rows.append(["Add salt and pepper noise and denoise it with a median filter size 5", "auto", str(lena), None, None])
        rows.append(["Detect edges with canny", "auto", str(lena), None, None])
        rows.append(["Show the high-pass filter in the frequency domain", "auto", str(lena), None, None])
    if box.exists() and scene.exists():
        rows.append(["Match the box with the scene using SIFT, ratio 0.75", "auto", str(box), str(scene), None])
    if measure_img.exists():
        rows.append(["How big is this object? It is 50 cm away", "auto", str(measure_img), None, None])
    fruit = find_fruit_example()
    if fruit:
        rows.append(["Is this fruit fresh or rotten?", "auto", fruit, None, None])
    return rows


# ----------------------------------------------------------------------
# 4. The interface
# ----------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="FCVG101 Vision Assistant") as demo:
        gr.Markdown("# Computer Vision Assistant\n"
                    "Tell me what to do with your image(s): *calibrate, filter noise, find edges, "
                    "match features, classify fruit freshness, measure an object*. "
                    "English and a few Vietnamese keywords work.")
        status = gr.Markdown(task2_model_status())

        with gr.Row():
            with gr.Column(scale=1):
                text = gr.Textbox(label="What do you want to do?", lines=2,
                                  placeholder="e.g. Add salt and pepper noise and denoise it with a median filter size 5")
                mode = gr.Dropdown(MODES, value="auto", label="Mode")
                img1 = gr.Image(label="Image 1 (main image / object)", type="numpy", height=260)
                img2 = gr.Image(label="Image 2 (optional: scene for feature matching)", type="numpy", height=200)
                files = gr.Files(label="Extra images (optional: 3+ chessboard photos to recalibrate)",
                                 file_count="multiple", type="filepath")
                with gr.Accordion("Advanced settings (the text request overrides these)", open=False):
                    filter_name = gr.Dropdown(["auto", "box", "gaussian", "median", "bilateral", "all"],
                                              value="auto", label="Filter")
                    noise = gr.Dropdown(["none", "gaussian", "saltpepper"], value="none", label="Add noise first")
                    ksize = gr.Slider(3, 21, value=5, step=2, label="Filter kernel size")
                    edge_method = gr.Dropdown(["auto", "canny", "sobel", "laplacian", "all"], value="auto",
                                              label="Edge method")
                    feature_method = gr.Dropdown(["auto", "sift", "orb", "harris"], value="auto",
                                                 label="Feature method")
                    ratio = gr.Slider(0.5, 0.95, value=0.75, step=0.05, label="Ratio test threshold")
                    distance_mm = gr.Number(label="Measure: camera-to-object distance (mm)", value=None)
                    px_per_mm = gr.Number(label="Measure: or known scale (pixels per mm)", value=None)
                run = gr.Button("Run", variant="primary")
            with gr.Column(scale=2):
                trace = gr.Markdown("*The tool-use trace appears here after you press Run.*", label="Tool-use trace")
                gallery = gr.Gallery(label="Result images", columns=3, object_fit="contain", height="auto")
                summary = gr.Markdown(label="Numbers and explanation")

        advanced = [filter_name, noise, ksize, edge_method, feature_method, ratio, distance_mm, px_per_mm]
        inputs = [text, mode, img1, img2, files] + advanced
        outputs = [gallery, summary, trace]
        run.click(handle_request, inputs, outputs)
        text.submit(handle_request, inputs, outputs)

        examples = build_examples()
        if examples:
            gr.Examples(examples, inputs=[text, mode, img1, img2, files], label="Examples (click, then press Run)")
        demo.load(task2_model_status, None, status)   # refresh the model status on every page load
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Computer vision assistant (Task 3)")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--open", action="store_true", help="open the browser automatically")
    args = parser.parse_args()
    build_ui().launch(server_name="127.0.0.1", server_port=args.port, inbrowser=args.open,
                      allowed_paths=[str(PROJECT_ROOT)])

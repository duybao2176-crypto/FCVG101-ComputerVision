# Task 3 - Computer Vision Assistant (application integration / tool use)

A small Gradio app. The user types what they want ("denoise this with a median filter", "is this fruit fresh?")
and uploads image(s). The app **understands the request**, **calls the matching tool** (Task 1 pipeline or
Task 2 model) and **shows the visualised result plus the numbers**, together with a visible *tool-use trace*
(which tool, why, which parameters).

## Launch

From the project root (`FCVG101_Coursework`):

```
pip install gradio          # once (opencv, numpy, torch, matplotlib are already needed by Task 1 / 2)
python Task3/app.py         # open http://127.0.0.1:7860
python Task3/app.py --port 7861 --open     # other port, open the browser automatically
```

Gradio is used instead of Streamlit (the coursework allows either). The app works from any folder
because it adds the project root to `sys.path` itself.

Other commands:

```
python src/task3/router.py       # router self-check (29 sentences + 10 parameter checks)
python Task3/test_tools.py [out_folder]   # runs every tool on the sample data (optionally saves PNGs)
python Task3/make_samples.py     # re-creates Task3/samples/measure_demo.png
```

## Architecture

```
 user text ---------------+                                   +--> Task 1.1  calibration.py   (calibrate, measure)
 image 1 / image 2 / files|                                   |--> Task 1.2  filtering.py     (filter, edges)
 mode dropdown            v                                   |--> Task 1.3  features.py      (features)
                    +-----------+  tool name + params   +-----------+   +--> Task 2    predict.py       (classify)
   Task3/app.py --> |  router   | --------------------> |   tools   |---+
   (Gradio UI)      | router.py |  (+ confidence, why)  | tools.py  |
        ^           +-----------+                       +-----------+
        |                                                     |
        |   ToolResult(title, images, text, data)             |
        +-----------------------------------------------------+
        gallery of images + markdown with the numbers + tool-use trace
```

Files:

| File | Role |
|---|---|
| `Task3/app.py` | Gradio interface: inputs, Run button, outputs, examples, friendly error handling |
| `src/task3/router.py` | **Intent recognition**: text -> `Route(tool, confidence, params, matched keywords, clarification)` |
| `src/task3/tools.py` | **Tools**: one thin wrapper per capability, all with the signature `tool(images, params) -> ToolResult` |
| `Task3/test_tools.py` | Headless smoke test of every tool |
| `Task3/make_samples.py`, `Task3/samples/` | Synthetic photo for the measure demo |

## Flow: text -> router -> tool -> result

1. **Input.** The UI collects the text, up to two images, optional extra files (chessboard photos) and the mode.
2. **Router** (mode = *Auto*): `route(text, n_images)` in `src/task3/router.py`
   * normalises the text (lower case; "median filter **size 5**" is removed from the scoring text because
     "size" there is a parameter and not a measurement request; "low-pass filter" is turned into "lowpass" so the word
     *filter* does not vote for the filter tool),
   * scores every tool by adding up the weights of the keyword patterns found (strong 3, medium 2, weak 1;
     English and a few Vietnamese words such as *hiệu chuẩn, lọc nhiễu, biên, so khớp, tươi, hỏng, đo*),
   * `confidence = margin * strength` where `margin = top / (top + second)` and `strength = min(1, top / 3)`:
     a clear winner with strong evidence gives about 1.0, a tie gives 0.5 or less, one weak keyword gives 0.33,
   * if the confidence is below **0.55** or there is no image, it returns a **clarification question** instead of guessing,
   * `extract_params` reads settings from the text: kernel size, filter type, noise type, `ratio 0.7`, `sift`/`orb`/`harris`,
     `canny` / `sobel`, `low-pass` / `high-pass`, `cutoff 20`, `50 cm away`, `5 px/mm` ...
   * with a manual mode, the router is skipped but the parameters are still read from the text.
3. **Tool**: `run_tool(name, images, params)` calls the Task 1 / Task 2 code. Text parameters override the
   "Advanced settings" panel, which overrides the defaults.
4. **Result**: the tool returns a `ToolResult`; the app shows the images in a gallery, the markdown (numbers and a short
   explanation) and the trace line (`Tool chosen`, `Why`, `Parameters`, `Runs`).
5. **Errors**: expected problems (no model yet, no image, no chessboard found) raise `ToolError` with a friendly
   message; anything else is logged in the console and shown as a short message, never as a traceback.

## Modes and the modules they use

| Mode (tool) | What it does | Task module |
|---|---|---|
| `calibrate` | Undistorts the image with the saved `Task1/outputs/calibration.json` (shows K, distortion, RMS). With 3+ chessboard photos in *Extra images* it recalibrates (9 x 6 inner corners) and shows the detected corners | `src/task1/calibration.py` (`load_calibration`, `find_chessboard_corners`, `calibrate_camera`, `undistort_image`) |
| `filter` | Optionally adds Gaussian / salt-and-pepper noise, then box / Gaussian / median / bilateral (or all four); PSNR and SSIM against the clean image when noise was added | `src/task1/filtering.py` (`add_*_noise`, `apply_filter`, `psnr`, `ssim`) |
| `edges` | Sobel / Laplacian / Canny (or all), or FFT low-pass / high-pass with the spectrum | `src/task1/filtering.py` (`detect_edges`, `frequency_filter`, `fft_spectrum`) |
| `features` | Two images: SIFT/ORB + ratio test + RANSAC, match picture, box outline, counts. One image: keypoints, or Harris corners (from scratch) | `src/task1/features.py` (`match_images`, `detect_keypoints`, `harris_from_scratch`) |
| `classify` | Fruit class, freshness, confidence, top-3 bar chart, Grad-CAM overlay | `src/task2/predict.py` (`load_model`, `predict_image`, `gradcam`) |
| `measure` | Segments the largest object (Otsu + contours), reports its rotated-rectangle size in mm | calibration (K, distortion) + OpenCV + pinhole relation |

### The measure tool, honestly

* The image is undistorted with the Task 1 calibration first.
* Scale from **either** a user scale (`5 px/mm`, e.g. from a ruler in the picture) **or** the pinhole relation
  `size_mm = size_px * Z / f` with `f` (pixels) from the calibration and the camera-to-object distance `Z` entered by the user
  ("it is 50 cm away"). If no distance is given, `Z = 500 mm` is assumed and the output says so.
* Assumptions: one object on a plain background, flat and facing the camera, all at distance Z.
  The relative error of the result equals the relative error of Z. Tilted or 3-D objects give wrong numbers.
* If the image size differs from the calibration size (640 x 480), K is rescaled; this is only valid for a resized
  photo of the same camera.

## How to add a new tool

1. In `src/task3/tools.py` write `def my_tool(images, params) -> ToolResult` (convert RGB -> BGR with `to_bgr` if you call
   OpenCV / Task 1 code, return RGB images).
2. Register it: add `"my_tool": my_tool` to `TOOLS`, a description to `TOOL_BACKENDS` and its parameter names to `TOOL_PARAMS`.
3. In `src/task3/router.py` add `"my_tool"` to `TOOL_NAMES` and a list of `(regex, weight)` keywords to `KEYWORDS`;
   optionally read parameters in `extract_params`. Add 2-3 sentences to `SELF_TESTS` and run `python src/task3/router.py`.
4. In `Task3/app.py` add `("My tool", "my_tool")` to `MODES` (and an example row if you like).

## Example prompts to demo

| Prompt (+ images) | Expected |
|---|---|
| "Calibrate the camera and undistort this image" + `Task1/Chestboard_data/left01.jpg` | tool `calibrate`, before/after, K and distortion, RMS 0.41 px |
| "Add salt and pepper noise and denoise it with a median filter size 5" + `Task1/noise_data/lena.jpg` | tool `filter`, noisy 18.21 dB -> median 30.89 dB, SSIM 0.856 |
| "Match the box with the scene using SIFT, ratio 0.75" + `box.png`, `box_in_scene.png` | tool `features`, 80 matches, 75 inliers, green outline around the box |
| "How big is this object? It is 50 cm away" + `Task3/samples/measure_demo.png` | tool `measure`, about 281 x 188 mm (true value 280 x 187) |
| "Is this fruit fresh or rotten?" + a fruit photo | tool `classify` (needs the Task 2 model), class, confidence, Grad-CAM |
| "hello" | no tool: the app asks what you want |

## Notes and limits

* **Classification needs the trained Task 2 model** (tested with the real model once it was produced: Fresh/Rotten apple photos are classified correctly) (`Task2/outputs/model_best.pt` + `model_meta.json`). Until they exist the
  classify tool shows a clear message and every other mode works; the status line at the top of the page and the tool
  start working as soon as the files appear (no restart needed for the tool).
* The router is rule-based on purpose: it is fast, needs no internet and every decision can be explained
  (the matched keywords are shown in the trace). Its weakness is vocabulary: a request without any known keyword
  gets a clarification question.
* **Replacing the router with an LLM (optional idea, not implemented, no API dependency here):** keep `tools.py` unchanged and
  replace `route()` by a function that sends the text plus the list `TOOL_NAMES` / `TOOL_PARAMS` to a language model and
  asks for a JSON answer `{"tool": ..., "params": {...}}` (function calling / tool use). The rest of the app only needs the
  same `Route` object back, and the rule-based router can stay as an offline fallback.
* Calibration recalibration does **not** overwrite `Task1/outputs/calibration.json`; the new values are only used for the current request.

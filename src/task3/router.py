"""Intent router: free-text request -> (tool name, confidence, parameters).

This is a transparent, rule-based router (no machine learning, no API):

    1. Normalise the text (lower case, remove "size 5"-style parameter phrases).
    2. For every tool, add up the weights of the keyword patterns that occur
       in the text (a pattern counts once).  Weights: strong 3, medium 2, weak 1.
    3. confidence = margin * strength
           margin   = top / (top + second)   -> how clearly the winner beats the runner-up
           strength = min(1, top / 3)        -> a single weak keyword is never enough
    4. If confidence < CONFIDENCE_THRESHOLD, or there is no image to work on,
       the router does NOT guess: it returns a clarification question instead.
    5. Parameters (kernel size, ratio, SIFT/ORB, ...) are read from the text with
       small regular expressions in `extract_params`.

Run `python src/task3/router.py` to execute the self-check at the bottom.
"""

from __future__ import annotations

import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Allow `python src/task3/router.py` (script mode) as well as `import src.task3.router`.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

TOOL_NAMES = ["calibrate", "filter", "edges", "features", "classify", "measure"]
CONFIDENCE_THRESHOLD = 0.55   # below this we ask the user instead of guessing

STRONG, MEDIUM, WEAK = 3, 2, 1

# ----------------------------------------------------------------------
# Keyword table: tool -> list of (regex, weight).
# English and a few Vietnamese keywords (with and without diacritics).
# Patterns are prefix matches on purpose ("calibrat" matches calibrate/calibration).
# ----------------------------------------------------------------------
KEYWORDS: Dict[str, List[Tuple[str, int]]] = {
    "calibrate": [
        (r"calibrat", STRONG), (r"undistort", STRONG), (r"intrinsic", STRONG),
        (r"hi[ệe]u chu[ẩa]n", STRONG), (r"camera matrix", STRONG),
        (r"distortion", MEDIUM), (r"chessboard|checkerboard", MEDIUM), (r"pinhole", MEDIUM),
        (r"\blens\b", WEAK), (r"\bm[ée]o\b|bi[ếe]n d[ạa]ng", MEDIUM), (r"focal", WEAK),
    ],
    "filter": [
        (r"denois|de-nois", STRONG), (r"\bnoise|\bnoisy", STRONG), (r"\bmedian", STRONG),
        (r"bilateral", STRONG), (r"\bblur", STRONG), (r"smooth", MEDIUM),
        (r"l[ọo]c nhi[ễe]u|nhi[ễe]u|kh[ửu] nhi[ễe]u|l[àa]m m[ờo]", STRONG),
        (r"\bfilter", MEDIUM), (r"\bl[ọo]c\b", MEDIUM), (r"salt|pepper|mu[ốo]i ti[êe]u", MEDIUM),
        (r"gaussian", WEAK), (r"\bbox filter|\bmean filter", MEDIUM),
    ],
    "edges": [
        (r"\bedges?\b", STRONG), (r"canny", STRONG), (r"sobel", STRONG), (r"laplacian", STRONG),
        (r"\bfft\b|fourier|frequenc", STRONG), (r"lowpass|highpass|low-pass|high-pass", STRONG),
        (r"t[ầa]n s[ốo]", STRONG), (r"\bbi[êe]n\b|\bc[ạa]nh\b", MEDIUM),
        (r"gradient", MEDIUM), (r"outline|contour", WEAK), (r"spectrum", MEDIUM),
    ],
    "features": [
        (r"\bmatch", STRONG), (r"keypoint|key point", STRONG), (r"\bsift\b", STRONG),
        (r"\borb\b", STRONG), (r"harris", STRONG), (r"descriptor", STRONG),
        (r"so kh[ớo]p|gh[ée]p [ảa]nh|[đd][ặa]c tr[ưu]ng", STRONG),
        (r"\bfeatures?\b", STRONG), (r"corners?\b", MEDIUM), (r"homography|ransac", MEDIUM),
        (r"find .* in (the |this )?(scene|image)", MEDIUM), (r"\blocali[sz]e", WEAK),
    ],
    "classify": [
        (r"classif", STRONG), (r"\bfresh", STRONG), (r"rotten|\brot\b", STRONG),
        (r"\bripe|spoil|stale|decay", STRONG), (r"t[ươu][ơo]i|h[ỏo]ng|th[ốo]i|ph[âa]n lo[ạa]i|\bch[íi]n\b", STRONG),
        (r"fruit|apple|banana|orange", MEDIUM), (r"good to eat|edible|safe to eat", MEDIUM),
        (r"what is (this|it)\b", WEAK), (r"gradcam|grad-cam", MEDIUM),
    ],
    "measure": [
        (r"measure|\bmeasuring", STRONG), (r"\bđo\b|\bdo (?:kich|dac|chieu)", STRONG), (r"k[íi]ch th[ưu][ớo]c", STRONG),
        (r"how (big|large|long|wide|tall|far)", STRONG), (r"dimension", STRONG),
        (r"\bwidth|\bheight|\blength|chi[ềe]u (d[àa]i|r[ộo]ng|cao)", MEDIUM),
        (r"\bsize\b", MEDIUM), (r"millimet|centimet|\bmm\b|\bcm\b", WEAK),
        (r"real[- ]world", MEDIUM), (r"\bdistance", WEAK), (r"\bruler\b", MEDIUM),
    ],
}


@dataclass
class Route:
    """Answer of the router (the 'tool-use trace' shown in the UI)."""
    tool: Optional[str]                 # None = could not decide
    confidence: float                   # 0..1
    params: Dict = field(default_factory=dict)
    matched: List[str] = field(default_factory=list)   # keywords that triggered the tool
    scores: Dict[str, int] = field(default_factory=dict)
    clarification: Optional[str] = None  # question for the user when tool is None

    def as_tuple(self) -> Tuple[Optional[str], float, Dict]:
        return self.tool, self.confidence, self.params


# ----------------------------------------------------------------------
# Step 1: normalise
# ----------------------------------------------------------------------
def normalise(text: str) -> str:
    """Lower-case and remove phrases that would otherwise confuse the scoring."""
    t = text.lower()
    # "median filter size 5" -> the word "size" here is a filter parameter, not a measurement.
    t = re.sub(r"\b(?:size|kernel|ksize)\s*[=:]?\s*\d+", " ", t)
    # "low-pass filter" is a frequency tool, so make sure the word "filter" does not vote for 'filter'.
    t = re.sub(r"\b(low|high|band)[- ]?pass\s+filter(?:ing)?", r"\1pass", t)
    t = re.sub(r"frequency[- ]domain\s+filter(?:ing)?|frequency\s+filter(?:ing)?", "frequency", t)
    t = re.sub(r"\b(sobel|canny|laplacian)\s+filter", r"\1", t)
    return t


# ----------------------------------------------------------------------
# Step 2-3: score
# ----------------------------------------------------------------------
def score_tools(text: str, n_images: int = 1) -> Tuple[Dict[str, int], Dict[str, List[str]]]:
    """Return (score per tool, matched keywords per tool)."""
    t = normalise(text)
    scores = {name: 0 for name in TOOL_NAMES}
    matched: Dict[str, List[str]] = {name: [] for name in TOOL_NAMES}
    for tool, patterns in KEYWORDS.items():
        for pattern, weight in patterns:
            m = re.search(pattern, t)
            if m:
                scores[tool] += weight          # each pattern counts once
                matched[tool].append(m.group(0).strip())
    # Context hint: two images and nothing else to go on usually means "match them".
    if n_images >= 2:
        scores["features"] += WEAK
        matched["features"].append("(2 images uploaded)")
    return scores, matched


def confidence_from_scores(scores: Dict[str, int]) -> Tuple[Optional[str], float]:
    """Winner and its confidence = margin * strength (see module docstring)."""
    ranked = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)
    (best, top), (_, second) = ranked[0], ranked[1]
    if top == 0:
        return None, 0.0
    margin = top / (top + second)
    strength = min(1.0, top / STRONG)
    return best, round(margin * strength, 2)


# ----------------------------------------------------------------------
# Step 5: parameters from the text
# ----------------------------------------------------------------------
def _first_number(pattern: str, text: str, cast=float):
    m = re.search(pattern, text)
    return cast(m.group(1)) if m else None


def _to_mm(value: float, unit: str) -> float:
    return value * {"mm": 1.0, "cm": 10.0, "m": 1000.0}[unit]


def extract_params(text: str, tool: Optional[str] = None) -> Dict:
    """Read optional parameters from the sentence.  Only keys that were found are returned."""
    t = text.lower()
    p: Dict = {}

    # --- generic: kernel size ("size 5", "kernel 7", "7x7") -------------
    k = _first_number(r"(?:size|kernel|ksize)\s*(?:of|=|:)?\s*(\d+)", t, int) \
        or _first_number(r"\b(\d+)\s*[x×]\s*\d+\b", t, int)
    if k:
        p["ksize"] = k if k % 2 == 1 else k + 1       # filter kernels must be odd

    # --- filter ---------------------------------------------------------
    if re.search(r"median|trung v[ịi]", t):
        p["filter"] = "median"
    elif re.search(r"bilateral", t):
        p["filter"] = "bilateral"
    elif re.search(r"gaussian blur|gaussian filter|l[ọo]c gaussian", t) or \
            (re.search(r"gaussian", t) and not re.search(r"gaussian noise", t)):
        p["filter"] = "gaussian"
    elif re.search(r"\bbox (?:filter|blur)|\bmean (?:filter|blur)|average (?:filter|blur)|trung b[ìi]nh", t):
        p["filter"] = "box"
    if re.search(r"compare|all filters|every filter|so s[áa]nh", t):
        p["filter"] = "all"
    if re.search(r"salt|pepper|mu[ốo]i ti[êe]u|impulse", t):
        p["noise"] = "saltpepper"
    elif re.search(r"gaussian noise|add noise|noisy|th[êe]m nhi[ễe]u", t):
        p["noise"] = "gaussian"
    sigma = _first_number(r"sigma\s*[=:]?\s*(\d+(?:\.\d+)?)", t)
    if sigma:
        p["noise_level"] = sigma

    # --- edges / frequency --------------------------------------------
    if re.search(r"sobel|gradient", t):
        p["edge_method"] = "sobel"
    elif re.search(r"laplacian", t):
        p["edge_method"] = "laplacian"
    elif re.search(r"canny", t):
        p["edge_method"] = "canny"
    if re.search(r"high[- ]?pass", t):
        p["frequency"] = "highpass"
    elif re.search(r"low[- ]?pass", t):
        p["frequency"] = "lowpass"
    elif re.search(r"\bfft\b|fourier|frequenc|spectrum|t[ầa]n s[ốo]", t):
        p["frequency"] = "lowpass"
    cutoff = _first_number(r"cut-?off\s*(?:of|=|:)?\s*(\d+)", t, int)
    if cutoff:
        p["cutoff"] = cutoff
    low = _first_number(r"\blow(?:er)?\s*(?:threshold)?\s*[=:]?\s*(\d+)", t, int)
    high = _first_number(r"\bhigh(?:er)?\s*(?:threshold)?\s*[=:]?\s*(\d+)", t, int)
    if low is not None:
        p["canny_low"] = low
    if high is not None:
        p["canny_high"] = high

    # --- features ------------------------------------------------------
    if re.search(r"\borb\b", t):
        p["method"] = "orb"
    elif re.search(r"harris", t):
        p["method"] = "harris"
    elif re.search(r"\bsift\b", t):
        p["method"] = "sift"
    ratio = _first_number(r"ratio\s*(?:test)?\s*(?:of|=|:|is)?\s*(0?\.\d+|1(?:\.0+)?)", t)
    if ratio:
        p["ratio"] = ratio

    # --- calibrate -------------------------------------------------------
    if re.search(r"re-?calibrat|new calibration|calibrate (from|using|with) (my|the|these|new)|from (my |the |these )?(chessboard|checkerboard)", t):
        p["recalibrate"] = True
    sq = re.search(r"square(?: size)?\s*(?:of|=|:)?\s*(\d+(?:\.\d+)?)\s*(mm|cm)", t)
    if sq:
        p["square_size_mm"] = _to_mm(float(sq.group(1)), sq.group(2))

    # --- measure -----------------------------------------------------------
    dist = re.search(r"(\d+(?:\.\d+)?)\s*(mm|cm|m)\s*(?:away|distance|from (?:the )?camera)", t) or \
        re.search(r"(?:distance|at|z)\s*(?:of|=|:|is)?\s*(\d+(?:\.\d+)?)\s*(mm|cm|m)\b", t)
    if dist:
        p["distance_mm"] = _to_mm(float(dist.group(1)), dist.group(2))
    scale = re.search(r"(\d+(?:\.\d+)?)\s*(?:px|pixels?)\s*(?:per|/)\s*mm", t)
    if scale:
        p["px_per_mm"] = float(scale.group(1))
    return p


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------
def route(text: str, n_images: int = 1) -> Route:
    """Decide which tool the user wants.

    Args:
        text: the user's free-text request.
        n_images: how many images were uploaded (the router will not pick a
            tool when there is nothing to work on).
    """
    scores, matched = score_tools(text, n_images)
    best, confidence = confidence_from_scores(scores)

    if n_images < 1:
        return Route(None, 0.0, {}, [], scores,
                     "Please upload an image first (or choose one of the examples), then tell me what to do.")
    if best is None or confidence < CONFIDENCE_THRESHOLD:
        return Route(None, confidence, {}, [], scores,
                     "I am not sure what you want. Do you want to: calibrate the camera / undistort, "
                     "filter noise, detect edges, match features, classify fruit freshness, or measure an object? "
                     "Pick a mode from the dropdown or rephrase your request.")
    return Route(best, confidence, extract_params(text, best), matched[best], scores, None)


# ----------------------------------------------------------------------
# Self-check:  python src/task3/router.py
# ----------------------------------------------------------------------
SELF_TESTS = [
    # (sentence, n_images, expected tool or None for "ask the user")
    ("Calibrate the camera and undistort this image", 1, "calibrate"),
    ("remove the lens distortion", 1, "calibrate"),
    ("Hiệu chuẩn camera giúp mình", 1, "calibrate"),
    ("Add salt and pepper noise and denoise it with a median filter size 5", 1, "filter"),
    ("blur the image with a gaussian filter", 1, "filter"),
    ("Lọc nhiễu bằng median", 1, "filter"),
    ("compare box, gaussian, median and bilateral denoising", 1, "filter"),
    ("use a box filter of size 7", 1, "filter"),
    ("Detect edges with canny", 1, "edges"),
    ("show the sobel gradient", 1, "edges"),
    ("apply a high-pass filter in the frequency domain", 1, "edges"),
    ("Phát hiện biên ảnh", 1, "edges"),
    ("Match the box with the scene using SIFT, ratio 0.7", 2, "features"),
    ("find keypoints with orb", 1, "features"),
    ("Harris corners please", 1, "features"),
    ("so khớp đặc trưng hai ảnh", 2, "features"),
    ("match these two images", 2, "features"),
    ("Is this apple fresh or rotten?", 1, "classify"),
    ("what fruit is this", 1, "classify"),
    ("Quả này tươi hay hỏng?", 1, "classify"),
    ("classify the fruit and show grad-cam", 1, "classify"),
    ("How wide is the object? It is 50 cm away", 1, "measure"),
    ("Đo kích thước vật thể", 1, "measure"),
    ("measure the length of this object", 1, "measure"),
    ("", 2, None),                            # two images but no text: a weak hint is not enough, ask
    ("hello", 1, None),                       # no keyword -> ask
    ("do something nice", 1, None),           # vague -> ask
    ("classify this", 0, None),               # no image -> ask for an upload
    ("edges and noise", 1, None),             # two tools tie -> ask
]


def self_check() -> bool:
    ok_all = True
    for text, n_images, expected in SELF_TESTS:
        r = route(text, n_images)
        ok = r.tool == expected
        ok_all &= ok
        print(f"[{'OK' if ok else 'FAIL'}] {text!r:75} -> {r.tool} (conf {r.confidence:.2f}) "
              f"expected {expected} {r.params if r.params else ''}")
    # A few parameter-extraction checks.
    checks = [
        ("median filter size 5", "ksize", 5),
        ("match the box with the scene", "filter", None),   # "box" here is the object, not a box filter
        ("box filter size 7", "filter", "box"),
        ("median filter size 4", "ksize", 5),       # even sizes are bumped to odd
        ("match with SIFT ratio 0.7", "ratio", 0.7),
        ("use orb", "method", "orb"),
        ("canny low 50 high 150", "canny_high", 150),
        ("object is 50 cm away", "distance_mm", 500.0),
        ("add salt and pepper noise", "noise", "saltpepper"),
        ("high-pass cutoff 20", "frequency", "highpass"),
    ]
    for text, key, expected in checks:
        got = extract_params(text).get(key)
        ok = got == expected
        ok_all &= ok
        print(f"[{'OK' if ok else 'FAIL'}] params {text!r:40} -> {key} = {got} (expected {expected})")
    print("ALL PASSED" if ok_all else "SOME FAILED")
    return ok_all


if __name__ == "__main__":
    sys.exit(0 if self_check() else 1)

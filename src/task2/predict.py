"""Inference helpers for Task 3 (Gradio app): load model, predict, Grad-CAM.

Typical use:
    model = load_model()                       # reads Task2/outputs/model_best.pt
    result = predict_image(model, pil_image)   # dict with class, confidence, ...
    tensor = preprocess_image(model, pil_image)
    overlay = gradcam(model, tensor)           # numpy RGB image (H, W, 3), uint8
"""
import json
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps

from . import config
from .model import build_model, get_target_layer
from .transforms import denormalize, get_eval_transforms

ImageLike = Union[Image.Image, np.ndarray, str, Path]


def load_model(model_dir: Optional[Union[str, Path]] = None,
               device: Optional[torch.device] = None, prefix: str = "") -> torch.nn.Module:
    """Load {prefix}model_best.pt + {prefix}model_meta.json from model_dir (default: config.OUT_DIR).

    Returns the model in eval mode on `device`. The metadata dict (class_names,
    img_size, mean, std, architecture, ...) is attached as `model.meta`.
    """
    model_dir = Path(model_dir) if model_dir else config.OUT_DIR
    weights_path = model_dir / f"{prefix}model_best.pt"
    meta_path = model_dir / f"{prefix}model_meta.json"
    if not weights_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing model_best.pt / model_meta.json in {model_dir}. "
                                f"Run Task2/03_train.ipynb first.")
    meta = json.loads(meta_path.read_text())
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pretrained=False: we load our own trained weights, no download needed.
    model = build_model(len(meta["class_names"]), pretrained=False,
                        architecture=meta["architecture"], dropout=meta.get("dropout", 0.0))
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device).eval()
    model.meta = meta
    return model


def _to_pil(image: ImageLike) -> Image.Image:
    """Accept a PIL image, a numpy array (RGB uint8) or a file path -> upright RGB PIL image."""
    if isinstance(image, (str, Path)):
        image = Image.open(image)
    elif isinstance(image, np.ndarray):
        image = Image.fromarray(image.astype(np.uint8))
    # Phone photos may carry an EXIF rotation flag; apply it like training did.
    return ImageOps.exif_transpose(image).convert("RGB")


def preprocess_image(model: torch.nn.Module, image: ImageLike) -> torch.Tensor:
    """Apply the same evaluation transforms as in training -> tensor (1, 3, H, W)."""
    meta = model.meta
    transform = get_eval_transforms(meta["img_size"], meta["mean"], meta["std"])
    return transform(_to_pil(image)).unsqueeze(0)


@torch.no_grad()
def predict_image(model: torch.nn.Module, image: ImageLike, top_k: int = 3) -> dict:
    """Classify one image.

    Returns dict(class_name, fruit, freshness, confidence,
                 probabilities={class: prob}, top3=[{class_name, probability}, ...]).
    """
    class_names: List[str] = model.meta["class_names"]
    device = next(model.parameters()).device
    model.eval()
    probs = torch.softmax(model(preprocess_image(model, image).to(device)), dim=1)[0].cpu()
    best = int(probs.argmax())
    freshness, fruit = config.parse_class_name(class_names[best])
    top = probs.topk(min(top_k, len(class_names)))
    return {
        "class_name": class_names[best],
        "fruit": fruit,
        "freshness": freshness,
        "confidence": float(probs[best]),
        "probabilities": {name: float(p) for name, p in zip(class_names, probs)},
        "top3": [{"class_name": class_names[int(i)], "probability": float(p)}
                 for p, i in zip(top.values, top.indices)],
    }


def gradcam_heatmap(model: torch.nn.Module, img_tensor: torch.Tensor,
                    target_class: Optional[int] = None) -> np.ndarray:
    """Grad-CAM heatmap in 0..1, shape (H, W), same size as the input image.

    Idea: take the feature maps A of the last conv stage, weight each channel by
    the average gradient of the class score w.r.t. that channel, sum, ReLU.
    Regions with large values pushed the prediction towards the target class.
    """
    device = next(model.parameters()).device
    if img_tensor.dim() == 3:
        img_tensor = img_tensor.unsqueeze(0)
    x = img_tensor.to(device).clone().requires_grad_(True)  # makes sure a graph exists
    stored = {}

    def forward_hook(module, inputs, output):
        stored["activations"] = output
        # Capture the gradient flowing back into these activations.
        output.register_hook(lambda grad: stored.__setitem__("gradients", grad))

    handle = get_target_layer(model).register_forward_hook(forward_hook)
    try:
        with torch.enable_grad():
            model.eval()
            scores = model(x)
            if target_class is None:
                target_class = int(scores.argmax(1))
            model.zero_grad()
            scores[0, target_class].backward()
    finally:
        handle.remove()                                      # always clean up the hook

    weights = stored["gradients"].mean(dim=(2, 3), keepdim=True)        # one weight per channel
    cam = F.relu((weights * stored["activations"]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=x.shape[2:], mode="bilinear", align_corners=False)[0, 0]
    cam = cam.detach().cpu().numpy()
    return (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)           # scale to 0..1


def gradcam(model: torch.nn.Module, img_tensor: torch.Tensor,
            target_class: Optional[int] = None, alpha: float = 0.45) -> np.ndarray:
    """Grad-CAM overlay: the (de-normalized) image blended with a colour heatmap.

    img_tensor is a normalized tensor (3,H,W) or (1,3,H,W), e.g. from preprocess_image.
    Returns a uint8 RGB numpy array (H, W, 3).
    """
    heatmap = gradcam_heatmap(model, img_tensor, target_class)
    tensor = img_tensor[0] if img_tensor.dim() == 4 else img_tensor
    meta = getattr(model, "meta", {})
    rgb = denormalize(tensor.cpu(), meta.get("mean", config.IMAGENET_MEAN),
                      meta.get("std", config.IMAGENET_STD))
    rgb = (rgb.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
    colour = cv2.applyColorMap((heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    colour = cv2.cvtColor(colour, cv2.COLOR_BGR2RGB)          # OpenCV uses BGR order
    return (alpha * colour + (1 - alpha) * rgb).astype(np.uint8)

"""Task 1.2 - Filtering, denoising, gradients, edges and frequency-domain filtering.

Conventions
-----------
* Colour images are BGR uint8 (OpenCV convention).
* "gray" arguments are 2D uint8 arrays.
* No plotting / printing here: functions only return arrays or plain values.
"""

from __future__ import annotations

from typing import Dict

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


# ======================================================================
# 1. Synthetic noise
# ======================================================================
def add_gaussian_noise(img: np.ndarray, sigma: float = 25.0, seed: int = 0) -> np.ndarray:
    """Add zero-mean Gaussian noise (sensor-like noise) to an image.

    Args:
        img: uint8 image.
        sigma: standard deviation of the noise in grey levels (0-255 scale).
        seed: random seed so results are reproducible.
    """
    rng = np.random.default_rng(seed)
    noise = rng.normal(0.0, sigma, img.shape)
    noisy = img.astype(np.float64) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)  # keep valid pixel range


def add_salt_pepper_noise(img: np.ndarray, amount: float = 0.05, seed: int = 0) -> np.ndarray:
    """Add impulse ("salt & pepper") noise: random pixels become 0 or 255.

    Args:
        img: uint8 image.
        amount: fraction of pixels corrupted (half salt, half pepper).
        seed: random seed.
    """
    rng = np.random.default_rng(seed)
    noisy = img.copy()
    h, w = img.shape[:2]
    r = rng.random((h, w))        # one random number per pixel position
    noisy[r < amount / 2] = 0     # pepper (black)
    noisy[r > 1 - amount / 2] = 255  # salt (white)
    return noisy


# ======================================================================
# 2. Smoothing / denoising filters
# ======================================================================
def apply_filter(img: np.ndarray, name: str, **params) -> np.ndarray:
    """Apply one of the standard smoothing filters.

    Args:
        img: uint8 image (gray or BGR).
        name: 'box', 'gaussian', 'median' or 'bilateral'.
        **params:
            box:       ksize (default 5)
            gaussian:  ksize (default 5), sigma (default 0 -> OpenCV picks it from ksize)
            median:    ksize (default 5, must be odd)
            bilateral: d (9), sigma_color (75), sigma_space (75)

    Returns:
        Filtered image, same shape and dtype as the input.
    """
    if name == "box":
        k = params.get("ksize", 5)
        return cv2.blur(img, (k, k))  # every pixel = mean of its k x k neighbours
    if name == "gaussian":
        k = params.get("ksize", 5)
        return cv2.GaussianBlur(img, (k, k), params.get("sigma", 0))
    if name == "median":
        return cv2.medianBlur(img, params.get("ksize", 5))  # non-linear: sorts the neighbours
    if name == "bilateral":
        return cv2.bilateralFilter(
            img, params.get("d", 9), params.get("sigma_color", 75), params.get("sigma_space", 75)
        )
    raise ValueError(f"Unknown filter '{name}'. Use box, gaussian, median or bilateral.")


# ======================================================================
# 3. Quality metrics
# ======================================================================
def psnr(reference: np.ndarray, test: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio in dB (higher = closer to the reference)."""
    return float(peak_signal_noise_ratio(reference, test, data_range=255))


def ssim(reference: np.ndarray, test: np.ndarray) -> float:
    """Structural Similarity Index in [-1, 1] (1 = identical). Works for gray or BGR."""
    channel_axis = 2 if reference.ndim == 3 else None
    return float(
        structural_similarity(reference, test, data_range=255, channel_axis=channel_axis)
    )


# ======================================================================
# 4. Convolution written from scratch
# ======================================================================
def convolve2d_from_scratch(gray: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """Slide a kernel over a grayscale image (same operation as cv2.filter2D).

    Like cv2.filter2D this computes *correlation* (kernel is NOT flipped);
    for symmetric kernels (box, Gaussian) this equals true convolution.
    Borders are handled by mirror padding (cv2's default BORDER_REFLECT_101).

    Args:
        gray: 2D array.
        kernel: 2D array with odd height and width.

    Returns:
        float64 array, same shape as `gray` (not clipped / rounded).
    """
    kh, kw = kernel.shape
    pad_y, pad_x = kh // 2, kw // 2
    padded = np.pad(gray.astype(np.float64), ((pad_y, pad_y), (pad_x, pad_x)), mode="reflect")

    h, w = gray.shape
    out = np.zeros((h, w), dtype=np.float64)
    # For each kernel weight, take the image shifted by (dy, dx), multiply, accumulate.
    # This is the sum  out(y,x) = sum_{dy,dx} kernel(dy,dx) * image(y+dy, x+dx).
    for dy in range(kh):
        for dx in range(kw):
            out += kernel[dy, dx] * padded[dy:dy + h, dx:dx + w]
    return out


# ======================================================================
# 5. Gradients and edges
# ======================================================================
def _to_gray(img: np.ndarray) -> np.ndarray:
    """BGR -> gray (pass-through if already 2D)."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img


def sobel_gradients(gray: np.ndarray, ksize: int = 3) -> Dict[str, np.ndarray]:
    """Sobel derivatives.

    Returns:
        dict with float32 arrays: gx, gy, magnitude = sqrt(gx^2+gy^2),
        direction (degrees, 0-360, measured from the +x axis).
    """
    gray = _to_gray(gray)
    # CV_32F so negative derivatives are not clipped (uint8 would lose them).
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=ksize)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=ksize)
    magnitude = cv2.magnitude(gx, gy)
    direction = np.degrees(np.arctan2(gy, gx)) % 360
    return {"gx": gx, "gy": gy, "magnitude": magnitude, "direction": direction}


def laplacian(gray: np.ndarray, ksize: int = 3) -> np.ndarray:
    """Absolute Laplacian (second derivative) as uint8 - highlights edges and blobs."""
    gray = _to_gray(gray)
    lap = cv2.Laplacian(gray, cv2.CV_32F, ksize=ksize)
    return cv2.convertScaleAbs(lap)  # |lap| clipped to 0..255


def canny_edges(
    gray: np.ndarray, low: int = 100, high: int = 200, blur_sigma: float = 0.0
) -> np.ndarray:
    """Canny edge detector (binary uint8 map, edges = 255).

    Args:
        low, high: hysteresis thresholds. Pixels above `high` are strong edges;
            pixels between low and high are kept only if connected to a strong one.
        blur_sigma: if > 0, Gaussian-blur first (reduces noise-induced edges).
    """
    gray = _to_gray(gray)
    if blur_sigma > 0:
        gray = cv2.GaussianBlur(gray, (0, 0), blur_sigma)  # (0,0) -> size from sigma
    return cv2.Canny(gray, low, high)


def detect_edges(img: np.ndarray, method: str = "canny", **params) -> np.ndarray:
    """Dispatcher returning a uint8 edge image for 'sobel', 'laplacian' or 'canny'.

    Extra params are passed to the underlying function
    (canny: low, high, blur_sigma; sobel/laplacian: ksize).
    """
    gray = _to_gray(img)
    if method == "canny":
        return canny_edges(gray, **params)
    if method == "sobel":
        mag = sobel_gradients(gray, **params)["magnitude"]
        return cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    if method == "laplacian":
        return laplacian(gray, **params)
    raise ValueError(f"Unknown edge method '{method}'. Use sobel, laplacian or canny.")


# ======================================================================
# 6. Frequency domain
# ======================================================================
def fft_spectrum(gray: np.ndarray) -> np.ndarray:
    """Log-magnitude spectrum of the 2D FFT, with the zero frequency in the centre.

    Returns:
        float32 array, same shape as the input (log(1 + |F|)).
    """
    gray = _to_gray(gray)
    F = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))  # move DC to the centre
    return np.log1p(np.abs(F)).astype(np.float32)  # log: magnitudes span many orders


def frequency_mask(
    shape: tuple, kind: str = "lowpass", cutoff: float = 30.0, type: str = "gaussian"
) -> np.ndarray:
    """Build a centred frequency-domain mask H(u,v) with values in [0, 1].

    Args:
        shape: (height, width) of the image.
        kind: 'lowpass' (keep small frequencies) or 'highpass' (keep large ones).
        cutoff: cut-off radius D0 in frequency pixels from the centre.
        type: 'ideal' (hard circle) or 'gaussian' (smooth fall-off).
    """
    h, w = shape
    v, u = np.ogrid[:h, :w]
    # D = distance from the spectrum centre (the zero frequency).
    D2 = (u - w // 2) ** 2 + (v - h // 2) ** 2
    if type == "ideal":
        lowpass = (D2 <= cutoff ** 2).astype(np.float64)
    elif type == "gaussian":
        lowpass = np.exp(-D2 / (2.0 * cutoff ** 2))
    else:
        raise ValueError("type must be 'ideal' or 'gaussian'")
    if kind == "lowpass":
        return lowpass
    if kind == "highpass":
        return 1.0 - lowpass  # complement of the low-pass
    raise ValueError("kind must be 'lowpass' or 'highpass'")


def frequency_filter(
    gray: np.ndarray, kind: str = "lowpass", cutoff: float = 30.0, type: str = "gaussian"
) -> np.ndarray:
    """Filter an image in the frequency domain: FFT -> multiply by mask -> inverse FFT.

    Returns:
        uint8 image. For high-pass the result is signed, so its absolute value
        is stretched to 0..255 to be visible.
    """
    gray = _to_gray(gray)
    F = np.fft.fftshift(np.fft.fft2(gray.astype(np.float64)))
    H = frequency_mask(gray.shape, kind, cutoff, type)
    result = np.real(np.fft.ifft2(np.fft.ifftshift(F * H)))  # back to the image domain
    if kind == "highpass":
        result = cv2.normalize(np.abs(result), None, 0, 255, cv2.NORM_MINMAX)
    return np.clip(result, 0, 255).astype(np.uint8)

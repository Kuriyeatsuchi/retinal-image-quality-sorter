"""
Image-quality measurements for retinal fundus photos.

Every measurement is taken *inside the retina only*: a fundus photo is a
bright circle on a black background, and including that background would
drag averages down by an amount that depends on the camera's framing.

Measurements
------------
brightness   Mean HSV value (0-255) inside the retina.
diameter_px  Diameter of the retina in the original image, in pixels.
             A direct measure of resolution, independent of JPEG compression.
sharpness    Variance of the Laplacian of the green channel (x1000 for
             readability), after the retina is placed on a square canvas,
             resized to a fixed size and blurred. The resize makes photos
             from different cameras comparable. The blur is deliberately
             strong: it removes sensor grain and JPEG compression artifacts,
             so the score reflects structures at the scale of blood vessels
             rather than how the file was saved. The green channel is used
             because vessels have the most contrast there.
"""

from dataclasses import dataclass, asdict

import cv2
import numpy as np

# Size the retina is resampled to before measuring sharpness, so the score
# does not depend on the camera's resolution.
SHARPNESS_SIZE = 512
# Blur (Gaussian sigma, in pixels at SHARPNESS_SIZE) applied before the
# Laplacian. Lighter blur (e.g. 1.5 px at 768) let JPEG blocking artifacts
# raise the score of blurry photos above that of sharp ones; at this scale
# re-saving a photo barely changes its score.
SHARPNESS_BLUR_SIGMA = 3.5
# Raw scores are small; multiply for readability.
SHARPNESS_SCALE = 1000.0
# Only the central part of the retina is used for sharpness, so the hard
# edge between the retina and the black background is not counted.
SHARPNESS_INNER_FRACTION = 0.85
# Pixels darker than this (max of B, G, R) are treated as background.
BACKGROUND_LEVEL = 15
# The retina must cover at least this fraction of the image to be accepted.
MIN_RETINA_AREA_FRACTION = 0.15


@dataclass
class Thresholds:
    min_brightness: float = 50      # below: underexposed
    max_brightness: float = 240     # above: saturated (washed-out photos are also caught as blurry)
    min_diameter_px: int = 400      # below: thumbnail-sized, too few pixels to grade
    min_sharpness: float = 3.5      # below: blurry / out of focus


@dataclass
class Measurements:
    retina_found: bool
    brightness: float = 0.0
    diameter_px: float = 0.0
    sharpness: float = 0.0


def find_retina(image_bgr, work_size=1024):
    """
    Locate the circular retina.

    Returns (cx, cy, r) in original-image pixels, or None if no retina is found.
    Detection runs on a downscaled copy for speed.
    """
    h, w = image_bgr.shape[:2]
    scale = work_size / max(h, w)
    small = cv2.resize(image_bgr, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

    _, mask = cv2.threshold(small.max(axis=2), BACKGROUND_LEVEL, 255, cv2.THRESH_BINARY)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < MIN_RETINA_AREA_FRACTION * small.shape[0] * small.shape[1]:
        return None

    (cx, cy), r = cv2.minEnclosingCircle(largest)
    return cx / scale, cy / scale, r / scale


def circle_mask(shape, cx, cy, r):
    mask = np.zeros(shape[:2], np.uint8)
    cv2.circle(mask, (int(round(cx)), int(round(cy))), int(round(r)), 255, -1)
    return mask > 0


def measure(image_bgr):
    """Compute all quality measurements for one image."""
    retina = find_retina(image_bgr)
    if retina is None:
        return Measurements(retina_found=False)
    cx, cy, r = retina

    # Brightness inside the retina (slightly shrunk to avoid the dark rim).
    hsv_value = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    brightness = float(hsv_value[circle_mask(image_bgr.shape, cx, cy, r * 0.95)].mean())

    # Sharpness: place the retina on a square canvas, resample to a fixed size,
    # denoise, Laplacian. Fundus cameras often cut off the top and bottom of
    # the retina; padding (instead of stretching the visible part to a square)
    # keeps the geometry the same for every photo.
    sharpness = _sharpness(image_bgr[:, :, 1], cx, cy, r)

    return Measurements(
        retina_found=True,
        brightness=brightness,
        diameter_px=2 * r,
        sharpness=sharpness,
    )


def _sharpness(green, cx, cy, r):
    h, w = green.shape
    side = max(1, int(round(2 * r)))
    x0, y0 = int(round(cx - r)), int(round(cy - r))
    canvas = np.zeros((side, side), np.uint8)
    inside = np.zeros((side, side), np.uint8)
    sx0, sy0 = max(0, x0), max(0, y0)
    sx1, sy1 = min(w, x0 + side), min(h, y0 + side)
    canvas[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = green[sy0:sy1, sx0:sx1]
    inside[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = 255

    n = SHARPNESS_SIZE
    g = cv2.resize(canvas, (n, n), interpolation=cv2.INTER_AREA).astype(np.float32)
    g = cv2.GaussianBlur(g, (0, 0), SHARPNESS_BLUR_SIGMA)
    # Measure only where the photo has real pixels: inside the central circle
    # and away from any edge where the image was cut off.
    inside = cv2.resize(inside, (n, n), interpolation=cv2.INTER_NEAREST)
    inside = cv2.erode(inside, np.ones((9, 9), np.uint8)) > 0
    region = circle_mask((n, n), n / 2, n / 2, n / 2 * SHARPNESS_INNER_FRACTION) & inside
    return float(cv2.Laplacian(g, cv2.CV_32F)[region].var()) * SHARPNESS_SCALE


def decide(m: Measurements, t: Thresholds):
    """Return (is_gradeable, reason) for a set of measurements."""
    if not m.retina_found:
        return False, "no retina detected"
    if m.brightness < t.min_brightness:
        return False, "too dark"
    if m.brightness > t.max_brightness:
        return False, "too bright"
    if m.diameter_px < t.min_diameter_px:
        return False, "resolution too low"
    if m.sharpness < t.min_sharpness:
        return False, "blurry"
    return True, "passed all checks"


def legacy_decide(image_bgr, file_size_bytes, brightness_limit=35, size_threshold_bytes=109_000):
    """
    The original (v2) rule, kept for comparison: whole-image brightness plus
    file size.
    """
    value = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)[:, :, 2]
    if int(value.mean()) < brightness_limit:
        return False, "too dark"
    if file_size_bytes < size_threshold_bytes:
        return False, "file too small"
    return True, "passed both checks"


def as_dict(m: Measurements):
    d = asdict(m)
    for key in ("brightness", "diameter_px", "sharpness"):
        d[key] = round(d[key], 2 if key == "sharpness" else 1)
    return d

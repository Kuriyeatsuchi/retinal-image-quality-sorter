"""
Robustness check: does the decision depend on how a photo was *saved*, or
on what is *in* it?

For every image in INPUT_DIR this script creates re-saved copies that keep
the visual content the same but change the file:

  smaller_jpeg   shrunk to at most 2000 px wide (never enlarged), JPEG quality 60
                 (much smaller file)
  larger_jpeg    saved again at JPEG quality 100  (much larger file)

A good quality check should give every copy the same decision as the
original. The script compares the original v2 rule (file size + whole-image
brightness) with the current rule (resolution + sharpness + brightness
inside the retina).

Usage:
    python experiments/robustness_check.py INPUT_DIR
"""

import os
import sys
import tempfile
from collections import Counter

import cv2

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import quality  # noqa: E402
from sort_images import list_images, find_duplicates  # noqa: E402

MAX_WIDTH = 2000


def shrink(img, max_width=MAX_WIDTH):
    """Downscale to at most max_width pixels wide; smaller images are left as they are."""
    h, w = img.shape[:2]
    if w <= max_width:
        return img
    return cv2.resize(img, (max_width, round(h * max_width / w)), interpolation=cv2.INTER_AREA)


VARIANTS = {
    "smaller_jpeg": lambda img: (shrink(img), [cv2.IMWRITE_JPEG_QUALITY, 60]),
    "larger_jpeg": lambda img: (img, [cv2.IMWRITE_JPEG_QUALITY, 100]),
}


def decisions(path, thresholds):
    image = cv2.imread(path)
    size = os.path.getsize(path)
    new_ok, _ = quality.decide(quality.measure(image), thresholds)
    old_ok, _ = quality.legacy_decide(image, size)
    return old_ok, new_ok, size


def main():
    input_dir = sys.argv[1]
    thresholds = quality.Thresholds()
    paths = list_images(input_dir)
    duplicates = find_duplicates(paths)
    paths = [p for p in paths if os.path.basename(p) not in duplicates]

    flips = {"v2": Counter(), "v3": Counter()}
    print(f"{'image':<14}{'variant':<14}{'size':>9}   {'v2 rule':<22}v3 rule")
    with tempfile.TemporaryDirectory() as tmp:
        for path in paths:
            image = cv2.imread(path)
            old0, new0, size0 = decisions(path, thresholds)
            name = os.path.basename(path)[:12]
            print(f"{name:<14}{'original':<14}{size0 / 1024:>7.0f}KB   "
                  f"{'gradeable' if old0 else 'ungradeable':<22}{'gradeable' if new0 else 'ungradeable'}")
            for variant, make in VARIANTS.items():
                out, params = make(image)
                out_path = os.path.join(tmp, f"{variant}.jpg")
                cv2.imwrite(out_path, out, params)
                old, new, size = decisions(out_path, thresholds)
                mark_old = "  CHANGED" if old != old0 else ""
                mark_new = "  CHANGED" if new != new0 else ""
                print(f"{'':<14}{variant:<14}{size / 1024:>7.0f}KB   "
                      f"{('gradeable' if old else 'ungradeable') + mark_old:<22}"
                      f"{('gradeable' if new else 'ungradeable') + mark_new}")
                flips["v2"][variant] += old != old0
                flips["v3"][variant] += new != new0

    n = len(paths)
    print(f"\n{n} unique images. Decisions that changed just by re-saving the file:")
    for rule in ("v2", "v3"):
        parts = ", ".join(f"{v}: {flips[rule][v]}/{n}" for v in VARIANTS)
        print(f"  {rule} rule  ->  {parts}")


if __name__ == "__main__":
    main()

"""
Retinal image quality sorter.

Sorts retinal fundus photos into "gradeable" and "ungradeable" folders using
two quick checks:

  1. Brightness: the mean of the HSV "V" (value) channel. Photos darker than
     the limit are underexposed and can't be reliably read by a doctor.
  2. File size: very small files are usually low-resolution or heavily
     compressed, so they are treated as ungradeable.

A photo is gradeable only if it passes both checks.

Usage:
    python sort_images.py INPUT_DIR [--output OUTPUT_DIR]
                          [--brightness-limit 35] [--size-threshold-kb 109]
                          [--move] [--report results.csv]
"""

import argparse
import csv
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

GRADEABLE = "gradeable"
UNGRADEABLE = "ungradeable"


def mean_brightness(image_bgr):
    """Average of the HSV value channel, on a 0-255 scale."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return float(np.mean(hsv[:, :, 2]))


def classify(path, brightness_limit, size_threshold_bytes):
    """Return a dict describing the decision for one image, or None if unreadable."""
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return None

    brightness = mean_brightness(image)
    size = os.path.getsize(path)

    if int(brightness) < brightness_limit:
        label, reason = UNGRADEABLE, "too dark"
    elif size < size_threshold_bytes:
        label, reason = UNGRADEABLE, "file too small"
    else:
        label, reason = GRADEABLE, "passed both checks"

    return {
        "file": os.path.basename(path),
        "label": label,
        "reason": reason,
        "brightness": round(brightness, 1),
        "size_kb": round(size / 1024, 1),
    }


def sort_images(input_dir, output_dir, brightness_limit, size_threshold_bytes, move=False):
    folders = {
        GRADEABLE: os.path.join(output_dir, "Gradeable"),
        UNGRADEABLE: os.path.join(output_dir, "Ungradeable"),
    }
    for folder in folders.values():
        os.makedirs(folder, exist_ok=True)

    paths = sorted(
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS
        and os.path.isfile(os.path.join(input_dir, name))
    )

    def process(path):
        result = classify(path, brightness_limit, size_threshold_bytes)
        if result is None:
            return {"file": os.path.basename(path), "label": "skipped",
                    "reason": "could not read image", "brightness": "", "size_kb": ""}
        destination = os.path.join(folders[result["label"]], result["file"])
        (shutil.move if move else shutil.copy2)(path, destination)
        return result

    with ThreadPoolExecutor() as executor:
        return list(executor.map(process, paths))


def main():
    parser = argparse.ArgumentParser(description="Sort retinal images by gradeability.")
    parser.add_argument("input_dir", help="Folder containing the retinal images")
    parser.add_argument("--output", default="sorted",
                        help="Where to create Gradeable/ and Ungradeable/ (default: ./sorted)")
    parser.add_argument("--brightness-limit", type=int, default=35,
                        help="Minimum mean brightness, 0-255 (default: 35)")
    parser.add_argument("--size-threshold-kb", type=float, default=109,
                        help="Minimum file size in KB (default: 109)")
    parser.add_argument("--move", action="store_true",
                        help="Move files instead of copying them")
    parser.add_argument("--report", help="Optional CSV file to write per-image results to")
    args = parser.parse_args()

    start = time.time()
    results = sort_images(
        args.input_dir,
        args.output,
        args.brightness_limit,
        int(args.size_threshold_kb * 1000),
        move=args.move,
    )

    for r in results:
        print(f"{r['label']:<12} {r['file']}  ({r['reason']}; brightness={r['brightness']}, size={r['size_kb']} KB)")

    counts = {label: sum(r["label"] == label for r in results)
              for label in (GRADEABLE, UNGRADEABLE, "skipped")}
    print(f"\n{len(results)} images: {counts[GRADEABLE]} gradeable, "
          f"{counts[UNGRADEABLE]} ungradeable, {counts['skipped']} skipped "
          f"in {time.time() - start:.2f}s")

    if args.report:
        with open(args.report, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["file", "label", "reason", "brightness", "size_kb"])
            writer.writeheader()
            writer.writerows(results)
        print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()

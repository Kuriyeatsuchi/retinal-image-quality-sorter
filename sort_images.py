"""
Retinal image quality sorter.

Sorts retinal fundus photos into "Gradeable" and "Ungradeable" folders.
A photo is gradeable only if, measured inside the retina, it is:

  - neither too dark nor washed out    (brightness)
  - high enough resolution             (retina diameter in pixels)
  - in focus                           (sharpness)

Exact duplicate files are detected and reported, and can be skipped.

Usage:
    python sort_images.py INPUT_DIR [--output OUTPUT_DIR] [--report results.csv]
                          [--min-brightness 50] [--max-brightness 210]
                          [--min-diameter 1000] [--min-sharpness 0.15]
                          [--skip-duplicates] [--move] [--legacy]
"""

import argparse
import csv
import hashlib
import os
import shutil
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import cv2

import quality

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
REPORT_FIELDS = ["file", "label", "reason", "duplicate_of",
                 "brightness", "diameter_px", "sharpness", "size_kb"]


def list_images(input_dir):
    return sorted(
        os.path.join(input_dir, name)
        for name in os.listdir(input_dir)
        if os.path.splitext(name)[1].lower() in SUPPORTED_EXTENSIONS
        and os.path.isfile(os.path.join(input_dir, name))
    )


def file_hash(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def find_duplicates(paths):
    """Map each duplicate file name to the name of the first file with identical content."""
    first_seen, duplicate_of = {}, {}
    with ThreadPoolExecutor() as executor:
        hashes = list(executor.map(file_hash, paths))
    for path, digest in zip(paths, hashes):
        name = os.path.basename(path)
        if digest in first_seen:
            duplicate_of[name] = first_seen[digest]
        else:
            first_seen[digest] = name
    return duplicate_of


def assess(path, thresholds, legacy=False):
    """Measure one image and decide whether it is gradeable."""
    name = os.path.basename(path)
    size = os.path.getsize(path)
    image = cv2.imread(path, cv2.IMREAD_COLOR)
    if image is None:
        return {"file": name, "label": "skipped", "reason": "could not read image",
                "size_kb": round(size / 1024, 1)}

    m = quality.measure(image)
    if legacy:
        ok, reason = quality.legacy_decide(image, size)
    else:
        ok, reason = quality.decide(m, thresholds)

    return {
        "file": name,
        "label": "gradeable" if ok else "ungradeable",
        "reason": reason,
        **{k: v for k, v in quality.as_dict(m).items() if k != "retina_found"},
        "size_kb": round(size / 1024, 1),
    }


def sort_images(input_dir, output_dir, thresholds, move=False,
                skip_duplicates=False, legacy=False, copy_files=True):
    paths = list_images(input_dir)
    duplicate_of = find_duplicates(paths)

    to_assess = [p for p in paths
                 if not (skip_duplicates and os.path.basename(p) in duplicate_of)]
    with ThreadPoolExecutor() as executor:
        results = list(executor.map(lambda p: assess(p, thresholds, legacy), to_assess))

    for r in results:
        r["duplicate_of"] = duplicate_of.get(r["file"], "")
    if skip_duplicates:
        results += [{"file": name, "label": "skipped", "reason": "duplicate",
                     "duplicate_of": original}
                    for name, original in duplicate_of.items()]

    if copy_files:
        folders = {"gradeable": os.path.join(output_dir, "Gradeable"),
                   "ungradeable": os.path.join(output_dir, "Ungradeable")}
        for folder in folders.values():
            os.makedirs(folder, exist_ok=True)
        transfer = shutil.move if move else shutil.copy2
        for r in results:
            if r["label"] in folders:
                transfer(os.path.join(input_dir, r["file"]),
                         os.path.join(folders[r["label"]], r["file"]))

    return sorted(results, key=lambda r: r["file"])


def write_report(results, path):
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=REPORT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)


def main():
    defaults = quality.Thresholds()
    parser = argparse.ArgumentParser(description="Sort retinal images by gradeability.")
    parser.add_argument("input_dir", help="Folder containing the retinal images")
    parser.add_argument("--output", default="sorted",
                        help="Where to create Gradeable/ and Ungradeable/ (default: ./sorted)")
    parser.add_argument("--report", help="Write per-image measurements and decisions to this CSV")
    parser.add_argument("--min-brightness", type=float, default=defaults.min_brightness,
                        help=f"Minimum brightness inside the retina, 0-255 (default: {defaults.min_brightness})")
    parser.add_argument("--max-brightness", type=float, default=defaults.max_brightness,
                        help=f"Maximum brightness inside the retina, 0-255 (default: {defaults.max_brightness})")
    parser.add_argument("--min-diameter", type=int, default=defaults.min_diameter_px,
                        help=f"Minimum retina diameter in pixels (default: {defaults.min_diameter_px})")
    parser.add_argument("--min-sharpness", type=float, default=defaults.min_sharpness,
                        help=f"Minimum sharpness score (default: {defaults.min_sharpness})")
    parser.add_argument("--skip-duplicates", action="store_true",
                        help="Only process the first copy of identical files")
    parser.add_argument("--move", action="store_true", help="Move files instead of copying them")
    parser.add_argument("--dry-run", action="store_true",
                        help="Assess and report without copying or moving any files")
    parser.add_argument("--legacy", action="store_true",
                        help="Use the original v2 rule (whole-image brightness + file size)")
    args = parser.parse_args()

    thresholds = quality.Thresholds(
        min_brightness=args.min_brightness,
        max_brightness=args.max_brightness,
        min_diameter_px=args.min_diameter,
        min_sharpness=args.min_sharpness,
    )

    start = time.time()
    results = sort_images(args.input_dir, args.output, thresholds,
                          move=args.move, skip_duplicates=args.skip_duplicates,
                          legacy=args.legacy, copy_files=not args.dry_run)

    for r in results:
        detail = ""
        if "sharpness" in r:
            detail = (f"  brightness={r['brightness']}, diameter={r['diameter_px']:.0f}px, "
                      f"sharpness={r['sharpness']}")
        dup = f"  [duplicate of {r['duplicate_of']}]" if r.get("duplicate_of") else ""
        print(f"{r['label']:<12} {r['file']}  ({r['reason']}){detail}{dup}")

    counts = Counter(r["label"] for r in results)
    reasons = Counter(r["reason"] for r in results if r["label"] == "ungradeable")
    n_dup = sum(1 for r in results if r.get("duplicate_of"))
    print(f"\n{len(results)} images: {counts['gradeable']} gradeable, "
          f"{counts['ungradeable']} ungradeable, {counts['skipped']} skipped "
          f"in {time.time() - start:.2f}s")
    if reasons:
        print("Ungradeable because: " + ", ".join(f"{k} ({v})" for k, v in reasons.most_common()))
    if n_dup:
        print(f"{n_dup} files are exact duplicates of another file")

    if args.report:
        write_report(results, args.report)
        print(f"Report written to {args.report}")


if __name__ == "__main__":
    main()

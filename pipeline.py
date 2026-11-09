"""
Two-step screening pipeline:

  1. Quality check: is the photo gradeable?          (quality.py)
  2. Disease grading: how severe is the retinopathy?  (grading/)

Only photos that pass step 1 are graded; the rest are flagged for re-imaging.

NOT A MEDICAL DEVICE. This is a research and education project. It has not
been clinically validated and must not be used to diagnose patients or make
referral decisions.

Usage:
    python pipeline.py IMAGE_DIR --grader results/grader/grader.joblib [--out results.csv]
"""

import argparse
import csv
import os
import sys

import cv2
import joblib

import quality
from sort_images import list_images

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "grading"))
from features import FeatureExtractor  # noqa: E402

DISCLAIMER = ("RESEARCH / EDUCATION ONLY - not a medical device and not clinically validated. "
              "Do not use these results to diagnose patients or make referral decisions.")


def main():
    p = argparse.ArgumentParser(description="Quality check, then grade diabetic retinopathy.")
    p.add_argument("image_dir")
    p.add_argument("--grader", required=True, help="Model saved by grading/train_grader.py")
    p.add_argument("--out", help="Write results to this CSV")
    args = p.parse_args()

    print(DISCLAIMER + "\n")
    bundle = joblib.load(args.grader)
    model, grade_names = bundle["model"], bundle["grade_names"]
    extractor = FeatureExtractor()
    thresholds = quality.Thresholds()

    rows = []
    for path in list_images(args.image_dir):
        name = os.path.basename(path)
        image = cv2.imread(path)
        if image is None:
            rows.append({"file": name, "quality": "unreadable"})
            continue
        ok, reason = quality.decide(quality.measure(image), thresholds)
        row = {"file": name, "quality": "gradeable" if ok else f"ungradeable ({reason})"}
        if ok:
            proba = model.predict_proba(extractor(image)[None])[0]
            grade = int(proba.argmax())
            row.update({"grade": grade, "grade_name": grade_names[grade],
                        "confidence": round(float(proba[grade]), 3),
                        "refer": "yes" if grade >= 2 else "no"})
        else:
            row.update({"refer": "re-image"})
        rows.append(row)

        summary = (f"grade {row['grade']} ({row['grade_name']}, {row['confidence']:.0%})"
                   if ok else "-> re-image")
        print(f"{name:<45} {row['quality']:<32} {summary}")

    if args.out:
        fields = ["file", "quality", "grade", "grade_name", "confidence", "refer"]
        with open(args.out, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        print(f"\nResults written to {args.out}")
    print("\n" + DISCLAIMER)


if __name__ == "__main__":
    main()

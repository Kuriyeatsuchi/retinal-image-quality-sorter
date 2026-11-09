"""
Measure how well the quality sorter agrees with human gradability labels,
and optionally search for better thresholds.

Labels CSV: one row per image, with a file-name column and a label column.
Labels are read as gradeable if they look like "gradable", "gradeable",
"yes", "1" or "true", and as ungradeable otherwise. The column names default
to "Image" and "gradability" (the format of the SDAIA Athka metadata) and can
be changed with --file-col / --label-col.

Usage:
    python evaluate_sorter.py IMAGE_DIR LABELS_CSV [--tune] [--report measurements.csv]

With --tune, thresholds are searched with 5-fold cross-validation so the
reported accuracy is not measured on the same images the thresholds were
chosen on.
"""

import argparse
import itertools
import os

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

import quality
from sort_images import assess

POSITIVE = {"gradable", "gradeable", "yes", "y", "1", "true", "good"}

SEARCH_GRID = {
    "min_brightness": [30, 40, 50, 60, 70],
    "max_brightness": [200, 220, 240, 256],
    "min_diameter_px": [0, 400, 600, 800, 1000],
    "min_sharpness": [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 7.0, 10.0],
}


def load_labels(path, file_col, label_col, image_dir):
    table = pd.read_csv(path).dropna(subset=[file_col, label_col])
    on_disk = set(os.listdir(image_dir))
    table = table[table[file_col].astype(str).isin(on_disk)]
    if table.empty:
        raise SystemExit("None of the labelled files were found in the image folder.")
    labels = table[label_col].astype(str).str.strip().str.lower().isin(POSITIVE)
    return dict(zip(table[file_col].astype(str), labels))


def measure_all(image_dir, files):
    rows = []
    for i, name in enumerate(files, 1):
        r = assess(os.path.join(image_dir, name), quality.Thresholds())
        rows.append(r)
        if i % 100 == 0 or i == len(files):
            print(f"  measured {i}/{len(files)}", flush=True)
    return pd.DataFrame(rows)


def predict(df, t: quality.Thresholds):
    return (
        df["brightness"].ge(t.min_brightness)
        & df["brightness"].le(t.max_brightness)
        & df["diameter_px"].ge(t.min_diameter_px)
        & df["sharpness"].ge(t.min_sharpness)
    ).to_numpy()


def scores(y_true, y_pred):
    y_true, y_pred = np.asarray(y_true, bool), np.asarray(y_pred, bool)
    tp = np.sum(y_pred & y_true)
    tn = np.sum(~y_pred & ~y_true)
    fp = np.sum(y_pred & ~y_true)
    fn = np.sum(~y_pred & y_true)
    sens = tp / max(1, tp + fn)          # gradeable photos correctly kept
    spec = tn / max(1, tn + fp)          # ungradeable photos correctly rejected
    return {
        "accuracy": (tp + tn) / len(y_true),
        "balanced_accuracy": (sens + spec) / 2,
        "kept_gradeable": sens,
        "rejected_ungradeable": spec,
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
    }


def best_thresholds(df, y):
    best, best_score = None, -1
    keys = list(SEARCH_GRID)
    for values in itertools.product(*(SEARCH_GRID[k] for k in keys)):
        t = quality.Thresholds(**dict(zip(keys, values)))
        s = scores(y, predict(df, t))["balanced_accuracy"]
        if s > best_score:
            best, best_score = t, s
    return best


def print_scores(title, s):
    print(f"\n{title}")
    print(f"  accuracy                        {s['accuracy']:.3f}")
    print(f"  balanced accuracy               {s['balanced_accuracy']:.3f}")
    print(f"  gradeable photos kept           {s['kept_gradeable']:.3f}")
    print(f"  ungradeable photos rejected     {s['rejected_ungradeable']:.3f}")
    print(f"  confusion: kept-good {s['tp']}, rejected-bad {s['tn']}, "
          f"kept-bad {s['fp']}, rejected-good {s['fn']}")


def main():
    p = argparse.ArgumentParser(description="Evaluate the sorter against gradability labels.")
    p.add_argument("image_dir")
    p.add_argument("labels_csv")
    p.add_argument("--file-col", default="Image")
    p.add_argument("--label-col", default="gradability")
    p.add_argument("--tune", action="store_true", help="Search thresholds with cross-validation")
    p.add_argument("--report", help="Save per-image measurements, labels and decisions to this CSV")
    args = p.parse_args()

    labels = load_labels(args.labels_csv, args.file_col, args.label_col, args.image_dir)
    files = sorted(labels)
    y = np.array([labels[f] for f in files])
    print(f"{len(files)} labelled images: {y.sum()} gradeable, {(~y).sum()} ungradeable")

    df = measure_all(args.image_dir, files)
    df["true_label"] = np.where(y, "gradeable", "ungradeable")
    readable = df["label"].ne("skipped").to_numpy()
    if not readable.all():
        print(f"  {np.sum(~readable)} images could not be read and are counted as rejected")
    df[["brightness", "diameter_px", "sharpness"]] = df[["brightness", "diameter_px", "sharpness"]].fillna(-1)

    default = quality.Thresholds()
    print_scores("Default thresholds", scores(y, predict(df, default)))

    if args.tune:
        folds = StratifiedKFold(5, shuffle=True, random_state=0)
        oof = np.zeros(len(y), bool)
        for train_idx, test_idx in folds.split(df, y):
            t = best_thresholds(df.iloc[train_idx], y[train_idx])
            oof[test_idx] = predict(df.iloc[test_idx], t)
        print_scores("Tuned thresholds (5-fold cross-validated estimate)", scores(y, oof))
        final = best_thresholds(df, y)
        print("\nThresholds tuned on all labelled images (use these going forward):")
        for k, v in vars(final).items():
            print(f"  --{k.replace('_px', '').replace('_', '-')} {v}")

    if args.report:
        df["predicted_default"] = np.where(predict(df, default), "gradeable", "ungradeable")
        df.to_csv(args.report, index=False)
        print(f"\nPer-image results written to {args.report}")


if __name__ == "__main__":
    main()

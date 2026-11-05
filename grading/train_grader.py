"""
Train and evaluate a diabetic-retinopathy severity grader.

Grades follow the usual 0-4 scale:
    0 no DR, 1 mild, 2 moderate, 3 severe, 4 proliferative

Pipeline: pretrained ResNet18 features (see features.py) -> standardisation
-> logistic regression. The regularisation strength is chosen by 5-fold
cross-validation on train+validation combined; the test set is used once, at
the end, for the reported numbers.

Each split is given as a CSV with an image id column and a grade column,
plus the folder containing the images. Rows whose image is missing from the
folder are skipped.

Usage:
    python grading/train_grader.py \
        --train-csv train.csv --train-dir train_images \
        --val-csv valid.csv   --val-dir val_images \
        --test-csv test.csv   --test-dir test_images \
        --out results/grader
"""

import argparse
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, cohen_kappa_score, confusion_matrix,
                             roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FeatureExtractor  # noqa: E402

GRADE_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative"]
C_GRID = [0.01, 0.03, 0.1, 0.3]
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp")


def load_split(csv_path, image_dir, id_col, label_col):
    table = pd.read_csv(csv_path)
    on_disk = {os.path.splitext(f)[0]: f for f in os.listdir(image_dir)
               if f.lower().endswith(IMAGE_EXTENSIONS)}
    ids, paths, labels = [], [], []
    for image_id, label in zip(table[id_col].astype(str), table[label_col]):
        key = os.path.splitext(image_id)[0]
        if key in on_disk:
            ids.append(key)
            paths.append(os.path.join(image_dir, on_disk[key]))
            labels.append(int(label))
    skipped = len(table) - len(ids)
    print(f"{os.path.basename(csv_path)}: {len(ids)} images found"
          + (f", {skipped} listed but missing (skipped)" if skipped else ""))
    return ids, paths, np.array(labels)


def cached_features(extractor, ids, paths, cache_path):
    if cache_path and os.path.exists(cache_path):
        cached = np.load(cache_path, allow_pickle=False)
        if list(cached["ids"]) == ids:
            return cached["features"]
    features = extractor.from_files(paths)
    if cache_path:
        np.savez(cache_path, ids=np.array(ids), features=features)
    return features


def make_model(c):
    return make_pipeline(StandardScaler(),
                         LogisticRegression(C=c, class_weight="balanced", max_iter=5000))


def binary_metrics(y_true, score, predicted):
    tp = int(np.sum(predicted & y_true))
    tn = int(np.sum(~predicted & ~y_true))
    return {
        "auc": round(float(roc_auc_score(y_true, score)), 3),
        "sensitivity": round(tp / max(1, int(y_true.sum())), 3),
        "specificity": round(tn / max(1, int((~y_true).sum())), 3),
        "positives": int(y_true.sum()),
        "negatives": int((~y_true).sum()),
    }


def evaluate(y, proba):
    pred = proba.argmax(1)
    majority = np.bincount(y).argmax()
    return {
        "n_images": int(len(y)),
        "grade_counts": np.bincount(y, minlength=5).tolist(),
        "quadratic_weighted_kappa": round(float(cohen_kappa_score(y, pred, weights="quadratic")), 3),
        "accuracy": round(float(accuracy_score(y, pred)), 3),
        "accuracy_always_majority_grade": round(float(np.mean(y == majority)), 3),
        "recall_per_grade": {GRADE_NAMES[g]: round(float(np.mean(pred[y == g] == g)), 3)
                             for g in range(5) if np.any(y == g)},
        "any_dr_vs_none": binary_metrics(y >= 1, 1 - proba[:, 0], pred >= 1),
        "referable_dr_grade2plus": binary_metrics(y >= 2, proba[:, 2:].sum(1), pred >= 2),
        "confusion_matrix": confusion_matrix(y, pred, labels=range(5)).tolist(),
    }


def save_confusion_plot(cm, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = np.array(cm)
    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(5), GRADE_NAMES, rotation=30, ha="right")
    ax.set_yticks(range(5), GRADE_NAMES)
    ax.set_xlabel("Predicted grade")
    ax.set_ylabel("Doctor's grade")
    for i in range(5):
        for j in range(5):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                    color="white" if cm[i, j] > cm.max() / 2 else "black")
    ax.set_title("Test set confusion matrix")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    for split in ("train", "val", "test"):
        p.add_argument(f"--{split}-csv", required=split != "val")
        p.add_argument(f"--{split}-dir", required=split != "val")
    p.add_argument("--id-col", default="id_code")
    p.add_argument("--label-col", default="diagnosis")
    p.add_argument("--out", default="results/grader", help="Folder for the model and metrics")
    p.add_argument("--cache", default="features_cache", help="Folder for cached features")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    os.makedirs(args.cache, exist_ok=True)
    extractor = FeatureExtractor()

    def split_data(name):
        ids, paths, y = load_split(getattr(args, f"{name}_csv"), getattr(args, f"{name}_dir"),
                                   args.id_col, args.label_col)
        X = cached_features(extractor, ids, paths, os.path.join(args.cache, f"{name}.npz"))
        return X, y

    X_train, y_train = split_data("train")
    if args.val_csv:
        X_val, y_val = split_data("val")
        X_train, y_train = np.vstack([X_train, X_val]), np.concatenate([y_train, y_val])
    X_test, y_test = split_data("test")

    print(f"\nChoosing regularisation by 5-fold cross-validation on {len(y_train)} images ...")
    folds = StratifiedKFold(5, shuffle=True, random_state=0)
    cv_scores = {}
    for c in C_GRID:
        proba = cross_val_predict(make_model(c), X_train, y_train, cv=folds, method="predict_proba")
        cv_scores[c] = cohen_kappa_score(y_train, proba.argmax(1), weights="quadratic")
        print(f"  C={c:<5} cross-validated kappa = {cv_scores[c]:.3f}")
    best_c = max(cv_scores, key=cv_scores.get)

    model = make_model(best_c).fit(X_train, y_train)
    metrics = evaluate(y_test, model.predict_proba(X_test))
    metrics["chosen_C"] = best_c
    metrics["cross_validated_kappa"] = round(float(cv_scores[best_c]), 3)
    metrics["n_training_images"] = int(len(y_train))

    joblib.dump({"model": model, "grade_names": GRADE_NAMES}, os.path.join(args.out, "grader.joblib"))
    with open(os.path.join(args.out, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    save_confusion_plot(metrics["confusion_matrix"], os.path.join(args.out, "confusion_matrix.png"))

    m = metrics
    print(f"\nTest set ({m['n_images']} images, grade counts {m['grade_counts']}):")
    print(f"  Quadratic weighted kappa : {m['quadratic_weighted_kappa']}")
    print(f"  Accuracy                 : {m['accuracy']}  "
          f"(always predicting the most common grade: {m['accuracy_always_majority_grade']})")
    for key, title in (("any_dr_vs_none", "Any DR vs none"), ("referable_dr_grade2plus", "Referable DR (2+)")):
        b = m[key]
        print(f"  {title:<25}: AUC {b['auc']}, sensitivity {b['sensitivity']}, specificity {b['specificity']}")
    print(f"  Recall per grade         : {m['recall_per_grade']}")
    print(f"\nSaved model and metrics to {args.out}/")


if __name__ == "__main__":
    main()

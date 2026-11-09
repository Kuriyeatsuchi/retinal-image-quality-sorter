"""
How much do the grader's results depend on the particular train/test split?

The headline numbers in the README come from one fixed split. This script
pools all labelled photos (train, validation and test) and runs repeated
stratified 5-fold cross-validation: every photo is used for testing exactly
once per repeat, by a model that never saw it. The regularisation strength is
chosen inside each training fold (nested cross-validation), so the held-out
photos never influence any choice.

It reuses the cached features written by grading/train_grader.py.

Usage:
    python experiments/repeated_cv.py --cache features_cache \
        --labels train.csv valid.csv test.csv [--repeats 5]
"""

import argparse
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, cohen_kappa_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "grading"))
from train_grader import C_GRID, make_model  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)


def choose_c(X, y, seed):
    folds = StratifiedKFold(3, shuffle=True, random_state=seed)
    scores = {}
    for c in C_GRID:
        proba = cross_val_predict(make_model(c), X, y, cv=folds, method="predict_proba")
        scores[c] = cohen_kappa_score(y, proba.argmax(1), weights="quadratic")
    return max(scores, key=scores.get)


def metrics(y, proba):
    pred = proba.argmax(1)
    return {
        "kappa": cohen_kappa_score(y, pred, weights="quadratic"),
        "accuracy": accuracy_score(y, pred),
        "any_dr_auc": roc_auc_score(y >= 1, 1 - proba[:, 0]),
        "referable_dr_auc": roc_auc_score(y >= 2, proba[:, 2:].sum(1)),
        "any_dr_sensitivity": np.mean((pred >= 1)[y >= 1]),
        "any_dr_specificity": np.mean((pred == 0)[y == 0]),
        "referable_dr_sensitivity": np.mean((pred >= 2)[y >= 2]),
        "referable_dr_specificity": np.mean((pred < 2)[y < 2]),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--id-col", default="id_code")
    p.add_argument("--label-col", default="diagnosis")
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()

    grades = {}
    for path in args.labels:
        t = pd.read_csv(path)
        grades.update(dict(zip(t[args.id_col].astype(str), t[args.label_col])))

    X, y = [], []
    for name in ("train", "val", "test"):
        path = os.path.join(args.cache, f"{name}.npz")
        if os.path.exists(path):
            z = np.load(path)
            X.append(z["features"])
            y.extend(grades[str(i)] for i in z["ids"])
    X, y = np.vstack(X), np.array(y)
    print(f"{len(y)} photos pooled, grade counts {np.bincount(y, minlength=5).tolist()}")
    print(f"{args.repeats} x 5-fold cross-validation, regularisation chosen inside each training fold\n")

    runs = []
    for repeat in range(args.repeats):
        proba = np.zeros((len(y), 5))
        chosen = []
        for train, test in StratifiedKFold(5, shuffle=True, random_state=repeat).split(X, y):
            c = choose_c(X[train], y[train], seed=repeat)
            chosen.append(c)
            proba[test] = make_model(c).fit(X[train], y[train]).predict_proba(X[test])
        m = metrics(y, proba)
        runs.append(m)
        print(f"repeat {repeat + 1}: kappa {m['kappa']:.3f}  any-DR AUC {m['any_dr_auc']:.3f}  "
              f"referable AUC {m['referable_dr_auc']:.3f}  accuracy {m['accuracy']:.3f}  "
              f"(C per fold: {chosen})", flush=True)

    print("\nAcross repeats (mean, and lowest-highest):")
    for key in runs[0]:
        v = np.array([r[key] for r in runs])
        print(f"  {key:<26} {v.mean():.3f}   ({v.min():.3f}-{v.max():.3f})")


if __name__ == "__main__":
    main()

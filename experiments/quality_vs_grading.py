"""
Does the quality sorter's decision relate to how well photos can be graded?

Uses the cached grader features (written by grading/train_grader.py) to get
an honest prediction for every image: out-of-fold (5-fold) predictions for
train+validation images, and the final model's predictions for test images.
These are then grouped by the sorter's decision (from a sort_images.py
--report CSV covering the same images).

Comparing groups with different amounts of disease is misleading (healthy
photos are easier to grade), so results are also shown for diseased photos
only.

Usage:
    python experiments/quality_vs_grading.py --cache features_cache \
        --labels train.csv valid.csv test.csv --sorter-report sorter_report.csv
"""

import argparse
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache", required=True)
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--sorter-report", required=True)
    p.add_argument("--id-col", default="id_code")
    p.add_argument("--label-col", default="diagnosis")
    p.add_argument("--C", type=float, default=0.3)
    args = p.parse_args()

    grades = {}
    for path in args.labels:
        t = pd.read_csv(path)
        grades.update(dict(zip(t[args.id_col].astype(str), t[args.label_col])))

    def load(name):
        z = np.load(os.path.join(args.cache, f"{name}.npz"))
        ids = [str(i) for i in z["ids"]]
        return ids, z["features"], np.array([grades[i] for i in ids])

    model = lambda: make_pipeline(StandardScaler(), LogisticRegression(  # noqa: E731
        C=args.C, class_weight="balanced", max_iter=5000))

    ids_tr, X_tr, y_tr = load("train")
    if os.path.exists(os.path.join(args.cache, "val.npz")):
        ids_va, X_va, y_va = load("val")
        ids_tr, X_tr, y_tr = ids_tr + ids_va, np.vstack([X_tr, X_va]), np.concatenate([y_tr, y_va])
    ids_te, X_te, y_te = load("test")

    oof = cross_val_predict(model(), X_tr, y_tr, method="predict_proba",
                            cv=StratifiedKFold(5, shuffle=True, random_state=0))
    test = model().fit(X_tr, y_tr).predict_proba(X_te)

    df = pd.DataFrame({"id": ids_tr + ids_te,
                       "grade": np.concatenate([y_tr, y_te]),
                       "predicted": np.vstack([oof, test]).argmax(1)})
    report = pd.read_csv(args.sorter_report)
    report["id"] = report["file"].str.rsplit(".", n=1).str[0]
    df = df.merge(report[["id", "reason", "diameter_px"]], on="id")

    def summary(group):
        exact = np.mean(group.grade == group.predicted)
        within1 = np.mean(abs(group.grade - group.predicted) <= 1)
        kappa = (cohen_kappa_score(group.grade, group.predicted, weights="quadratic")
                 if group.grade.nunique() > 1 else float("nan"))
        return f"n={len(group):<5} exact={exact:.3f}  within-one-grade={within1:.3f}  kappa={kappa:.3f}"

    df["resolution"] = pd.cut(df.diameter_px, [0, 700, 1000, 1500, 10_000],
                              labels=["<700 px", "700-1000 px", "1000-1500 px", ">1500 px"])
    for title, data in (("All photos", df), ("Diseased photos only (grade >= 1)", df[df.grade > 0])):
        print(f"\n{title}\n  by sorter decision:")
        for key, g in data.groupby("reason"):
            print(f"    {key:<22} {summary(g)}")
        print("  by retina diameter:")
        for key, g in data.groupby("resolution", observed=True):
            print(f"    {key:<22} {summary(g)}")


if __name__ == "__main__":
    main()

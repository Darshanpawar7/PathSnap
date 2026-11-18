"""Step 3 - train and compare candidate classifiers from different algorithm families.

Protocol
  * fitting data: 78 training trips (2 excluded, see config), every 3rd GPS point
  * cross-validation: 5-fold GroupKFold by trip (no trip in both fit and validation)
  * development comparison: all points of the 10 student_test trips
  * the evaluator partition is never touched

Outputs
  results/model_cv_folds.csv       per-model, per-fold CV metrics
  results/model_comparison.csv     consolidated table (CV + student_test)
  results/model_configs.json       configuration of every model
  results/figures/models_*.png     comparison plots, confusion matrices, loss curves
"""
import json
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import GroupKFold
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from _paths import FIGURES, PROCESSED, RESULTS, SEED, STUDENT_TEST_DIR
from pathsnap.config import EXCLUDED_TRAIN_TRIPS, TRAIN_POINT_STRIDE
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.io import load_trip
from pathsnap.metrics import candidate_metrics, route_continuity, top1_accuracy, top1_table

KNN_MAX_ROWS = 80_000  # KNN predicts by scanning stored rows: cap its memory for speed

MODELS = {
    "LogisticRegression": ("Linear", make_pipeline(
        StandardScaler(), LogisticRegression(C=1.0, max_iter=1000))),
    "GaussianNB": ("Probabilistic", make_pipeline(StandardScaler(), GaussianNB())),
    "KNN": ("Instance-based", make_pipeline(
        StandardScaler(), KNeighborsClassifier(n_neighbors=25, weights="distance", n_jobs=2))),
    "DecisionTree": ("Tree-based", DecisionTreeClassifier(
        max_depth=12, min_samples_leaf=50, random_state=SEED)),
    "RandomForest": ("Ensemble (bagging)", RandomForestClassifier(
        n_estimators=150, max_depth=16, min_samples_leaf=20, n_jobs=3, random_state=SEED)),
    "LightGBM": ("Boosting", LGBMClassifier(
        n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
        colsample_bytree=0.8, n_jobs=3, random_state=SEED, verbose=-1)),
    "MLP": ("Neural network", make_pipeline(StandardScaler(), MLPClassifier(
        hidden_layer_sizes=(64, 32), alpha=1e-4, batch_size=512, learning_rate_init=1e-3,
        max_iter=40, early_stopping=True, n_iter_no_change=5, random_state=SEED))),
}


def fit_rows(name: str, X: np.ndarray, y: np.ndarray, rng_seed: int):
    if name == "KNN" and len(X) > KNN_MAX_ROWS:
        idx = np.random.default_rng(rng_seed).choice(len(X), KNN_MAX_ROWS, replace=False)
        return X[idx], y[idx]
    return X, y


def run_fold(name, model, fold, tr, va, train_df, n_val_points):
    X, y = train_df[FEATURE_COLUMNS].to_numpy(), train_df.label.to_numpy()
    Xf, yf = fit_rows(name, X[tr], y[tr], SEED + fold)
    m = clone(model)
    t0 = time.time()
    m.fit(Xf, yf)
    fit_s = time.time() - t0
    proba = m.predict_proba(X[va])[:, 1]
    met = candidate_metrics(y[va], proba)
    top = top1_table(train_df.iloc[va], proba)
    met["top1"], met["top1_undirected"] = top1_accuracy(top, n_val_points)
    train_proba = m.predict_proba(Xf[:200_000])[:, 1]
    met["train_macro_f1"] = candidate_metrics(yf[:200_000], train_proba)["macro_f1"]
    met.update(model=name, fold=fold, fit_s=fit_s)
    return met


def main():
    cands = pd.read_pickle(PROCESSED / "candidates.pkl")
    points = pd.read_pickle(PROCESSED / "points.pkl")

    is_train = (cands.split == "train") & ~cands.trip_id.isin(EXCLUDED_TRAIN_TRIPS)
    train_df = cands[is_train & (cands.obs % TRAIN_POINT_STRIDE == 0)].reset_index(drop=True)
    test_df = cands[cands.split == "student_test"].reset_index(drop=True)
    tr_points = points[(points.split == "train") & ~points.trip_id.isin(EXCLUDED_TRAIN_TRIPS)
                       & (points.obs % TRAIN_POINT_STRIDE == 0)]
    te_points = points[points.split == "student_test"]
    print(f"fit rows={len(train_df):,} ({train_df.trip_id.nunique()} trips, {len(tr_points):,} points), "
          f"student_test rows={len(test_df):,} ({len(te_points):,} points)")

    test_nets = {}
    for tid in test_df.trip_id.unique():
        arcs = load_trip(STUDENT_TEST_DIR, tid, with_route=False).arcs
        test_nets[tid] = (arcs[:, 0], arcs[:, 1])

    gkf = GroupKFold(n_splits=5)
    splits = list(gkf.split(train_df, groups=train_df.trip_id))
    val_counts = [tr_points.trip_id.isin(train_df.trip_id.iloc[va].unique()).sum() for _, va in splits]

    fold_rows, summary, histories = [], [], {}
    X_tr, y_tr = train_df[FEATURE_COLUMNS].to_numpy(), train_df.label.to_numpy()
    X_te, y_te = test_df[FEATURE_COLUMNS].to_numpy(), test_df.label.to_numpy()
    fig_cm, axes_cm = plt.subplots(1, len(MODELS), figsize=(3.2 * len(MODELS), 3.2))

    for i, (name, (family, model)) in enumerate(MODELS.items()):
        t0 = time.time()
        folds = Parallel(n_jobs=5)(
            delayed(run_fold)(name, model, f, tr, va, train_df, val_counts[f])
            for f, (tr, va) in enumerate(splits)
        )
        fold_rows += folds
        cv = pd.DataFrame(folds)

        Xf, yf = fit_rows(name, X_tr, y_tr, SEED)
        final = clone(model)
        t1 = time.time()
        final.fit(Xf, yf)
        fit_s = time.time() - t1
        t1 = time.time()
        proba = final.predict_proba(X_te)[:, 1]
        infer_s = time.time() - t1

        te = candidate_metrics(y_te, proba)
        top = top1_table(test_df, proba)
        te_top1, te_top1_u = top1_accuracy(top, len(te_points))
        cont = np.mean([
            route_continuity(g.sort_values("obs").arc.to_numpy(), *test_nets[tid])
            for tid, g in top.groupby("trip_id")
        ])
        cm = confusion_matrix(y_te, (proba >= 0.5).astype(int))
        axes_cm[i].imshow(cm, cmap="Blues")
        for (r, c), v in np.ndenumerate(cm):
            axes_cm[i].text(c, r, f"{v:,}", ha="center", va="center", fontsize=8)
        axes_cm[i].set_title(name, fontsize=9)
        axes_cm[i].set_xticks([0, 1]); axes_cm[i].set_yticks([0, 1])
        axes_cm[i].set_xlabel("predicted"); axes_cm[i].set_ylabel("actual")

        est = final[-1] if hasattr(final, "steps") else final
        if name == "MLP":
            histories[name] = {"train_loss": list(est.loss_curve_),
                               "val_accuracy": list(est.validation_scores_)}

        summary.append({
            "model": name, "family": family,
            "cv_macro_f1": cv.macro_f1.mean(), "cv_macro_f1_std": cv.macro_f1.std(),
            "cv_top1": cv.top1.mean(), "cv_log_loss": cv.log_loss.mean(),
            "cv_train_macro_f1": cv.train_macro_f1.mean(),
            "test_accuracy": te["accuracy"], "test_precision": te["precision"], "test_recall": te["recall"],
            "test_f1": te["f1"], "test_macro_f1": te["macro_f1"], "test_log_loss": te["log_loss"],
            "test_top1": te_top1, "test_top1_undirected": te_top1_u, "test_route_continuity": cont,
            "fit_time_s": fit_s, "infer_time_s": infer_s,
            "infer_ms_per_1k_points": infer_s / len(te_points) * 1e6,
        })
        print(f"[{name:18s}] CV macroF1={cv.macro_f1.mean():.4f}±{cv.macro_f1.std():.4f} "
              f"CV top1={cv.top1.mean():.4f} | test macroF1={te['macro_f1']:.4f} top1={te_top1:.4f} "
              f"logloss={te['log_loss']:.4f} cont={cont:.4f} | {time.time() - t0:.0f}s", flush=True)

    fig_cm.tight_layout(); fig_cm.savefig(FIGURES / "models_confusion_matrices.png", dpi=130); plt.close(fig_cm)

    pd.DataFrame(fold_rows).to_csv(RESULTS / "model_cv_folds.csv", index=False)
    comp = pd.DataFrame(summary)
    comp.to_csv(RESULTS / "model_comparison.csv", index=False)
    configs = {n: {"family": f, "params": {k: str(v) for k, v in m.get_params().items()
                                           if "__" in k or not hasattr(m, "steps")}}
               for n, (f, m) in MODELS.items()}
    configs["_protocol"] = {"seed": SEED, "cv": "GroupKFold(5) by trip", "train_point_stride": TRAIN_POINT_STRIDE,
                            "excluded_train_trips": list(EXCLUDED_TRAIN_TRIPS), "knn_max_rows": KNN_MAX_ROWS,
                            "features": FEATURE_COLUMNS}
    (RESULTS / "model_configs.json").write_text(json.dumps(configs, indent=2), encoding="utf-8")
    (RESULTS / "model_histories.json").write_text(json.dumps(histories), encoding="utf-8")

    # ------------------------------------------------------------- plots
    folds_df = pd.DataFrame(fold_rows)
    order = comp.sort_values("test_top1", ascending=False).model.tolist()
    fig, ax = plt.subplots(figsize=(9, 4))
    data = [folds_df[folds_df.model == m].macro_f1 for m in order]
    ax.boxplot(data, tick_labels=order)
    ax.set_title("Cross-validation Macro F1 per fold (GroupKFold by trip, train)")
    ax.tick_params(axis="x", rotation=20)
    fig.tight_layout(); fig.savefig(FIGURES / "models_cv_macro_f1.png", dpi=130); plt.close(fig)

    c = comp.set_index("model").loc[order]
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    x = np.arange(len(order))
    axes[0].bar(x - 0.2, c.test_top1, 0.4, label="Top-1 accuracy")
    axes[0].bar(x + 0.2, c.test_macro_f1, 0.4, label="Macro F1")
    axes[0].set_ylim(0.5, 1); axes[0].set_title("student_test: Top-1 accuracy & Macro F1"); axes[0].legend()
    axes[1].bar(x, c.test_log_loss, color="#d73027"); axes[1].set_title("student_test: log loss (lower is better)")
    axes[2].bar(x, c.infer_ms_per_1k_points, color="#4575b4"); axes[2].set_yscale("log")
    axes[2].set_title("Inference time (ms per 1000 GPS points, log)")
    for a in axes:
        a.set_xticks(x); a.set_xticklabels(order, rotation=25, ha="right", fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "models_student_test_comparison.png", dpi=130); plt.close(fig)

    if "MLP" in histories:
        h = histories["MLP"]
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(h["train_loss"], label="training loss (log loss)")
        ax2 = ax.twinx()
        ax2.plot(h["val_accuracy"], color="#d73027", label="validation accuracy (10% hold-out)")
        ax.set_xlabel("epoch"); ax.set_ylabel("training loss"); ax2.set_ylabel("validation accuracy")
        ax.set_title("MLP training history (final fit)")
        fig.legend(loc="upper center", bbox_to_anchor=(0.5, 0.88))
        fig.tight_layout(); fig.savefig(FIGURES / "models_mlp_history.png", dpi=130); plt.close(fig)

    pd.set_option("display.width", 250)
    print(comp.round(4).to_string(index=False))


if __name__ == "__main__":
    main()

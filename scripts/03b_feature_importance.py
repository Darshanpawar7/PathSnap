"""Step 3b - which features does the selected model rely on?

1. LightGBM gain importance (same settings and fit data as 03_train_models.py)
2. Group ablation: refit without one feature group at a time and measure
   student_test Top-1. Analysis only - the feature set is not changed from it.

Outputs
  results/feature_importance.csv, results/feature_group_ablation.csv
  results/figures/feature_importance.png
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from lightgbm import LGBMClassifier

from _paths import FIGURES, PROCESSED, RESULTS, SEED
from pathsnap.config import EXCLUDED_TRAIN_TRIPS, TRAIN_POINT_STRIDE
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.metrics import top1_accuracy, top1_table

GROUPS = {
    "Distance": ["dist", "dist_sq", "rank", "dist_minus_min", "dist_ratio_min", "dist_z", "n_cands"],
    "Position on arc": ["proj_t", "beyond_end", "log_arc_len"],
    "Direction": ["heading_valid", "cos_dir", "abs_cos", "along_progress_m"],
    "Motion": ["speed_mps", "dt_prev", "dt_next", "turn_deg"],
    "Network": ["two_way", "to_out_degree", "from_in_degree"],
    "Neighbours": ["win2_seg_mean", "win5_line_mean", "nbr_cand_share"],
}


def model():
    return LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=63, subsample=0.8, subsample_freq=1,
                          colsample_bytree=0.8, n_jobs=8, random_state=SEED, verbose=-1)


def main():
    cands = pd.read_pickle(PROCESSED / "candidates.pkl")
    points = pd.read_pickle(PROCESSED / "points.pkl")
    is_train = (cands.split == "train") & ~cands.trip_id.isin(EXCLUDED_TRAIN_TRIPS)
    train_df = cands[is_train & (cands.obs % TRAIN_POINT_STRIDE == 0)].reset_index(drop=True)
    test_df = cands[cands.split == "student_test"].reset_index(drop=True)
    n_test_points = int((points.split == "student_test").sum())

    def fit_score(cols):
        m = model().fit(train_df[cols], train_df.label)
        top = top1_table(test_df, m.predict_proba(test_df[cols])[:, 1])
        return m, top1_accuracy(top, n_test_points)

    t0 = time.time()
    full, (top1, top1_u) = fit_score(FEATURE_COLUMNS)
    imp = pd.DataFrame({
        "feature": FEATURE_COLUMNS,
        "gain": full.booster_.feature_importance("gain"),
        "splits": full.booster_.feature_importance("split"),
    })
    imp["gain_share"] = imp.gain / imp.gain.sum()
    group_of = {f: g for g, fs in GROUPS.items() for f in fs}
    imp["group"] = imp.feature.map(group_of)
    imp = imp.sort_values("gain_share", ascending=False)
    imp.to_csv(RESULTS / "feature_importance.csv", index=False)

    rows = [{"setting": "all 24 features", "n_features": len(FEATURE_COLUMNS), "test_top1": top1, "test_top1_undirected": top1_u}]
    for g, fs in GROUPS.items():
        cols = [c for c in FEATURE_COLUMNS if c not in fs]
        _, (t, tu) = fit_score(cols)
        rows.append({"setting": f"without {g}", "n_features": len(cols), "test_top1": t, "test_top1_undirected": tu})
        print(f"without {g:16s} top1={t:.4f} (undirected {tu:.4f})", flush=True)
    abl = pd.DataFrame(rows)
    abl["drop_vs_all_points"] = (abl.test_top1.iloc[0] - abl.test_top1) * 100
    abl.to_csv(RESULTS / "feature_group_ablation.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    top = imp.head(15).iloc[::-1]
    axes[0].barh(top.feature, top.gain_share * 100, color="#2166ac")
    axes[0].set_title("LightGBM gain importance, top 15 features (% of total gain)")
    a = abl.iloc[1:].sort_values("drop_vs_all_points")
    axes[1].barh(a.setting, a.drop_vs_all_points, color="#d73027")
    axes[1].set_title("Top-1 lost on student_test when a feature group is removed (points)")
    fig.tight_layout(); fig.savefig(FIGURES / "feature_importance.png", dpi=130); plt.close(fig)

    print(imp.head(12).round(4).to_string(index=False))
    print(abl.round(4).to_string(index=False))
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

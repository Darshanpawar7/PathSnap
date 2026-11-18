"""Step 5 - HMM / Viterbi route reconstruction: decoder tuning and evaluation.

1. Out-of-fold (OOF) probabilities for every training point: each trip is scored
   by a tuned LightGBM that did not see that trip (5-fold GroupKFold by trip).
2. Decoder grid search (beta, emission weight) on those OOF training lattices only.
3. Student-test comparison: classifier only vs classifier + HMM, per trip and pooled.
4. Writes artifacts/pipeline_config.json (used by the API) and checks that the
   end-to-end MapMatcher on raw files reproduces the same predictions.

Outputs
  results/hmm_decoder_grid.csv, results/hmm_student_test_per_trip.csv,
  results/hmm_student_test_summary.csv, results/figures/hmm_*.png,
  artifacts/pipeline_config.json
"""
import json
import time
from datetime import date

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from lightgbm import LGBMClassifier
from sklearn.model_selection import GroupKFold

from _paths import ARTIFACTS, FIGURES, PROCESSED, RESULTS, SEED, STUDENT_TEST_DIR, TRAIN_DIR
from pathsnap import hmm as H
from pathsnap.config import CANDIDATE_K, CANDIDATE_RADIUS_M, EXCLUDED_TRAIN_TRIPS, TRAIN_POINT_STRIDE
from pathsnap.features import FEATURE_COLUMNS
from pathsnap.geo import LocalProjection
from pathsnap.io import load_trip
from pathsnap.metrics import route_continuity, route_set_scores
from pathsnap.network import RoadNetwork
from pathsnap.pipeline import CONFIG_FILE, MapMatcher

BETAS = [1, 2, 4, 8, 16, 32]
WEIGHTS = [0.5, 1.0, 2.0]


def trip_lattice(split_dir, trip_id, obs, arc, pos, proba):
    trip = load_trip(split_dir, trip_id, with_route=True)
    proj = LocalProjection.around(trip.track[:, :2])
    xy = proj.forward(trip.track[:, :2])
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    lat = H.build_lattice(net, xy, trip.track[:, 2], obs, arc, pos, proba)
    return trip_id, lat, len(xy)


def evaluate_trip(split_dir, trip_id, lat, n, ref_arc, beta, w):
    trip = load_trip(split_dir, trip_id, with_route=True)
    proj = LocalProjection.around(trip.track[:, :2])
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    clf = H.classifier_only(lat, n)
    hm, breaks = H.decode(lat, n, beta, w)
    row = {"trip_id": trip_id, "n_points": n, "breaks": breaks}
    for name, pred in (("classifier", clf), ("hmm", hm)):
        ok = pred >= 0
        rev = np.where(ok, net.reverse_arc[np.where(ok, pred, 0)], -1)
        route = np.asarray(H.reconstruct_route(net, pred), dtype=np.int64)
        rs = route_set_scores(route, trip.route, net.length)
        row.update({
            f"{name}_correct": int(np.sum(ok & (pred == ref_arc))),
            f"{name}_correct_undirected": int(np.sum(ok & ((pred == ref_arc) | (rev == ref_arc)))),
            f"{name}_continuity": route_continuity(pred, net.frm, net.to),
            f"{name}_route_precision": rs["route_precision"], f"{name}_route_recall": rs["route_recall"],
            f"{name}_route_f1": rs["route_f1"],
        })
    row["classifier_accuracy"] = row["classifier_correct"] / n
    row["hmm_accuracy"] = row["hmm_correct"] / n
    return row, hm


def main():
    t_start = time.time()
    best = json.loads((RESULTS / "bayes_opt_best_params.json").read_text())["best_params"]
    cands = pd.read_pickle(PROCESSED / "candidates.pkl")
    points = pd.read_pickle(PROCESSED / "points.pkl")
    feats = FEATURE_COLUMNS

    # ------------------------------------------------ 1. OOF probabilities on train
    tr_all = cands[(cands.split == "train") & ~cands.trip_id.isin(EXCLUDED_TRAIN_TRIPS)].reset_index(drop=True)
    oof = np.zeros(len(tr_all))
    trips = tr_all.trip_id.to_numpy()
    for f, (fit_idx, val_idx) in enumerate(GroupKFold(n_splits=5).split(tr_all, groups=trips)):
        fit_rows = tr_all.iloc[fit_idx]
        fit_rows = fit_rows[fit_rows.obs % TRAIN_POINT_STRIDE == 0]
        m = LGBMClassifier(**{"subsample_freq": 1, **best}, n_jobs=16, random_state=SEED, verbose=-1)
        m.fit(fit_rows[feats], fit_rows.label)
        oof[val_idx] = m.predict_proba(tr_all.iloc[val_idx][feats])[:, 1]
        print(f"OOF fold {f} done ({time.time() - t_start:.0f}s)", flush=True)
    tr_all["proba"] = oof

    jobs = [(TRAIN_DIR, tid, g.obs.to_numpy(), g.arc.to_numpy(), g.proj_t.to_numpy().astype(float), g.proba.to_numpy())
            for tid, g in tr_all.groupby("trip_id")]
    lattices = Parallel(n_jobs=8)(delayed(trip_lattice)(*j) for j in jobs)
    print(f"train lattices built ({time.time() - t_start:.0f}s)", flush=True)
    ref = {tid: g.sort_values("obs").ref_arc.to_numpy()
           for tid, g in points[points.split == "train"].groupby("trip_id")}

    # ------------------------------------------------ 2. decoder grid on OOF train
    total_pts = sum(n for _, _, n in lattices)
    clf_correct = sum(np.sum(H.classifier_only(lat, n) == ref[tid]) for tid, lat, n in lattices)
    grid = []
    for beta in BETAS:
        for w in WEIGHTS:
            correct = sum(np.sum(H.decode(lat, n, beta, w)[0] == ref[tid]) for tid, lat, n in lattices)
            grid.append({"beta": beta, "emission_weight": w, "train_oof_hmm_accuracy": correct / total_pts})
            print(f"beta={beta:>3} w={w}: {correct / total_pts:.4f}", flush=True)
    grid = pd.DataFrame(grid)
    grid["train_oof_classifier_accuracy"] = clf_correct / total_pts
    grid.to_csv(RESULTS / "hmm_decoder_grid.csv", index=False)
    b = grid.loc[grid.train_oof_hmm_accuracy.idxmax()]
    beta, w = float(b.beta), float(b.emission_weight)
    print(f"best decoder: beta={beta} w={w} train OOF HMM={b.train_oof_hmm_accuracy:.4f} "
          f"(classifier only {clf_correct / total_pts:.4f})")

    # ------------------------------------------------ 3. student-test comparison
    bundle = joblib.load(ARTIFACTS / "lightgbm_tuned.joblib")
    te = cands[cands.split == "student_test"].reset_index(drop=True)
    te["proba"] = bundle["model"].predict_proba(te[feats])[:, 1]
    jobs = [(STUDENT_TEST_DIR, tid, g.obs.to_numpy(), g.arc.to_numpy(), g.proj_t.to_numpy().astype(float), g.proba.to_numpy())
            for tid, g in te.groupby("trip_id")]
    te_lat = Parallel(n_jobs=8)(delayed(trip_lattice)(*j) for j in jobs)
    te_ref = {tid: g.sort_values("obs").ref_arc.to_numpy()
              for tid, g in points[points.split == "student_test"].groupby("trip_id")}
    rows, hmm_preds = [], {}
    for tid, lat, n in te_lat:
        r, hm = evaluate_trip(STUDENT_TEST_DIR, tid, lat, n, te_ref[tid], beta, w)
        rows.append(r); hmm_preds[tid] = hm
    per_trip = pd.DataFrame(rows)
    per_trip.to_csv(RESULTS / "hmm_student_test_per_trip.csv", index=False)
    N = per_trip.n_points.sum()
    summary = pd.DataFrame([{
        "method": name,
        "pointwise_accuracy": per_trip[f"{key}_correct"].sum() / N,
        "pointwise_accuracy_undirected": per_trip[f"{key}_correct_undirected"].sum() / N,
        "route_continuity_mean": per_trip[f"{key}_continuity"].mean(),
        "route_precision_mean": per_trip[f"{key}_route_precision"].mean(),
        "route_recall_mean": per_trip[f"{key}_route_recall"].mean(),
        "route_f1_mean": per_trip[f"{key}_route_f1"].mean(),
        "trips_improved_vs_classifier": int((per_trip.hmm_accuracy > per_trip.classifier_accuracy).sum()) if key == "hmm" else None,
    } for name, key in (("Classifier only (tuned LightGBM)", "classifier"), ("Classifier + HMM/Viterbi", "hmm"))])
    summary.to_csv(RESULTS / "hmm_student_test_summary.csv", index=False)

    # ------------------------------------------------ 4. pipeline config + end-to-end check
    config = {
        "model_name": "LightGBM (Bayesian-tuned)", "model_file": "lightgbm_tuned.joblib",
        "features": feats, "candidate_radius_m": CANDIDATE_RADIUS_M, "candidate_k": CANDIDATE_K,
        "hmm": {"beta": beta, "emission_weight": w, "break_gap_s": H.BREAK_GAP_S,
                "min_cutoff_m": H.MIN_CUTOFF_M, "max_cutoff_m": H.MAX_CUTOFF_M},
        "lightgbm_params": bundle["params"],
        "trained_on": "train partition (78 trips, every 3rd GPS point); decoder tuned on out-of-fold train predictions",
        "created": date.today().isoformat(),
    }
    (ARTIFACTS / CONFIG_FILE).write_text(json.dumps(config, indent=2), encoding="utf-8")
    matcher = MapMatcher.load(ARTIFACTS)
    e2e = []
    for tid in sorted(hmm_preds):
        trip = load_trip(STUDENT_TEST_DIR, tid, with_route=False)
        res = matcher.match(trip)
        e2e.append({"trip_id": tid, "same_as_eval": bool(np.array_equal(res.hmm_arcs, hmm_preds[tid])),
                    **{f"time_{k}_s": v for k, v in res.timings_s.items()}})
    e2e = pd.DataFrame(e2e)
    e2e.to_csv(RESULTS / "pipeline_end_to_end_timing.csv", index=False)

    # ------------------------------------------------ plots
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.8))
    piv = grid.pivot(index="emission_weight", columns="beta", values="train_oof_hmm_accuracy")
    im = axes[0].imshow(piv.values, cmap="viridis", aspect="auto")
    axes[0].set_xticks(range(len(piv.columns))); axes[0].set_xticklabels(piv.columns)
    axes[0].set_yticks(range(len(piv.index))); axes[0].set_yticklabels(piv.index)
    for (i, j), v in np.ndenumerate(piv.values):
        axes[0].text(j, i, f"{v:.3f}", ha="center", va="center", color="w", fontsize=8)
    axes[0].set_xlabel("beta (m)"); axes[0].set_ylabel("emission weight")
    axes[0].set_title("Decoder grid: HMM accuracy on out-of-fold train trips")
    fig.colorbar(im, ax=axes[0])
    pt = per_trip.sort_values("trip_id")
    x = np.arange(len(pt))
    axes[1].bar(x - 0.2, pt.classifier_accuracy * 100, 0.4, label="classifier only", color="#9e9ac8")
    axes[1].bar(x + 0.2, pt.hmm_accuracy * 100, 0.4, label="classifier + HMM", color="#f2a900")
    axes[1].set_xticks(x); axes[1].set_xticklabels([str(int(t)) for t in pt.trip_id])
    axes[1].set_xlabel("student_test trip"); axes[1].set_ylabel("pointwise accuracy (%)")
    axes[1].set_ylim(max(0, min(pt.classifier_accuracy.min(), pt.hmm_accuracy.min()) * 100 - 5), 100)
    axes[1].legend(); axes[1].set_title("Student-test accuracy per trip")
    fig.tight_layout(); fig.savefig(FIGURES / "hmm_decoder_and_per_trip.png", dpi=130); plt.close(fig)

    pd.set_option("display.width", 220)
    print(summary.round(4).to_string(index=False))
    print(per_trip[["trip_id", "n_points", "classifier_accuracy", "hmm_accuracy", "classifier_continuity",
                    "hmm_continuity", "hmm_route_f1", "breaks"]].round(4).to_string(index=False))
    print(e2e.round(2).to_string(index=False))
    print(f"done in {time.time() - t_start:.0f}s")


if __name__ == "__main__":
    main()

"""Step 2 - preprocessing, labelling, candidate generation and feature engineering.

For every trip (train + student_test):
  1. project GPS + network to metres (local tangent plane)
  2. align the reference route to GPS points -> one reference arc per point
  3. candidate-radius study (train only): how often is the reference arc
     among the K nearest arcs within R metres?  -> picks R, K
  4. generate candidates with the chosen R, K and compute features

Outputs
  data/processed/candidates.pkl   one row per (point, candidate arc)
  data/processed/points.pkl       one row per GPS point (label info)
  results/candidate_radius_study.csv  + figure
"""
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from _paths import FIGURES, PROCESSED, RESULTS, STUDENT_TEST_DIR, TRAIN_DIR
from pathsnap.candidates import generate_candidates
from pathsnap.config import CANDIDATE_K, CANDIDATE_RADIUS_M, EXCLUDED_TRAIN_TRIPS
from pathsnap.features import FEATURE_COLUMNS, candidate_features
from pathsnap.geo import LocalProjection
from pathsnap.io import list_trips, load_trip
from pathsnap.labels import align_route
from pathsnap.network import RoadNetwork

WIDE_RADIUS, WIDE_K = 100.0, 16


def process_trip(split: str, split_dir, trip_id: str):
    trip = load_trip(split_dir, trip_id)
    proj = LocalProjection.around(trip.track[:, :2])
    xy = proj.forward(trip.track[:, :2])
    t = trip.track[:, 2]
    net = RoadNetwork(trip.nodes, trip.arcs, proj)
    ref_arc, ref_dist, _ = align_route(net, xy, trip.route)

    # where does the reference arc rank among all nearby arcs?
    wide = generate_candidates(net, xy, WIDE_RADIUS, WIDE_K, fallback_factor=1.0)
    hit = wide.arc == ref_arc[wide.obs]
    true_rank = np.full(len(xy), 999)
    true_dist = np.full(len(xy), np.inf)
    true_rank[wide.obs[hit]] = wide.rank[hit]
    true_dist[wide.obs[hit]] = wide.dist[hit]

    c = generate_candidates(net, xy, CANDIDATE_RADIUS_M, CANDIDATE_K)
    feats = candidate_features(net, xy, t, c)
    feats["label"] = (c.arc == ref_arc[c.obs]).astype(np.int8)
    feats["two_way_match"] = (net.reverse_arc[c.arc] == ref_arc[c.obs]).astype(np.int8)
    feats.insert(0, "trip_id", trip_id)
    feats.insert(0, "split", split)

    points = pd.DataFrame({
        "split": split, "trip_id": trip_id, "obs": np.arange(len(xy)),
        "lon": trip.track[:, 0], "lat": trip.track[:, 1], "t": t,
        "ref_arc": ref_arc, "ref_dist": ref_dist, "true_rank": true_rank, "true_dist": true_dist,
        "n_cands": np.bincount(c.obs, minlength=len(xy)),
    })
    return feats, points


def radius_study(points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for r in (20, 30, 40, 50, 60, 80, 100):
        for k in (2, 4, 6, 8, 10, 12, 16):
            found = (points.true_dist <= r) & (points.true_rank < k)
            rows.append({"radius_m": r, "k": k, "candidate_miss_rate": 1 - found.mean()})
    return pd.DataFrame(rows)


def main():
    t0 = time.time()
    jobs = [("train", TRAIN_DIR, t) for t in list_trips(TRAIN_DIR)] + \
           [("student_test", STUDENT_TEST_DIR, t) for t in list_trips(STUDENT_TEST_DIR)]
    out = Parallel(n_jobs=8)(delayed(process_trip)(*j) for j in jobs)
    cands = pd.concat([o[0] for o in out], ignore_index=True)
    points = pd.concat([o[1] for o in out], ignore_index=True)
    cands.to_pickle(PROCESSED / "candidates.pkl")
    points.to_pickle(PROCESSED / "points.pkl")

    keep = ~points.trip_id.isin(EXCLUDED_TRAIN_TRIPS)
    tr_pts = points[(points.split == "train") & keep]
    study = radius_study(tr_pts)
    study.to_csv(RESULTS / "candidate_radius_study.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for r, sub in study.groupby("radius_m"):
        ax.plot(sub.k, sub.candidate_miss_rate * 100, marker="o", label=f"R = {r} m")
    ax.axvline(CANDIDATE_K, color="k", ls=":")
    ax.set_xlabel("K (max candidates per GPS point)")
    ax.set_ylabel("candidate miss rate (%)")
    ax.set_title("Reference arc missing from candidate set (train partition)")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(FIGURES / "candidate_radius_study.png", dpi=130); plt.close(fig)

    for split in ("train", "student_test"):
        c = cands[(cands.split == split) & ~cands.trip_id.isin(EXCLUDED_TRAIN_TRIPS)]
        p = points[(points.split == split) & keep]
        found = c.groupby(["trip_id", "obs"]).label.max().sum()
        print(f"{split}: points={len(p):,} candidate rows={len(c):,} "
              f"mean cands/point={len(c) / len(p):.2f} positive rate={c.label.mean():.3f} "
              f"candidate miss rate={1 - found / len(p):.4f}")
    print("chosen R=%.0f m, K=%d" % (CANDIDATE_RADIUS_M, CANDIDATE_K))
    print(study.pivot(index="radius_m", columns="k", values="candidate_miss_rate").round(4).to_string())
    print(f"features: {len(FEATURE_COLUMNS)} -> {FEATURE_COLUMNS}")
    print(f"done in {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()

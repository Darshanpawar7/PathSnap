"""Feature engineering for (GPS point, candidate arc) pairs.

The classifier answers: "is this candidate arc the road the vehicle was on at
this point?". Distance alone cannot separate parallel roads or the two
directions of a two-way street, so we add movement direction, speed and the
fit of neighbouring points to the same arc.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .candidates import CandidateSet
from .geo import bearing_deg, point_segment
from .network import RoadNetwork

HEADING_HALF_WINDOW = 2       # points each side used to estimate movement direction
MIN_MOVE_FOR_HEADING_M = 3.0  # below this the vehicle is ~stationary: heading unreliable

FEATURE_COLUMNS = [
    # distance to the road
    "dist", "dist_sq", "rank", "dist_minus_min", "dist_ratio_min", "dist_z", "n_cands",
    # where along the arc the point projects
    "proj_t", "beyond_end", "log_arc_len",
    # movement direction vs road direction
    "heading_valid", "cos_dir", "abs_cos", "along_progress_m",
    # vehicle dynamics / sampling
    "speed_mps", "dt_prev", "dt_next", "turn_deg",
    # road-network context
    "two_way", "to_out_degree", "from_in_degree",
    # neighbouring points (temporal context)
    "win2_seg_mean", "win5_line_mean", "nbr_cand_share",
]


def _shift_idx(n: int, offset: int) -> np.ndarray:
    return np.clip(np.arange(n) + offset, 0, n - 1)


def observation_features(xy: np.ndarray, t: np.ndarray) -> pd.DataFrame:
    """Per-GPS-point movement features."""
    n = len(xy)
    h = HEADING_HALF_WINDOW
    i0, i1 = _shift_idx(n, -h), _shift_idx(n, h)
    dx, dy = xy[i1, 0] - xy[i0, 0], xy[i1, 1] - xy[i0, 1]
    move = np.hypot(dx, dy)
    dt_win = np.maximum(t[i1] - t[i0], 1e-6)
    heading = bearing_deg(dx, dy)

    # turn angle between the incoming and outgoing direction at each point
    j0, j1 = _shift_idx(n, -3), _shift_idx(n, 3)
    b_in = bearing_deg(xy[:, 0] - xy[j0, 0], xy[:, 1] - xy[j0, 1])
    b_out = bearing_deg(xy[j1, 0] - xy[:, 0], xy[j1, 1] - xy[:, 1])
    turn = np.abs((b_out - b_in + 180) % 360 - 180)

    dt_prev = np.r_[1.0, np.diff(t)]
    dt_next = np.r_[np.diff(t), 1.0]
    return pd.DataFrame({
        "heading": heading, "move": move, "dx": dx, "dy": dy,
        "heading_valid": (move >= MIN_MOVE_FOR_HEADING_M).astype(np.float32),
        "speed_mps": move / dt_win,
        "dt_prev": np.clip(dt_prev, 0, 600), "dt_next": np.clip(dt_next, 0, 600),
        "turn_deg": turn,
    })


def candidate_features(net: RoadNetwork, xy: np.ndarray, t: np.ndarray, c: CandidateSet) -> pd.DataFrame:
    n = len(xy)
    obs, arc = c.obs, c.arc
    of = observation_features(xy, t)

    # ---- distance-based, relative to the other candidates of the same point
    df = pd.DataFrame({"obs": obs, "arc": arc, "dist": c.dist, "rank": c.rank, "proj_t": c.t})
    g = df.groupby("obs")["dist"]
    dmin = g.transform("min").to_numpy()
    dmean = g.transform("mean").to_numpy()
    dstd = g.transform("std").fillna(0).to_numpy()
    df["dist_sq"] = c.dist ** 2
    df["dist_minus_min"] = c.dist - dmin
    df["dist_ratio_min"] = (c.dist + 1.0) / (dmin + 1.0)
    df["dist_z"] = (c.dist - dmean) / (dstd + 1.0)
    df["n_cands"] = g.transform("size").to_numpy()
    df["beyond_end"] = ((c.t <= 0.0) | (c.t >= 1.0)).astype(np.float32)
    df["log_arc_len"] = np.log1p(net.length[arc])

    # ---- direction agreement between movement and the (directed) arc
    hv = of["heading_valid"].to_numpy()[obs]
    diff = np.radians(of["heading"].to_numpy()[obs] - net.bearing[arc])
    df["heading_valid"] = hv
    df["cos_dir"] = np.where(hv > 0, np.cos(diff), 0.0)
    df["abs_cos"] = np.where(hv > 0, np.abs(np.cos(diff)), 0.0)
    ux = (net.bx[arc] - net.ax[arc]) / np.maximum(net.length[arc], 1e-6)
    uy = (net.by[arc] - net.ay[arc]) / np.maximum(net.length[arc], 1e-6)
    df["along_progress_m"] = of["dx"].to_numpy()[obs] * ux + of["dy"].to_numpy()[obs] * uy

    for col in ("speed_mps", "dt_prev", "dt_next", "turn_deg"):
        df[col] = of[col].to_numpy()[obs]

    # ---- network context
    df["two_way"] = (net.reverse_arc[arc] >= 0).astype(np.float32)
    df["to_out_degree"] = net.out_degree[net.to[arc]]
    df["from_in_degree"] = net.in_degree[net.frm[arc]]

    # ---- temporal context: do the neighbouring points also fit this arc?
    seg_sum = np.zeros(len(df))
    for o in (-2, -1, 1, 2):
        j = _shift_idx(n, o)[obs]
        d, _ = point_segment(xy[j, 0], xy[j, 1], net.ax[arc], net.ay[arc], net.bx[arc], net.by[arc])
        seg_sum += d
    df["win2_seg_mean"] = (seg_sum + c.dist) / 5.0

    line_sum = np.zeros(len(df))
    nx, ny = -uy, ux  # unit normal of the arc
    for o in range(-5, 6):
        j = _shift_idx(n, o)[obs]
        line_sum += np.abs((xy[j, 0] - net.ax[arc]) * nx + (xy[j, 1] - net.ay[arc]) * ny)
    df["win5_line_mean"] = line_sum / 11.0

    # share of the 4 neighbouring points that also have this arc as a candidate
    key = obs.astype(np.int64) * (net.n_arcs + 1) + arc
    key_sorted = np.sort(key)
    share = np.zeros(len(df))
    for o in (-2, -1, 1, 2):
        j = _shift_idx(n, o)[obs]
        k2 = j.astype(np.int64) * (net.n_arcs + 1) + arc
        pos = np.clip(np.searchsorted(key_sorted, k2), 0, len(key_sorted) - 1)
        share += key_sorted[pos] == k2
    df["nbr_cand_share"] = share / 4.0

    for col in FEATURE_COLUMNS:
        df[col] = df[col].astype(np.float32)
    return df

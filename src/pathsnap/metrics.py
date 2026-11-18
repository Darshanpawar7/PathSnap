"""Evaluation metrics shared by model comparison, HMM evaluation and the API."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score, f1_score, log_loss, precision_score, recall_score,
)


def candidate_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float = 0.5) -> dict:
    """Binary 'is this candidate the reference arc?' metrics."""
    y_pred = (proba >= threshold).astype(int)
    p = np.clip(proba, 1e-7, 1 - 1e-7)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "log_loss": log_loss(y_true, p, labels=[0, 1]),
    }


def top1_table(cands: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    """Best-scoring candidate per GPS point: columns trip_id, obs, arc, label, two_way_match."""
    df = cands[["trip_id", "obs", "arc", "label", "two_way_match"]].copy()
    df["score"] = score
    idx = df.groupby(["trip_id", "obs"], sort=False)["score"].idxmax()
    return df.loc[idx].sort_values(["trip_id", "obs"]).reset_index(drop=True)


def top1_accuracy(top: pd.DataFrame, n_points: int) -> tuple[float, float]:
    """(exact arc, arc or its reverse) accuracy over all n_points GPS points.

    Points without any candidate count as wrong, so the candidate miss rate
    is included in this number.
    """
    exact = top.label.sum() / n_points
    undirected = (top.label | top.two_way_match).sum() / n_points
    return float(exact), float(undirected)


def route_continuity(pred_arcs: np.ndarray, arc_from: np.ndarray, arc_to: np.ndarray) -> float:
    """Share of consecutive point pairs whose predicted arcs are the same arc
    or directly connected (end node of one = start node of the next)."""
    pred_arcs = np.asarray(pred_arcs)
    ok = pred_arcs >= 0
    a, b = pred_arcs[:-1], pred_arcs[1:]
    valid = ok[:-1] & ok[1:]
    if valid.sum() == 0:
        return float("nan")
    a, b = a[valid], b[valid]
    good = (a == b) | (arc_to[a] == arc_from[b])
    return float(good.mean())


def route_set_scores(pred_arcs: np.ndarray, ref_route: np.ndarray, arc_len: np.ndarray) -> dict:
    """Sequence-level overlap between predicted and reference arcs (length weighted)."""
    p = np.unique(pred_arcs[pred_arcs >= 0])
    r = np.unique(ref_route)
    inter = np.intersect1d(p, r)
    lp, lr, li = arc_len[p].sum(), arc_len[r].sum(), arc_len[inter].sum()
    prec = li / lp if lp > 0 else 0.0
    rec = li / lr if lr > 0 else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    return {"route_precision": float(prec), "route_recall": float(rec), "route_f1": float(f1)}

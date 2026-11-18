"""Candidate generation: for every GPS point, the K nearest arcs within a radius."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geo import point_segment
from .network import RoadNetwork


@dataclass(frozen=True)
class CandidateSet:
    obs: np.ndarray    # observation (GPS point) index of each candidate row
    arc: np.ndarray    # arc id
    dist: np.ndarray   # perpendicular distance point -> arc (m)
    t: np.ndarray      # projection position along the arc, 0..1
    rank: np.ndarray   # 0 = nearest candidate of that point
    n_obs: int

    def __len__(self) -> int:
        return len(self.obs)


def generate_candidates(
    net: RoadNetwork, xy: np.ndarray, radius: float, k: int, fallback_factor: float = 4.0
) -> CandidateSet:
    pi, ai = net.arcs_near(xy, radius)
    # Points with nothing in range get a second, wider search so a GPS outlier
    # does not leave a hole in the sequence.
    missing = np.setdiff1d(np.arange(len(xy)), pi)
    if len(missing):
        pj, aj = net.arcs_near(xy[missing], radius * fallback_factor)
        pi = np.concatenate([pi, missing[pj]])
        ai = np.concatenate([ai, aj])

    d, t = point_segment(xy[pi, 0], xy[pi, 1], net.ax[ai], net.ay[ai], net.bx[ai], net.by[ai])
    order = np.lexsort((ai, d, pi))
    pi, ai, d, t = pi[order], ai[order], d[order], t[order]
    starts = np.searchsorted(pi, pi, side="left")
    rank = np.arange(len(pi)) - starts
    keep = rank < k
    return CandidateSet(pi[keep], ai[keep], d[keep], t[keep], rank[keep], len(xy))

"""HMM over candidate arcs, decoded with Viterbi.

Hidden state at GPS point i  = which of its candidate arcs the vehicle is on.
Emission                     = classifier probability of that candidate,
                               normalised over the point's candidates.
Transition a (point i) -> b (point i+1)
                             = how well the network distance from a to b
                               matches the straight-line distance between
                               the two GPS points (Newson & Krumm style):
                               log P = -|d_route - d_gps| / beta
                               b unreachable from a within the cutoff -> impossible.
Breaks                       = a time gap longer than BREAK_GAP_S, or a step
                               where no transition is possible, restarts the
                               chain (uniform transition) instead of failing.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import RoadNetwork

BREAK_GAP_S = 120.0      # longer gaps: do not trust network distance, restart the chain
MIN_CUTOFF_M = 300.0     # Dijkstra search limit = max(MIN_CUTOFF_M, 3 * gps step + 200 m)
MAX_CUTOFF_M = 8000.0
CUTOFF_STEP_M = 250.0    # cutoffs are rounded up to this step so cached searches get reused


@dataclass
class Lattice:
    obs: np.ndarray              # GPS point index of each step (points with >= 1 candidate)
    arcs: list                   # per step: candidate arc ids
    pos: list                    # per step: projection position t (0..1) of the point on each arc
    emis: list                   # per step: log of normalised classifier probability
    disc: list                   # per transition: |d_route - d_gps| matrix (inf = impossible) or None = break
    gaps: np.ndarray             # per transition: seconds between the two points


def _cutoff(d_gps: float) -> float:
    c = max(MIN_CUTOFF_M, 3.0 * d_gps + 200.0)
    return min(MAX_CUTOFF_M, np.ceil(c / CUTOFF_STEP_M) * CUTOFF_STEP_M)


def build_lattice(net: RoadNetwork, xy: np.ndarray, t: np.ndarray,
                  obs: np.ndarray, arc: np.ndarray, pos: np.ndarray, proba: np.ndarray) -> Lattice:
    """obs/arc/pos/proba: one entry per candidate row, sorted by obs."""
    order = np.argsort(obs, kind="stable")
    obs, arc, pos, proba = obs[order], arc[order], pos[order], proba[order]
    steps, starts = np.unique(obs, return_index=True)
    ends = np.r_[starts[1:], len(obs)]

    arcs, poss, emis = [], [], []
    for s, e in zip(starts, ends):
        p = np.clip(proba[s:e], 1e-6, None)
        arcs.append(arc[s:e])
        poss.append(pos[s:e])
        emis.append(np.log(p / p.sum()))

    disc, gaps = [], np.zeros(max(len(steps) - 1, 0))
    for k in range(len(steps) - 1):
        i, j = steps[k], steps[k + 1]
        gaps[k] = t[j] - t[i]
        if gaps[k] > BREAK_GAP_S:
            disc.append(None)
            continue
        d_gps = float(np.hypot(*(xy[j] - xy[i])))
        cutoff = _cutoff(d_gps)
        A, B = arcs[k], arcs[k + 1]
        ta, tb = poss[k], poss[k + 1]
        len_a, len_b = net.length[A], net.length[B]
        from_b = net.frm[B]
        D = np.full((len(A), len(B)), np.inf)
        for ai, a in enumerate(A):
            reach = net.dists_from(int(net.to[a]), cutoff)
            sp = np.fromiter((reach.get(int(n), np.inf) for n in from_b), float, len(B))
            d_route = (1.0 - ta[ai]) * len_a[ai] + sp + tb * len_b
            same = B == a
            if same.any():
                d_route[same] = np.abs(tb[same] - ta[ai]) * len_a[ai]
            D[ai] = np.abs(d_route - d_gps)
        disc.append(D)
    return Lattice(steps, arcs, poss, emis, disc, gaps)


def viterbi(lat: Lattice, beta: float, emission_weight: float) -> tuple[np.ndarray, int]:
    """Most likely candidate index per step. Returns (choice per step, number of chain breaks)."""
    n = len(lat.obs)
    if n == 0:
        return np.empty(0, dtype=np.int64), 0
    score = emission_weight * lat.emis[0]
    back, breaks = [], 0
    for k in range(n - 1):
        D = lat.disc[k]
        if D is None:
            trans = np.zeros((len(score), len(lat.emis[k + 1])))
            breaks += 1
        else:
            trans = -D / beta
            if not np.isfinite(trans).any():
                trans = np.zeros_like(trans)
                breaks += 1
        total = score[:, None] + trans
        best_prev = np.argmax(total, axis=0)
        new = total[best_prev, np.arange(total.shape[1])]
        if not np.isfinite(new).any():
            # nothing at this point is reachable from any surviving state: restart the chain here
            best_prev = np.full(total.shape[1], int(np.argmax(score)))
            new = np.zeros(total.shape[1])
            breaks += 1
        score = new + emission_weight * lat.emis[k + 1]
        score = score - score.max()  # keep numbers small
        back.append(best_prev)
    choice = np.empty(n, dtype=np.int64)
    choice[-1] = int(np.argmax(score))
    for k in range(n - 2, -1, -1):
        choice[k] = back[k][choice[k + 1]]
    return choice, breaks


def decode(lat: Lattice, n_points: int, beta: float, emission_weight: float) -> tuple[np.ndarray, int]:
    """Arc per GPS point (-1 where a point had no candidates) and the break count."""
    choice, breaks = viterbi(lat, beta, emission_weight)
    out = np.full(n_points, -1, dtype=np.int64)
    for k, c in enumerate(choice):
        out[lat.obs[k]] = lat.arcs[k][c]
    return out, breaks


def classifier_only(lat: Lattice, n_points: int) -> np.ndarray:
    out = np.full(n_points, -1, dtype=np.int64)
    for k in range(len(lat.obs)):
        out[lat.obs[k]] = lat.arcs[k][int(np.argmax(lat.emis[k]))]
    return out


def reconstruct_route(net: RoadNetwork, point_arcs: np.ndarray, max_fill_m: float = 3000.0) -> list[int]:
    """Ordered, de-duplicated arc list; holes between consecutive matched arcs
    are filled with the shortest network path when one exists."""
    seq = [int(a) for a in point_arcs if a >= 0]
    route: list[int] = []
    for a in seq:
        if route and route[-1] == a:
            continue
        if route:
            prev = route[-1]
            if net.to[prev] != net.frm[a]:
                nodes = net.shortest_node_path(int(net.to[prev]), int(net.frm[a]), max_fill_m)
                if nodes:
                    route.extend(net.arc_id(u, v) for u, v in zip(nodes[:-1], nodes[1:]))
        route.append(a)
    return [a for a in route if a >= 0]

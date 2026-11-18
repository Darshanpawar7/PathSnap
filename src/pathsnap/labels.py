"""Turning the reference route into one reference arc per GPS point.

The .route file is an ordered list of arcs with no link to individual GPS
points. We align the two with a monotone dynamic programme: point i is given
route position j(i) such that j never goes backwards and the summed
point-to-arc distance is minimal. Going forward only matters because routes
can pass the same place twice (loops, out-and-back trips).
"""
from __future__ import annotations

import numpy as np

from .geo import point_segment
from .network import RoadNetwork


def align_route(net: RoadNetwork, xy: np.ndarray, route: np.ndarray, chunk: int = 2000):
    """Return (ref_arc per point, distance to it, route position per point)."""
    n, r = len(xy), len(route)
    if n == 0 or r == 0:
        return np.full(n, -1), np.full(n, np.nan), np.full(n, -1)
    ax, ay = net.ax[route], net.ay[route]
    bx, by = net.bx[route], net.by[route]

    back = np.empty((n, r), dtype=np.int32)
    prev = None
    dist_rows = np.empty((n, r), dtype=np.float32)
    for s in range(0, n, chunk):
        e = min(n, s + chunk)
        d, _ = point_segment(xy[s:e, 0:1], xy[s:e, 1:2], ax, ay, bx, by)
        dist_rows[s:e] = d
    for i in range(n):
        d = dist_rows[i]
        if prev is None:
            cost = d.astype(np.float64)
            back[i] = np.arange(r)
        else:
            # best previous position <= j  (prefix minimum with arg)
            pm = np.minimum.accumulate(prev)
            idx = np.arange(r)
            is_new = np.r_[True, prev[1:] < pm[:-1]]
            arg = np.maximum.accumulate(np.where(is_new, idx, 0))
            cost = pm + d
            back[i] = arg
        prev = cost
    pos = np.empty(n, dtype=np.int64)
    pos[-1] = int(np.argmin(prev))
    for i in range(n - 1, 0, -1):
        pos[i - 1] = back[i, pos[i]]
    ref = route[pos]
    return ref, dist_rows[np.arange(n), pos].astype(np.float64), pos

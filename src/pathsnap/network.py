"""Road network: projected geometry, spatial lookup and bounded shortest paths."""
from __future__ import annotations

import heapq

import numpy as np
import shapely

from .geo import LocalProjection, bearing_deg


class RoadNetwork:
    """Directed road graph of one trip, in local metric coordinates.

    Every arc is a straight segment between two nodes, so arc geometry is just
    its two end points. Arc ids are the 0-based row index of the .arcs file.
    """

    def __init__(self, nodes_lonlat: np.ndarray, arcs: np.ndarray, proj: LocalProjection):
        self.proj = proj
        self.nodes_lonlat = nodes_lonlat
        self.node_xy = proj.forward(nodes_lonlat) if len(nodes_lonlat) else np.empty((0, 2))
        self.arcs = arcs.astype(np.int64)
        self.n_nodes = len(nodes_lonlat)
        self.n_arcs = len(arcs)

        frm, to = self.arcs[:, 0], self.arcs[:, 1]
        self.frm, self.to = frm, to
        self.ax, self.ay = self.node_xy[frm, 0], self.node_xy[frm, 1]
        self.bx, self.by = self.node_xy[to, 0], self.node_xy[to, 1]
        self.length = np.hypot(self.bx - self.ax, self.by - self.ay)
        self.bearing = bearing_deg(self.bx - self.ax, self.by - self.ay)

        self.out_degree = np.bincount(frm, minlength=self.n_nodes)
        self.in_degree = np.bincount(to, minlength=self.n_nodes)
        self.reverse_arc = self._reverse_lookup()

        order = np.argsort(frm, kind="stable")
        self._indptr = np.searchsorted(frm[order], np.arange(self.n_nodes + 1)).tolist()
        self._adj_to = to[order].tolist()
        self._adj_w = self.length[order].tolist()
        self._tree: shapely.STRtree | None = None
        self._sp_cache: dict[int, tuple[float, dict[int, float]]] = {}

    # ------------------------------------------------------------------ lookup
    def _reverse_lookup(self) -> np.ndarray:
        n = max(self.n_nodes, 1)
        key = self.frm * n + self.to
        rkey = self.to * n + self.frm
        order = np.argsort(key)
        pos = np.searchsorted(key[order], rkey)
        pos = np.clip(pos, 0, len(key) - 1)
        found = key[order][pos] == rkey
        rev = np.full(self.n_arcs, -1, dtype=np.int64)
        rev[found] = order[pos[found]]
        return rev

    def _index(self) -> shapely.STRtree:
        if self._tree is None:
            coords = np.stack(
                [np.column_stack([self.ax, self.ay]), np.column_stack([self.bx, self.by])], axis=1
            )
            self._tree = shapely.STRtree(shapely.linestrings(coords))
        return self._tree

    def arcs_near(self, xy: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
        """All (point_index, arc_id) pairs with the arc within `radius` metres."""
        if len(xy) == 0 or self.n_arcs == 0:
            return np.empty(0, np.int64), np.empty(0, np.int64)
        pts = shapely.points(xy)
        pi, ai = self._index().query(pts, predicate="dwithin", distance=radius)
        return pi.astype(np.int64), ai.astype(np.int64)

    # ----------------------------------------------------------- shortest path
    def dists_from(self, src: int, cutoff: float) -> dict[int, float]:
        """Network distances (m) from node `src` to all nodes within `cutoff`."""
        cached = self._sp_cache.get(src)
        if cached is not None and cached[0] >= cutoff:
            return cached[1]
        dist = {src: 0.0}
        heap = [(0.0, src)]
        indptr, adj_to, adj_w = self._indptr, self._adj_to, self._adj_w
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, np.inf):
                continue
            for k in range(indptr[u], indptr[u + 1]):
                v = adj_to[k]
                nd = d + adj_w[k]
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    heapq.heappush(heap, (nd, v))
        self._sp_cache[src] = (cutoff, dist)
        return dist

    def shortest_node_path(self, src: int, dst: int, cutoff: float) -> list[int] | None:
        """Node sequence of the shortest path src -> dst (None if beyond cutoff)."""
        if src == dst:
            return [src]
        dist = {src: 0.0}
        prev: dict[int, int] = {}
        heap = [(0.0, src)]
        indptr, adj_to, adj_w = self._indptr, self._adj_to, self._adj_w
        while heap:
            d, u = heapq.heappop(heap)
            if u == dst:
                break
            if d > dist.get(u, np.inf):
                continue
            for k in range(indptr[u], indptr[u + 1]):
                v = adj_to[k]
                nd = d + adj_w[k]
                if nd <= cutoff and nd < dist.get(v, np.inf):
                    dist[v] = nd
                    prev[v] = u
                    heapq.heappush(heap, (nd, v))
        if dst not in dist:
            return None
        path = [dst]
        while path[-1] != src:
            path.append(prev[path[-1]])
        return path[::-1]

    def arc_between(self, u: int, v: int) -> int:
        """Shortest arc id going u -> v, or -1."""
        best, best_len = -1, np.inf
        for k in range(self._indptr[u], self._indptr[u + 1]):
            if self._adj_to[k] == v and self._adj_w[k] < best_len:
                best_len = self._adj_w[k]
        if best_len == np.inf:
            return -1
        cand = np.flatnonzero((self.frm == u) & (self.to == v))
        if len(cand):
            best = int(cand[np.argmin(self.length[cand])])
        return best

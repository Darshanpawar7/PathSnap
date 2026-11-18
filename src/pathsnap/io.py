"""Reading the raw trip files.

Each trip lives in its own folder <split>/<trip_id>/ with four headerless,
whitespace-separated files:
  .track  lon lat t        GPS observations (t in seconds)
  .nodes  lon lat          road-network nodes, row index = node id
  .arcs   from to          directed road segments, row index = arc id
  .route  arc_id           reference route (ordered arc ids), server-side only
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Trip:
    trip_id: str
    track: np.ndarray            # (N, 3) float64: lon, lat, t
    nodes: np.ndarray            # (M, 2) float64: lon, lat
    arcs: np.ndarray             # (A, 2) int64: from_node, to_node
    route: np.ndarray | None     # (R,) int64 arc ids, or None when not loaded


def _read_table(path: Path, ncols: int, dtype) -> np.ndarray:
    if not path.exists() or path.stat().st_size == 0:
        return np.empty((0, ncols), dtype=dtype)
    df = pd.read_csv(path, sep=r"\s+", header=None, engine="c")
    if df.shape[1] != ncols:
        raise ValueError(f"{path.name}: expected {ncols} columns, found {df.shape[1]}")
    return df.to_numpy(dtype=dtype)


def list_trips(split_dir: str | Path) -> list[str]:
    split_dir = Path(split_dir)
    if not split_dir.exists():
        return []
    return sorted(p.name for p in split_dir.iterdir() if p.is_dir())


def load_trip(split_dir: str | Path, trip_id: str, with_route: bool = True) -> Trip:
    folder = Path(split_dir) / trip_id
    if not folder.is_dir():
        raise FileNotFoundError(f"trip folder not found: {folder}")
    track = _read_table(folder / f"{trip_id}.track", 3, np.float64)
    nodes = _read_table(folder / f"{trip_id}.nodes", 2, np.float64)
    arcs = _read_table(folder / f"{trip_id}.arcs", 2, np.int64)
    route = None
    if with_route:
        r = _read_table(folder / f"{trip_id}.route", 1, np.int64)
        route = r[:, 0] if r.size else np.empty(0, dtype=np.int64)
    return Trip(trip_id, track, nodes, arcs, route)


def load_route(split_dir: str | Path, trip_id: str) -> np.ndarray:
    """Reference route only (used server-side for scoring, never for prediction)."""
    r = _read_table(Path(split_dir) / trip_id / f"{trip_id}.route", 1, np.int64)
    return r[:, 0] if r.size else np.empty(0, dtype=np.int64)

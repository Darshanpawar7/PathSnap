"""Geographic helpers.

Longitude/latitude are angles, so we never measure distance in degrees.
Each trip covers at most a few tens of kilometres, so we project it onto a
local tangent plane (equirectangular projection centred on the trip). Over
that extent the distance error is well under 1 %, which is far smaller than
GPS noise, and it is fast and dependency-free. Haversine is kept for checks.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EARTH_RADIUS_M = 6_371_008.8


@dataclass(frozen=True)
class LocalProjection:
    lon0: float
    lat0: float

    @classmethod
    def around(cls, lonlat: np.ndarray) -> "LocalProjection":
        lonlat = np.asarray(lonlat, dtype=float)
        return cls(float(np.nanmean(lonlat[:, 0])), float(np.nanmean(lonlat[:, 1])))

    def forward(self, lonlat: np.ndarray) -> np.ndarray:
        lonlat = np.asarray(lonlat, dtype=float)
        k = np.pi / 180.0
        x = EARTH_RADIUS_M * (lonlat[:, 0] - self.lon0) * k * np.cos(self.lat0 * k)
        y = EARTH_RADIUS_M * (lonlat[:, 1] - self.lat0) * k
        return np.column_stack([x, y])

    def inverse(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        k = np.pi / 180.0
        lon = xy[:, 0] / (EARTH_RADIUS_M * k * np.cos(self.lat0 * k)) + self.lon0
        lat = xy[:, 1] / (EARTH_RADIUS_M * k) + self.lat0
        return np.column_stack([lon, lat])


def haversine_m(lon1, lat1, lon2, lat2) -> np.ndarray:
    lon1, lat1, lon2, lat2 = map(np.radians, (lon1, lat1, lon2, lat2))
    a = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_M * np.arcsin(np.sqrt(a))


def bearing_deg(dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    """Compass bearing (0 = north, clockwise) of a planar displacement."""
    return (np.degrees(np.arctan2(dx, dy)) + 360.0) % 360.0


def point_segment(px, py, ax, ay, bx, by):
    """Distance from points P to segments AB (all arrays broadcast together).

    Returns (distance, t) where t in [0, 1] is the projection position along AB.
    """
    vx, vy = bx - ax, by - ay
    wx, wy = px - ax, py - ay
    seg2 = vx * vx + vy * vy
    with np.errstate(invalid="ignore", divide="ignore"):
        t = np.where(seg2 > 0, (wx * vx + wy * vy) / seg2, 0.0)
    t = np.clip(t, 0.0, 1.0)
    cx, cy = ax + t * vx, ay + t * vy
    return np.hypot(px - cx, py - cy), t

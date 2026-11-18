"""End-to-end prediction for one trip - the exact code path the API runs.

match()  uses only the GPS track and the road network (never the route file).
score()  is called afterwards, server-side, to compare a prediction with the
         reference route.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np

from .candidates import generate_candidates
from .features import candidate_features
from .geo import LocalProjection
from .hmm import build_lattice, classifier_only, decode, reconstruct_route
from .io import Trip
from .labels import align_route
from .metrics import route_continuity, route_set_scores
from .network import RoadNetwork

MODEL_FILE = "lightgbm_tuned.joblib"
CONFIG_FILE = "pipeline_config.json"


@dataclass
class MatchResult:
    trip_id: str
    lonlat: np.ndarray           # (N, 2) GPS lon, lat
    t: np.ndarray
    xy: np.ndarray
    net: RoadNetwork
    hmm_arcs: np.ndarray         # arc per GPS point after HMM (-1 = no candidate)
    clf_arcs: np.ndarray         # arc per GPS point from the classifier alone
    route_arcs: list             # connected predicted route (HMM)
    n_candidates: int
    n_breaks: int
    timings_s: dict = field(default_factory=dict)


class MapMatcher:
    def __init__(self, model, features: list[str], config: dict):
        self.model = model
        self.features = features
        self.radius = float(config["candidate_radius_m"])
        self.k = int(config["candidate_k"])
        self.beta = float(config["hmm"]["beta"])
        self.emission_weight = float(config["hmm"]["emission_weight"])
        self.config = config

    @classmethod
    def load(cls, artifacts_dir: str | Path) -> "MapMatcher":
        artifacts_dir = Path(artifacts_dir)
        bundle = joblib.load(artifacts_dir / MODEL_FILE)
        config = json.loads((artifacts_dir / CONFIG_FILE).read_text(encoding="utf-8"))
        return cls(bundle["model"], bundle["features"], config)

    def match(self, trip: Trip) -> MatchResult:
        tm, t0 = {}, time.perf_counter()
        lonlat, t = trip.track[:, :2], trip.track[:, 2]
        proj = LocalProjection.around(lonlat)
        xy = proj.forward(lonlat)
        net = RoadNetwork(trip.nodes, trip.arcs, proj)
        tm["load_network"] = time.perf_counter() - t0

        t1 = time.perf_counter()
        c = generate_candidates(net, xy, self.radius, self.k)
        feats = candidate_features(net, xy, t, c)
        tm["candidates_features"] = time.perf_counter() - t1

        t1 = time.perf_counter()
        proba = self.model.predict_proba(feats[self.features])[:, 1]
        tm["classifier"] = time.perf_counter() - t1

        t1 = time.perf_counter()
        lat = build_lattice(net, xy, t, c.obs, c.arc, c.t, proba)
        hmm_arcs, breaks = decode(lat, len(xy), self.beta, self.emission_weight)
        clf_arcs = classifier_only(lat, len(xy))
        route = reconstruct_route(net, hmm_arcs)
        tm["hmm_viterbi"] = time.perf_counter() - t1
        tm["total"] = time.perf_counter() - t0
        return MatchResult(trip.trip_id, lonlat, t, xy, net, hmm_arcs, clf_arcs, route, len(c), breaks, tm)


def score(res: MatchResult, ref_route: np.ndarray) -> dict:
    """Compare a prediction with the reference route (server-side only)."""
    net = res.net
    ref_arc, _, _ = align_route(net, res.xy, ref_route)
    n = len(ref_arc)

    def pointwise(pred):
        ok = pred >= 0
        exact = float(np.mean(ok & (pred == ref_arc))) if n else 0.0
        rev = np.where(ok, net.reverse_arc[np.where(ok, pred, 0)], -1)
        undirected = float(np.mean(ok & ((pred == ref_arc) | (rev == ref_arc)))) if n else 0.0
        return exact, undirected

    h_exact, h_und = pointwise(res.hmm_arcs)
    c_exact, c_und = pointwise(res.clf_arcs)
    hmm_route = route_set_scores(np.asarray(res.route_arcs, dtype=np.int64), ref_route, net.length)
    clf_route = route_set_scores(np.asarray(reconstruct_route(net, res.clf_arcs), dtype=np.int64), ref_route, net.length)
    return {
        "n_points": n,
        "hmm_accuracy": h_exact, "hmm_accuracy_undirected": h_und,
        "classifier_accuracy": c_exact, "classifier_accuracy_undirected": c_und,
        "hmm_continuity": route_continuity(res.hmm_arcs, net.frm, net.to),
        "classifier_continuity": route_continuity(res.clf_arcs, net.frm, net.to),
        "route_precision": hmm_route["route_precision"], "route_recall": hmm_route["route_recall"],
        "route_f1": hmm_route["route_f1"],
        "classifier_route_f1": clf_route["route_f1"],
        "n_breaks": res.n_breaks,
    }

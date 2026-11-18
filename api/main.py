"""PathSnap FastAPI service (runs on laptop 1).

Start from the repository root:
    uvicorn api.main:app --host 0.0.0.0 --port 8000

Environment variables (optional):
    PATHSNAP_EVAL_DIR   folder with one sub-folder per evaluation trip
                        (default: data/eval/evaluator_test)
    PATHSNAP_ARTIFACTS  folder with lightgbm_tuned.joblib + pipeline_config.json
                        (default: artifacts)

Ground truth stays here: /predict runs the model on the GPS track and road
network only, then reads the reference route server-side to compute the
scores. Only scores are returned - never the reference route itself.
"""
from __future__ import annotations

import logging
import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.middleware.cors import CORSMiddleware  # noqa: E402
from pydantic import BaseModel, Field  # noqa: E402

from pathsnap.hmm import reconstruct_route  # noqa: E402
from pathsnap.io import list_trips, load_route, load_trip  # noqa: E402
from pathsnap.pipeline import MapMatcher, score  # noqa: E402

EVAL_DIR = Path(os.environ.get("PATHSNAP_EVAL_DIR", ROOT / "data" / "eval" / "evaluator_test"))
ARTIFACTS = Path(os.environ.get("PATHSNAP_ARTIFACTS", ROOT / "artifacts"))
MAX_GPS_POINTS_RETURNED = 4000

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-5s | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("pathsnap.api")
state: dict = {}


@asynccontextmanager
async def lifespan(_app: FastAPI):
    t0 = time.perf_counter()
    state["matcher"] = MapMatcher.load(ARTIFACTS)
    state["trips"] = list_trips(EVAL_DIR)
    cfg = state["matcher"].config
    log.info("model loaded: %s | HMM beta=%.1f emission_weight=%.2f | %.1fs",
             cfg.get("model_name", "LightGBM"), cfg["hmm"]["beta"], cfg["hmm"]["emission_weight"], time.perf_counter() - t0)
    log.info("evaluation trips in %s: %s", EVAL_DIR, ", ".join(state["trips"]) or "NONE FOUND")
    yield


app = FastAPI(title="PathSnap API", version="1.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])


class PredictRequest(BaseModel):
    trip_id: str = Field(..., min_length=1, max_length=32, examples=["00000015"])


def _count_lines(path: Path) -> int:
    with open(path, "rb") as fh:
        return sum(1 for line in fh if line.strip())


def _polylines(net, arcs: list[int]) -> list[list[list[float]]]:
    """Arc list -> connected [lat, lon] polylines (a new line starts where arcs are not connected)."""
    lines: list[list[list[float]]] = []
    ll = net.nodes_lonlat
    prev = None
    for a in arcs:
        f, t = int(net.frm[a]), int(net.to[a])
        if prev is None or int(net.to[prev]) != f:
            lines.append([[round(ll[f, 1], 6), round(ll[f, 0], 6)]])
        lines[-1].append([round(ll[t, 1], 6), round(ll[t, 0], 6)])
        prev = a
    return lines


@app.get("/health")
def health():
    cfg = state["matcher"].config
    return {"status": "ok", "model": cfg.get("model_name", "LightGBM"), "trips_available": len(state["trips"]),
            "candidate_radius_m": cfg["candidate_radius_m"], "candidate_k": cfg["candidate_k"], "hmm": cfg["hmm"]}


@app.get("/trips")
def trips():
    out = []
    for tid in state["trips"]:
        track = EVAL_DIR / tid / f"{tid}.track"
        out.append({"trip_id": tid, "n_points": _count_lines(track) if track.exists() else None})
    return {"trips": out}


@app.post("/predict")
def predict(req: PredictRequest, request: Request):
    tid = req.trip_id.strip()
    if tid not in state["trips"]:  # only known folder names: no path tricks
        log.warning("predict rejected: unknown trip_id=%r from %s", tid, request.client.host if request.client else "?")
        raise HTTPException(status_code=404, detail=f"unknown trip_id {tid!r}")
    client = request.client.host if request.client else "?"
    log.info("predict request: trip=%s from %s", tid, client)
    try:
        trip = load_trip(EVAL_DIR, tid, with_route=False)   # prediction never sees the route
        res = state["matcher"].match(trip)
        metrics = score(res, load_route(EVAL_DIR, tid))     # ground truth used here, server-side only
    except Exception as exc:  # report a clean error to the UI, full trace in the log
        log.exception("predict failed for trip=%s", tid)
        raise HTTPException(status_code=500, detail=f"prediction failed: {exc}") from exc

    net = res.net
    step = max(1, len(res.lonlat) // MAX_GPS_POINTS_RETURNED)
    gps = [[round(lat, 6), round(lon, 6)] for lon, lat in res.lonlat[::step]]
    log.info("predict done: trip=%s points=%d candidates=%d | HMM acc=%.3f (classifier only %.3f) | "
             "route F1=%.3f continuity=%.3f breaks=%d | %.2fs",
             tid, metrics["n_points"], res.n_candidates, metrics["hmm_accuracy"], metrics["classifier_accuracy"],
             metrics["route_f1"], metrics["hmm_continuity"], metrics["n_breaks"], res.timings_s["total"])
    return {
        "trip_id": tid,
        "n_points": int(metrics["n_points"]),
        "n_candidates": int(res.n_candidates),
        "gps": gps,
        "predicted_route": _polylines(net, res.route_arcs),
        "classifier_route": _polylines(net, reconstruct_route(net, res.clf_arcs)),
        "metrics": {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.items()},
        "timings_s": {k: round(v, 3) for k, v in res.timings_s.items()},
    }

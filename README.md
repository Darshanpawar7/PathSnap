# PathSnap — GPS map-matching with ML + HMM/Viterbi

UE24CS352A Machine Learning hackathon (PES University, 18-11-2026).

Noisy GPS points → for each point, which road segment (arc) was the vehicle on?

1. **Candidates** – for every GPS point, the nearest road arcs within a radius.
2. **Classifier** – a trained model scores each (GPS point, candidate arc) pair.
3. **HMM / Viterbi** – picks the most likely *connected* sequence of arcs.
4. **FastAPI** (laptop 1) serves predictions; **UI** (laptop 2) picks a trip and shows the route.

## Status

| Stage | Due | State |
| --- | --- | --- |
| Data processing + feature engineering | 11:30 eval | in progress |
| Model comparison (≥ 5 models) | 11:30 eval | in progress |
| Bayesian hyperparameter optimisation | 15:00 eval | not started |
| HMM / Viterbi | 15:00 eval | not started |
| FastAPI + UI deployment | 15:00 eval | not started |

## Setup

```bash
pip install -r requirements.txt
```

Put the dataset (from the Drive "Student folder.zip") here — it is not committed:

```
data/raw/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
data/raw/student_test/<trip_id>/...
```

## Layout

```
src/pathsnap/   library code (io, geo, network, candidates, labels, features, ...)
scripts/        numbered steps: 01_eda.py, 02_build_dataset.py, 03_train_models.py, ...
results/        CSVs, figures and maps produced by the scripts
```

Raw file formats (all headerless, whitespace separated):

| File | Columns |
| --- | --- |
| `.track` | lon, lat, t (s) — GPS points, ~1 Hz |
| `.nodes` | lon, lat — row index = node id |
| `.arcs` | from_node, to_node — row index = arc id |
| `.route` | arc_id — ordered reference route |

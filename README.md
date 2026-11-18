# PathSnap: GPS map-matching with ML + HMM/Viterbi

UE24CS352A Machine Learning hackathon, PES University (18-11-2026).

Noisy GPS points go in. For every point, PathSnap returns the road segment (arc) the vehicle was on, and a connected route.

1. **Candidates**: for each GPS point, the 10 nearest road arcs within 50 m.
2. **Classifier**: a Bayesian-tuned LightGBM scores each (GPS point, candidate arc) pair from 24 features.
3. **HMM + Viterbi**: candidate arcs are the hidden states. Classifier scores are the emissions; network driving distance compared with GPS distance gives the transitions.
4. **FastAPI** (laptop 1) serves predictions. A **browser UI** (laptop 2) picks a trip and shows the route and the scores.

Team: Darshan P Pawar (PES2UG24CS143), Harsh Gupta (PES2UG24CS181), Dhatri (PES2UG24CS158), Dhanush Sai Suprapadha (PES2UG24CS154).

## Results (student_test partition, 10 trips, 19,160 GPS points)

See `results/` for every number. Summary tables:

| File | What it holds |
| --- | --- |
| `results/model_comparison.csv` | 7 models, GroupKFold CV + student_test |
| `results/bayes_opt_comparison.csv` | LightGBM untuned vs Bayesian-tuned |
| `results/hmm_student_test_summary.csv` | classifier only vs classifier + HMM |
| `results/hmm_student_test_per_trip.csv` | the same, per trip |

Reports: [docs/PathSnap_Final_Report.pdf](docs/PathSnap_Final_Report.pdf) (final), [docs/PathSnap_Round1_Report.pdf](docs/PathSnap_Round1_Report.pdf) (round 1).

## Install

Python 3.12+:

```bash
pip install -r requirements.txt
```

Put the dataset here (it is not committed; it comes from the course Drive):

```text
data/raw/train/<trip_id>/<trip_id>.{track,nodes,arcs,route}
data/raw/student_test/<trip_id>/...
data/eval/evaluator_test/<trip_id>/...      <- evaluator trips, used only by the API
```

## Reproduce the experiments

Run from the repository root, in order. On Windows PowerShell, first run `$env:PYTHONPATH = "$PWD\src"`; on bash, run `export PYTHONPATH=$PWD/src`.

```bash
python scripts/01_eda.py                # dataset inventory, quality checks, plots, Folium EDA map
python scripts/02_build_dataset.py      # route-to-GPS labels, candidates, 24 features, radius study
python scripts/03_train_models.py       # 7 models, 5-fold GroupKFold by trip, student_test comparison
python scripts/03b_feature_importance.py  # LightGBM importance + feature-group ablation
python scripts/04_bayes_opt.py          # Optuna TPE search -> artifacts/lightgbm_tuned.joblib
python scripts/05_hmm_eval.py           # decoder tuning on out-of-fold train, HMM vs classifier, pipeline_config.json
python scripts/06_folium_maps.py        # Folium prediction maps for student_test trips
```

The random seed is 42 everywhere. The evaluator trips are never used by any of these scripts.

## Deployment (two laptops)

Both laptops must be on the **same network**. A phone hotspot, or laptop 1's Windows Mobile Hotspot, is the most reliable choice, because campus Wi-Fi often blocks laptop-to-laptop traffic.

### Laptop 1: model and API

1. Install the requirements (above). Copy `artifacts/` (the model and its config) and `data/eval/evaluator_test/`.
2. Double-click **`run_api.bat`**, or run:

   ```bash
   python -m uvicorn api.main:app --host 0.0.0.0 --port 8000
   ```

3. Note the IPv4 address that `run_api.bat` prints (for example `192.168.137.1`).
4. When Windows asks about the firewall, tick **Private and Public**, then click **Allow access**.
5. Check it works: open `http://localhost:8000/health` in a browser.

The terminal shows one log line per request (trip, accuracy, time). That is the API log the evaluators ask to see.

### Laptop 2: evaluator UI

1. Download this repository (Code → Download ZIP). Only the `ui/` folder is needed.
2. Double-click **`run_ui.bat`** (needs Python, no packages), or just open `ui/index.html` in Chrome or Edge.
3. Type `http://<laptop-1-IP>:8000` and press **Connect**. The ten evaluation trips appear.
4. Click a trip. The map shows the raw GPS points, the predicted route (amber), and the classifier-only route (dashed purple), with the scores beside it.

The map background tiles need internet on laptop 2. Without internet the routes still draw on a plain background.

### API

| Endpoint | Purpose |
| --- | --- |
| `GET /health` | model name and settings |
| `GET /trips` | evaluation trips available on laptop 1 |
| `POST /predict` `{"trip_id": "00000015"}` | runs the full pipeline and returns GPS points, predicted route, classifier-only route, metrics and timings |

The reference route is read only on the server, after prediction, to compute the metrics. It is never sent to the UI and never seen by the model.

Optional environment variables: `PATHSNAP_EVAL_DIR` (trip folder) and `PATHSNAP_ARTIFACTS` (model folder).

### If something goes wrong

| Problem | Fix |
| --- | --- |
| UI says "timed out" or "Failed to fetch" | Wrong IP or a different network. On laptop 2, open `http://<IP>:8000/health` directly. If that fails, it is the network or firewall, not the code. |
| Works on laptop 1 but not from laptop 2 | Windows firewall. As administrator: `netsh advfirewall firewall add rule name="PathSnap API" dir=in action=allow protocol=TCP localport=8000` |
| "DLL load failed ... Application Control policy" when starting | Windows Smart App Control checks new library files once. Run the command again. |
| No trips listed | `data/eval/evaluator_test/` is missing on laptop 1, or set `PATHSNAP_EVAL_DIR`. |

## Repository layout

```text
src/pathsnap/   io, geo (projection), network (graph, spatial index, shortest paths), labels (route alignment),
                candidates, features, metrics, hmm (HMM + Viterbi), pipeline (end-to-end MapMatcher), config
scripts/        01-06 numbered experiment steps
api/main.py     FastAPI service (laptop 1)
ui/             static evaluator page + Leaflet (laptop 2)
artifacts/      lightgbm_tuned.joblib, pipeline_config.json
results/        CSV results, logs, figures, Folium maps
docs/           reports and slides
```

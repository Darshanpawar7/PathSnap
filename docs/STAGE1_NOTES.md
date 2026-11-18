# Stage 1 notes — data, preprocessing, features, models (11:30 evaluation)

Plain-language notes so every team member can explain stage 1. Numbers come
from `results/eda_summary.txt`, `results/candidate_radius_study.csv` and
`results/model_comparison.csv` — re-run the scripts to reproduce them.

## 1. What the data is

Each trip is a folder with four files (no headers, whitespace separated):

| File | What it holds | How it links |
| --- | --- | --- |
| `.track` | GPS points: lon, lat, time (s) | ~1 point per second |
| `.nodes` | road junction/shape points: lon, lat | row number = node id |
| `.arcs` | road segments: from_node, to_node | row number = arc id; **directed** |
| `.route` | reference route: list of arc ids in driving order | **not** one row per GPS point |

Key point: the route is a list of roads, not a label per GPS point. We have to
work out which GPS point belongs to which route arc ourselves (section 3).

## 2. What we found (EDA, `scripts/01_eda.py`)

* 80 train trips (208,417 points), 10 student_test trips (19,160 points).
  The 10 evaluator trips are missing ids 15, 17, 23, 30, 49, 59, 67, 73, 87, 94.
* No missing values, no out-of-range coordinates, timestamps always increasing.
* 96.7 % of steps are exactly 1 s; 135 gaps longer than 30 s; trip 90 has a
  42-hour gap (two recordings joined).
* Trips are all over the world (lon −100 to 137), so we cannot use one fixed
  map projection — each trip gets its own.
* Column order is lon, lat (every GPS point falls inside its network's box).
* Networks: 649 to 307,392 arcs per trip; ~95 % of roads are two-way (stored as
  two arcs, one per direction); no isolated nodes; each network has a few
  small disconnected pieces but ~98 % of nodes are in the main piece.
* Reference routes are almost perfectly connected (99.94 % of consecutive arcs
  share a node).
* GPS noise: half the points are within 5 m of the reference road, 95 % within
  19 m, 99 % within 70 m.
* **Two broken trips** (excluded from training, kept in the EDA):
  * 33 — GPS sits on roads (7 m) but the reference route is 122 m away → the
    route does not describe this track (wrong labels).
  * 54 — GPS is 84 m from *any* road → the supplied network is missing the
    roads this vehicle used.

## 3. Preprocessing (`src/pathsnap/`)

1. **Projection (`geo.py`)** — degrees are not metres. Each trip is projected
   onto a flat local plane centred on the trip (equirectangular). Over a trip's
   size (a few km to ~50 km) the error is far below 1 %, much smaller than GPS
   noise, and it needs no extra library.
2. **Label alignment (`labels.py`)** — assigns each GPS point one reference arc.
   Dynamic programming: walk through the GPS points in time order, the route
   position may only stay or move forward, minimise total point-to-arc distance.
   "Only forward" stops a point on the way back from being matched to the same
   street on the way out.
3. **Candidate generation (`candidates.py`)** — for each GPS point, the arcs
   within R metres, nearest K kept. Chosen on **train only** from the radius
   study: R = 50 m, K = 10 → reference arc missing for only ~1.7 % of train
   points (0.3 % on student_test). Bigger R/K barely helps but doubles the rows.
   If a point has nothing within 50 m we search 200 m so it is never empty.
4. **Training sample** — every 3rd GPS point (1 Hz points are near-duplicates).

## 4. Features (`features.py`) — 24 per (GPS point, candidate arc)

| Group | Features | Why |
| --- | --- | --- |
| Distance | dist, dist², rank, dist − nearest, ratio to nearest, z-score, n_cands | the closest road is usually right, but *relative* closeness matters more than absolute |
| Position on arc | proj_t, beyond_end, log arc length | a point projecting past the end of a segment is likely on the next one |
| Direction | heading_valid, cos_dir, abs_cos, along_progress_m | separates the two directions of a two-way road and crossing streets |
| Motion | speed, dt_prev, dt_next, turn angle | fast + straight = main road; gaps make matching harder |
| Network | two_way, out-degree at end, in-degree at start | junctions are where mistakes happen |
| Neighbours | win2_seg_mean, win5_line_mean, nbr_cand_share | do the points before/after also fit this road? removes single-point noise |

Label = 1 if the candidate is the point's aligned reference arc, else 0
(~15 % positives).

## 5. Models (`scripts/03_train_models.py`)

Seven models from seven families, same features, same splits:
Logistic Regression (linear), Gaussian NB (probabilistic), KNN (instance),
Decision Tree (tree), Random Forest (bagging), LightGBM (boosting), MLP (neural).

* 5-fold **GroupKFold by trip** on the 78 usable train trips — a trip is never
  in both the fitting and the validation part, so scores are honest.
* Then refit on all train data and score the 10 student_test trips.
* Metrics: candidate accuracy/precision/recall/F1/macro-F1, log loss,
  observation-level **Top-1 accuracy** (pick the best candidate per GPS point,
  is it the reference arc?), route continuity, fit and inference time.
* **Primary metric = Top-1 accuracy** — it is exactly the decision the
  map-matcher makes. Log loss is the tie-breaker because the HMM uses the
  model's probabilities directly, so they must be well calibrated.

Results table: `results/model_comparison.csv`.

## 6. Results (from `results/model_comparison.csv`)

| Model | Family | CV macro F1 (±std) | CV Top-1 | Test Top-1 | Test macro F1 | Test log loss | Continuity | Inference (ms / 1k pts) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| **LightGBM** | Boosting | **0.930 ± 0.013** | **0.881** | **0.895** | **0.937** | **0.079** | 0.943 | 57 |
| MLP (64, 32) | Neural network | 0.925 ± 0.015 | 0.876 | 0.886 | 0.931 | 0.087 | 0.945 | 3 |
| Random Forest | Bagging | 0.929 ± 0.013 | 0.877 | 0.884 | 0.931 | 0.083 | 0.933 | 30 |
| Logistic Regression | Linear | 0.921 ± 0.014 | 0.875 | 0.882 | 0.922 | 0.103 | 0.938 | 0.9 |
| Decision Tree | Tree | 0.922 ± 0.016 | 0.872 | 0.879 | 0.927 | 0.103 | 0.929 | 0.5 |
| KNN (k = 25) | Instance | 0.921 ± 0.014 | 0.868 | 0.875 | 0.923 | 0.129 | 0.927 | 1078 |
| Gaussian NB | Probabilistic | 0.782 ± 0.008 | 0.856 | 0.860 | 0.779 | 0.978 | 0.911 | 4 |

**What the numbers say**

* **LightGBM is chosen.** It is best on the primary metric (Top-1) in CV *and*
  on student_test, and has the lowest log loss (best probabilities for the HMM).
  It trains in ~6 s. Inference is ~1 s for all 10 test trips, which is fine for the API.
* **Most remaining errors are direction mix-ups.** If picking the reverse arc of
  a two-way road is also counted as correct, LightGBM's Top-1 goes from 89.5 %
  to 94.1 %. A single GPS point cannot tell which way the car drives; the HMM
  (stage 2) uses the sequence and the network to fix this.
* **Overfitting evidence** (train vs CV macro F1):
  KNN 1.00 vs 0.92 — distance weighting memorises the training rows (each row
  is its own nearest neighbour), so it overfits badly and is also the slowest.
  LightGBM 0.961 vs 0.930 — a small gap; Bayesian tuning should target it.
  Decision Tree 0.935 vs 0.922, Random Forest 0.941 vs 0.929 — mild.
* **Gaussian NB under-fits.** Its independence assumption is broken here
  (dist, dist², rank, ratio are strongly correlated), so it double-counts
  evidence: very high recall (0.94), low precision (0.50), worst log loss.
* **MLP** stopped at its 40-epoch limit while validation accuracy was still
  rising (`results/figures/models_mlp_history.png`), so it is slightly
  under-trained rather than overfitting.
* All models stay within ~3 points of each other on Top-1. Feature design and
  the candidate set matter more than the algorithm; the HMM is where the next
  big gain should come from.

Figures: `results/figures/models_cv_macro_f1.png`,
`models_student_test_comparison.png`, `models_confusion_matrices.png`,
`models_mlp_history.png`.

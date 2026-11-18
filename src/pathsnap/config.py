"""Pipeline settings chosen from the training partition (see 02_build_dataset.py)."""

CANDIDATE_RADIUS_M = 50.0   # search radius around each GPS point
CANDIDATE_K = 10            # keep at most the K nearest arcs per point

# Training trips whose reference route clearly does not describe the GPS track
# (found in 01_eda.py): excluded from model fitting, kept in the EDA report.
#   00000033 GPS is ~7 m from roads but ~122 m from the reference route
#   00000054 GPS is ~84 m from any road in the supplied network
EXCLUDED_TRAIN_TRIPS = ("00000033", "00000054")

TRAIN_POINT_STRIDE = 3      # 1 Hz points are near-duplicates: use every 3rd for fitting

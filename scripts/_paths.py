"""Shared paths for the numbered scripts."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

DATA_RAW = ROOT / "data" / "raw"
TRAIN_DIR = DATA_RAW / "train"
STUDENT_TEST_DIR = DATA_RAW / "student_test"
PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
FIGURES = RESULTS / "figures"
MAPS = RESULTS / "maps"
ARTIFACTS = ROOT / "artifacts"

for d in (PROCESSED, RESULTS, FIGURES, MAPS, ARTIFACTS):
    d.mkdir(parents=True, exist_ok=True)

SEED = 42

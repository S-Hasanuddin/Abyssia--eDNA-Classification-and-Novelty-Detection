"""
config.py — Application configuration
"""
import os
from pathlib import Path

# ── Paths ────────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CHECKPOINT_DIR = Path(os.environ.get("CHECKPOINT_DIR", str(BASE_DIR / "checkpoints")))
UPLOAD_DIR = Path(os.environ.get("UPLOAD_DIR", str(BASE_DIR / "data" / "uploads")))
REPORT_DIR = Path(os.environ.get("REPORT_DIR", str(BASE_DIR / "reports")))
HISTORY_DB_PATH = Path(os.environ.get("HISTORY_DB_PATH", str(BASE_DIR / "data" / "historical_clusters.json")))

# Create dirs
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

# ── Model settings ───────────────────────────────────────────────────────────
DEVICE = os.environ.get("DEVICE", "cpu")  # override to "cuda" on GPU server
CONFIDENCE_THRESHOLD = float(os.environ.get("CONFIDENCE_THRESHOLD", "0.5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "32"))
HDBSCAN_MIN_CLUSTER = int(os.environ.get("HDBSCAN_MIN_CLUSTER", "5"))
HDBSCAN_MIN_SAMPLES = int(os.environ.get("HDBSCAN_MIN_SAMPLES", "3"))

# Checkpoint filename — use ft_final.pt (phase3_finetune output)
# Falls back to phase3_final.pt or phase2_latest.pt
CHECKPOINT_PRIORITY = ["ft_final.pt", "phase3_final.pt", "phase2_latest.pt"]

# ── Per-rank confidence thresholds ───────────────────────────────────────────
# Based on observed per-rank validation accuracy from training:
#   Domain 99.18% → low threshold fine
#   Kingdom 90.13% → modest threshold
#   Phylum  84.99% → moderate threshold
#   Class   81.55% → moderate threshold
#   Order   65.15% → higher threshold needed
#   Family  48.15% → highest threshold (model least reliable here)
RANK_THRESHOLDS = {
    "domain":  float(os.environ.get("THRESHOLD_DOMAIN",  "0.30")),
    "kingdom": float(os.environ.get("THRESHOLD_KINGDOM", "0.40")),
    "phylum":  float(os.environ.get("THRESHOLD_PHYLUM",  "0.50")),
    "class":   float(os.environ.get("THRESHOLD_CLASS",   "0.55")),
    "order":   float(os.environ.get("THRESHOLD_ORDER",   "0.65")),
    "family":  float(os.environ.get("THRESHOLD_FAMILY",  "0.72")),
}

# ── Novel routing ─────────────────────────────────────────────────────────────
# Minimum rank specificity required to route a sequence to classification
# rather than novel clustering. Sequences that only reach Domain/Kingdom/Phylum/Class
# confidence are considered novel candidates.
# Options: domain, kingdom, phylum, class, order, family
# Recommended: "order" — sequences need at least Order-level confidence to be
# classified; everything coarser goes to HDBSCAN novel detection.
MIN_CLASSIFICATION_RANK = os.environ.get("MIN_CLASSIFICATION_RANK", "order")

# Numeric specificity mapping (used for comparison in inference.py)
RANK_SPECIFICITY = {
    "domain":  0,
    "kingdom": 1,
    "phylum":  2,
    "class":   3,
    "order":   4,
    "family":  5,
}

# ── API settings ─────────────────────────────────────────────────────────────
MAX_UPLOAD_SIZE_MB = int(os.environ.get("MAX_UPLOAD_SIZE_MB", "100"))
CORS_ORIGINS = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://localhost:5173,http://localhost:8080,http://127.0.0.1:3000"
).split(",")

# ── Taxonomy ranks (global constant) ────────────────────────────────────────
RANKS = ['domain', 'kingdom', 'phylum', 'class', 'order', 'family']
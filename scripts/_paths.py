"""Shared repo-relative paths for AuditLM scripts."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
MODELS_DIR = REPO_ROOT / "models"
DATASETS_DIR = REPO_ROOT / "datasets"
EVAL_RESULTS_DIR = REPO_ROOT / "eval_results"


def model_path(*parts: str) -> Path:
    return MODELS_DIR.joinpath(*parts)

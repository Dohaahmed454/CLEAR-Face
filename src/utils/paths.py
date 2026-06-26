"""Canonical project paths for Milestone 2.

Importing this module makes ``src`` importable from any working directory
(useful when scripts are launched as ``python src/fer/train_primary_fer.py``).
"""

import sys
from pathlib import Path

# src/utils/paths.py -> parents[2] == repository root.
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Data
DATA_DIR = PROJECT_ROOT / "data" / "milestone_2"
METADATA_DIR = DATA_DIR / "metadata"
SYNTHETIC_DIR = DATA_DIR / "synthetic"
CLEAR_FACE_R_DIR = DATA_DIR / "clear_face_r"
VALIDATION_GALLERY_DIR = DATA_DIR / "validation_gallery"

# Models
MODELS_DIR = PROJECT_ROOT / "models"
FER_PRIMARY_DIR = MODELS_DIR / "fer_primary"
FER_COMPARISON_DIR = MODELS_DIR / "fer_comparison"

# Experiments
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments" / "milestone_2"
RESULTS_DIR = EXPERIMENTS_DIR / "results"
CONFIGS_DIR = EXPERIMENTS_DIR / "configs"

# Docs / reports
DOCS_DIR = PROJECT_ROOT / "docs" / "milestone_2"
REPORTS_DIR = DOCS_DIR / "reports"
FIGURES_DIR = DOCS_DIR / "figures"
GALLERY_DOC_DIR = DOCS_DIR / "validation_gallery"

# Common default dataset roots (used by the manifest builder when present).
KAGGLE_INPUT = Path("/kaggle/input")


def ensure_dir(path) -> Path:
    """Create ``path`` (and parents) if missing and return it as a ``Path``."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_project_root_to_path() -> None:
    """Ensure the repository root is importable as the top of ``src``."""
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)


# Make `import src...` work as a side effect of importing this module.
add_project_root_to_path()

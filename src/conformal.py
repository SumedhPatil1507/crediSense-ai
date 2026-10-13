"""Split-conformal probability intervals for binary risk scores."""
import json
import math
from pathlib import Path
from typing import Iterable

import numpy as np

DEFAULT_CALIBRATION_PATH = Path(__file__).resolve().parents[1] / "models" / "conformal_90_calibration.json"


def absolute_residuals(y_true: Iterable[int], probabilities: Iterable[float]) -> list[float]:
    """Return calibration nonconformity scores for binary outcomes."""
    labels = np.asarray(list(y_true), dtype=float)
    scores = np.asarray(list(probabilities), dtype=float)
    if labels.ndim != 1 or scores.ndim != 1 or labels.size != scores.size or not labels.size:
        raise ValueError("y_true and probabilities must be non-empty aligned vectors")
    if not np.isin(labels, [0.0, 1.0]).all():
        raise ValueError("y_true must contain only binary labels 0 and 1")
    if not np.isfinite(scores).all() or ((scores < 0) | (scores > 1)).any():
        raise ValueError("probabilities must be finite values in [0, 1]")
    return np.abs(labels - scores).tolist()


def conformal_interval(probability: float, calibration_scores: Iterable[float],
                        *, coverage: float = 0.90) -> tuple[float, float]:
    """Compute a finite-sample split-conformal interval, clipped to [0, 1]."""
    scores = np.asarray(list(calibration_scores), dtype=float)
    if not 0.0 < coverage < 1.0:
        raise ValueError("coverage must be between 0 and 1")
    if not 0.0 <= probability <= 1.0 or not math.isfinite(probability):
        raise ValueError("probability must be finite and in [0, 1]")
    if not scores.size or not np.isfinite(scores).all() or (scores < 0).any():
        raise ValueError("calibration_scores must be a non-empty finite non-negative sequence")
    rank = min(math.ceil((scores.size + 1) * coverage), scores.size)
    radius = float(np.partition(scores, rank - 1)[rank - 1])
    return max(0.0, probability - radius), min(1.0, probability + radius)


def save_calibration(y_true: Iterable[int], probabilities: Iterable[float],
                     path: str | Path = DEFAULT_CALIBRATION_PATH,
                     *, coverage: float = 0.90) -> list[float]:
    """Persist held-out calibration residuals for serving-time intervals."""
    scores = absolute_residuals(y_true, probabilities)
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps({"method": "split_conformal_absolute_residual",
                                  "coverage": coverage,
                                  "sample_count": len(scores),
                                  "absolute_residuals": scores}, indent=2),
                      encoding="utf-8")
    return scores


def load_calibration(path: str | Path = DEFAULT_CALIBRATION_PATH) -> list[float] | None:
    """Load calibration scores or return None when the model has no artifact."""
    target = Path(path)
    if not target.is_file():
        return None
    artifact = json.loads(target.read_text(encoding="utf-8"))
    scores = artifact.get("absolute_residuals") if isinstance(artifact, dict) else None
    if not isinstance(scores, list) or not scores:
        raise ValueError(f"Invalid conformal calibration artifact: {target}")
    # Validate eagerly rather than serving silently malformed bounds.
    np.asarray(scores, dtype=float)
    return [float(score) for score in scores]


def calibrated_interval(probability: float, *, path: str | Path = DEFAULT_CALIBRATION_PATH,
                        coverage: float = 0.90) -> tuple[float, float] | None:
    scores = load_calibration(path)
    return conformal_interval(probability, scores, coverage=coverage) if scores else None

"""Neuronpedia client for SAE-feature metadata (explanations, interp scores, top
tokens).

Feature selection in this repo is otherwise statistics-only (density band +
projected positives; see `data_prep/select_sae_features.py`). This module attaches the
*semantic* side -- the auto-interp explanation, its (best-effort) score, and the
feature's top-activating tokens -- so downstream selection can enforce "has an
explanation", "high interp score when available", and can build a concept-probe
set for the steerability test (`score_sae_steerability.py`).

Public API, no key required. We only ever GET a feature by (model, source,
index); nothing is uploaded. Responses are cached on disk so re-runs and the
steerability step do not re-hit the server (Neuronpedia asks people to use the S3
bulk export instead of hammering the per-feature endpoint -- the on-disk cache is
our politeness layer for the small density-selected candidate set).

Endpoint (confirmed):
  GET https://www.neuronpedia.org/api/feature/{model_id}/{source_id}/{index}
where for Gemma Scope canonical res SAEs
  model_id  = "gemma-2-2b"
  source_id = "{layer}-gemmascope-{site}-{width}"   e.g. "9-gemmascope-res-16k"
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

BASE_URL = "https://www.neuronpedia.org/api/feature"
DEFAULT_CACHE_DIR = Path("data/neuronpedia_cache")
_SCORE_KEYS = ("value", "score", "recall", "f1", "cosineSimilarity", "correlation")


def source_id_for(layer: int, site: str = "res", width: str = "16k") -> str:
    """Neuronpedia source id for a Gemma Scope canonical SAE at (layer, site, width)."""
    return f"{layer}-gemmascope-{site}-{width}"


@dataclass
class FeatureInfo:
    """Normalized subset of a Neuronpedia feature record used by selection + steering."""
    index: int
    explanation: Optional[str]
    explanation_model: Optional[str]
    interp_score: Optional[float]
    n_explanations: int
    pos_str: list
    neg_str: list
    frac_nonzero: Optional[float]
    max_act: Optional[float]
    has_vector: bool
    default_steer_strength: Optional[float]

    @property
    def has_explanation(self) -> bool:
        return bool(self.explanation and self.explanation.strip())

    def to_row(self) -> dict:
        return asdict(self)


def _score_of(explanation: dict) -> Optional[float]:
    """Best numeric score from one explanation's `scores[]`, or None if empty/absent.

    Neuronpedia stores several score *types* per explanation with inconsistent field
    names, and for many Gemma Scope features `scores` is `[]`. We scan defensively and
    return the max numeric value found under any known key."""
    scores = explanation.get("scores") or []
    found = []
    for s in scores:
        if not isinstance(s, dict):
            continue
        for k in _SCORE_KEYS:
            v = s.get(k)
            if isinstance(v, (int, float)):
                found.append(float(v))
                break
    return max(found) if found else None


def parse_feature(raw: dict) -> FeatureInfo:
    """Normalize a raw Neuronpedia feature JSON into a FeatureInfo.

    Explanation selection: prefer an explanation that carries a numeric score;
    otherwise fall back to the first. `interp_score` is the max score across ALL
    explanations (so a feature is credited with its best available score)."""
    explanations = raw.get("explanations") or []

    best_expl = None
    best_score = None
    for e in explanations:
        sc = _score_of(e)
        if sc is not None and (best_score is None or sc > best_score):
            best_score, best_expl = sc, e
    if best_expl is None and explanations:
        best_expl = explanations[0]

    def _num(x):
        return float(x) if isinstance(x, (int, float)) else None

    return FeatureInfo(
        index=int(raw.get("index")),
        explanation=(best_expl or {}).get("description"),
        explanation_model=(best_expl or {}).get("explanationModelName"),
        interp_score=best_score,
        n_explanations=len(explanations),
        pos_str=list(raw.get("pos_str") or []),
        neg_str=list(raw.get("neg_str") or []),
        frac_nonzero=_num(raw.get("frac_nonzero")),
        max_act=_num(raw.get("maxActApprox")),
        has_vector=bool(raw.get("hasVector")),
        default_steer_strength=_num(raw.get("vectorDefaultSteerStrength")),
    )


def fetch_feature_raw(
    model_id: str,
    source_id: str,
    index: int,
    *,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    session=None,
    timeout: float = 30.0,
    max_retries: int = 4,
    rate_limit_s: float = 0.2,
    use_cache: bool = True,
) -> Optional[dict]:
    """GET one feature record, cached on disk. Returns the raw JSON dict, or None on
    a persistent 404 / failure. Retries transient errors with exponential backoff."""
    cache_path = Path(cache_dir) / source_id / f"{index}.json"
    if use_cache and cache_path.exists():
        try:
            return json.loads(cache_path.read_text())
        except (json.JSONDecodeError, OSError):
            pass

    import requests
    sess = session or requests.Session()
    url = f"{BASE_URL}/{model_id}/{source_id}/{index}"

    last_err = None
    for attempt in range(max_retries):
        try:
            resp = sess.get(url, timeout=timeout,
                            headers={"Accept": "application/json"})
            if resp.status_code == 404:
                return None
            if resp.status_code == 429 or resp.status_code >= 500:
                raise requests.HTTPError(f"HTTP {resp.status_code}")
            resp.raise_for_status()
            data = resp.json()
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_text(json.dumps(data))
            time.sleep(rate_limit_s)
            return data
        except Exception as e:
            last_err = e
            time.sleep(min(2.0 ** attempt, 15.0))
    print(f"[neuronpedia] giving up on {source_id}/{index}: {last_err}")
    return None


def fetch_feature(
    model_id: str,
    source_id: str,
    index: int,
    **kwargs,
) -> Optional[FeatureInfo]:
    """Fetch + parse one feature. None if the feature could not be retrieved."""
    raw = fetch_feature_raw(model_id, source_id, index, **kwargs)
    return parse_feature(raw) if raw is not None else None

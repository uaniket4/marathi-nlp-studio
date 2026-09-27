"""Model lifecycle: load the NER model and tokenizer exactly once.

The heavy objects are cached at module level so the FastAPI app can load them
during startup and reuse them for every request. Never load per-request.
"""

from __future__ import annotations

import json
import os
from typing import Dict, Optional

import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

from .config import get_settings


class NERModel:
    """Wraps a token-classification model + tokenizer and its metadata."""

    def __init__(self, model_name: str, max_length: int) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(model_name)
        self.model.eval()  # inference mode; disables dropout
        self.id2label: Dict[int, str] = {
            int(k): v for k, v in self.model.config.id2label.items()
        }
        if not self.tokenizer.is_fast:
            raise RuntimeError(
                "A fast tokenizer is required for offset mapping during inference."
            )

    @property
    def entity_types(self) -> list[str]:
        """Distinct display entity types this model can emit (excluding O)."""
        from .inference import to_display_type, normalize_label

        types = set()
        for lab in self.id2label.values():
            etype, _ = normalize_label(lab)
            if etype is not None:
                types.add(to_display_type(etype))
        return sorted(types)


_model: Optional[NERModel] = None
_load_error: Optional[str] = None


def load_model() -> None:
    """Load the model at startup. Records an error string instead of raising."""
    global _model, _load_error
    s = get_settings()
    try:
        _model = NERModel(s.MODEL_NAME, s.MAX_LENGTH)
        _load_error = None
    except Exception as exc:  # noqa: BLE001 - surfaced via /health, not a crash
        _model = None
        _load_error = f"{type(exc).__name__}: {exc}"


def get_model() -> Optional[NERModel]:
    return _model


def get_load_error() -> Optional[str]:
    return _load_error


def load_metrics() -> Optional[dict]:
    """Read evaluation metrics from METRICS_PATH if present."""
    s = get_settings()
    if os.path.isfile(s.METRICS_PATH):
        try:
            with open(s.METRICS_PATH, "r", encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:  # noqa: BLE001
            return None
    return None


# Torch runs single-threaded inference deterministically here; keep it lean.
torch.set_grad_enabled(False)

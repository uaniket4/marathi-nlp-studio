"""Central configuration via environment variables.

All tunables live here so nothing is hardcoded across modules. Values may be
overridden with a ``.env`` file (see ``.env.example``).
"""

from __future__ import annotations

import os
from functools import lru_cache

try:  # optional; harmless if python-dotenv is absent
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


class Settings:
    """Runtime settings shared by training, evaluation and the API."""

    # --- Model ---
    # Demo/inference model: the official MahaNER-fine-tuned model (real, trained
    # on this exact dataset). A locally trained model can override this by
    # pointing MODEL_NAME at ./backend/models/marathi-ner.
    MODEL_NAME: str = _get("MODEL_NAME", "l3cube-pune/marathi-ner")
    # Base model used by train.py when fine-tuning from scratch.
    BASE_MODEL: str = _get("BASE_MODEL", "l3cube-pune/marathi-bert-v2")
    MAX_LENGTH: int = int(_get("MAX_LENGTH", "256"))

    # --- Paths ---
    DATA_DIR: str = _get(
        "DATA_DIR",
        os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "data")),
    )
    OUTPUT_DIR: str = _get(
        "OUTPUT_DIR",
        os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "models", "marathi-ner")
        ),
    )
    METRICS_PATH: str = _get(
        "METRICS_PATH",
        os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "models", "metrics.json")
        ),
    )

    # --- Training hyperparameters ---
    EPOCHS: float = float(_get("EPOCHS", "3"))
    TRAIN_BATCH_SIZE: int = int(_get("TRAIN_BATCH_SIZE", "16"))
    EVAL_BATCH_SIZE: int = int(_get("EVAL_BATCH_SIZE", "32"))
    LEARNING_RATE: float = float(_get("LEARNING_RATE", "3e-5"))
    WEIGHT_DECAY: float = float(_get("WEIGHT_DECAY", "0.01"))
    SEED: int = int(_get("SEED", "42"))

    # --- API ---
    MAX_TEXT_CHARS: int = int(_get("MAX_TEXT_CHARS", "5000"))
    CORS_ORIGINS: str = _get("CORS_ORIGINS", "http://localhost:5173,http://127.0.0.1:5173")

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()

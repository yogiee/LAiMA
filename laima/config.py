from __future__ import annotations

import json
from pathlib import Path
from typing import Any

CONFIG_DIR = Path.home() / ".config" / "laima"
CONFIG_FILE = CONFIG_DIR / "config.json"
MODELS_FILE = CONFIG_DIR / "models.json"

DEFAULTS: dict[str, Any] = {
    "host": "127.0.0.1",
    "port": 8080,
    "ctx_size": 8192,
    "n_gpu_layers": 99,
    "n_gpu_layers_draft": 99,
    "threads": 8,
    "draft_min": 0,
    "draft_max": 16,
    # Optional: path to a llama.cpp source tree, used to refresh conversion scripts
    "llama_cpp_path": None,
}


def is_first_run() -> bool:
    return not CONFIG_FILE.exists()


def load_config() -> dict[str, Any]:
    if CONFIG_FILE.exists():
        try:
            with open(CONFIG_FILE) as f:
                saved = json.load(f)
            return {**DEFAULTS, **saved}
        except (json.JSONDecodeError, OSError):
            pass
    return DEFAULTS.copy()


def save_config(cfg: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(CONFIG_FILE, "w") as f:
        json.dump(cfg, f, indent=2)


def load_models() -> dict[str, Any]:
    if MODELS_FILE.exists():
        try:
            with open(MODELS_FILE) as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def save_models(models: dict[str, Any]) -> None:
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    with open(MODELS_FILE, "w") as f:
        json.dump(models, f, indent=2)

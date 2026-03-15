from __future__ import annotations

import json
import os
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

MANIFEST_ROOT = (
    Path.home() / ".ollama" / "models" / "manifests" / "registry.ollama.ai"
)
BLOB_ROOT = Path.home() / ".ollama" / "models" / "blobs"

GGUF_MAGIC = b"GGUF"
OLLAMA_MODEL_MEDIA_TYPE = "application/vnd.ollama.image.model"


# ---------------------------------------------------------------------------
# Ollama model discovery
# ---------------------------------------------------------------------------

def get_ollama_model_names() -> list[str]:
    """Return model names reported by `ollama list`."""
    try:
        result = subprocess.run(
            ["ollama", "list"],
            capture_output=True,
            text=True,
            timeout=15,
        )
        names: list[str] = []
        for line in result.stdout.strip().splitlines()[1:]:  # skip header
            parts = line.split()
            if parts:
                names.append(parts[0])
        return names
    except (subprocess.TimeoutExpired, FileNotFoundError, subprocess.SubprocessError):
        return []


# ---------------------------------------------------------------------------
# Manifest → blob resolution
# ---------------------------------------------------------------------------

def _find_manifest_path(family: str, tag: str) -> Optional[Path]:
    """
    Locate the Ollama manifest file for a model.

    Ollama stores manifests at:
      registry.ollama.ai/library/<name>/<tag>   (unnamespaced models)
      registry.ollama.ai/<ns>/<name>/<tag>       (namespaced models)
    """
    if not MANIFEST_ROOT.exists():
        return None

    # Fast path: unnamespaced library model
    if "/" not in family:
        direct = MANIFEST_ROOT / "library" / family / tag
        if direct.is_file():
            return direct

    # Walk to handle namespaced models and edge cases
    target_name = family.split("/")[-1].lower()
    target_ns = family.split("/")[0].lower() if "/" in family else None

    for root, _dirs, files in os.walk(MANIFEST_ROOT):
        if tag in files:
            root_path = Path(root)
            model_part = root_path.name.lower()
            ns_part = root_path.parent.name.lower()
            if model_part == target_name:
                if target_ns is None or ns_part == target_ns:
                    return root_path / tag

    return None


def _find_gguf_blob(model_name: str) -> Optional[Path]:
    """Return the GGUF blob path for an Ollama model, or None."""
    if ":" in model_name:
        family, tag = model_name.rsplit(":", 1)
    else:
        family, tag = model_name, "latest"

    manifest_path = _find_manifest_path(family, tag)
    if not manifest_path:
        return None

    try:
        manifest: dict = json.loads(manifest_path.read_text())
    except (json.JSONDecodeError, OSError):
        return None

    # Prefer the layer explicitly typed as the model weights
    digest: Optional[str] = None
    for layer in manifest.get("layers", []):
        if layer.get("mediaType") == OLLAMA_MODEL_MEDIA_TYPE:
            digest = layer.get("digest")
            break

    # Fallback: first layer
    if not digest:
        layers = manifest.get("layers", [])
        if layers:
            digest = layers[0].get("digest")

    if not digest:
        return None

    blob_path = BLOB_ROOT / digest.replace(":", "-")
    return blob_path if blob_path.exists() else None


# ---------------------------------------------------------------------------
# GGUF validation
# ---------------------------------------------------------------------------

def _is_gguf(path: Path) -> bool:
    """Check GGUF magic bytes (fast, no subprocess)."""
    try:
        with open(path, "rb") as f:
            return f.read(4) == GGUF_MAGIC
    except OSError:
        return False


def _size_gb(path: Path) -> Optional[float]:
    try:
        return path.stat().st_size / (1024 ** 3)
    except OSError:
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def scan_models(
    existing: dict[str, Any],
) -> tuple[dict[str, Any], int, int, int]:
    """
    Scan installed Ollama models and update the model database.

    Returns ``(updated_db, new_count, removed_count, unchanged_count)``.
    """
    current_names = set(get_ollama_model_names())
    existing_names = set(existing.keys())

    new_names = current_names - existing_names
    removed_names = existing_names - current_names
    unchanged_names = current_names & existing_names

    # Keep surviving Ollama entries plus any custom (non-Ollama) models unchanged
    updated: dict[str, Any] = {
        k: v for k, v in existing.items()
        if k in current_names or v.get("origin") == "custom"
    }

    for name in sorted(new_names):
        blob = _find_gguf_blob(name)
        compatible = bool(blob and _is_gguf(blob))
        size = _size_gb(blob) if blob else None

        updated[name] = {
            "gguf_path": str(blob) if blob else None,
            "gguf_compatible": compatible,
            "size_gb": round(size, 2) if size is not None else None,
            "last_checked": datetime.now().isoformat(),
        }

    return updated, len(new_names), len(removed_names), len(unchanged_names)


def add_custom_model(models: dict[str, Any], name: str, gguf_path: str) -> dict[str, Any]:
    """
    Register a manually converted / quantised model in the database.
    These entries are preserved across Ollama scans (``origin == "custom"``).
    """
    size = _size_gb(Path(gguf_path))
    models[name] = {
        "gguf_path": gguf_path,
        "gguf_compatible": True,
        "size_gb": round(size, 2) if size is not None else None,
        "last_checked": datetime.now().isoformat(),
        "origin": "custom",
    }
    return models


def get_compatible_models(models: dict[str, Any]) -> list[dict[str, Any]]:
    """
    Return a sorted list of GGUF-compatible model dicts ready for display.

    Each entry has: ``name``, ``path``, ``size_str``, ``label``.
    """
    result: list[dict[str, Any]] = []
    for name, info in models.items():
        if not info.get("gguf_compatible"):
            continue
        size = info.get("size_gb")
        size_str = f"{size:.1f} GB" if size is not None else "? GB"
        result.append(
            {
                "name": name,
                "path": info["gguf_path"],
                "size_str": size_str,
            }
        )
    result.sort(key=lambda x: x["name"])

    # Build aligned labels after sorting (consistent column width)
    max_len = max((len(m["name"]) for m in result), default=0)
    for m in result:
        m["label"] = f"{m['name']:<{max_len + 2}}{m['size_str']:>8}"

    return result

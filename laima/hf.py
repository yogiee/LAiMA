from __future__ import annotations

import sys
import termios
from pathlib import Path
from typing import Any, Optional

import questionary

from laima.config import load_models, save_models
from laima.models import add_custom_model
from laima.ui import console

_QSTYLE = questionary.Style(
    [
        ("qmark",       "fg:#61afef bold"),
        ("question",    "bold"),
        ("answer",      "fg:#98c379 bold"),
        ("pointer",     "fg:#61afef bold"),
        ("highlighted", "fg:#61afef bold"),
        ("selected",    "fg:#98c379"),
        ("separator",   "fg:#5c6370"),
        ("instruction", "fg:#5c6370"),
    ]
)


def _flush_stdin() -> None:
    try:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def hf_pull_flow(cfg: dict[str, Any]) -> None:  # noqa: ARG001
    """Interactive: enter an HF Hub repo ID, download GGUF or trigger convert."""
    _flush_stdin()

    try:
        from huggingface_hub import list_repo_files, hf_hub_download  # noqa: F401
    except ImportError:
        console.print("[error]✖[/error]  huggingface_hub not installed.")
        console.print("   Run: pip install -e . (to pick up the updated dependencies)")
        return

    console.print()
    repo_id = questionary.text(
        "HuggingFace model ID:",
        instruction="(e.g. bartowski/Qwen3-0.6B-GGUF  or  Qwen/Qwen3.5-0.8B)",
        style=_QSTYLE,
        validate=lambda x: bool(x.strip()) or "Please enter a model ID.",
    ).ask()
    if not repo_id:
        return
    repo_id = repo_id.strip()

    console.print()
    with console.status("[muted]Checking repository…[/muted]", spinner="dots"):
        try:
            from huggingface_hub import list_repo_files
            all_files = list(list_repo_files(repo_id))
        except Exception as exc:
            console.print(f"[error]✖[/error]  Could not access [bold]{repo_id}[/bold]: {exc}")
            return

    gguf_files = sorted(f for f in all_files if f.lower().endswith(".gguf"))

    if gguf_files:
        _download_gguf_flow(repo_id, gguf_files)
    else:
        console.print(
            f"[warning]⚠[/warning]   No GGUF files found in [bold]{repo_id}[/bold]."
        )
        console.print("   The model needs to be converted to GGUF first.")
        console.print()
        _flush_stdin()
        if questionary.confirm(
            "Convert this model to GGUF now?", default=True, style=_QSTYLE
        ).ask():
            from laima.convert import convert_hf_flow
            convert_hf_flow(cfg, prefill_model_id=repo_id)


# ---------------------------------------------------------------------------
# GGUF download
# ---------------------------------------------------------------------------

def _download_gguf_flow(repo_id: str, gguf_files: list[str]) -> None:
    from huggingface_hub import hf_hub_download

    _flush_stdin()

    choices = [
        questionary.Choice(title=Path(f).name, value=f) for f in gguf_files
    ]
    choices.append(questionary.Choice(title="← Back", value="__back__"))

    console.print()
    selected = questionary.select(
        f"Select GGUF file from {repo_id}:",
        choices=choices,
        style=_QSTYLE,
        use_search_filter=True,
        use_jk_keys=False,
    ).ask()

    if selected is None or selected == "__back__":
        return

    default_outdir = str(Path.home() / "models")
    _flush_stdin()
    outdir_str = questionary.text(
        "Download to:",
        default=default_outdir,
        style=_QSTYLE,
    ).ask()
    if outdir_str is None:
        return
    outdir = Path(outdir_str.strip() or default_outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    filename = Path(selected).name
    dest = outdir / filename

    console.print()
    console.print(f"  [bold]Repo  [/bold]  {repo_id}")
    console.print(f"  [bold]File  [/bold]  {filename}")
    console.print(f"  [bold]Dest  [/bold]  {dest}")
    console.print()

    _flush_stdin()
    if not questionary.confirm("Download?", default=True, style=_QSTYLE).ask():
        return

    console.print()
    console.rule("[dim]Downloading from HuggingFace Hub…[/dim]", style="dim")

    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=selected,
            local_dir=str(outdir),
        )
        console.rule(style="dim")
        console.print()
        console.print(f"[success]✔[/success]  Downloaded → [bold]{local_path}[/bold]")

        stem = Path(filename).stem
        models = load_models()
        updated = add_custom_model(models, stem, local_path)
        save_models(updated)
        console.print(f"[success]✔[/success]  Added [bold]{stem}[/bold] to model list.")
        console.print()

    except Exception as exc:
        console.rule(style="dim")
        console.print()
        console.print(f"[error]✖[/error]  Download failed: {exc}")
        console.print()

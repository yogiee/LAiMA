from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import termios
from pathlib import Path
from typing import Any, Optional

import questionary

from laima.config import load_models, save_config, save_models
from laima.models import add_custom_model
from laima.ui import console

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

# Bundled conversion scripts — copied from llama.cpp during install (or setup)
SCRIPTS_DIR = Path.home() / ".config" / "laima" / "scripts"
_SCRIPTS_READY = SCRIPTS_DIR / ".ready"
_HF_SCRIPT = SCRIPTS_DIR / "convert_hf_to_gguf.py"

# Dedicated venv for heavy conversion deps (torch, transformers, etc.)
CONVERSION_VENV = Path.home() / ".config" / "laima" / "conversion-venv"
_VENV_READY = CONVERSION_VENV / ".ready"

# Packages for the conversion venv.
# Using standard PyPI torch on macOS (supports MPS/Metal natively).
# numpy: prefer 1.26.x but allow 2.x on Python 3.13+ where 1.26.x is unavailable.
# torch: prefer 2.6+ but allow newer releases for Python 3.13+.
_CONV_DEPS = [
    "numpy>=1.26.4",
    "sentencepiece>=0.1.98,<0.3.0",
    "transformers>=4.57.1,<5.0.0",
    "gguf>=0.1.0",
    "protobuf>=4.21.0,<5.0.0",
    "huggingface_hub",
    "torch>=2.6.0",
]

# ---------------------------------------------------------------------------
# Quantisation choices
# ---------------------------------------------------------------------------

QUANT_CHOICES = [
    ("Q4_K_M", "4-bit K-quant, medium  — recommended · best balance of size & quality"),
    ("Q5_K_M", "5-bit K-quant, medium  — higher quality, ~25 % larger than Q4_K_M"),
    ("Q8_0",   "8-bit                  — near-lossless, ~2× size of Q4_K_M"),
    ("Q3_K_M", "3-bit K-quant, medium  — smaller, noticeable quality loss"),
    ("Q2_K",   "2-bit K-quant          — very small, significant quality loss"),
    ("F16",    "16-bit float           — full precision, large file"),
]

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

# ---------------------------------------------------------------------------
# Script management
# ---------------------------------------------------------------------------

def is_scripts_ready() -> bool:
    return _SCRIPTS_READY.exists() and _HF_SCRIPT.exists()


def copy_scripts_from(source: str | Path) -> bool:
    """
    Copy convert_hf_to_gguf.py and gguf-py/ from a llama.cpp source tree
    into ~/.config/laima/scripts/.  Returns True on success.
    """
    src = Path(source)
    script_src = src / "convert_hf_to_gguf.py"
    gguf_py_src = src / "gguf-py"

    if not script_src.is_file():
        console.print(f"[error]✖[/error]  convert_hf_to_gguf.py not found in {src}")
        return False

    SCRIPTS_DIR.mkdir(parents=True, exist_ok=True)

    shutil.copy2(script_src, _HF_SCRIPT)

    dest_gguf_py = SCRIPTS_DIR / "gguf-py"
    if dest_gguf_py.exists():
        shutil.rmtree(dest_gguf_py)
    if gguf_py_src.is_dir():
        shutil.copytree(gguf_py_src, dest_gguf_py)

    _SCRIPTS_READY.touch()
    return True


def detect_llama_cpp_path() -> Optional[str]:
    """Scan common dirs for a llama.cpp clone (contains convert_hf_to_gguf.py)."""
    search_roots = [
        Path.home() / "WORK",
        Path.home() / "repos",
        Path.home() / "src",
        Path.home() / "code",
        Path.home() / "projects",
        Path.home(),
    ]
    for root in search_roots:
        if not root.is_dir():
            continue
        try:
            for child in sorted(root.iterdir()):
                if child.is_dir() and (child / "convert_hf_to_gguf.py").is_file():
                    return str(child)
        except PermissionError:
            continue
    return None


def _ensure_scripts(cfg: dict[str, Any]) -> bool:
    """
    Make sure bundled scripts are present.  If not, attempt to copy them from
    llama_cpp_path (config) or auto-detect, then ask the user.
    Returns True if scripts are ready.
    """
    if is_scripts_ready():
        return True

    console.print()
    console.print("[warning]⚠[/warning]   Conversion scripts not yet installed.")

    # Try stored llama_cpp_path first
    stored = cfg.get("llama_cpp_path")
    if stored and Path(stored, "convert_hf_to_gguf.py").is_file():
        if copy_scripts_from(stored):
            console.print("[success]✔[/success]  Scripts installed from saved path.")
            return True

    # Auto-detect
    with console.status("[dim]Searching for llama.cpp…[/dim]", spinner="dots"):
        detected = detect_llama_cpp_path()

    if detected:
        console.print(f"[dim]Found llama.cpp at[/dim] [bold]{detected}[/bold]")
        if questionary.confirm("Copy conversion scripts from here?", default=True, style=_QSTYLE).ask():
            cfg["llama_cpp_path"] = detected
            save_config(cfg)
            return copy_scripts_from(detected)

    # Ask manually
    path = questionary.text(
        "Enter path to your llama.cpp source directory:",
        style=_QSTYLE,
        validate=lambda x: (
            Path(x, "convert_hf_to_gguf.py").is_file()
            or "convert_hf_to_gguf.py not found at that path."
        ),
    ).ask()

    if path:
        cfg["llama_cpp_path"] = path
        save_config(cfg)
        return copy_scripts_from(path)

    return False


# ---------------------------------------------------------------------------
# Conversion venv
# ---------------------------------------------------------------------------

def is_venv_ready() -> bool:
    return _VENV_READY.exists() and (CONVERSION_VENV / "bin" / "python").exists()


def setup_conversion_venv() -> bool:
    """Create ~/.config/laima/conversion-venv/ and install deps. Returns True on success."""
    console.print()
    console.print("[bold]Setting up conversion environment[/bold]")
    console.print(
        f"[dim]A dedicated venv will be created at:\n"
        f"  {CONVERSION_VENV}\n\n"
        "This installs PyTorch + HuggingFace Transformers.\n"
        "Expect a large download (~2–4 GB) on first run.[/dim]"
    )
    console.print()

    if not questionary.confirm("Proceed?", default=True, style=_QSTYLE).ask():
        return False

    venv_pip = CONVERSION_VENV / "bin" / "pip"

    console.print("[dim]Creating virtual environment…[/dim]")
    r = subprocess.run([sys.executable, "-m", "venv", str(CONVERSION_VENV)])
    if r.returncode != 0:
        console.print("[error]✖[/error]  Failed to create venv.")
        return False

    subprocess.run([str(venv_pip), "install", "--quiet", "--upgrade", "pip"], check=False)

    console.print("[dim]Installing dependencies (torch last — largest package)…[/dim]")
    r = subprocess.run([str(venv_pip), "install"] + _CONV_DEPS)
    if r.returncode != 0:
        console.print("[error]✖[/error]  Dependency installation failed.")
        return False

    _VENV_READY.touch()
    console.print("[success]✔[/success]  Conversion environment ready.")
    return True


def _ensure_venv() -> bool:
    if is_venv_ready():
        return True
    return setup_conversion_venv()


# ---------------------------------------------------------------------------
# HuggingFace → GGUF
# ---------------------------------------------------------------------------

def _sanitize(model_id: str) -> str:
    """Turn a HF model ID or local path into a safe filename stem."""
    name = Path(model_id).name if (os.sep in model_id) else model_id.split("/")[-1]
    return re.sub(r"[^\w.-]", "-", name)


def convert_hf_flow(cfg: dict[str, Any], prefill_model_id: Optional[str] = None) -> None:
    """Interactive: HuggingFace model ID (or local dir) → GGUF."""
    _flush_stdin()
    if not _ensure_scripts(cfg):
        console.print("[error]✖[/error]  Conversion scripts unavailable — cannot continue.")
        return

    if not _ensure_venv():
        return

    console.print()

    # Model source
    model_src = questionary.text(
        "HuggingFace model ID or local model directory:",
        instruction="(e.g. microsoft/Phi-3-mini-4k-instruct  or  /path/to/model)",
        default=prefill_model_id or "",
        style=_QSTYLE,
        validate=lambda x: bool(x.strip()) or "Please enter a model ID or path.",
    ).ask()
    if not model_src:
        return
    model_src = model_src.strip()

    # Output type
    outtype = questionary.select(
        "Output type:",
        choices=[
            questionary.Choice("auto  — let the script decide (recommended)", value="auto"),
            questionary.Choice("f16   — 16-bit float, full precision",        value="f16"),
            questionary.Choice("q8_0  — 8-bit quantised",                    value="q8_0"),
        ],
        style=_QSTYLE,
    ).ask()
    if not outtype:
        return

    # Output directory
    default_outdir = str(Path.home() / "models")
    outdir_str = questionary.text(
        "Output directory:",
        default=default_outdir,
        style=_QSTYLE,
    ).ask()
    if outdir_str is None:
        return
    outdir = Path(outdir_str.strip() or default_outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    stem = _sanitize(model_src)
    outfile = outdir / f"{stem}-{outtype}.gguf"

    # Confirm
    console.print()
    console.print(f"  [bold]Source[/bold]  {model_src}")
    console.print(f"  [bold]Type  [/bold]  {outtype}")
    console.print(f"  [bold]Output[/bold]  {outfile}")
    console.print()
    if not questionary.confirm("Start conversion?", default=True, style=_QSTYLE).ask():
        return

    # Run — script finds gguf-py relative to itself (same directory)
    venv_python = str(CONVERSION_VENV / "bin" / "python")
    console.print()
    console.rule("[dim]convert_hf_to_gguf.py[/dim]", style="dim")

    # Detect whether model_src is a local path or a HuggingFace Hub ID.
    # If it doesn't contain a path separator and doesn't exist as a directory,
    # treat it as an HF Hub ID and pass --remote so the script downloads it.
    is_local = Path(model_src).is_dir()
    cmd = [venv_python, str(_HF_SCRIPT), model_src,
           "--outfile", str(outfile), "--outtype", outtype]
    if not is_local:
        cmd.append("--remote")

    result = subprocess.run(cmd, cwd=str(SCRIPTS_DIR))
    console.rule(style="dim")
    console.print()

    if result.returncode != 0:
        console.print("[error]✖[/error]  Conversion failed — check output above.")
        return

    console.print(f"[success]✔[/success]  Saved → [bold]{outfile}[/bold]")
    _offer_add_to_list(str(outfile), stem)


# ---------------------------------------------------------------------------
# Quantise GGUF → smaller GGUF
# ---------------------------------------------------------------------------

def quantize_flow(_cfg: dict[str, Any]) -> None:
    """Interactive: existing GGUF → quantised GGUF via llama-quantize."""
    _flush_stdin()
    if not _cmd_exists("llama-quantize"):
        console.print()
        console.print("[error]✖[/error]  [bold]llama-quantize[/bold] not found in PATH.")
        console.print("    It is part of llama.cpp — build it and ensure it is on your PATH.")
        console.print()
        return

    console.print()

    # Source: pick from model list or enter manually
    models_db = load_models()
    compatible = sorted(
        (name, info["gguf_path"])
        for name, info in models_db.items()
        if info.get("gguf_compatible") and info.get("gguf_path")
    )
    source_choices = [
        questionary.Choice(title=name, value=path)
        for name, path in compatible
    ]
    source_choices.append(questionary.Choice(title="Enter path manually…", value="__manual__"))

    src_path: Optional[str] = questionary.select(
        "Source GGUF:",
        choices=source_choices,
        style=_QSTYLE,
        use_search_filter=True,
        use_jk_keys=False,
    ).ask()
    if not src_path:
        return

    if src_path == "__manual__":
        src_path = questionary.text(
            "Path to source GGUF:",
            style=_QSTYLE,
            validate=lambda x: Path(x).is_file() or "File not found.",
        ).ask()
        if not src_path:
            return

    src = Path(src_path)

    # Quantisation type
    quant_type: Optional[str] = questionary.select(
        "Quantisation type:",
        choices=[
            questionary.Choice(title=f"{q:<10} {desc}", value=q)
            for q, desc in QUANT_CHOICES
        ],
        style=_QSTYLE,
    ).ask()
    if not quant_type:
        return

    # Output path
    default_out = str(src.parent / f"{src.stem}-{quant_type.lower()}.gguf")
    out_str = questionary.text(
        "Output file:",
        default=default_out,
        style=_QSTYLE,
    ).ask()
    if not out_str:
        return
    out = Path(out_str.strip())

    # Confirm
    console.print()
    console.print(f"  [bold]Input [/bold]  {src}")
    console.print(f"  [bold]Output[/bold]  {out}")
    console.print(f"  [bold]Quant [/bold]  {quant_type}")
    console.print()
    if not questionary.confirm("Start quantisation?", default=True, style=_QSTYLE).ask():
        return

    # Run
    console.print()
    console.rule("[dim]llama-quantize[/dim]", style="dim")
    result = subprocess.run(["llama-quantize", str(src), str(out), quant_type])
    console.rule(style="dim")
    console.print()

    if result.returncode != 0:
        console.print("[error]✖[/error]  Quantisation failed — check output above.")
        return

    console.print(f"[success]✔[/success]  Saved → [bold]{out}[/bold]")
    _offer_add_to_list(str(out), out.stem)


# ---------------------------------------------------------------------------
# Refresh scripts from a new llama.cpp source
# ---------------------------------------------------------------------------

def refresh_scripts_flow(cfg: dict[str, Any]) -> None:
    """Copy fresh conversion scripts from a llama.cpp source tree."""
    console.print()
    stored = cfg.get("llama_cpp_path", "")

    path = questionary.text(
        "Path to llama.cpp source directory:",
        default=stored or "",
        style=_QSTYLE,
        validate=lambda x: (
            Path(x, "convert_hf_to_gguf.py").is_file()
            or "convert_hf_to_gguf.py not found at that path."
        ),
    ).ask()
    if not path:
        return

    cfg["llama_cpp_path"] = path
    save_config(cfg)

    if copy_scripts_from(path):
        console.print("[success]✔[/success]  Conversion scripts updated.")
    console.print()


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def _offer_add_to_list(gguf_path: str, suggested_name: str) -> None:
    """Offer to register the new file in laima's model list."""
    console.print()
    if not questionary.confirm(
        "Add to laima's model list?", default=True, style=_QSTYLE
    ).ask():
        return

    name = questionary.text(
        "Display name:",
        default=suggested_name,
        style=_QSTYLE,
        validate=lambda x: bool(x.strip()) or "Please enter a name.",
    ).ask()
    if not name:
        return

    models = load_models()
    models = add_custom_model(models, name.strip(), gguf_path)
    save_models(models)
    console.print(f"[success]✔[/success]  [bold]{name}[/bold] added to model list.")


def _cmd_exists(name: str) -> bool:
    return subprocess.run(["which", name], capture_output=True).returncode == 0


# ---------------------------------------------------------------------------
# Submenu
# ---------------------------------------------------------------------------

def _flush_stdin() -> None:
    try:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass


def conversion_menu(cfg: dict[str, Any]) -> None:
    while True:
        console.print()
        _flush_stdin()
        choice = questionary.select(
            "Convert Model:",
            choices=[
                questionary.Choice("1.  HuggingFace → GGUF",          value="hf"),
                questionary.Choice("2.  Quantise GGUF",                value="quant"),
                questionary.Choice("3.  Refresh conversion scripts",   value="refresh"),
                questionary.Choice("← Back",                           value="__back__"),
            ],
            style=_QSTYLE,
        ).ask()

        if choice is None or choice == "__back__":
            break
        elif choice == "hf":
            convert_hf_flow(cfg)
        elif choice == "quant":
            quantize_flow(cfg)
        elif choice == "refresh":
            refresh_scripts_flow(cfg)

from __future__ import annotations

import os
import subprocess
import sys
import termios
from typing import Any, Optional

import questionary
from rich.panel import Panel

from laima.config import (
    DEFAULTS,
    is_first_run,
    load_config,
    load_models,
    save_config,
    save_models,
)
from laima.convert import conversion_menu
from laima.hf import hf_pull_flow
from laima.models import get_compatible_models, scan_models
from laima.server import run_server
from laima.ui import console

LLAMA_CPP_INSTALL_URL = "https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md"
OLLAMA_INSTALL_URL = "https://ollama.com"

# questionary style — matches rich's blue palette
_QSTYLE = questionary.Style(
    [
        ("qmark", "fg:#61afef bold"),
        ("question", "bold"),
        ("answer", "fg:#98c379 bold"),
        ("pointer", "fg:#61afef bold"),
        ("highlighted", "fg:#61afef bold"),
        ("selected", "fg:#98c379"),
        ("separator", "fg:#5c6370"),
        ("instruction", "fg:#5c6370"),
    ]
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _banner() -> None:
    console.print()
    console.rule("[bold blue]LAiMA[/bold blue]  ·  llama.cpp Launcher", style="blue")
    console.print()


def _cmd_exists(name: str) -> bool:
    return (
        subprocess.run(["which", name], capture_output=True).returncode == 0
    )


def _check_deps() -> bool:
    ok = True
    if not _cmd_exists("llama-server"):
        console.print("[error]✖[/error]  [bold]llama-server[/bold] not found in PATH.")
        console.print(f"    Install llama.cpp: [cyan]{LLAMA_CPP_INSTALL_URL}[/cyan]")
        ok = False
    if not _cmd_exists("ollama"):
        console.print("[error]✖[/error]  [bold]ollama[/bold] not found in PATH.")
        console.print(f"    Install Ollama: [cyan]{OLLAMA_INSTALL_URL}[/cyan]")
        ok = False
    return ok


# ---------------------------------------------------------------------------
# First-run setup
# ---------------------------------------------------------------------------

def _first_run(cfg: dict[str, Any]) -> dict[str, Any]:
    console.print("[bold]First-run setup[/bold]\n")

    if not _check_deps():
        console.print()
        console.print("[warning]⚠[/warning]   Fix the issues above and re-run LAiMA.")
        sys.exit(1)

    console.print("[success]✔[/success]   Dependencies OK")
    console.print("[muted]Scanning Ollama models…[/muted]")

    with console.status("[muted]Scanning…[/muted]", spinner="dots"):
        updated, new_count, _, _ = scan_models({})

    save_models(updated)
    save_config(cfg)

    compat = sum(1 for v in updated.values() if v.get("gguf_compatible"))
    console.print(
        f"[success]✔[/success]   Found [bold]{new_count}[/bold] Ollama model(s), "
        f"[bold]{compat}[/bold] GGUF-compatible."
    )
    console.print()
    return cfg


# ---------------------------------------------------------------------------
# Update model list
# ---------------------------------------------------------------------------

def _do_update_models() -> None:
    with console.status("[muted]Scanning Ollama models…[/muted]", spinner="dots"):
        models = load_models()
        updated, new_c, rm_c, unch_c = scan_models(models)
        save_models(updated)

    console.print(
        f"[success]✔[/success]   Scan complete — "
        f"[bold]{new_c}[/bold] new, "
        f"[bold]{rm_c}[/bold] removed, "
        f"[bold]{unch_c}[/bold] unchanged."
    )
    console.print()


# ---------------------------------------------------------------------------
# Pull model from Ollama
# ---------------------------------------------------------------------------

def _do_pull_model() -> None:
    console.print()
    name = questionary.text(
        "Enter model name (e.g. qwen3:4b):",
        style=_QSTYLE,
        validate=lambda x: bool(x.strip()) or "Please enter a model name.",
    ).ask()

    if not name:
        return

    name = name.strip()
    console.print()
    console.rule(f"[muted]Pulling {name}[/muted]", style="dim")

    try:
        result = subprocess.run(["ollama", "pull", name])
        console.rule(style="dim")
        console.print()

        if result.returncode == 0:
            console.print(f"[success]✔[/success]   Pulled [bold]{name}[/bold] successfully.")
            ans = questionary.confirm(
                "Update model list now?", default=True, style=_QSTYLE
            ).ask()
            if ans:
                _do_update_models()
        else:
            console.print(
                f"[error]✖[/error]   Pull failed (exit code {result.returncode})."
            )
    except FileNotFoundError:
        console.print("[error]✖[/error]   ollama not found.")

    console.print()


# ---------------------------------------------------------------------------
# Model selection
# ---------------------------------------------------------------------------

def _select_model(
    prompt: str, compatible: list[dict[str, Any]]
) -> Optional[dict[str, Any]]:
    if not compatible:
        console.print("[warning]⚠[/warning]   No GGUF-compatible models found.")
        console.print(
            "   Run [bold]Update Model List[/bold] or [bold]Pull Model from Ollama[/bold] first."
        )
        console.print()
        return None

    _BACK = "__back__"
    choices = [
        questionary.Choice(title=m["label"], value=m) for m in compatible
    ]
    choices.append(questionary.Choice(title="← Back", value=_BACK))

    console.print()
    result = questionary.select(
        prompt,
        choices=choices,
        style=_QSTYLE,
        use_search_filter=True,
        use_jk_keys=False,
    ).ask()
    return None if (result is None or result == _BACK) else result


def _select_port(cfg: dict[str, Any]) -> Optional[int]:
    console.print()
    raw = questionary.text(
        f"Port (default {cfg['port']}):",
        default=str(cfg["port"]),
        style=_QSTYLE,
        validate=lambda x: (
            (x.isdigit() and 1 <= int(x) <= 65535)
            or "Enter a valid port number (1–65535)."
        ),
    ).ask()

    return int(raw) if raw is not None else None


# ---------------------------------------------------------------------------
# Mode flows
# ---------------------------------------------------------------------------

def _do_single(cfg: dict[str, Any]) -> None:
    models = load_models()
    compatible = get_compatible_models(models)

    model = _select_model("Select model:", compatible)
    if model is None:
        return

    port = _select_port(cfg)
    if port is None:
        return

    console.print()
    console.print(
        f"[success]►[/success]  [bold]{model['name']}[/bold]   port [bold]{port}[/bold]"
    )

    run_server(
        model_path=model["path"],
        model_label=model["name"],
        port=port,
        cfg=cfg,
    )


def _do_speculative(cfg: dict[str, Any]) -> None:
    models = load_models()
    compatible = get_compatible_models(models)

    draft = _select_model("Select draft model (smaller/faster):", compatible)
    if draft is None:
        return

    target = _select_model("Select target model (larger/better):", compatible)
    if target is None:
        return

    port = _select_port(cfg)
    if port is None:
        return

    console.print()
    console.print(f"[success]►[/success]  Draft  [bold]{draft['name']}[/bold]")
    console.print(f"[success]►[/success]  Target [bold]{target['name']}[/bold]")
    console.print(f"[success]►[/success]  Port   [bold]{port}[/bold]")

    run_server(
        model_path=target["path"],
        model_label=target["name"],
        port=port,
        cfg=cfg,
        draft_path=draft["path"],
        draft_label=draft["name"],
    )


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------

_SETTINGS_META = [
    ("host",               "Host",                       str,  None),
    ("port",               "Default port",               int,  (1, 65535)),
    ("ctx_size",           "Context size",               int,  (128, 131072)),
    ("n_gpu_layers",       "GPU layers (main)",          int,  (0, 999)),
    ("n_gpu_layers_draft", "GPU layers (draft)",         int,  (0, 999)),
    ("threads",            "CPU threads",                int,  (1, 512)),
    ("draft_min",          "Draft min tokens",           int,  (0, 64)),
    ("draft_max",          "Draft max tokens",           int,  (1, 256)),
    ("llama_cpp_path",     "llama.cpp source path",      str,  None),
]


def _validate_setting(raw: str, typ: type, rng: Optional[tuple]) -> bool | str:
    if typ is int:
        if not raw.isdigit():
            return "Please enter a whole number."
        if rng and not (rng[0] <= int(raw) <= rng[1]):
            return f"Enter a number between {rng[0]} and {rng[1]}."
    return True


def _do_settings(cfg: dict[str, Any]) -> dict[str, Any]:
    while True:
        console.print()
        choices = [
            questionary.Choice(
                title=f"{label:<30} {cfg.get(key) or '(not set)'}",
                value=key,
            )
            for key, label, *_ in _SETTINGS_META
        ]
        choices.append(questionary.Choice(title="← Back", value="__back__"))

        key = questionary.select(
            "Settings:", choices=choices, style=_QSTYLE
        ).ask()

        if key is None or key == "__back__":
            break

        for k, label, typ, rng in _SETTINGS_META:
            if k != key:
                continue

            current_val = cfg.get(key)
            raw = questionary.text(
                f"{label} (current: {current_val or 'not set'}):",
                default=str(current_val) if current_val is not None else "",
                style=_QSTYLE,
                validate=lambda x, _t=typ, _r=rng: _validate_setting(x, _t, _r),
            ).ask()

            if raw is not None:
                cfg[key] = typ(raw) if raw.strip() else None
                save_config(cfg)
                console.print(f"[success]✔[/success]   {label} → [bold]{cfg[key]}[/bold]")
            break

    return cfg


# ---------------------------------------------------------------------------
# Main menu
# ---------------------------------------------------------------------------

def _flush_stdin() -> None:
    """Discard any buffered keypresses before presenting a menu."""
    try:
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
    except Exception:
        pass


def _do_list_models() -> None:
    from rich.table import Table

    models = load_models()
    compatible = get_compatible_models(models)

    if not compatible:
        console.print()
        console.print("[warning]⚠[/warning]   No models in database. Run Update Model List first.")
        console.print()
        return

    table = Table(
        show_header=True,
        header_style="bold blue",
        border_style="dim",
        show_lines=False,
        pad_edge=True,
    )
    table.add_column("Name",   style="bold",  no_wrap=True)
    table.add_column("Size",   justify="right")
    table.add_column("Origin", justify="center")
    table.add_column("GGUF",   justify="center")

    for m in compatible:
        raw = models.get(m["name"], {})
        size = raw.get("size_gb")
        size_str = f"{size:.1f} GB" if size else "—"
        origin = raw.get("origin", "ollama")
        gguf_mark = "[green]✔[/green]" if raw.get("gguf_compatible") else "[red]✖[/red]"
        table.add_row(m["name"], size_str, origin, gguf_mark)

    console.print()
    console.print(table)
    console.print(f"  [dim]{len(compatible)} model(s)[/dim]")
    console.print()
    _flush_stdin()
    questionary.press_any_key_to_continue(style=_QSTYLE).ask()


def _main_loop(cfg: dict[str, Any]) -> None:
    while True:
        _banner()
        _flush_stdin()  # prevent buffered Enter from auto-selecting on return

        choice = questionary.select(
            "What would you like to do?",
            choices=[
                questionary.Choice("1.  Start — Single Model",          value="single"),
                questionary.Choice("2.  Start — Speculative Decoding",  value="spec"),
                questionary.Choice("3.  List Models",                   value="list"),
                questionary.Choice("4.  Update Model List",             value="update"),
                questionary.Choice("5.  Pull Model from Ollama",        value="pull_ollama"),
                questionary.Choice("6.  Pull Model from HuggingFace",   value="pull_hf"),
                questionary.Choice("7.  Convert Model",                 value="convert"),
                questionary.Choice("8.  Settings",                      value="settings"),
                questionary.Choice("9.  Exit",                          value="exit"),
            ],
            style=_QSTYLE,
            use_search_filter=False,
        ).ask()

        if choice is None or choice == "exit":
            console.print()
            break
        elif choice == "single":
            _do_single(cfg)
        elif choice == "spec":
            _do_speculative(cfg)
        elif choice == "list":
            _do_list_models()
        elif choice == "update":
            _do_update_models()
        elif choice == "pull_ollama":
            _do_pull_model()
        elif choice == "pull_hf":
            hf_pull_flow(cfg)
        elif choice == "convert":
            conversion_menu(cfg)
        elif choice == "settings":
            cfg = _do_settings(cfg)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    # Ensure Homebrew and common local bins are on PATH
    _inject_path("/opt/homebrew/bin", "/usr/local/bin", str(os.path.expanduser("~/.local/bin")))

    cfg = load_config()

    if is_first_run():
        _banner()
        cfg = _first_run(cfg)

    else:
        # Silently pick up any new Ollama models added since last run
        with console.status("", spinner="dots"):
            models = load_models()
            updated, new_c, _, _ = scan_models(models)
            if new_c:
                save_models(updated)

    _main_loop(cfg)


def _inject_path(*dirs: str) -> None:
    current = os.environ.get("PATH", "")
    extra = ":".join(d for d in dirs if d not in current.split(":"))
    if extra:
        os.environ["PATH"] = extra + ":" + current


if __name__ == "__main__":
    main()

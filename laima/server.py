from __future__ import annotations

import os
import re
import select
import signal
import subprocess
import sys
import tempfile
import termios
import threading
import time
import tty
from pathlib import Path
from typing import Any, Optional

import psutil
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from laima.ui import console


# ---------------------------------------------------------------------------
# System stats
# ---------------------------------------------------------------------------

def _gpu_percent_apple() -> Optional[int]:
    """
    Apple Silicon GPU utilisation via ioreg (no sudo required).
    Returns 0-100 or None if unavailable.
    """
    try:
        result = subprocess.run(
            ["ioreg", "-r", "-d", "1", "-n", "AGXAccelerator"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        m = re.search(r'"GPU Activity\(%\)"\s*=\s*(\d+)', result.stdout)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    return None


def _bar(pct: float, width: int = 20) -> str:
    pct = max(0.0, min(100.0, pct))
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


# ---------------------------------------------------------------------------
# Dashboard panel
# ---------------------------------------------------------------------------

def _build_panel(
    model_label: str,
    port: int,
    pid: int,
    start_time: float,
    draft_label: Optional[str],
) -> Panel:
    elapsed = int(time.time() - start_time)
    h, m, s = elapsed // 3600, (elapsed % 3600) // 60, elapsed % 60
    uptime = f"{h:02d}:{m:02d}:{s:02d}"

    cpu_pct = psutil.cpu_percent(interval=None)
    mem = psutil.virtual_memory()
    mem_used_gb = mem.used / 1024 ** 3
    mem_total_gb = mem.total / 1024 ** 3

    gpu_pct = _gpu_percent_apple()

    proc_mem_gb = 0.0
    try:
        proc_mem_gb = psutil.Process(pid).memory_info().rss / 1024 ** 3
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        pass

    t = Text()

    t.append("● RUNNING", style="bold green")
    t.append(f"   PID {pid} · Uptime {uptime}\n", style="dim")
    t.append("\n")

    if draft_label:
        t.append("Draft   ", style="bold")
        t.append(f"{draft_label}\n")
        t.append("Target  ", style="bold")
        t.append(f"{model_label}\n")
    else:
        t.append("Model   ", style="bold")
        t.append(f"{model_label}\n")

    t.append("URL     ", style="bold")
    t.append(f"http://localhost:{port}\n", style="cyan")
    t.append("\n")

    t.append("CPU   ", style="bold")
    t.append(f"{_bar(cpu_pct)}  {cpu_pct:5.1f}%\n")

    t.append("MEM   ", style="bold")
    t.append(f"{_bar(mem.percent)}  {mem_used_gb:.1f}/{mem_total_gb:.0f} GB\n")

    if gpu_pct is not None:
        t.append("GPU   ", style="bold")
        t.append(f"{_bar(float(gpu_pct))}  {gpu_pct:5d}%\n")

    proc_pct = (proc_mem_gb / mem_total_gb * 100) if mem_total_gb else 0.0
    t.append("PROC  ", style="bold")
    t.append(f"{_bar(proc_pct)}  {proc_mem_gb:.1f} GB\n")

    t.append("\n")
    t.append("Press ", style="dim")
    t.append("Q", style="bold")
    t.append(" or ", style="dim")
    t.append("Ctrl+C", style="bold")
    t.append(" to stop the server", style="dim")

    return Panel(
        t,
        title="[bold blue]LAiMA[/bold blue]",
        border_style="blue",
        padding=(1, 2),
    )


# ---------------------------------------------------------------------------
# Error diagnosis
# ---------------------------------------------------------------------------

_HINTS: list[tuple[str, str]] = [
    (
        "wrong array length",
        "Model/llama.cpp version mismatch. Try re-pulling the model via "
        "'Pull Model from Ollama', or rebuild llama.cpp from the latest source.",
    ),
    (
        "key not found in model",
        "llama-server does not recognise a required model key — your llama.cpp build "
        "is likely too old for this model architecture. Rebuild llama.cpp from the "
        "latest source (git pull && cmake --build).",
    ),
    (
        "failed to load model",
        "llama-server could not load the GGUF file. Check the model path is valid "
        "and the file is not corrupted.",
    ),
    (
        "unknown argument",
        "llama-server rejected an unrecognised flag. Check your llama.cpp version "
        "is recent enough, or adjust settings in laima's Settings menu.",
    ),
    (
        "out of memory",
        "Not enough RAM/VRAM to load this model. Try reducing GPU layers "
        "(Settings → GPU layers) or choose a smaller/more-quantised model.",
    ),
    (
        "address already in use",
        "Port is already occupied by another process. Choose a different port.",
    ),
]


def _diagnose(log: str) -> Optional[str]:
    log_lower = log.lower()
    for pattern, hint in _HINTS:
        if pattern in log_lower:
            return hint
    return None


# ---------------------------------------------------------------------------
# Server lifecycle
# ---------------------------------------------------------------------------

def run_server(
    model_path: str,
    model_label: str,
    port: int,
    cfg: dict[str, Any],
    draft_path: Optional[str] = None,
    draft_label: Optional[str] = None,
) -> None:
    """Launch llama-server and show a live stats dashboard until stopped."""

    logfile = tempfile.mktemp(prefix="laima-", suffix=".log")

    cmd = [
        "llama-server",
        "--host", cfg["host"],
        "--port", str(port),
        "--ctx-size", str(cfg["ctx_size"]),
        "--n-gpu-layers", str(cfg["n_gpu_layers"]),
        "--threads", str(cfg["threads"]),
        "--model", model_path,
        "--alias", model_label,
        # Note: --log-disable is intentionally omitted — it suppresses error
        # messages too, making crash diagnosis impossible.
    ]

    if draft_path:
        cmd += [
            "--model-draft", draft_path,
            "--n-gpu-layers-draft", str(cfg["n_gpu_layers_draft"]),
            "--draft-min", str(cfg["draft_min"]),
            "--draft-max", str(cfg["draft_max"]),
        ]

    console.print()
    console.print("[dim]Starting llama-server…[/dim]")

    # Write the exact command to the top of the logfile for easy manual replay
    with open(logfile, "w") as lf:
        lf.write("CMD: " + " ".join(cmd) + "\n\n")
        proc = subprocess.Popen(cmd, stdout=lf, stderr=lf)

    start_time = time.time()

    # Poll for up to 5 s to detect an immediate crash
    for _ in range(10):
        time.sleep(0.5)
        if proc.poll() is not None:
            break

    if proc.poll() is not None:
        console.print(f"[bold red]✖[/bold red]  Server failed to start (exit code {proc.returncode}).\n")
        # Print raw log via sys.stdout to avoid Rich mangling llama.cpp's
        # bracket-heavy output (e.g. "[llama]", "n_ctx = 8192", etc.)
        try:
            log_text = Path(logfile).read_text()
        except OSError:
            log_text = "(log unavailable)"
        sys.stdout.write("─" * 80 + "\n")
        sys.stdout.write(log_text)
        sys.stdout.write("\n" + "─" * 80 + "\n")
        sys.stdout.write(f"Full log saved at: {logfile}\n")
        hint = _diagnose(log_text)
        if hint:
            sys.stdout.write(f"\nHint: {hint}\n")
        sys.stdout.write("\n")
        sys.stdout.flush()
        return  # keep logfile for inspection

    # --- install signal handlers ---
    stop_event = threading.Event()
    orig_sigint = signal.getsignal(signal.SIGINT)
    orig_sigterm = signal.getsignal(signal.SIGTERM)

    def _handle_sig(sig: int, frame: Any) -> None:  # noqa: ANN001
        stop_event.set()

    signal.signal(signal.SIGINT, _handle_sig)
    signal.signal(signal.SIGTERM, _handle_sig)

    # --- set terminal to cbreak + no-echo for single-keypress detection ---
    fd = sys.stdin.fileno()
    old_term = termios.tcgetattr(fd)
    try:
        tty.setcbreak(fd)
        new_term = termios.tcgetattr(fd)
        new_term[3] &= ~termios.ECHO  # disable echo
        termios.tcsetattr(fd, termios.TCSADRAIN, new_term)

        with Live(console=console, auto_refresh=False, screen=False) as live:
            while not stop_event.is_set():
                # Non-blocking check for keypress
                if select.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1)
                    if ch.lower() == "q":
                        break

                if proc.poll() is not None:
                    live.stop()
                    sys.stdout.write("\n✖  Server died unexpectedly.\n")
                    sys.stdout.write(f"   Log: {logfile}\n\n")
                    sys.stdout.flush()
                    return  # keep logfile

                live.update(
                    _build_panel(
                        model_label=model_label,
                        port=port,
                        pid=proc.pid,
                        start_time=start_time,
                        draft_label=draft_label,
                    )
                )
                live.refresh()
                time.sleep(1)

    except KeyboardInterrupt:
        pass
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_term)
        signal.signal(signal.SIGINT, orig_sigint)
        signal.signal(signal.SIGTERM, orig_sigterm)

    # --- shutdown ---
    console.print()
    console.print("[yellow]⚠[/yellow]   Stopping server…")
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        console.print("[yellow]⚠[/yellow]   Forcing kill…")
        proc.kill()
        proc.wait()
    console.print("[green]✔[/green]   Server stopped.")
    _cleanup_log(logfile)


def _cleanup_log(logfile: str) -> None:
    try:
        os.unlink(logfile)
    except OSError:
        pass



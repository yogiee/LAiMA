# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

LAiMA is a personal macOS terminal app for launching local LLMs via [llama.cpp](https://github.com/ggml-org/llama.cpp) server. It discovers models from a local Ollama installation, validates GGUF compatibility, and provides an interactive TUI for server management.

## Setup & Installation

```bash
./install.sh          # one-time: creates .venv, installs deps, symlinks ~/.local/bin/laima
laima                 # run from anywhere
```

**Dependencies** (installed automatically into `.venv`): `rich`, `questionary`, `psutil`
**Runtime requirements**: `llama-server` and `ollama` must be on `PATH` (checked at startup).

## Architecture

```
laima/
├── main.py      # Entry point + all menu flows (single model, speculative, pull, settings)
├── config.py    # Config R/W (~/.config/laima/config.json & models.json), defaults
├── models.py    # Ollama model scanning, manifest parsing, GGUF magic-byte validation
├── server.py    # llama-server subprocess, live stats dashboard, clean shutdown
└── ui.py        # Shared Rich console + theme
```

### Key data files (runtime, not in repo)

| File | Purpose |
|---|---|
| `~/.config/laima/config.json` | User settings (port, ctx_size, gpu_layers, threads, etc.) |
| `~/.config/laima/models.json` | Model database cache: gguf_path, gguf_compatible, size_gb |

### Model resolution (`models.py`)

Ollama stores model weights as GGUF blobs under `~/.ollama/models/blobs/`. The path is resolved by:
1. Walking `~/.ollama/models/manifests/registry.ollama.ai/` to find the manifest for `<family>/<tag>`
2. Selecting the layer with `mediaType == "application/vnd.ollama.image.model"` (fallback: first layer)
3. Converting the digest `sha256:xxxx` → filename `sha256-xxxx`
4. Reading the first 4 bytes to verify the GGUF magic (`GGUF` = `0x47475546`)

### Server dashboard (`server.py`)

`run_server()` launches `llama-server` as a subprocess with `--log-disable`. A `rich.Live` panel refreshes every second showing CPU/MEM/GPU/PROC bars. Terminal is put into `cbreak + no-echo` mode so pressing `Q` stops the server immediately (no Enter needed). `SIGINT`/`SIGTERM` are also handled.

GPU stats use `ioreg -r -d 1 -n AGXAccelerator` (Apple Silicon only, no sudo required); returns `None` on Intel or if unavailable.

### First-run & model scanning

- On first run: checks for `llama-server` and `ollama`, scans all Ollama models, writes config and models.json.
- On every subsequent run: silently scans for *new* Ollama models (skips existing entries) and adds them to the cache.
- "Update Model List" (menu option 3) does a full re-scan with user-visible results.

## llama-server flags used

| Flag | Config key | Default |
|---|---|---|
| `--host` | `host` | `127.0.0.1` |
| `--port` | `port` | `8080` |
| `--ctx-size` | `ctx_size` | `8192` |
| `--n-gpu-layers` | `n_gpu_layers` | `99` |
| `--n-gpu-layers-draft` | `n_gpu_layers_draft` | `99` |
| `--threads` | `threads` | `8` |
| `--draft-min` | `draft_min` | `0` |
| `--draft-max` | `draft_max` | `16` |

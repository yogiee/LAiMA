# LAiMA — llama.cpp Launcher

A polished macOS terminal app for running local LLMs via [llama.cpp](https://github.com/ggml-org/llama.cpp).

LAiMA handles the full lifecycle: discover models, pull from Ollama or HuggingFace, convert to GGUF if needed, and launch `llama-server` with a live stats dashboard.

![Python](https://img.shields.io/badge/python-3.10%2B-blue)
![Platform](https://img.shields.io/badge/platform-macOS-lightgrey)
![License](https://img.shields.io/badge/license-MIT-green)

---

## Features

### Server modes
- **Single Model** — launch any GGUF-compatible model via `llama-server`
- **Speculative Decoding** — pair a small draft model with a larger target for faster inference

### Live dashboard
While the server runs, LAiMA shows a real-time stats panel:
```
● RUNNING   PID 12345 · Uptime 00:03:42

Model   qwen3:4b
URL     http://localhost:8080

CPU   ████████░░░░░░░░░░░░   42.3%
MEM   ███████████░░░░░░░░░   18.2/64 GB
GPU   ██████░░░░░░░░░░░░░░   31%
PROC  █░░░░░░░░░░░░░░░░░░░    2.1 GB

Press Q or Ctrl+C to stop the server
```
Press `Q` or `Ctrl+C` to stop. Crash logs are shown automatically with plain-English diagnosis hints.

### Model management
- **Ollama discovery** — automatically finds all GGUF-compatible models installed via Ollama
- **Pull from Ollama** — run `ollama pull` directly from the app
- **Pull from HuggingFace** — browse a repo's GGUF variants (e.g. `bartowski/Qwen3-0.6B-GGUF`), pick a quantization, and download it; if the repo has no GGUF files, LAiMA offers to convert automatically
- **List Models** — table view of all models with size, origin, and compatibility

### Conversion
- **HuggingFace → GGUF** — converts any HF model using `convert_hf_to_gguf.py` (isolated conversion venv, supports remote download)
- **Quantize GGUF** — re-quantize an existing GGUF to a smaller format via `llama-quantize`

---

## Requirements

| Dependency | Notes |
|---|---|
| macOS | Apple Silicon recommended (GPU stats via `ioreg`) |
| Python 3.10+ | Tested on 3.14 |
| [`llama-server`](https://github.com/ggml-org/llama.cpp/blob/master/docs/build.md) | Must be in PATH |
| [Ollama](https://ollama.com) | Optional — needed for Ollama model discovery and pull |

---

## Install

```bash
git clone https://github.com/yogiee/LAiMA.git
cd LAiMA
./install.sh
```

The installer:
1. Creates a `.venv` and installs all Python dependencies
2. Places a `laima` launcher at `~/.local/bin/laima`
3. Copies conversion scripts from a bundled `llama.cpp` reference clone (if present)

> **Note:** If `~/.local/bin` is not in your PATH, add this to `~/.zshrc`:
> ```bash
> export PATH="$HOME/.local/bin:$PATH"
> ```

Then run:
```bash
laima
```

---

## Configuration

Settings are stored in `~/.config/laima/config.json` and editable via the **Settings** menu:

| Setting | Default | Description |
|---|---|---|
| `host` | `127.0.0.1` | llama-server bind address |
| `port` | `8080` | Default port |
| `ctx_size` | `4096` | Context window size |
| `n_gpu_layers` | `99` | GPU layers for main model |
| `n_gpu_layers_draft` | `99` | GPU layers for draft model |
| `threads` | `8` | CPU threads |
| `draft_min` | `5` | Min speculative draft tokens |
| `draft_max` | `15` | Max speculative draft tokens |
| `llama_cpp_path` | — | Path to llama.cpp source (for conversion scripts) |

---

## Usage

```
1.  Start — Single Model
2.  Start — Speculative Decoding
3.  List Models
4.  Update Model List
5.  Pull Model from Ollama
6.  Pull Model from HuggingFace
7.  Convert Model
8.  Settings
9.  Exit
```

---

## License

MIT

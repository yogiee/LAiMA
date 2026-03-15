#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$SCRIPT_DIR/.venv"
BIN_DIR="$HOME/.local/bin"
LAUNCHER="$BIN_DIR/laima"

echo
echo "  LAiMA Installer"
echo "  ==============="
echo

# Require Python 3.10+
PY=$(command -v python3 || true)
if [[ -z "$PY" ]]; then
    echo "  ✖  python3 not found. Install Python 3.10+ and try again."
    exit 1
fi

PY_VER=$("$PY" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
PY_MAJOR=$(echo "$PY_VER" | cut -d. -f1)
PY_MINOR=$(echo "$PY_VER" | cut -d. -f2)

if (( PY_MAJOR < 3 || (PY_MAJOR == 3 && PY_MINOR < 10) )); then
    echo "  ✖  Python 3.10+ required (found $PY_VER)."
    exit 1
fi

echo "  Using Python $PY_VER at $PY"
echo

# Create venv
echo "  Creating virtual environment…"
"$PY" -m venv "$VENV"

# Install package (editable) — picks up requirements from pyproject.toml
echo "  Installing dependencies…"
"$VENV/bin/pip" install --quiet --upgrade pip
"$VENV/bin/pip" install --quiet -e "$SCRIPT_DIR"

# Create launcher
mkdir -p "$BIN_DIR"
cat > "$LAUNCHER" << LAUNCHER_SCRIPT
#!/usr/bin/env bash
exec "$VENV/bin/laima" "\$@"
LAUNCHER_SCRIPT
chmod +x "$LAUNCHER"

echo "  ✔  Installed to $LAUNCHER"
echo

# Copy conversion scripts from the bundled llama.cpp reference clone (if present)
LLAMA_CPP_REF="$SCRIPT_DIR/llama.cpp"
SCRIPTS_DEST="$HOME/.config/laima/scripts"

if [[ -f "$LLAMA_CPP_REF/convert_hf_to_gguf.py" ]]; then
    echo "  Copying conversion scripts from llama.cpp reference…"
    mkdir -p "$SCRIPTS_DEST"
    cp "$LLAMA_CPP_REF/convert_hf_to_gguf.py" "$SCRIPTS_DEST/"
    if [[ -d "$LLAMA_CPP_REF/gguf-py" ]]; then
        rm -rf "$SCRIPTS_DEST/gguf-py"
        cp -r "$LLAMA_CPP_REF/gguf-py" "$SCRIPTS_DEST/"
    fi
    touch "$SCRIPTS_DEST/.ready"
    echo "  ✔  Conversion scripts installed to $SCRIPTS_DEST"
else
    echo "  Note: llama.cpp reference folder not found — HF→GGUF conversion"
    echo "        will prompt for your llama.cpp path on first use."
fi
echo

# PATH reminder
if [[ ":${PATH}:" != *":${BIN_DIR}:"* ]]; then
    echo "  Note: $BIN_DIR is not in your PATH."
    echo "  Add the following to your ~/.zshrc:"
    echo
    echo "    export PATH=\"\$HOME/.local/bin:\$PATH\""
    echo
fi

echo "  Run 'laima' to start."
echo

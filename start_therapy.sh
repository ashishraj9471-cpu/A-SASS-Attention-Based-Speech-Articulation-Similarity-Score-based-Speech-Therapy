#!/bin/bash

# =====================================================
# A-SASS Speech Therapy — One-Click Launcher
# =====================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$SCRIPT_DIR/venv"
APP_FILE="$SCRIPT_DIR/app.py"

echo "============================================"
echo "  A-SASS Speech Therapy System"
echo "  One-Click Launcher"
echo "============================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] Python 3 not found! Please install Python 3.10+"
    exit 1
fi

echo "[✓] Python detected: $(python3 --version)"

# Create venv if missing
if [ ! -f "$VENV_DIR/bin/activate" ]; then
    echo ""
    echo "[+] Creating virtual environment (first time only)..."
    python3 -m venv "$VENV_DIR"
fi

# Activate
source "$VENV_DIR/bin/activate"

# Install requirements
echo ""
echo "[+] Checking dependencies (may take a few minutes on first run)..."
pip install --upgrade pip -q
pip install -r "$SCRIPT_DIR/requirements.txt" -q

# Verify app exists
if [ ! -f "$APP_FILE" ]; then
    echo "[ERROR] app.py not found in: $SCRIPT_DIR"
    exit 1
fi

# Launch
echo ""
echo "[✓] All ready! Starting Streamlit..."
echo "[i] First run will download Whisper models (~1-3 GB)"
echo "[i] Your browser will open automatically"
echo "============================================"
echo ""

# Open browser after delay
(sleep 10 && open http://localhost:8501 || xdg-open http://localhost:8501 || true) &

# Run
streamlit run "$APP_FILE" --server.headless=false

echo ""
echo "[✓] Session ended."
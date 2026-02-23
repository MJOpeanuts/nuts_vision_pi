#!/usr/bin/env bash
# run_pi.sh – launch nuts_vision_pi on Raspberry Pi 4
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Ensure the display is available (for autostart from systemd use DISPLAY=:0)
export DISPLAY="${DISPLAY:-:0}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

# Activate virtualenv if it exists next to this script
if [[ -f "$SCRIPT_DIR/venv/bin/activate" ]]; then
    source "$SCRIPT_DIR/venv/bin/activate"
fi

exec python3 "$SCRIPT_DIR/rpi_app/main.py" "$@"

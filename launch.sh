#!/usr/bin/env bash
# launch.sh – Lanceur tout-en-un pour nuts_vision_pi sur Raspberry Pi 4
#
# Utilisation : double-cliquez sur ce fichier dans le gestionnaire de fichiers,
# ou exécutez-le depuis un terminal :  ./launch.sh
#
# Ce script :
#   1. Crée un environnement virtuel Python si nécessaire.
#   2. Installe / met à jour les dépendances (requirements.txt).
#   3. Lance l'application.
#
# Compatible : clé USB, carte microSD, dossier home.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ── Affichage / Qt ────────────────────────────────────────────────────────────
export DISPLAY="${DISPLAY:-:0}"
export QT_QPA_PLATFORM="${QT_QPA_PLATFORM:-xcb}"

# ── Environnement virtuel ─────────────────────────────────────────────────────
VENV_DIR="$SCRIPT_DIR/venv"
VENV_FALLBACK_DIR="$HOME/.local/share/nuts_vision_pi/venv"

if [[ ! -f "$VENV_DIR/bin/activate" ]]; then
    echo "[nuts_vision_pi] Création de l'environnement virtuel..."
    if ! python3 -m venv --system-site-packages "$VENV_DIR" 2>/dev/null; then
        # The filesystem (e.g. FAT32 on a USB key) does not support symlinks.
        # Fall back to a writable location on the local drive.
        VENV_DIR="$VENV_FALLBACK_DIR"
        echo "[nuts_vision_pi] Repli sur $VENV_DIR (système de fichiers non POSIX)..."
        mkdir -p "$VENV_DIR"
        python3 -m venv --system-site-packages "$VENV_DIR"
    fi
fi

# shellcheck source=/dev/null
source "$VENV_DIR/bin/activate"

# ── Dépendances ───────────────────────────────────────────────────────────────
# Réinstalle les dépendances uniquement si requirements.txt a changé.
STAMP="$VENV_DIR/.last_install"
if [[ ! -f "$STAMP" || "$SCRIPT_DIR/requirements.txt" -nt "$STAMP" ]]; then
    echo "[nuts_vision_pi] Installation des dépendances..."
    pip install --quiet --requirement "$SCRIPT_DIR/requirements.txt"
    touch "$STAMP"
else
    echo "[nuts_vision_pi] Dépendances déjà à jour."
fi

# ── Lancement ─────────────────────────────────────────────────────────────────
echo "[nuts_vision_pi] Démarrage de l'application..."
exec python3 "$SCRIPT_DIR/rpi_app/main.py" "$@"

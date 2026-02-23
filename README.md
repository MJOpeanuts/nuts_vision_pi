# nuts_vision_pi

Application embarquée de détection de composants électroniques sur **Raspberry Pi 4** (aarch64).

## Architecture

```
nuts_vision_pi/
├── best.onnx               ← modèle YOLOv8 exporté en ONNX
├── crop.py                 ← utilitaire de découpe (réutilisé)
├── pipeline.py             ← pipeline complet (référence)
├── visualize.py            ← visualisation (référence)
├── requirements.txt
├── run_pi.sh               ← lanceur
└── rpi_app/
    ├── main.py             ← point d'entrée
    ├── ui.py               ← interface PyQt6 (800×480)
    ├── camera.py           ← wrapper Picamera2
    ├── detector_onnx.py    ← inférence ONNX Runtime
    ├── sqlite_db.py        ← base SQLite (disque externe)
    └── storage.py          ← gestion du disque externe
```

### Stockage

Les données sont écrites sur le premier disque externe monté sous
`/media/<USER>/` qui contient (ou accepte la création de) un dossier
`nuts_vision/`.  
Structure générée :

```
/media/<USER>/<disk>/nuts_vision/
├── nuts_vision.db          ← index SQLite
└── jobs/
    └── scan_YYYYMMDD_HHMMSS/
        ├── input.jpg
        ├── result.jpg      ← image annotée
        ├── metadata.json
        └── crops/
            ├── 000_IC.jpg
            └── ...
```

Si aucun disque externe n'est disponible, les données tombent dans
`~/nuts_vision_fallback/` (microSD).

---

## Installation sur Raspberry Pi OS (aarch64)

### 1. Dépendances système

```bash
sudo apt update
sudo apt install -y \
    python3-pip python3-venv \
    python3-pyqt6 \
    python3-picamera2 \
    python3-opencv \
    libatlas-base-dev        # numpy optimisé
```

### 2. Environnement Python

```bash
git clone https://github.com/MJOpeanuts/nuts_vision_pi.git
cd nuts_vision_pi

python3 -m venv --system-site-packages venv
source venv/bin/activate

pip install -r requirements.txt
```

> **Note :** `--system-site-packages` permet de réutiliser `picamera2`
> et `PyQt6` installés par apt.

### 3. Lancement

#### Option A – lanceur tout-en-un (recommandé)

```bash
chmod +x launch.sh
./launch.sh
```

`launch.sh` crée automatiquement le virtualenv, installe les dépendances si
nécessaire, puis démarre l'application. Vous pouvez le double-cliquer depuis
le gestionnaire de fichiers de Raspberry Pi OS.

#### Option B – raccourci bureau (`.desktop`)

Copiez `nuts_vision_pi.desktop` sur le bureau du Raspberry Pi :

```bash
cp nuts_vision_pi.desktop ~/Desktop/
chmod +x ~/Desktop/nuts_vision_pi.desktop
```

> **Note :** Si le projet n'est pas dans `/home/pi/nuts_vision_pi`, modifiez
> les lignes `Exec=` et `Path=` du fichier `.desktop` en conséquence.

#### Option C – lanceur classique

```bash
chmod +x run_pi.sh
./run_pi.sh
```

Options disponibles :

| Option | Description | Défaut |
|--------|-------------|--------|
| `--model PATH` | Chemin vers le modèle ONNX | `best.onnx` (racine du dépôt) |
| `--conf FLOAT` | Seuil de confiance | `0.25` |

---

## Utilisation

1. **Prévisualisation** – la caméra CSI démarre automatiquement au lancement.
2. **SCAN** – appuyer sur le bouton bleu pour capturer une image haute
   résolution, lancer la détection et sauvegarder les résultats.
3. **Historique** – appuyer sur « Historique » pour parcourir les jobs
   précédents, voir l'image annotée et les crops.

---

## Démarrage automatique (systemd)

Créer `/etc/systemd/system/nuts_vision_pi.service` :

```ini
[Unit]
Description=nuts_vision_pi – IC detector
After=graphical-session.target

[Service]
User=pi
Environment=DISPLAY=:0
Environment=QT_QPA_PLATFORM=xcb
WorkingDirectory=/home/pi/nuts_vision_pi
ExecStart=/home/pi/nuts_vision_pi/run_pi.sh
Restart=on-failure

[Install]
WantedBy=graphical-session.target
```

Puis :

```bash
sudo systemctl daemon-reload
sudo systemctl enable nuts_vision_pi.service
sudo systemctl start  nuts_vision_pi.service
```

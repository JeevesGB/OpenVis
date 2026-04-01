# settings.py
import json
from pathlib import Path

# ====================== CONSTANTS ======================
CHUNK = 1024
RATE = 44100
HISTORY = 3

# ====================== SETTINGS ======================
SETTINGS_FILE = Path("dat/sve/settings.json")

DEFAULT_SETTINGS = {
    "theme":      "grey",
    "osc_color":  "#939e9e",
    "bg_color":   "#0a0a12",
    "grid_color": "#1e1e32",
    "zoom":       1.0,
    "thickness":  2,
    "bar_count":  128,
    "gain":       1.0,
    "peak_hold":  False,
    "window_fn":  "hanning",
    "device":     None,
}

THEMES = {
    "green":  {"osc_color": "#00e678", "bg_color": "#0a0a12", "grid_color": "#1e1e32"},
    "cyan":   {"osc_color": "#00cfff", "bg_color": "#070e14", "grid_color": "#0d2233"},
    "amber":  {"osc_color": "#ffbb00", "bg_color": "#110e00", "grid_color": "#2a2200"},
    "purple": {"osc_color": "#cc88ff", "bg_color": "#0d0814", "grid_color": "#1e1230"},
    "red":    {"osc_color": "#ff5555", "bg_color": "#120808", "grid_color": "#2a1010"},
    "white":  {"osc_color": "#ffffff", "bg_color": "#0a0a0a", "grid_color": "#222222"},
    "custom": {},
}

NOTE_NAMES = ['C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#', 'A', 'A#', 'B']


class Settings:
    def __init__(self):
        self.data = dict(DEFAULT_SETTINGS)
        self.load()

    def load(self):
        try:
            if SETTINGS_FILE.exists():
                self.data.update(json.loads(SETTINGS_FILE.read_text()))
        except Exception:
            pass

    def save(self):
        try:
            SETTINGS_FILE.parent.mkdir(parents=True, exist_ok=True)
            SETTINGS_FILE.write_text(json.dumps(self.data, indent=2))
        except Exception:
            pass

    def get(self, k, default=None):
        return self.data.get(k, default)

    def __getitem__(self, k):
        return self.data[k]

    def __setitem__(self, k, v):
        self.data[k] = v
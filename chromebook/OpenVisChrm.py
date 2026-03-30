import sys
import json
import time
import threading
import numpy as np
import sounddevice as sd
from pathlib import Path
from collections import deque
from scipy.signal import find_peaks

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QComboBox, QPushButton, QColorDialog, QGroupBox,
    QScrollArea, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QObject
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPolygonF
from PyQt6.QtCore import QPointF

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
CHUNK         = 1024
RATE          = 44100
HISTORY       = 3
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
    "device":     None,  # Chromebook-safe
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

NOTE_NAMES = ['C','C#','D','D#','E','F','F#','G','G#','A','A#','B']


# ---------------------------------------------------------------------------
# Settings
# ---------------------------------------------------------------------------
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

    def __getitem__(self, k): return self.data[k]
    def __setitem__(self, k, v): self.data[k] = v


# ---------------------------------------------------------------------------
# Audio engine (Chromebook-safe)
# ---------------------------------------------------------------------------
class AudioEngine:
    def __init__(self, device=None):
        self.buffer  = deque(maxlen=HISTORY)
        self._latest = np.zeros(CHUNK, dtype=np.float32)
        self._lock   = threading.Lock()

        for _ in range(HISTORY):
            self.buffer.append(np.zeros(CHUNK, dtype=np.float32))

        try:
            if device is None:
                device = sd.default.device[0]

            info = sd.query_devices(device, 'input')
            self._ch = min(2, int(info['max_input_channels']))

            self.stream = sd.InputStream(
                samplerate=RATE,
                blocksize=CHUNK,
                channels=self._ch,
                dtype='float32',
                device=device,
                callback=self._callback,
            )
            self.stream.start()
            print(f"[Audio] Using device {device}")

        except Exception as e:
            print("[Audio ERROR]", e)
            print("Falling back to default mic")

            self.stream = sd.InputStream(
                samplerate=RATE,
                blocksize=CHUNK,
                channels=1,
                dtype='float32',
                callback=self._callback,
            )
            self.stream.start()

    def _callback(self, indata, frames, time, status):
        mono = indata.mean(axis=1)
        with self._lock:
            self._latest = mono.copy()
            self.buffer.append(mono.copy())

    def read(self):
        with self._lock:
            return self._latest.copy()

    def smoothed(self):
        with self._lock:
            return np.concatenate(list(self.buffer))

    def close(self):
        self.stream.stop()
        self.stream.close()


# ---------------------------------------------------------------------------
# Simple analyser
# ---------------------------------------------------------------------------
class AnalysisEngine(QObject):
    results_ready = pyqtSignal(dict)

    def process(self, samples):
        peak = float(np.max(np.abs(samples)))
        rms  = float(np.sqrt(np.mean(samples**2)))
        self.results_ready.emit({
            "peak": peak,
            "rms": rms
        })


# ---------------------------------------------------------------------------
# Oscilloscope
# ---------------------------------------------------------------------------
class OscilloscopeWidget(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.samples = np.zeros(CHUNK)

    def update_samples(self, samples):
        self.samples = samples
        self.update()

    def paintEvent(self, e):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(self.settings["bg_color"]))

        pen = QPen(QColor(self.settings["osc_color"]), 2)
        p.setPen(pen)

        w, h = self.width(), self.height()
        mid = h // 2

        for i in range(len(self.samples)-1):
            x1 = int(i / len(self.samples) * w)
            x2 = int((i+1) / len(self.samples) * w)
            y1 = int(mid - self.samples[i] * h * 0.4)
            y2 = int(mid - self.samples[i+1] * h * 0.4)
            p.drawLine(x1, y1, x2, y2)


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.settings = Settings()
        self.audio = AudioEngine(device=self.settings["device"])
        self.analyser = AnalysisEngine()

        self.osc = OscilloscopeWidget(self.settings)
        self.setCentralWidget(self.osc)

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(30)

        self.setWindowTitle("OpenViz (Chromebook)")
        self.resize(800, 400)

    def tick(self):
        samples = self.audio.read()
        self.osc.update_samples(samples)

    def closeEvent(self, e):
        self.audio.close()
        e.accept()


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
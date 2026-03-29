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
    QScrollArea, QSplitter, QFileDialog
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
    "device":     22,
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

    def get(self, k, default=None):
        return self.data.get(k, default)

    def __getitem__(self, k):    return self.data[k]
    def __setitem__(self, k, v): self.data[k] = v


# ---------------------------------------------------------------------------
# Audio engine
# ---------------------------------------------------------------------------
class AudioEngine:
    def __init__(self, device=None):
        self.buffer  = deque(maxlen=HISTORY)
        self._latest = np.zeros(CHUNK, dtype=np.float32)
        self._stereo = np.zeros((CHUNK, 2), dtype=np.float32)
        self._lock   = threading.Lock()
        for _ in range(HISTORY):
            self.buffer.append(np.zeros(CHUNK, dtype=np.float32))

        info     = sd.query_devices(device if device is not None else sd.default.device[0], 'input')
        self._ch = min(2, int(info['max_input_channels']))

        self.stream = sd.InputStream(
            samplerate=RATE, blocksize=CHUNK,
            channels=self._ch, dtype='float32',
            device=device, callback=self._callback,
        )
        self.stream.start()

    def _callback(self, indata, frames, time, status):
        with self._lock:
            if indata.shape[1] >= 2:
                self._stereo = indata[:, :2].copy()
                mono = indata.mean(axis=1)
            else:
                mono = indata[:, 0].copy()
                self._stereo = np.column_stack([mono, mono])
            self._latest = mono.copy()
            self.buffer.append(mono.copy())

    def read(self):
        with self._lock:
            return self._latest.copy()

    def read_stereo(self):
        with self._lock:
            return self._stereo.copy()

    def smoothed(self):
        with self._lock:
            return np.concatenate(list(self.buffer))

    def close(self):
        self.stream.stop()
        self.stream.close()


# ---------------------------------------------------------------------------
# Analysis engine
# ---------------------------------------------------------------------------
class AnalysisEngine(QObject):
    results_ready = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._energy_history = deque(maxlen=43)
        self._bpm            = 0.0
        self._beat_times     = deque(maxlen=16)
        self._lufs_history   = deque(maxlen=300)
        self._clip_count     = 0

    def process(self, samples: np.ndarray):
        clipped = bool(np.any(np.abs(samples) >= 0.999))
        if clipped:
            self._clip_count += 1
        else:
            self._clip_count = max(0, self._clip_count - 1)

        note_str = self._detect_pitch(samples)
        bpm      = self._detect_bpm(samples)
        lufs     = self._measure_lufs(samples)
        self._lufs_history.append(lufs)

        self.results_ready.emit({
            "clipped":      clipped,
            "clip_hold":    self._clip_count > 0,
            "note":         note_str,
            "bpm":          bpm,
            "lufs":         lufs,
            "lufs_history": list(self._lufs_history),
        })

    def _detect_pitch(self, samples):
        corr  = np.correlate(samples, samples, mode='full')
        corr  = corr[len(corr) // 2:]
        d     = np.diff(corr)
        start = next((i for i in range(len(d)-1) if d[i] < 0), None)
        if start is None:
            return "—"
        peaks, _ = find_peaks(corr[start:], height=0.1 * corr[0])
        if len(peaks) == 0:
            return "—"
        period = peaks[0] + start
        if period == 0:
            return "—"
        freq = RATE / period
        if freq < 50 or freq > 4000:
            return "—"
        midi   = 69 + 12 * np.log2(freq / 440.0)
        note   = NOTE_NAMES[int(round(midi)) % 12]
        octave = (int(round(midi)) // 12) - 1
        return f"{note}{octave}"

    def _detect_bpm(self, samples):
        energy = float(np.mean(samples ** 2))
        self._energy_history.append(energy)
        if len(self._energy_history) < 4:
            return self._bpm
        mean_e = np.mean(self._energy_history)
        if energy > 1.5 * mean_e and energy > 0.001:
            self._beat_times.append(time.time())
        if len(self._beat_times) >= 4:
            intervals = np.diff(list(self._beat_times))
            intervals = intervals[intervals < 2.0]
            if len(intervals) > 0:
                self._bpm = float(np.clip(60.0 / np.mean(intervals), 40, 220))
        return self._bpm

    def _measure_lufs(self, samples):
        ms = np.mean(samples ** 2)
        if ms < 1e-10:
            return -70.0
        return float(np.clip(-0.691 + 10 * np.log10(ms), -70, 0))


# ---------------------------------------------------------------------------
# Shared painter helpers
# ---------------------------------------------------------------------------
def draw_grid(painter, rect, color, divisions=8):
    painter.setPen(QPen(color, 0.5))
    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
    for i in range(1, divisions):
        painter.drawLine(x + w * i // divisions, y, x + w * i // divisions, y + h)
        painter.drawLine(x, y + h * i // divisions, x + w, y + h * i // divisions)
    painter.drawRect(rect)


# ---------------------------------------------------------------------------
# Oscilloscope
# ---------------------------------------------------------------------------
class OscilloscopeWidget(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.samples  = np.zeros(CHUNK * HISTORY, dtype=np.float32)
        self.peak_pos = None
        self.peak_ttl = 0
        self.setMinimumHeight(140)

    def update_samples(self, samples):
        self.samples = samples * self.settings["gain"]
        self.update()

    def _color(self, key): return QColor(self.settings[key])

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -2)
        painter.fillRect(rect, self._color("bg_color"))
        draw_grid(painter, rect, self._color("grid_color"))

        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        mid_y = y + h / 2
        painter.setPen(QPen(QColor(50, 50, 80), 1))
        painter.drawLine(x, int(mid_y), x + w, int(mid_y))

        visible = max(1, int(len(self.samples) / self.settings["zoom"]))
        samples = self.samples[:visible]
        osc_col = self._color("osc_color")
        pen = QPen(osc_col, self.settings["thickness"])
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        points  = QPolygonF()
        max_amp = -1
        max_idx = 0
        for i, s in enumerate(samples):
            px = x + (i / len(samples)) * w
            py = max(float(y), min(float(y + h), mid_y - float(s) * h * 0.45))
            points.append(QPointF(px, py))
            if abs(float(s)) > max_amp:
                max_amp = abs(float(s))
                max_idx = i

        if len(points) > 1:
            painter.drawPolyline(points)

        if self.settings["peak_hold"]:
            if max_amp > 0.01:
                px = x + (max_idx / max(len(samples), 1)) * w
                if self.peak_ttl <= 0 or px != self.peak_pos:
                    self.peak_pos = px
                    self.peak_ttl = 60
            if self.peak_pos is not None and self.peak_ttl > 0:
                pc = QColor(osc_col)
                pc.setAlpha(min(255, self.peak_ttl * 4))
                painter.setPen(QPen(pc, 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(self.peak_pos), y, int(self.peak_pos), y + h)
                self.peak_ttl -= 1

        tc = QColor(140, 140, 170)
        painter.setPen(QPen(tc))
        painter.setFont(QFont("Monospace", 10))
        painter.drawText(x + 8, y + 18, "OSCILLOSCOPE")
        painter.drawText(x + w - 90, y + 18, f"gain x{self.settings['gain']:.1f}")
        painter.end()


# ---------------------------------------------------------------------------
# Spectrum analyser
# ---------------------------------------------------------------------------
class SpectrumWidget(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.settings     = settings
        self.samples      = np.zeros(CHUNK, dtype=np.float32)
        self._n           = settings["bar_count"]
        self.smoothed_fft = np.zeros(self._n)
        self.peak_bars    = np.zeros(self._n)
        self.peak_ttl     = np.zeros(self._n, dtype=int)
        self.setMinimumHeight(140)

    def _ensure_size(self, n):
        if self._n != n:
            self._n           = n
            self.smoothed_fft = np.zeros(n)
            self.peak_bars    = np.zeros(n)
            self.peak_ttl     = np.zeros(n, dtype=int)

    def update_samples(self, samples):
        self.samples = samples * self.settings["gain"]
        self.update()

    def _color(self, key): return QColor(self.settings[key])

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -2)
        painter.fillRect(rect, self._color("bg_color"))
        draw_grid(painter, rect, self._color("grid_color"))

        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        n = self.settings["bar_count"]
        self._ensure_size(n)

        windows = {
            "hanning": np.hanning, "hamming": np.hamming,
            "blackman": np.blackman, "bartlett": np.bartlett,
            "none": lambda sz: np.ones(sz),
        }
        window   = windows.get(self.settings.get("window_fn", "hanning"), np.hanning)(len(self.samples))
        fft      = np.abs(np.fft.rfft(self.samples * window))
        fft      = np.log1p(fft)
        max_bin  = int(len(fft) * 16000 / (RATE / 2))
        fft      = fft[:max_bin]

        indices = np.logspace(0, np.log10(max(len(fft), 2)), n + 1).astype(int)
        indices = np.clip(indices, 0, len(fft) - 1)
        bars    = np.array([
            fft[indices[i]:indices[i+1]].mean() if indices[i+1] > indices[i] else fft[indices[i]]
            for i in range(n)
        ])
        bars /= (bars.max() + 1e-9)
        self.smoothed_fft = self.smoothed_fft * 0.75 + bars * 0.25

        if self.settings["peak_hold"]:
            for i in range(n):
                if self.smoothed_fft[i] >= self.peak_bars[i]:
                    self.peak_bars[i] = self.smoothed_fft[i]
                    self.peak_ttl[i]  = 90
                elif self.peak_ttl[i] > 0:
                    self.peak_ttl[i] -= 1
                else:
                    self.peak_bars[i] = max(0.0, self.peak_bars[i] - 0.01)

        bar_w   = w / n
        osc_col = self._color("osc_color")

        for i, val in enumerate(self.smoothed_fft):
            bar_h = int(val * h * 0.92)
            bx    = x + i * bar_w
            color = QColor(
                max(0, min(255, int(val * 80))),
                max(0, min(255, int(100 + val * 155))),
                max(0, min(255, int(255 - val * 200))),
            )
            painter.fillRect(int(bx)+1, y+h-bar_h, max(1, int(bar_w)-1), bar_h, color)

            if self.settings["peak_hold"] and self.peak_ttl[i] > 0:
                ph = int(self.peak_bars[i] * h * 0.92)
                pc = QColor(osc_col)
                pc.setAlpha(min(255, int(self.peak_ttl[i]) * 3))
                painter.fillRect(int(bx)+1, y+h-ph-2, max(1, int(bar_w)-1), 2, pc)

        tc = QColor(140, 140, 170)
        painter.setFont(QFont("Monospace", 9))
        for freq in [100, 500, 1000, 4000, 8000, 16000]:
            frac  = np.log10(freq / 100) / np.log10(16000 / 100)
            lx    = x + int(frac * w)
            label = f"{freq//1000}k" if freq >= 1000 else str(freq)
            painter.setPen(QPen(self._color("grid_color"), 0.5))
            painter.drawLine(lx, y+h-20, lx, y+h)
            painter.setPen(QPen(tc))
            painter.drawText(lx-10, y+h-4, label)

        painter.setPen(QPen(tc))
        painter.setFont(QFont("Monospace", 10))
        painter.drawText(x+8, y+18, "SPECTRUM ANALYSER")
        painter.end()


# ---------------------------------------------------------------------------
# LUFS history graph
# ---------------------------------------------------------------------------
class LUFSGraphWidget(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.history  = []
        self.setMinimumHeight(100)
        self.setMaximumHeight(140)

    def update_history(self, history):
        self.history = history
        self.update()

    def _color(self, key): return QColor(self.settings[key])

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -2)
        painter.fillRect(rect, self._color("bg_color"))
        draw_grid(painter, rect, self._color("grid_color"), divisions=6)

        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

        if len(self.history) > 1:
            osc_col = self._color("osc_color")
            painter.setPen(QPen(osc_col, 1.5))
            pts = QPolygonF()
            for i, val in enumerate(self.history):
                px = x + (i / max(len(self.history)-1, 1)) * w
                py = y + h - ((val + 70) / 70) * h
                py = max(float(y), min(float(y+h), py))
                pts.append(QPointF(px, py))
            painter.drawPolyline(pts)

        tc = QColor(140, 140, 170)
        painter.setPen(QPen(tc))
        painter.setFont(QFont("Monospace", 9))
        painter.drawText(x+6, y+14, "LUFS")
        painter.drawText(x+w-46, y+14, " 0 dB")
        painter.drawText(x+w-52, y+h-4, "-70 dB")
        painter.end()


# ---------------------------------------------------------------------------
# Analysis readout panel
# ---------------------------------------------------------------------------
class AnalysisPanel(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.setFixedWidth(220)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        clip_box = QGroupBox("Clipping")
        clip_lay = QVBoxLayout(clip_box)
        self.clip_light = QLabel("  OK  ")
        self.clip_light.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.clip_light.setFixedHeight(32)
        self.clip_light.setFont(QFont("Monospace", 11))
        self._set_clip(False)
        clip_lay.addWidget(self.clip_light)
        layout.addWidget(clip_box)

        note_box = QGroupBox("Pitch")
        note_lay = QVBoxLayout(note_box)
        self.note_label = QLabel("—")
        self.note_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.note_label.setFont(QFont("Monospace", 28))
        self.note_label.setStyleSheet("color: #00e678;")
        note_lay.addWidget(self.note_label)
        layout.addWidget(note_box)

        bpm_box = QGroupBox("BPM")
        bpm_lay = QVBoxLayout(bpm_box)
        self.bpm_label = QLabel("—")
        self.bpm_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.bpm_label.setFont(QFont("Monospace", 28))
        self.bpm_label.setStyleSheet("color: #00e678;")
        bpm_lay.addWidget(self.bpm_label)
        layout.addWidget(bpm_box)

        lufs_box = QGroupBox("Loudness")
        lufs_lay = QVBoxLayout(lufs_box)
        self.lufs_label = QLabel("— dBLUFS")
        self.lufs_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.lufs_label.setFont(QFont("Monospace", 14))
        self.lufs_label.setStyleSheet("color: #00e678;")
        lufs_lay.addWidget(self.lufs_label)
        layout.addWidget(lufs_box)

        layout.addStretch()

    def _set_clip(self, clipping):
        if clipping:
            self.clip_light.setText(" CLIP ")
            self.clip_light.setStyleSheet(
                "background: #ff2222; color: white; border-radius: 4px;")
        else:
            self.clip_light.setText("  OK  ")
            self.clip_light.setStyleSheet(
                "background: #003310; color: #00e678; border-radius: 4px;")

    def update_results(self, results):
        self._set_clip(results["clip_hold"])
        self.note_label.setText(results["note"])
        bpm = results["bpm"]
        self.bpm_label.setText(f"{bpm:.0f}" if bpm > 0 else "—")
        self.lufs_label.setText(f"{results['lufs']:.1f} dB")


# ---------------------------------------------------------------------------
# Settings panel
# ---------------------------------------------------------------------------
class SettingsPanel(QScrollArea):
    def __init__(self, settings, on_device_change):
        super().__init__()
        self.settings = settings
        self.setMinimumWidth(500)
        self.setMaximumWidth(600)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner  = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(10)
        layout.setContentsMargins(10, 10, 10, 10)

        dev_box = QGroupBox("Input device")
        dev_lay = QVBoxLayout(dev_box)
        self.device_combo = QComboBox()
        self._populate_devices()
        dev_lay.addWidget(self.device_combo)
        ab = QPushButton("Apply device")
        ab.clicked.connect(lambda: on_device_change(self.device_combo.currentData()))
        dev_lay.addWidget(ab)
        layout.addWidget(dev_box)

        theme_box = QGroupBox("Theme")
        theme_lay = QVBoxLayout(theme_box)
        self.theme_combo = QComboBox()
        for t in THEMES:
            self.theme_combo.addItem(t.capitalize(), t)
        self.theme_combo.setCurrentText(settings["theme"].capitalize())
        self.theme_combo.currentIndexChanged.connect(self._on_theme)
        theme_lay.addWidget(self.theme_combo)
        self.osc_btn  = self._color_btn("Waveform colour",   "osc_color")
        self.bg_btn   = self._color_btn("Background colour", "bg_color")
        self.grid_btn = self._color_btn("Grid colour",       "grid_color")
        for b in [self.osc_btn, self.bg_btn, self.grid_btn]:
            theme_lay.addWidget(b)
        layout.addWidget(theme_box)

        osc_box = QGroupBox("Oscilloscope")
        osc_lay = QVBoxLayout(osc_box)
        osc_lay.addWidget(QLabel("Zoom"))
        osc_lay.addWidget(self._slider(10, 80, int(settings["zoom"]*10),
                                       lambda v: self._set("zoom", v/10.0)))
        osc_lay.addWidget(QLabel("Line thickness"))
        osc_lay.addWidget(self._slider(1, 6, settings["thickness"],
                                       lambda v: self._set("thickness", v)))
        layout.addWidget(osc_box)

        spec_box = QGroupBox("Spectrum")
        spec_lay = QVBoxLayout(spec_box)
        spec_lay.addWidget(QLabel("Bar count"))
        self.bars_combo = QComboBox()
        for n in [64, 128, 256, 512]:
            self.bars_combo.addItem(str(n), n)
        self.bars_combo.setCurrentText(str(settings["bar_count"]))
        self.bars_combo.currentIndexChanged.connect(
            lambda: self._set("bar_count", self.bars_combo.currentData()))
        spec_lay.addWidget(self.bars_combo)
        spec_lay.addWidget(QLabel("Window function"))
        self.win_combo = QComboBox()
        for fn in ["hanning", "hamming", "blackman", "bartlett", "none"]:
            self.win_combo.addItem(fn.capitalize(), fn)
        self.win_combo.setCurrentText(settings.get("window_fn", "hanning").capitalize())
        self.win_combo.currentIndexChanged.connect(
            lambda: self._set("window_fn", self.win_combo.currentData()))
        spec_lay.addWidget(self.win_combo)
        layout.addWidget(spec_box)

        gain_box = QGroupBox("Gain")
        gain_lay = QVBoxLayout(gain_box)
        self.gain_label = QLabel(f"x{settings['gain']:.1f}")
        gain_lay.addWidget(self.gain_label)
        gain_lay.addWidget(self._slider(1, 80, int(settings["gain"]*10), self._on_gain))
        layout.addWidget(gain_box)

        peak_box = QGroupBox("Peak hold")
        peak_lay = QVBoxLayout(peak_box)
        self.peak_btn = QPushButton()
        self._update_peak_btn()
        self.peak_btn.clicked.connect(self._toggle_peak)
        peak_lay.addWidget(self.peak_btn)
        layout.addWidget(peak_box)

        save_btn = QPushButton("Save settings")
        save_btn.clicked.connect(self._save)
        layout.addWidget(save_btn)
        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #00e678; font-size: 11px;")
        layout.addWidget(self.status_label)

        layout.addStretch()
        self.setWidget(inner)

    def _populate_devices(self):
        self.device_combo.clear()
        for i, d in enumerate(sd.query_devices()):
            if d['max_input_channels'] > 0:
                self.device_combo.addItem(f"{i}: {d['name']}", i)
        default = sd.default.device[0]
        saved   = self.settings["device"]
        for j in range(self.device_combo.count()):
            if self.device_combo.itemData(j) == (saved if saved is not None else default):
                self.device_combo.setCurrentIndex(j)
                break

    def _slider(self, lo, hi, val, cb):
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        s.valueChanged.connect(cb)
        return s

    def _color_btn(self, label, key):
        btn = QPushButton(label)
        btn.setStyleSheet(f"background: {self.settings[key]}; color: white;")
        btn.clicked.connect(lambda _, k=key, b=btn: self._pick_color(k, b))
        return btn

    def _pick_color(self, key, btn):
        col = QColorDialog.getColor(QColor(self.settings[key]), self)
        if col.isValid():
            self.settings[key]     = col.name()
            self.settings["theme"] = "custom"
            self.theme_combo.setCurrentText("Custom")
            btn.setStyleSheet(f"background: {col.name()}; color: white;")

    def _on_theme(self):
        key = self.theme_combo.currentData()
        self.settings["theme"] = key
        if key != "custom":
            for k, v in THEMES.get(key, {}).items():
                self.settings[k] = v
            self.osc_btn.setStyleSheet(f"background: {self.settings['osc_color']}; color: white;")
            self.bg_btn.setStyleSheet(f"background: {self.settings['bg_color']}; color: white;")
            self.grid_btn.setStyleSheet(f"background: {self.settings['grid_color']}; color: white;")

    def _set(self, k, v): self.settings[k] = v

    def _on_gain(self, v):
        self.settings["gain"] = v / 10.0
        self.gain_label.setText(f"x{v/10.0:.1f}")

    def _toggle_peak(self):
        self.settings["peak_hold"] = not self.settings["peak_hold"]
        self._update_peak_btn()

    def _update_peak_btn(self):
        on = self.settings["peak_hold"]
        self.peak_btn.setText("On" if on else "Off")
        self.peak_btn.setStyleSheet("color: #00e678;" if on else "color: #888;")

    def _save(self):
        self.settings["device"] = self.device_combo.currentData()
        self.settings.save()
        self.status_label.setText("Saved!")
        QTimer.singleShot(2000, lambda: self.status_label.setText(""))


# ---------------------------------------------------------------------------
# Status bar
# ---------------------------------------------------------------------------
class StatusBar(QWidget):
    def __init__(self):
        super().__init__()
        self.setFixedHeight(32)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        self.stats_label = QLabel("Peak: 0.000   RMS: 0.000")
        self.stats_label.setStyleSheet("color: #8888aa; font-family: monospace; font-size: 12px;")
        layout.addWidget(self.stats_label)
        layout.addStretch()

    def update_stats(self, samples):
        peak = float(np.max(np.abs(samples)))
        rms  = float(np.sqrt(np.mean(samples ** 2)))
        self.stats_label.setText(f"Peak: {peak:.3f}   RMS: {rms:.3f}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = Settings()
        self.setWindowTitle("OpenViz")
        self.setStyleSheet("background: #0a0a12; color: #aaaacc;")

        self.audio    = AudioEngine(device=self.settings["device"])
        self.analyser = AnalysisEngine()

        self.osc        = OscilloscopeWidget(self.settings)
        self.spec       = SpectrumWidget(self.settings)
        self.lufs_graph = LUFSGraphWidget(self.settings)
        self.analysis   = AnalysisPanel(self.settings)
        self.settings_panel = SettingsPanel(self.settings, self.change_device)
        self.statusbar  = StatusBar()

        self.analyser.results_ready.connect(self.analysis.update_results)
        self.analyser.results_ready.connect(
            lambda r: self.lufs_graph.update_history(r["lufs_history"]))

        viz_splitter = QSplitter(Qt.Orientation.Vertical)
        viz_splitter.addWidget(self.osc)
        viz_splitter.addWidget(self.spec)
        viz_splitter.addWidget(self.lufs_graph)
        viz_splitter.setSizes([280, 280, 140])

        viz_wrap = QWidget()
        vl = QVBoxLayout(viz_wrap)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(4)
        vl.addWidget(viz_splitter, stretch=1)
        vl.addWidget(self.statusbar)

        right_splitter = QSplitter(Qt.Orientation.Vertical)
        right_splitter.addWidget(self.analysis)
        right_splitter.addWidget(self.settings_panel)
        right_splitter.setSizes([340, 500])

        root_splitter = QSplitter(Qt.Orientation.Horizontal)
        root_splitter.addWidget(viz_wrap)
        root_splitter.addWidget(right_splitter)
        root_splitter.setSizes([1100, 220])

        self.setCentralWidget(root_splitter)

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)
        self.showMaximized()

    def change_device(self, idx):
        self.audio.close()
        self.audio = AudioEngine(device=idx)

    def tick(self):
        samples  = self.audio.read()
        smoothed = self.audio.smoothed()
        self.osc.update_samples(smoothed)
        self.spec.update_samples(samples)
        self.statusbar.update_stats(samples)
        threading.Thread(target=self.analyser.process, args=(samples,), daemon=True).start()

    def closeEvent(self, event):
        self.timer.stop()
        self.audio.close()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    win = MainWindow()
    win.show()
    sys.exit(app.exec())
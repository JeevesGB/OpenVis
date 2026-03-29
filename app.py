import sys
import json
import numpy as np
import sounddevice as sd
from pathlib import Path
from collections import deque
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QSlider, QComboBox, QPushButton, QColorDialog, QGroupBox,
    QScrollArea
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPolygonF
from PyQt6.QtCore import QPointF

# --- Constants ---
CHUNK   = 1024
RATE    = 44100
HISTORY = 3
SETTINGS_FILE = Path("dat/sve/settings.json")
DEFAULT_SETTINGS = {
    "theme":      "green",
    "osc_color":  "#00e678",
    "bg_color":   "#0a0a12",
    "grid_color": "#1e1e32",
    "zoom":       1.0,
    "thickness":  2,
    "bar_count":  128,
    "gain":       1.0,
    "peak_hold":  True,
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


# ---------------------------------------------------------------------------
# Settings manager
# ---------------------------------------------------------------------------
class Settings:
    def __init__(self):
        self.data = dict(DEFAULT_SETTINGS)
        self.load()

    def load(self):
        try:
            if SETTINGS_FILE.exists():
                saved = json.loads(SETTINGS_FILE.read_text())
                self.data.update(saved)
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


# ---------------------------------------------------------------------------
# Audio engine
# ---------------------------------------------------------------------------
class AudioEngine:
    def __init__(self, device=None):
        self.buffer  = deque(maxlen=HISTORY)
        self._latest = np.zeros(CHUNK, dtype=np.float32)
        for _ in range(HISTORY):
            self.buffer.append(np.zeros(CHUNK, dtype=np.float32))

        device_info = sd.query_devices(device if device is not None else sd.default.device[0], 'input')
        channels = min(2, int(device_info['max_input_channels']))

        self.stream = sd.InputStream(
            samplerate=RATE,
            blocksize=CHUNK,
            channels=channels,
            dtype='float32',
            device=device,
            callback=self._callback,
        )
        self.stream.start()

    def _callback(self, indata, frames, time, status):
        samples = indata.mean(axis=1).copy() if indata.shape[1] > 1 else indata[:, 0].copy()
        self._latest = samples
        self.buffer.append(samples)

    def read(self):
        return self._latest.copy()

    def smoothed(self):
        return np.concatenate(list(self.buffer))

    def close(self):
        self.stream.stop()
        self.stream.close()


# ---------------------------------------------------------------------------
# Oscilloscope
# ---------------------------------------------------------------------------
class OscilloscopeWidget(QWidget):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings = settings
        self.samples  = np.zeros(CHUNK * HISTORY, dtype=np.float32)
        self.peak_pos = None
        self.peak_ttl = 0
        self.setMinimumHeight(180)

    def update_samples(self, samples):
        self.samples = samples * self.settings["gain"]
        self.update()

    def _color(self, key):
        return QColor(self.settings[key])

    def draw_grid(self, painter, rect):
        painter.setPen(QPen(self._color("grid_color"), 0.5))
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        for i in range(1, 8):
            painter.drawLine(x + w * i // 8, y, x + w * i // 8, y + h)
            painter.drawLine(x, y + h * i // 8, x + w, y + h * i // 8)
        painter.drawRect(rect)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -2)
        painter.fillRect(rect, self._color("bg_color"))
        self.draw_grid(painter, rect)

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
            py = mid_y - float(s) * h * 0.45
            py = max(float(y), min(float(y + h), py))
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
                alpha = min(255, self.peak_ttl * 4)
                peak_col = QColor(osc_col)
                peak_col.setAlpha(alpha)
                painter.setPen(QPen(peak_col, 1, Qt.PenStyle.DashLine))
                painter.drawLine(int(self.peak_pos), y, int(self.peak_pos), y + h)
                self.peak_ttl -= 1

        text_col = QColor(140, 140, 170)
        painter.setPen(QPen(text_col))
        painter.setFont(QFont("Monospace", 10))
        painter.drawText(x + 8, y + 18, "OSCILLOSCOPE")
        painter.drawText(x + w - 90, y + 18, f"gain x{self.settings['gain']:.1f}")
        painter.end()


# ---------------------------------------------------------------------------
# Spectrum analyser
# ---------------------------------------------------------------------------
class SpectrumWidget(QWidget):
    def __init__(self, settings: Settings):
        super().__init__()
        self.settings     = settings
        self.samples      = np.zeros(CHUNK, dtype=np.float32)
        self._n           = settings["bar_count"]
        self.smoothed_fft = np.zeros(self._n)
        self.peak_bars    = np.zeros(self._n)
        self.peak_ttl     = np.zeros(self._n, dtype=int)
        self.setMinimumHeight(180)

    def _ensure_size(self, n):
        """Resize internal arrays whenever bar_count changes."""
        if self._n != n:
            self._n           = n
            self.smoothed_fft = np.zeros(n)
            self.peak_bars    = np.zeros(n)
            self.peak_ttl     = np.zeros(n, dtype=int)

    def update_samples(self, samples):
        self.samples = samples * self.settings["gain"]
        self.update()

    def _color(self, key):
        return QColor(self.settings[key])

    def draw_grid(self, painter, rect):
        painter.setPen(QPen(self._color("grid_color"), 0.5))
        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        for i in range(1, 8):
            painter.drawLine(x + w * i // 8, y, x + w * i // 8, y + h)
            painter.drawLine(x, y + h * i // 8, x + w, y + h * i // 8)
        painter.drawRect(rect)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -2)
        painter.fillRect(rect, self._color("bg_color"))
        self.draw_grid(painter, rect)

        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()

        # Ensure arrays match current bar_count before any maths
        n = self.settings["bar_count"]
        self._ensure_size(n)

        win_name = self.settings.get("window_fn", "hanning")
        windows  = {
            "hanning":  np.hanning,
            "hamming":  np.hamming,
            "blackman": np.blackman,
            "bartlett": np.bartlett,
            "none":     lambda sz: np.ones(sz),
        }
        window   = windows.get(win_name, np.hanning)(len(self.samples))
        windowed = self.samples * window
        fft      = np.abs(np.fft.rfft(windowed))
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

        # Both arrays are guaranteed to be size n here
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
            by    = y + h - bar_h
            r     = int(val * 80)
            g     = int(100 + val * 155)
            b     = int(255 - val * 200)
            color = QColor(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b)))
            painter.fillRect(int(bx) + 1, int(by), max(1, int(bar_w) - 1), bar_h, color)

            if self.settings["peak_hold"] and self.peak_ttl[i] > 0:
                ph     = int(self.peak_bars[i] * h * 0.92)
                alpha  = min(255, int(self.peak_ttl[i]) * 3)
                pk_col = QColor(osc_col)
                pk_col.setAlpha(alpha)
                painter.fillRect(int(bx) + 1, y + h - ph - 2, max(1, int(bar_w) - 1), 2, pk_col)

        text_col = QColor(140, 140, 170)
        painter.setFont(QFont("Monospace", 9))
        for freq in [100, 500, 1000, 4000, 8000, 16000]:
            frac  = np.log10(freq / 100) / np.log10(16000 / 100)
            lx    = x + int(frac * w)
            label = f"{freq // 1000}k" if freq >= 1000 else str(freq)
            painter.setPen(QPen(self._color("grid_color"), 0.5))
            painter.drawLine(lx, y + h - 20, lx, y + h)
            painter.setPen(QPen(text_col))
            painter.drawText(lx - 10, y + h - 4, label)

        painter.setPen(QPen(text_col))
        painter.setFont(QFont("Monospace", 10))
        painter.drawText(x + 8, y + 18, "SPECTRUM ANALYSER")
        painter.end()


# ---------------------------------------------------------------------------
# Settings panel
# ---------------------------------------------------------------------------
class SettingsPanel(QScrollArea):
    def __init__(self, settings: Settings, osc, spec, on_device_change):
        super().__init__()
        self.settings = settings
        self.setMinimumWidth(220)
        self.setMaximumWidth(300)  # optional cap
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner  = QWidget()
        layout = QVBoxLayout(inner)
        layout.setSpacing(12)
        layout.setContentsMargins(10, 10, 10, 10)

        # -- Device --
        dev_box = QGroupBox("Input device")
        dev_lay = QVBoxLayout(dev_box)
        self.device_combo = QComboBox()
        self._populate_devices()
        dev_lay.addWidget(self.device_combo)
        apply_btn = QPushButton("Apply device")
        apply_btn.clicked.connect(lambda: on_device_change(self.device_combo.currentData()))
        dev_lay.addWidget(apply_btn)
        layout.addWidget(dev_box)

        # -- Theme --
        theme_box = QGroupBox("Theme")
        theme_lay = QVBoxLayout(theme_box)
        self.theme_combo = QComboBox()
        for t in THEMES:
            self.theme_combo.addItem(t.capitalize(), t)
        self.theme_combo.setCurrentText(settings["theme"].capitalize())
        self.theme_combo.currentIndexChanged.connect(self._on_theme)
        theme_lay.addWidget(self.theme_combo)
        self.osc_color_btn  = self._color_btn("Waveform colour",   "osc_color")
        self.bg_color_btn   = self._color_btn("Background colour", "bg_color")
        self.grid_color_btn = self._color_btn("Grid colour",       "grid_color")
        theme_lay.addWidget(self.osc_color_btn)
        theme_lay.addWidget(self.bg_color_btn)
        theme_lay.addWidget(self.grid_color_btn)
        layout.addWidget(theme_box)

        # -- Oscilloscope --
        osc_box = QGroupBox("Oscilloscope")
        osc_lay = QVBoxLayout(osc_box)
        osc_lay.addWidget(QLabel("Zoom"))
        osc_lay.addWidget(self._slider(10, 80, int(settings["zoom"] * 10),
                                       lambda v: self._set("zoom", v / 10.0)))
        osc_lay.addWidget(QLabel("Line thickness"))
        osc_lay.addWidget(self._slider(1, 6, settings["thickness"],
                                       lambda v: self._set("thickness", v)))
        layout.addWidget(osc_box)

        # -- Spectrum --
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
        self.window_combo = QComboBox()
        for fn in ["hanning", "hamming", "blackman", "bartlett", "none"]:
            self.window_combo.addItem(fn.capitalize(), fn)
        self.window_combo.setCurrentText(settings.get("window_fn", "hanning").capitalize())
        self.window_combo.currentIndexChanged.connect(
            lambda: self._set("window_fn", self.window_combo.currentData()))
        spec_lay.addWidget(self.window_combo)
        layout.addWidget(spec_box)

        # -- Gain --
        gain_box = QGroupBox("Gain")
        gain_lay = QVBoxLayout(gain_box)
        self.gain_label = QLabel(f"x{settings['gain']:.1f}")
        gain_lay.addWidget(self.gain_label)
        gain_lay.addWidget(self._slider(1, 80, int(settings["gain"] * 10),
                                        self._on_gain))
        layout.addWidget(gain_box)

        # -- Peak hold --
        peak_box = QGroupBox("Peak hold")
        peak_lay = QVBoxLayout(peak_box)
        self.peak_btn = QPushButton()
        self._update_peak_btn()
        self.peak_btn.clicked.connect(self._toggle_peak)
        peak_lay.addWidget(self.peak_btn)
        layout.addWidget(peak_box)

        # -- Save --
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

    def _slider(self, lo, hi, val, callback):
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(lo, hi)
        s.setValue(val)
        s.valueChanged.connect(callback)
        return s

    def _color_btn(self, label, key):
        btn = QPushButton(label)
        btn.setStyleSheet(f"background: {self.settings[key]}; color: white;")
        btn.clicked.connect(lambda _, k=key, b=btn: self._pick_color(k, b))
        return btn

    def _pick_color(self, key, btn):
        col = QColorDialog.getColor(QColor(self.settings[key]), self, "Pick colour")
        if col.isValid():
            self.settings[key]     = col.name()
            self.settings["theme"] = "custom"
            self.theme_combo.setCurrentText("Custom")
            btn.setStyleSheet(f"background: {col.name()}; color: white;")

    def _on_theme(self):
        key = self.theme_combo.currentData()
        self.settings["theme"] = key
        if key != "custom" and key in THEMES:
            for k, v in THEMES[key].items():
                self.settings[k] = v
            self.osc_color_btn.setStyleSheet(f"background: {self.settings['osc_color']}; color: white;")
            self.bg_color_btn.setStyleSheet(f"background: {self.settings['bg_color']}; color: white;")
            self.grid_color_btn.setStyleSheet(f"background: {self.settings['grid_color']}; color: white;")

    def _set(self, key, val):
        self.settings[key] = val

    def _on_gain(self, v):
        val = v / 10.0
        self.settings["gain"] = val
        self.gain_label.setText(f"x{val:.1f}")

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
        self.setFixedHeight(24)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(8, 0, 8, 0)
        self.label = QLabel("Peak: 0.000   RMS: 0.000")
        self.label.setStyleSheet("color: #8888aa; font-family: monospace; font-size: 12px;")
        layout.addWidget(self.label)
        layout.addStretch()

    def update_stats(self, samples):
        peak = float(np.max(np.abs(samples)))
        rms  = float(np.sqrt(np.mean(samples ** 2)))
        self.label.setText(f"Peak: {peak:.3f}   RMS: {rms:.3f}")


# ---------------------------------------------------------------------------
# Main window
# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = Settings()
        self.setWindowTitle("OpenVis")
        self.setStyleSheet("background: #0a0a12; color: #aaaacc;")

        self.audio  = AudioEngine(device=self.settings["device"])
        self.osc    = OscilloscopeWidget(self.settings)
        self.spec   = SpectrumWidget(self.settings)
        self.panel  = SettingsPanel(self.settings, self.osc, self.spec, self.change_device)
        self.status = StatusBar()

        viz = QWidget()
        viz_lay = QVBoxLayout(viz)
        viz_lay.setContentsMargins(0, 0, 0, 0)
        viz_lay.setSpacing(4)
        viz_lay.addWidget(self.osc, stretch=1)
        viz_lay.addWidget(self.spec, stretch=1)
        viz_lay.addWidget(self.status)

        central = QWidget()
        root = QHBoxLayout(central)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)
        root.addWidget(viz, stretch=1)
        root.addWidget(self.panel)
        self.setCentralWidget(central)

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)
        self.showMaximized()

    def change_device(self, device_index):
        self.audio.close()
        self.audio = AudioEngine(device=device_index)

    def tick(self):
        samples  = self.audio.read()
        smoothed = self.audio.smoothed()
        self.osc.update_samples(smoothed)
        self.spec.update_samples(samples)
        self.status.update_stats(samples)

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
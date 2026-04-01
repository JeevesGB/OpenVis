# panels.py
import sounddevice as sd   # Required for querying audio devices

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QLabel, QSlider,
    QComboBox, QPushButton, QColorDialog, QGroupBox,
    QScrollArea
)
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QFont
from settings import THEMES


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


class SettingsPanel(QScrollArea):
    def __init__(self, settings, on_device_change):
        super().__init__()
        self.settings = settings
        self.on_device_change = on_device_change

        self.setMinimumWidth(500)
        self.setMaximumWidth(600)
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        inner = QWidget()
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

        self.osc_btn = self._color_btn("Waveform colour", "osc_color")
        self.bg_btn = self._color_btn("Background colour", "bg_color")
        self.grid_btn = self._color_btn("Grid colour", "grid_color")

        for b in [self.osc_btn, self.bg_btn, self.grid_btn]:
            theme_lay.addWidget(b)
        layout.addWidget(theme_box)

        osc_box = QGroupBox("Oscilloscope")
        osc_lay = QVBoxLayout(osc_box)
        osc_lay.addWidget(QLabel("Zoom"))
        osc_lay.addWidget(self._slider(10, 80, int(settings["zoom"] * 10),
                                       lambda v: self._set("zoom", v / 10.0)))
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
        gain_lay.addWidget(self._slider(1, 80, int(settings["gain"] * 10), self._on_gain))
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
        try:
            devices = sd.query_devices()
            for i, d in enumerate(devices):
                if d['max_input_channels'] > 0:
                    self.device_combo.addItem(f"{i}: {d['name']}", i)
        except Exception as e:
            print(f"Warning: Could not query audio devices: {e}")
            self.device_combo.addItem("No audio devices found", None)

        default = sd.default.device[0] if hasattr(sd.default, 'device') else 0
        saved = self.settings.get("device")

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
            self.settings[key] = col.name()
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

    def _set(self, k, v):
        self.settings[k] = v

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
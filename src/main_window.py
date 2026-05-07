# main_window.py
import sys
import threading
from pathlib import Path
from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QWidget, QVBoxLayout,
    QPushButton, QHBoxLayout
)
from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QIcon
from settings import Settings
from audio import AudioEngine
from analysis import AnalysisEngine
from widgets import OscilloscopeWidget, SpectrumWidget, LUFSGraphWidget
from panels import AnalysisPanel
from status_bar import StatusBar
from settings_window import SettingsWindow
from analysis_window import AnalysisWindow

_HERE = Path(__file__).parent
APP_ICON = str(_HERE / "dat" / "img" / "ovis-square.png")

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = Settings()

        self.setWindowTitle("OpenViz")
        self.setStyleSheet("background: #0a0a12; color: #aaaacc;")
        self.setWindowIcon(QIcon(APP_ICON))

        self.audio = AudioEngine(device=self.settings["device"])
        self.analyser = AnalysisEngine()

        self.osc = OscilloscopeWidget(self.settings)
        self.spec = SpectrumWidget(self.settings)
        self.lufs_graph = LUFSGraphWidget(self.settings)

        self.statusbar = StatusBar()

        self.analyser.results_ready.connect(
            lambda r: self.lufs_graph.update_history(r["lufs_history"])
        )

        viz_splitter = QSplitter(Qt.Orientation.Vertical)
        viz_splitter.addWidget(self.osc)
        viz_splitter.addWidget(self.spec)
        viz_splitter.addWidget(self.lufs_graph)
        viz_splitter.setSizes([300, 300, 160])

        viz_wrap = QWidget()
        vl = QVBoxLayout(viz_wrap)
        vl.setContentsMargins(0, 0, 0, 0)
        vl.setSpacing(4)
        vl.addWidget(viz_splitter, stretch=1)
        vl.addWidget(self.statusbar)

        self.setCentralWidget(viz_wrap)

        self.settings_btn = QPushButton("⚙")
        self.settings_btn.setStyleSheet("font-size: 14px;")
        self.settings_btn.setToolTip("Settings")
        self.analysis_btn = QPushButton("📊")
        self.analysis_btn.setStyleSheet("font-size: 14px;")
        self.analysis_btn.setToolTip("Analysis")

        self.settings_btn.setFixedHeight(20)
        self.settings_btn.setFixedWidth(20)
        self.analysis_btn.setFixedHeight(20)
        self.analysis_btn.setFixedWidth(20)

        self.settings_btn.clicked.connect(self.open_settings)
        self.analysis_btn.clicked.connect(self.open_analysis)


        _toggle_style_on  = "font-size: 11px; color: #aaaacc; padding: 0 6px;"
        _toggle_style_off = "font-size: 11px; color: #444466; padding: 0 6px;"

        self.osc_toggle = QPushButton("OSC")
        self.osc_toggle.setCheckable(True)
        self.osc_toggle.setChecked(self.settings.get("show_osc", True))
        self.osc_toggle.setFixedHeight(20)
        self.osc_toggle.setToolTip("Show / hide oscilloscope")

        self.spec_toggle = QPushButton("SPEC")
        self.spec_toggle.setCheckable(True)
        self.spec_toggle.setChecked(self.settings.get("show_spec", True))
        self.spec_toggle.setFixedHeight(20)
        self.spec_toggle.setToolTip("Show / hide spectrum analyser")

        self.lufs_toggle = QPushButton("LUFS")
        self.lufs_toggle.setCheckable(True)
        self.lufs_toggle.setChecked(self.settings.get("show_lufs", True))
        self.lufs_toggle.setFixedHeight(20)
        self.lufs_toggle.setToolTip("Show / hide LUFS graph")

        def _apply_toggle_style(btn):
            btn.setStyleSheet(_toggle_style_on if btn.isChecked() else _toggle_style_off)

        def _make_toggle(btn, widget, key):
            def _on_toggle(checked):
                widget.setVisible(checked)
                self.settings[key] = checked
                _apply_toggle_style(btn)
            btn.toggled.connect(_on_toggle)
            widget.setVisible(btn.isChecked())
            _apply_toggle_style(btn)

        _make_toggle(self.osc_toggle,  self.osc,        "show_osc")
        _make_toggle(self.spec_toggle, self.spec,       "show_spec")
        _make_toggle(self.lufs_toggle, self.lufs_graph, "show_lufs")

        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(2)
        btn_layout.addStretch()
        btn_layout.addWidget(self.osc_toggle)
        btn_layout.addWidget(self.spec_toggle)
        btn_layout.addWidget(self.lufs_toggle)
        btn_layout.addSpacing(8)
        btn_layout.addWidget(self.analysis_btn)
        btn_layout.addWidget(self.settings_btn)

        btn_widget = QWidget()
        btn_widget.setLayout(btn_layout)
        self.statusBar().addPermanentWidget(btn_widget)

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

        self.resize(1280, 920)
        self.setMinimumSize(600, 600)
        self.show()

    def open_settings(self):
        self.settings_window = SettingsWindow(self.settings, self.change_device)
        self.settings_window.show()

    def open_analysis(self):
        self.analysis_window = AnalysisWindow(self.analyser)
        self.analysis_window.show()

    def change_device(self, idx):
        self.audio.close()
        self.audio = AudioEngine(device=idx)

    def tick(self):
        samples = self.audio.read()
        smoothed = self.audio.smoothed()

        self.osc.update_samples(smoothed)
        self.spec.update_samples(samples)
        self.statusbar.update_stats(samples)

        threading.Thread(
            target=self.analyser.process,
            args=(samples,),
            daemon=True
        ).start()

    def closeEvent(self, event):
        self.timer.stop()
        self.audio.close()
        event.accept()
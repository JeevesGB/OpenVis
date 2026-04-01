# main_window.py
import sys
import threading
from PyQt6.QtWidgets import (
    QMainWindow, QSplitter, QWidget, QVBoxLayout,
    QPushButton, QHBoxLayout
)
from PyQt6.QtCore import QTimer, Qt
from settings import Settings
from audio import AudioEngine
from analysis import AnalysisEngine
from widgets import OscilloscopeWidget, SpectrumWidget, LUFSGraphWidget
from panels import AnalysisPanel
from status_bar import StatusBar
from settings_window import SettingsWindow
from analysis_window import AnalysisWindow


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.settings = Settings()

        self.setWindowTitle("OpenViz")
        self.setStyleSheet("background: #0a0a12; color: #aaaacc;")

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

        self.settings_btn = QPushButton("⚙ Settings")
        self.analysis_btn = QPushButton("📊 Analysis")

        self.settings_btn.setFixedHeight(28)
        self.analysis_btn.setFixedHeight(28)

        self.settings_btn.clicked.connect(self.open_settings)
        self.analysis_btn.clicked.connect(self.open_analysis)

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        btn_layout.addWidget(self.analysis_btn)
        btn_layout.addWidget(self.settings_btn)

        btn_widget = QWidget()
        btn_widget.setLayout(btn_layout)
        self.statusBar().addPermanentWidget(btn_widget)

        self.timer = QTimer()
        self.timer.timeout.connect(self.tick)
        self.timer.start(33)

        self.resize(1380, 920)
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
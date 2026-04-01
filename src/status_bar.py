# status_bar.py
import numpy as np
from PyQt6.QtWidgets import QWidget, QHBoxLayout, QLabel
from PyQt6.QtGui import QFont


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
        rms = float(np.sqrt(np.mean(samples ** 2)))
        self.stats_label.setText(f"Peak: {peak:.3f}   RMS: {rms:.3f}")
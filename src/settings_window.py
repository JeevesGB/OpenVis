from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton
from panels import SettingsPanel


class SettingsWindow(QDialog):
    def __init__(self, settings, on_device_change):
        super().__init__()
        self.setWindowTitle("OpenViz - Settings")
        self.setMinimumWidth(560)
        self.setMinimumHeight(850)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)

        self.settings_panel = SettingsPanel(settings, on_device_change)
        layout.addWidget(self.settings_panel)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)
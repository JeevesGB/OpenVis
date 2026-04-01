from PyQt6.QtWidgets import QDialog, QVBoxLayout, QPushButton
from panels import AnalysisPanel


class AnalysisWindow(QDialog):
    def __init__(self, analyser):
        super().__init__()
        self.setWindowTitle("OpenViz - Analysis")
        self.setMinimumWidth(200)
        self.setMinimumHeight(420)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)

        self.analysis_panel = AnalysisPanel(analyser.settings if hasattr(analyser, 'settings') else None)


        layout.addWidget(self.analysis_panel)

        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.close)
        layout.addWidget(close_btn)

        analyser.results_ready.connect(self.analysis_panel.update_results)
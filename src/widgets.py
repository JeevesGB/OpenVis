# widgets.py
import numpy as np
from PyQt6.QtWidgets import QWidget
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPainter, QColor, QPen, QFont, QPolygonF
from PyQt6.QtCore import QPointF
from settings import CHUNK, RATE

CHUNK = 1024
RATE = 44100


def draw_grid(painter, rect, color, divisions=8):
    """Shared helper to draw grid"""
    painter.setPen(QPen(color, 0.5))
    x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
    for i in range(1, divisions):
        painter.drawLine(x + w * i // divisions, y, x + w * i // divisions, y + h)
        painter.drawLine(x, y + h * i // divisions, x + w, y + h * i // divisions)
    painter.drawRect(rect)


class OscilloscopeWidget(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.samples = np.zeros(CHUNK * 3, dtype=np.float32)
        self.peak_pos = None
        self.peak_ttl = 0
        self.setMinimumHeight(140)

    def update_samples(self, samples):
        self.samples = samples * self.settings["gain"]
        self.update()

    def _color(self, key):
        return QColor(self.settings[key])

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -2)

        painter.fillRect(rect, self._color("bg_color"))
        draw_grid(painter, rect, self._color("grid_color"))

        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        mid_y = y + h / 2

        # Center line
        painter.setPen(QPen(QColor(50, 50, 80), 1))
        painter.drawLine(x, int(mid_y), x + w, int(mid_y))

        visible = max(1, int(len(self.samples) / self.settings["zoom"]))
        samples = self.samples[:visible]

        osc_col = self._color("osc_color")
        pen = QPen(osc_col, self.settings["thickness"])
        pen.setCapStyle(Qt.PenCapStyle.RoundCap)
        pen.setJoinStyle(Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)

        points = QPolygonF()
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

        # Peak hold
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

        # Labels
        tc = QColor(140, 140, 170)
        painter.setPen(QPen(tc))
        painter.setFont(QFont("Monospace", 10))
        painter.drawText(x + 8, y + 18, "OSCILLOSCOPE")
        painter.drawText(x + w - 90, y + 18, f"gain x{self.settings['gain']:.1f}")
        painter.end()


class SpectrumWidget(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.samples = np.zeros(CHUNK, dtype=np.float32)
        self._n = settings["bar_count"]
        self.smoothed_fft = np.zeros(self._n)
        self.peak_bars = np.zeros(self._n)
        self.peak_ttl = np.zeros(self._n, dtype=int)
        self.setMinimumHeight(140)

    def _ensure_size(self, n):
        if self._n != n:
            self._n = n
            self.smoothed_fft = np.zeros(n)
            self.peak_bars = np.zeros(n)
            self.peak_ttl = np.zeros(n, dtype=int)

    def update_samples(self, samples):
        self.samples = samples * self.settings["gain"]
        self.update()

    def _color(self, key):
        return QColor(self.settings[key])

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(2, 2, -2, -2)

        painter.fillRect(rect, self._color("bg_color"))
        draw_grid(painter, rect, self._color("grid_color"))

        x, y, w, h = rect.x(), rect.y(), rect.width(), rect.height()
        n = self.settings["bar_count"]
        self._ensure_size(n)

        # Window function
        windows = {
            "hanning": np.hanning, "hamming": np.hamming,
            "blackman": np.blackman, "bartlett": np.bartlett,
            "none": lambda sz: np.ones(sz),
        }
        window = windows.get(self.settings.get("window_fn", "hanning"), np.hanning)(len(self.samples))

        fft = np.abs(np.fft.rfft(self.samples * window))
        fft = np.log1p(fft)
        max_bin = int(len(fft) * 16000 / (RATE / 2))
        fft = fft[:max_bin]

        indices = np.logspace(0, np.log10(max(len(fft), 2)), n + 1).astype(int)
        indices = np.clip(indices, 0, len(fft) - 1)

        bars = np.array([
            fft[indices[i]:indices[i+1]].mean() if indices[i+1] > indices[i] else fft[indices[i]]
            for i in range(n)
        ])
        bars /= (bars.max() + 1e-9)

        self.smoothed_fft = self.smoothed_fft * 0.75 + bars * 0.25

        # Peak hold
        if self.settings["peak_hold"]:
            for i in range(n):
                if self.smoothed_fft[i] >= self.peak_bars[i]:
                    self.peak_bars[i] = self.smoothed_fft[i]
                    self.peak_ttl[i] = 90
                elif self.peak_ttl[i] > 0:
                    self.peak_ttl[i] -= 1
                else:
                    self.peak_bars[i] = max(0.0, self.peak_bars[i] - 0.01)

        bar_w = w / n
        osc_col = self._color("osc_color")

        for i, val in enumerate(self.smoothed_fft):
            bar_h = int(val * h * 0.92)
            bx = x + i * bar_w
            color = QColor(
                max(0, min(255, int(val * 80))),
                max(0, min(255, int(100 + val * 155))),
                max(0, min(255, int(255 - val * 200))),
            )
            painter.fillRect(int(bx) + 1, y + h - bar_h, max(1, int(bar_w) - 1), bar_h, color)

            if self.settings["peak_hold"] and self.peak_ttl[i] > 0:
                ph = int(self.peak_bars[i] * h * 0.92)
                pc = QColor(osc_col)
                pc.setAlpha(min(255, int(self.peak_ttl[i]) * 3))
                painter.fillRect(int(bx) + 1, y + h - ph - 2, max(1, int(bar_w) - 1), 2, pc)

        # Frequency labels
        tc = QColor(140, 140, 170)
        painter.setFont(QFont("Monospace", 9))
        for freq in [100, 500, 1000, 4000, 8000, 16000]:
            frac = np.log10(freq / 100) / np.log10(16000 / 100)
            lx = x + int(frac * w)
            label = f"{freq//1000}k" if freq >= 1000 else str(freq)
            painter.setPen(QPen(self._color("grid_color"), 0.5))
            painter.drawLine(lx, y + h - 20, lx, y + h)
            painter.setPen(QPen(tc))
            painter.drawText(lx - 10, y + h - 4, label)

        painter.setPen(QPen(tc))
        painter.setFont(QFont("Monospace", 10))
        painter.drawText(x + 8, y + 18, "SPECTRUM ANALYSER")
        painter.end()


class LUFSGraphWidget(QWidget):
    def __init__(self, settings):
        super().__init__()
        self.settings = settings
        self.history = []
        self.setMinimumHeight(100)
        self.setMaximumHeight(140)

    def update_history(self, history):
        self.history = history
        self.update()

    def _color(self, key):
        return QColor(self.settings[key])

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
                px = x + (i / max(len(self.history) - 1, 1)) * w
                py = y + h - ((val + 70) / 70) * h
                py = max(float(y), min(float(y + h), py))
                pts.append(QPointF(px, py))
            painter.drawPolyline(pts)

        tc = QColor(140, 140, 170)
        painter.setPen(QPen(tc))
        painter.setFont(QFont("Monospace", 9))
        painter.drawText(x + 6, y + 14, "LUFS")
        painter.drawText(x + w - 46, y + 14, " 0 dB")
        painter.drawText(x + w - 52, y + h - 4, "-70 dB")
        painter.end()
import time
import numpy as np
from collections import deque
from scipy.signal import find_peaks
from PyQt6.QtCore import QObject, pyqtSignal
from settings import NOTE_NAMES, RATE


class AnalysisEngine(QObject):
    results_ready = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self._energy_history = deque(maxlen=43)
        self._bpm = 0.0
        self._beat_times = deque(maxlen=16)
        self._lufs_history = deque(maxlen=300)
        self._clip_count = 0

    def process(self, samples: np.ndarray):
        clipped = bool(np.any(np.abs(samples) >= 0.999))
        if clipped:
            self._clip_count += 1
        else:
            self._clip_count = max(0, self._clip_count - 1)

        note_str = self._detect_pitch(samples)
        bpm = self._detect_bpm(samples)
        lufs = self._measure_lufs(samples)
        self._lufs_history.append(lufs)

        self.results_ready.emit({
            "clipped": clipped,
            "clip_hold": self._clip_count > 0,
            "note": note_str,
            "bpm": bpm,
            "lufs": lufs,
            "lufs_history": list(self._lufs_history),
        })

    def _detect_pitch(self, samples):
        corr = np.correlate(samples, samples, mode='full')
        corr = corr[len(corr) // 2:]
        d = np.diff(corr)
        start = next((i for i in range(len(d) - 1) if d[i] < 0), None)
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

        midi = 69 + 12 * np.log2(freq / 440.0)
        note = NOTE_NAMES[int(round(midi)) % 12]
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
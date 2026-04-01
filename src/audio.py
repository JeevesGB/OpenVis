import threading
import numpy as np
import sounddevice as sd
from collections import deque
from settings import CHUNK, RATE, HISTORY


CHUNK = 1024
RATE = 44100
HISTORY = 3


class AudioEngine:
    def __init__(self, device=None):
        self.buffer = deque(maxlen=HISTORY)
        self._latest = np.zeros(CHUNK, dtype=np.float32)
        self._stereo = np.zeros((CHUNK, 2), dtype=np.float32)
        self._lock = threading.Lock()

        for _ in range(HISTORY):
            self.buffer.append(np.zeros(CHUNK, dtype=np.float32))

        def first_input_device():
            for i, d in enumerate(sd.query_devices()):
                if d['max_input_channels'] > 0:
                    try:
                        sd.query_devices(i, 'input')
                        return i
                    except Exception:
                        continue
            return None

        if device is not None:
            try:
                sd.query_devices(device, 'input')
            except Exception:
                device = None

        if device is None:
            default_input = sd.default.device[0]
            try:
                sd.query_devices(default_input, 'input')
                device = default_input
            except Exception:
                device = first_input_device()

        if device is None:
            raise RuntimeError("No input audio device found.")

        info = sd.query_devices(device, 'input')
        self._ch = min(2, int(info['max_input_channels']))

        self.stream = sd.InputStream(
            samplerate=RATE,
            blocksize=CHUNK,
            channels=self._ch,
            dtype='float32',
            device=device,
            callback=self._callback,
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
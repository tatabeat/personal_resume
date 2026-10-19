from __future__ import annotations

from pathlib import Path

import numpy as np
from scipy.io import wavfile


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(audio, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError("Only mono audio is supported")
    pcm = np.clip(array * 32767.0, -32768, 32767).astype(np.int16)
    wavfile.write(destination, int(sample_rate), pcm)
    return destination


def play_audio(audio: np.ndarray, sample_rate: int, *, blocking: bool = False) -> None:
    try:
        import sounddevice as sd
    except ImportError as exc:
        raise RuntimeError("sounddevice is not installed") from exc
    sd.stop()
    sd.play(np.asarray(audio, dtype=np.float32), int(sample_rate))
    if blocking:
        sd.wait()


def stop_audio() -> None:
    try:
        import sounddevice as sd
    except ImportError:
        return
    sd.stop()

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


@dataclass(slots=True)
class PhonemeToken:
    """One aligned text unit from the Mandarin frontend."""

    text: str
    pinyin: str | None
    initial: str = ""
    final: str = ""
    tone: int = 5
    kind: str = "syllable"  # syllable | pause | raw

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SegmentSpan:
    """Where one text token lives on the generated gesture timeline."""

    text: str
    pinyin: str | None
    initial: str
    final: str
    tone: int
    kind: str
    start_s: float
    end_s: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class GestureTimeline:
    """Time-aligned articulatory/acoustic control tracks."""

    rate_hz: int
    tracks: dict[str, np.ndarray]
    spans: list[SegmentSpan]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        lengths = {len(np.asarray(value)) for value in self.tracks.values()}
        if len(lengths) > 1:
            raise ValueError(f"Gesture tracks have inconsistent lengths: {sorted(lengths)}")
        for name, value in list(self.tracks.items()):
            array = np.asarray(value, dtype=np.float64)
            if array.ndim != 1:
                raise ValueError(f"Track {name!r} must be one-dimensional")
            if not np.all(np.isfinite(array)):
                raise ValueError(f"Track {name!r} contains NaN or Infinity")
            self.tracks[name] = array

    @property
    def frame_count(self) -> int:
        if not self.tracks:
            return 0
        return len(next(iter(self.tracks.values())))

    @property
    def duration_s(self) -> float:
        return self.frame_count / float(self.rate_hz)

    @property
    def times_s(self) -> np.ndarray:
        return np.arange(self.frame_count, dtype=np.float64) / float(self.rate_hz)

    def to_dict(self, *, include_tracks: bool = True) -> dict[str, Any]:
        data: dict[str, Any] = {
            "rate_hz": self.rate_hz,
            "duration_s": self.duration_s,
            "frame_count": self.frame_count,
            "spans": [span.to_dict() for span in self.spans],
            "metadata": self.metadata,
        }
        if include_tracks:
            data["tracks"] = {
                name: [float(item) for item in values]
                for name, values in self.tracks.items()
            }
        return data

from __future__ import annotations

import tkinter as tk
from typing import Callable

import numpy as np

from .models import GestureTimeline


class WaveformEditor(tk.Canvas):
    def __init__(
        self,
        master,
        values: list[float] | np.ndarray,
        on_change: Callable[[], None] | None = None,
        **kwargs,
    ):
        super().__init__(
            master,
            background="#101827",
            highlightthickness=1,
            highlightbackground="#475569",
            **kwargs,
        )
        self.values = np.asarray(values, dtype=np.float64).copy()
        self.on_change = on_change
        self._drag_index: int | None = None
        self.bind("<Configure>", lambda _event: self.redraw())
        self.bind("<Button-1>", self._on_press)
        self.bind("<B1-Motion>", self._on_drag)
        self.bind("<ButtonRelease-1>", lambda _event: setattr(self, "_drag_index", None))

    def set_values(self, values: list[float] | np.ndarray) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size < 8:
            raise ValueError("Waveform needs at least 8 points")
        # v2 profiles normally use 512 points.  Replacing the array rather than
        # forcing the old fixed 64-point shape lets native/high-resolution source
        # profiles remain intact in the editor.
        self.values = np.clip(array, -1.0, 1.0).copy()
        self.redraw()
        if self.on_change:
            self.on_change()

    def _bounds(self) -> tuple[float, float, float, float]:
        margin = 24.0
        return (
            margin,
            margin,
            max(margin + 2.0, self.winfo_width() - margin),
            max(margin + 2.0, self.winfo_height() - margin),
        )

    def _xy(self, index: int) -> tuple[float, float]:
        left, top, right, bottom = self._bounds()
        x = left + (right - left) * index / len(self.values)
        y = top + (1.0 - (self.values[index] + 1.0) * 0.5) * (bottom - top)
        return x, y

    def redraw(self) -> None:
        self.delete("all")
        left, top, right, bottom = self._bounds()
        middle = (top + bottom) * 0.5
        self.create_rectangle(left, top, right, bottom, outline="#475569")
        self.create_line(left, middle, right, middle, fill="#334155", dash=(5, 5))
        self.create_text(
            left + 4,
            top + 4,
            anchor="nw",
            fill="#94a3b8",
            text=f"一個週期的設計聲源（{len(self.values)} points）；這不是整句 WAV 波形",
        )
        coordinates: list[float] = []
        for index in range(len(self.values)):
            coordinates.extend(self._xy(index))
        coordinates.extend([right, self._xy(0)[1]])
        if len(coordinates) >= 4:
            self.create_line(*coordinates, fill="#38bdf8", width=2.0)
        # Hundreds of source samples should remain visible as a smooth curve;
        # drawing a handle for every one would turn the editor into a white band.
        handle_stride = max(1, int(np.ceil(len(self.values) / 96.0)))
        for index in range(0, len(self.values), handle_stride):
            x, y = self._xy(index)
            radius = 3
            self.create_oval(
                x - radius,
                y - radius,
                x + radius,
                y + radius,
                fill="#f8fafc",
                outline="#0284c7",
            )

    def _nearest_index(self, event: tk.Event) -> int:
        left, _top, right, _bottom = self._bounds()
        position = (event.x - left) / max(1.0, right - left)
        return int(np.clip(round(position * len(self.values)), 0, len(self.values) - 1))

    def _update(self, index: int, event: tk.Event) -> None:
        _left, top, _right, bottom = self._bounds()
        normalized = 1.0 - 2.0 * (event.y - top) / max(1.0, bottom - top)
        self.values[index] = float(np.clip(normalized, -1.0, 1.0))
        self.redraw()
        if self.on_change:
            self.on_change()

    def _on_press(self, event: tk.Event) -> None:
        self._drag_index = self._nearest_index(event)
        self._update(self._drag_index, event)

    def _on_drag(self, event: tk.Event) -> None:
        index = self._nearest_index(event)
        self._drag_index = index
        self._update(index, event)


class TimelineCanvas(tk.Canvas):
    def __init__(self, master, **kwargs):
        super().__init__(
            master,
            background="#0f172a",
            highlightthickness=1,
            highlightbackground="#475569",
            **kwargs,
        )
        self.timeline: GestureTimeline | None = None
        self.bind("<Configure>", lambda _event: self.redraw())

    def set_timeline(self, timeline: GestureTimeline | None) -> None:
        self.timeline = timeline
        self.redraw()

    def redraw(self) -> None:
        self.delete("all")
        width = max(2, self.winfo_width())
        height = max(2, self.winfo_height())
        margin_left = 56
        margin_right = 16
        margin_top = 24
        margin_bottom = 28
        plot_width = max(1, width - margin_left - margin_right)
        plot_height = max(1, height - margin_top - margin_bottom)

        self.create_rectangle(
            margin_left,
            margin_top,
            margin_left + plot_width,
            margin_top + plot_height,
            outline="#475569",
        )
        if self.timeline is None or self.timeline.frame_count < 2:
            self.create_text(width / 2, height / 2, fill="#94a3b8", text="尚未產生發聲構造時間軸")
            return

        timeline = self.timeline
        times = timeline.times_s
        duration = max(timeline.duration_s, 1e-6)
        f0 = timeline.tracks["f0_hz"]
        jaw = timeline.tracks["jaw_open"]
        lip = timeline.tracks["lip_round"]
        pressure = timeline.tracks["pressure"]

        voiced = timeline.tracks.get("voicing", np.ones_like(f0)) > 0.12
        f0_valid = f0[voiced & np.isfinite(f0) & (f0 > 0.0)]
        if f0_valid.size:
            # 使用實際最小／最大值，不再用 2～98 百分位裁切高峰。
            f0_data_min = float(np.min(f0_valid))
            f0_data_max = float(np.max(f0_valid))

            # 在上下各保留一點空間，避免曲線貼住圖框。
            f0_span = max(f0_data_max - f0_data_min, 1.0)
            f0_padding = max(5.0, f0_span * 0.08)
            f0_min = max(0.0, f0_data_min - f0_padding)
            f0_max = f0_data_max + f0_padding
        else:
            f0_data_min, f0_data_max = 50.0, 500.0
            f0_min, f0_max = 50.0, 500.0

        def x_of(time_value: float) -> float:
            return margin_left + plot_width * time_value / duration

        def y_upper(value: float) -> float:
            normalized = np.clip((value - f0_min) / (f0_max - f0_min), 0.0, 1.0)
            return margin_top + plot_height * 0.47 * (1.0 - normalized)

        def y_lower(value: float, lane: int) -> float:
            lane_top = margin_top + plot_height * (0.53 + lane * 0.15)
            lane_height = plot_height * 0.12
            return lane_top + lane_height * (1.0 - np.clip(value, 0.0, 1.0))

        stride = max(1, len(times) // max(400, plot_width))
        sample_indices = np.arange(0, len(times), stride)

        f0_points: list[float] = []
        for index in sample_indices:
            f0_points.extend([x_of(times[index]), y_upper(f0[index])])
        if len(f0_points) >= 4:
            self.create_line(*f0_points, fill="#38bdf8", width=2.0)

        for values, lane, color in (
            (jaw, 0, "#f59e0b"),
            (lip, 1, "#a78bfa"),
            (pressure, 2, "#34d399"),
        ):
            points: list[float] = []
            for index in sample_indices:
                points.extend([x_of(times[index]), y_lower(values[index], lane)])
            if len(points) >= 4:
                self.create_line(*points, fill=color, width=1.5)

        for span in timeline.spans:
            x = x_of(span.start_s)
            self.create_line(x, margin_top, x, margin_top + plot_height, fill="#334155")
            if span.kind == "syllable":
                label = span.pinyin or span.text
                self.create_text(
                    x + 3,
                    margin_top + 3,
                    anchor="nw",
                    fill="#cbd5e1",
                    text=label,
                    font=("Segoe UI", 8),
                )

        self.create_text(8, margin_top + 6, anchor="nw", fill="#38bdf8", text="F0")
        self.create_text(8, margin_top + plot_height * 0.55, anchor="nw", fill="#f59e0b", text="下顎")
        self.create_text(8, margin_top + plot_height * 0.70, anchor="nw", fill="#a78bfa", text="圓唇")
        self.create_text(8, margin_top + plot_height * 0.85, anchor="nw", fill="#34d399", text="肺壓")
        self.create_text(
            margin_left,
            height - 8,
            anchor="sw",
            fill="#94a3b8",
            text=f"0 s    時長 {duration:.2f} s",
        )
        self.create_text(
            margin_left + plot_width,
            height - 8,
            anchor="se",
            fill="#94a3b8",
            text=f"F0 顯示範圍 {f0_min:.0f}-{f0_max:.0f} Hz",
        )

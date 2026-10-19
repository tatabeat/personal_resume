from __future__ import annotations

import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Callable

import pyautogui


pyautogui.PAUSE = 0
pyautogui.FAILSAFE = False

_CLICK_RE = re.compile(r"^c\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$")
_PRESS_RE = re.compile(r"^p\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$")
_SLIDE_RE = re.compile(r"^s\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)\(\s*(-?\d+)\s*,\s*(-?\d+)\s*\)$")
_VALID_SINGLE = {"e", "n"}


def normalize_command(command: str) -> str:
    text = (command or "").strip()
    if not text:
        return "e"
    if text in _VALID_SINGLE:
        return text
    if _CLICK_RE.fullmatch(text):
        return text
    if _PRESS_RE.fullmatch(text):
        return text
    if _SLIDE_RE.fullmatch(text):
        return text
    return "e"


def parse_plan_text(raw_text: str, list_length: int) -> list[str]:
    target = max(1, int(list_length))
    text = (raw_text or "").strip()
    if not text:
        return ["e"] * target
    if text == "n":
        return ["n"] * target
    parts = [part.strip() for part in text.split("|") if part.strip()]
    normalized = [normalize_command(part) for part in parts]
    if not normalized:
        normalized = ["e"]
    if len(normalized) < target:
        normalized.extend(["e"] * (target - len(normalized)))
    return normalized[:target]


@dataclass
class ActionItem:
    due_time: float
    index: int
    offset_seconds: float
    command: str


class ActionQueue:
    def __init__(self) -> None:
        self._items: deque[ActionItem] = deque()
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def replace_future(self, start_time: float, commands: list[str], step_seconds: float) -> None:
        with self._lock:
            preserved = [item for item in self._items if item.due_time < start_time]
            new_items = [
                ActionItem(
                    due_time=start_time + index * step_seconds,
                    index=index,
                    offset_seconds=index * step_seconds,
                    command=normalize_command(command),
                )
                for index, command in enumerate(commands)
            ]
            merged = preserved + new_items
            merged.sort(key=lambda item: item.due_time)
            self._items = deque(merged)

    def pop_due(self, now: float) -> list[ActionItem]:
        due: list[ActionItem] = []
        with self._lock:
            while self._items and self._items[0].due_time <= now:
                due.append(self._items.popleft())
        return due

    def snapshot_text(self, now: float) -> str:
        with self._lock:
            items = list(self._items)
        if not items:
            return "empty"
        return "\n".join(f"{max(0.0, item.due_time-now):04.1f}s {item.command}" for item in items)

    def tail_due_time(self) -> float | None:
        with self._lock:
            return self._items[-1].due_time if self._items else None

    def remaining_seconds(self, now: float) -> float:
        tail = self.tail_due_time()
        return 0.0 if tail is None else max(0.0, tail - now)

    def next_due_in(self, now: float) -> float | None:
        with self._lock:
            return None if not self._items else max(0.0, self._items[0].due_time - now)


class ActionRunner:
    def __init__(
        self,
        hand_key: str,
        action_queue: ActionQueue,
        stop_event: threading.Event,
        origin: tuple[int, int] = (0, 0),
        execute_enabled: bool = True,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.hand_key = hand_key.upper()
        self.action_queue = action_queue
        self.stop_event = stop_event
        self.origin = (int(origin[0]), int(origin[1]))
        self.execute_enabled = execute_enabled
        self.log_fn = log_fn
        self._holding_left = False

    def _emit(self, message: str) -> None:
        print(message, flush=True)
        if self.log_fn is not None:
            self.log_fn(message)

    def _release_if_needed(self) -> None:
        if self._holding_left:
            pyautogui.mouseUp(button="left")
            self._holding_left = False

    def _to_screen_point(self, x: int, y: int) -> tuple[int, int]:
        return self.origin[0] + int(x), self.origin[1] + int(y)

    def _execute_click(self, x: int, y: int) -> None:
        self._release_if_needed()
        sx, sy = self._to_screen_point(x, y)
        pyautogui.click(x=sx, y=sy, button="left")

    def _ensure_holding(self, x: int, y: int) -> None:
        sx, sy = self._to_screen_point(x, y)
        if not self._holding_left:
            pyautogui.mouseDown(x=sx, y=sy, button="left")
            self._holding_left = True

    def _execute_press(self, x: int, y: int) -> None:
        self._ensure_holding(x, y)
        sx, sy = self._to_screen_point(x, y)
        pyautogui.moveTo(sx, sy, duration=0)

    def _execute_slide(self, x1: int, y1: int, x2: int, y2: int) -> None:
        self._ensure_holding(x1, y1)
        sx2, sy2 = self._to_screen_point(x2, y2)
        pyautogui.moveTo(sx2, sy2, duration=0)

    def _execute_command(self, command: str) -> None:
        text = normalize_command(command)
        if text in {"e", "n"}:
            self._release_if_needed()
            return
        match = _CLICK_RE.fullmatch(text)
        if match:
            self._execute_click(int(match.group(1)), int(match.group(2)))
            return
        match = _PRESS_RE.fullmatch(text)
        if match:
            self._execute_press(int(match.group(1)), int(match.group(2)))
            return
        match = _SLIDE_RE.fullmatch(text)
        if match:
            self._execute_slide(*map(int, match.groups()))
            return
        self._release_if_needed()

    def run(self) -> None:
        try:
            while not self.stop_event.is_set():
                now = time.monotonic()
                due_items = self.action_queue.pop_due(now)
                if not due_items:
                    next_due = self.action_queue.next_due_in(now)
                    sleep_for = 0.01 if next_due is None else max(0.001, min(0.01, next_due / 2))
                    self.stop_event.wait(sleep_for)
                    continue
                for item in due_items:
                    self._emit(f"[ACTION][{self.hand_key}] t+{item.offset_seconds:0.1f}s | now={time.monotonic():0.3f} | cmd={item.command}")
                    if self.execute_enabled:
                        self._execute_command(item.command)
        finally:
            if self.execute_enabled:
                self._release_if_needed()

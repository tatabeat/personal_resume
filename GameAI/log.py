import datetime
import re
from pathlib import Path
from typing import Iterable

_left_text = None
_right_text = None
_pending: list[tuple[str, str]] = []
_ALLOWED_PREFIXES = (
    "Preview",
    "[ACTION]",
    "[MIND]",
    "[DECIDE]",
    "[QUICK]",
    "[ERROR]",
)


def _is_allowed(msg: str) -> bool:
    text = (msg or "").strip()
    return any(text.startswith(prefix) for prefix in _ALLOWED_PREFIXES)


def _targets_for(msg: str) -> Iterable[str]:
    text = (msg or "").strip()
    if text.startswith("Preview"):
        return ("L", "R")
    if "[L]" in text or "_L_" in text:
        return ("L",)
    if "[R]" in text or "_R_" in text:
        return ("R",)
    return ("L", "R")


def init(left_text_widget, right_text_widget):
    global _left_text, _right_text
    _left_text = left_text_widget
    _right_text = right_text_widget
    if _pending:
        for side, line in _pending:
            _insert_line(side, line)
        _pending.clear()


def _insert_line(side: str, line: str) -> None:
    widget = _left_text if side == "L" else _right_text
    if widget is None:
        _pending.append((side, line))
        return
    widget.insert("end", line)
    widget.see("end")


def log(msg: str):
    if not _is_allowed(msg):
        return
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    line = f"[{ts}] {msg}\n"
    targets = tuple(_targets_for(msg))
    if _left_text is None and _right_text is None:
        for side in targets:
            _pending.append((side, line))
        return

    def _write():
        for side in targets:
            _insert_line(side, line)

    target_widget = _left_text or _right_text
    if target_widget is not None:
        target_widget.after(0, _write)


def sanitize_filename(name: str, max_len: int = 120) -> str:
    text = (name or "").strip().replace("\n", " ").replace("\r", " ")
    text = re.sub(r'[\\/:*?"<>|]', "_", text)
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"_+", "_", text).strip("._ ")
    if not text:
        text = "empty"
    return text[:max_len].rstrip("._ ") or "empty"


def save_stage_image(project_dir: Path, hand_key: str, stage_name: str, image, response_text: str) -> Path:
    folder = Path(project_dir) / "logs" / f"{hand_key}_{stage_name}"
    folder.mkdir(parents=True, exist_ok=True)
    stem = sanitize_filename(response_text)
    target = folder / f"{stem}.png"
    if not target.exists():
        image.save(target)
        return target
    suffix = 2
    while True:
        candidate = folder / f"{stem}_{suffix}.png"
        if not candidate.exists():
            image.save(candidate)
            return candidate
        suffix += 1

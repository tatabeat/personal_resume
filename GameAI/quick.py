from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Callable

import gpt
import log
import scshot
from action import ActionQueue, parse_plan_text
from mind import HandContext


PROJECT_DIR = Path(__file__).resolve().parent
IDLE_POLL_SECONDS = 0.05
DISABLED_POLL_SECONDS = 0.25


class QuickRunner:
    def __init__(
        self,
        hand_key: str,
        profile_dir: Path,
        region: tuple[int, int, int, int],
        context: HandContext,
        primary_queue: ActionQueue,
        partner_queue: ActionQueue,
        stop_event: threading.Event,
        log_fn: Callable[[str], None] | None = None,
    ) -> None:
        self.hand_key = hand_key.upper()
        self.profile_dir = Path(profile_dir)
        self.region = region
        self.context = context
        self.primary_queue = primary_queue
        self.partner_queue = partner_queue
        self.stop_event = stop_event
        self.log_fn = log_fn
        self.config_name = f"{self.hand_key}_quick"

    def _log(self, message: str) -> None:
        if self.log_fn is not None:
            self.log_fn(message)

    def _capture(self):
        left, top, width, height = self.region
        return scshot.sccshot(left, top, width, height)

    def _build_text(self, now: float) -> str:
        return (
            f"【控制器】\n{self.hand_key}\n\n"
            f"【最新 mind 結果】\n{self.context.last_mind_result or '無'}\n\n"
            f"【最新 decide 結果】\n{self.context.last_decide_result or '無'}\n\n"
            f"【本手目前 action_list】\n{self.primary_queue.snapshot_text(now)}\n\n"
            f"【另一手目前 action_list】\n{self.partner_queue.snapshot_text(now)}"
        )

    def run(self) -> None:
        while not self.stop_event.is_set():
            try:
                settings = gpt.get_stage_settings(self.config_name, profile_dir=self.profile_dir)
                if not settings["enabled"]:
                    if self.stop_event.wait(DISABLED_POLL_SECONDS):
                        break
                    continue
                now = time.monotonic()
                interval = max(0.05, float(settings["interval_seconds"]))
                if (now - self.context.last_quick_time) < interval:
                    if self.stop_event.wait(IDLE_POLL_SECONDS):
                        break
                    continue

                screenshot = self._capture()
                started = time.monotonic()
                result = gpt.call_with_image(
                    screenshot,
                    self._build_text(now),
                    config_name=self.config_name,
                    profile_dir=self.profile_dir,
                )
                elapsed = time.monotonic() - started
                if settings.get("log_images", True):
                    log.save_stage_image(PROJECT_DIR, self.hand_key, "quick", screenshot, result)
                plan = settings["plan"]
                commands = parse_plan_text(result, plan["list_length"])
                start_time = time.monotonic() + plan["start_offset_seconds"]
                self.primary_queue.replace_future(start_time, commands, plan["step_seconds"])
                self.context.last_quick_result = result.strip()
                self.context.last_quick_time = time.monotonic()
                self._log(f"[QUICK][{self.hand_key}] {elapsed:.3f} 秒 | {self.context.last_quick_result}")
            except Exception as e:
                self._log(f"[ERROR][{self.hand_key}] quick {e}")
                if self.stop_event.wait(0.5):
                    break

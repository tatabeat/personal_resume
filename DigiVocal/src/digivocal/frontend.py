from __future__ import annotations

import importlib.util
import unicodedata
from dataclasses import dataclass
from typing import Iterable

from .models import PhonemeToken
from .phonology import PINYIN_RE, parse_pinyin_syllable


PAUSE_PUNCTUATION = {
    "，",
    ",",
    "、",
    "。",
    ".",
    "！",
    "!",
    "？",
    "?",
    "；",
    ";",
    "：",
    ":",
    "…",
    "—",
    "-",
    "\n",
    "\t",
    " ",
}


def is_han(character: str) -> bool:
    if not character:
        return False
    code = ord(character[0])
    return (
        0x3400 <= code <= 0x4DBF
        or 0x4E00 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x20000 <= code <= 0x2FA1F
    )


def normalize_text(text: str) -> str:
    return unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")


@dataclass(slots=True)
class FrontendResult:
    tokens: list[PhonemeToken]
    backend_used: str
    warning: str | None = None


class MandarinFrontend:
    """Chinese text -> context-aware numbered Pinyin -> syllable tokens.

    ``pypinyin`` is the stable core backend. ``g2pW`` is optional and is used
    only when explicitly selected or when ``auto`` finds it installed.
    """

    def __init__(self, backend: str = "auto", *, tone_sandhi: bool = True):
        backend = backend.lower().strip()
        if backend not in {"auto", "pypinyin", "g2pw"}:
            raise ValueError("backend must be auto, pypinyin, or g2pw")
        self.backend = backend
        self.tone_sandhi = bool(tone_sandhi)
        self._g2pw_converter = None

    @staticmethod
    def available_backends() -> dict[str, bool]:
        return {
            "pypinyin": importlib.util.find_spec("pypinyin") is not None,
            "g2pw": importlib.util.find_spec("g2pw") is not None,
        }

    def convert(self, text: str) -> FrontendResult:
        text = normalize_text(text)
        if not text:
            return FrontendResult([], "none", "Input text is empty")

        selected = self.backend
        if selected == "auto":
            selected = "g2pw" if self.available_backends()["g2pw"] else "pypinyin"

        warning = None
        if selected == "g2pw":
            try:
                readings = self._convert_g2pw(text)
            except Exception as exc:
                if self.backend == "g2pw":
                    raise RuntimeError(f"g2pW failed: {exc}") from exc
                selected = "pypinyin"
                warning = f"g2pW failed; fell back to pypinyin: {exc}"
                readings = self._convert_pypinyin(text)
        else:
            readings = self._convert_pypinyin(text)

        tokens = self._align_tokens(text, readings)
        if self.tone_sandhi:
            self._apply_basic_tone_sandhi(tokens)
        return FrontendResult(tokens, selected, warning)

    @staticmethod
    def _set_surface_tone(token: PhonemeToken, tone: int) -> None:
        """Update a token's surface tone while preserving numbered Pinyin."""

        token.tone = int(tone)
        if token.pinyin:
            if token.pinyin[-1:].isdigit():
                token.pinyin = f"{token.pinyin[:-1]}{tone}"
            else:
                token.pinyin = f"{token.pinyin}{tone}"

    @classmethod
    def _apply_basic_tone_sandhi(cls, tokens: list[PhonemeToken]) -> None:
        """Apply deterministic Mandarin surface-tone rules.

        pypinyin exposes ``tone_sandhi=True`` in current releases, but older
        versions and some optional frontends may return lexical tones.  These
        guarded rules keep the renderer consistent without requiring a word
        segmenter.  They deliberately stop at punctuation and raw symbols.
        """

        run_start = 0
        while run_start < len(tokens):
            while run_start < len(tokens) and tokens[run_start].kind != "syllable":
                run_start += 1
            if run_start >= len(tokens):
                break

            run_end = run_start
            while run_end < len(tokens) and tokens[run_end].kind == "syllable":
                run_end += 1

            # Third-tone sandhi: every T3 immediately followed by T3 is
            # realized with a rising surface tone.  For a simple T3-T3-T3 run
            # this yields T2-T2-T3, a useful phrase-neutral approximation.
            for index in range(run_start, run_end - 1):
                if tokens[index].tone == 3 and tokens[index + 1].tone == 3:
                    cls._set_surface_tone(tokens[index], 2)

            # ``不`` becomes rising before a fourth tone.
            for index in range(run_start, run_end - 1):
                if tokens[index].text == "不" and tokens[index + 1].tone == 4:
                    cls._set_surface_tone(tokens[index], 2)

            # A conservative ``一`` rule.  Ordinals (第一) and isolated 一 are
            # left unchanged because they require lexical/prosodic context.
            for index in range(run_start, run_end - 1):
                token = tokens[index]
                previous_text = tokens[index - 1].text if index > run_start else ""
                if token.text != "一" or previous_text == "第":
                    continue
                next_tone = tokens[index + 1].tone
                if next_tone == 4:
                    cls._set_surface_tone(token, 2)
                elif next_tone in {1, 2, 3}:
                    cls._set_surface_tone(token, 4)

            run_start = run_end

    def _convert_g2pw(self, text: str) -> list[str | None]:
        from g2pw import G2PWConverter

        if self._g2pw_converter is None:
            self._g2pw_converter = G2PWConverter(
                style="pinyin",
                enable_non_tradional_chinese=True,
            )
        result = self._g2pw_converter(text)
        if isinstance(result, list) and result and isinstance(result[0], list):
            result = result[0]
        return list(result)

    def _convert_pypinyin(self, text: str) -> list[str | None]:
        try:
            from pypinyin import Style, pinyin
        except ImportError as exc:
            raise RuntimeError(
                "pypinyin is not installed. Run: python -m pip install pypinyin"
            ) from exc

        kwargs = dict(
            style=Style.TONE3,
            heteronym=False,
            neutral_tone_with_five=True,
            errors=lambda chars: list(chars),
        )
        try:
            result = pinyin(text, tone_sandhi=self.tone_sandhi, **kwargs)
        except TypeError:
            # Older versions do not expose tone_sandhi.
            result = pinyin(text, **kwargs)

        flattened: list[str | None] = []
        for item in result:
            if isinstance(item, (list, tuple)):
                flattened.append(str(item[0]) if item else None)
            else:
                flattened.append(str(item))
        return flattened

    def _align_tokens(self, text: str, readings: Iterable[str | None]) -> list[PhonemeToken]:
        readings_list = list(readings)
        if len(readings_list) != len(text):
            # Both official backends normally align per character. This guarded
            # fallback prevents a shifted pronunciation stream from being used.
            readings_list = self._realign_by_character(text, readings_list)

        tokens: list[PhonemeToken] = []
        for index, character in enumerate(text):
            reading = readings_list[index] if index < len(readings_list) else None

            if character in PAUSE_PUNCTUATION:
                tokens.append(PhonemeToken(character, None, kind="pause"))
                continue

            if reading is not None:
                reading = reading.strip().lower().replace("ü", "v")

            if reading and PINYIN_RE.match(reading):
                try:
                    parsed = parse_pinyin_syllable(reading)
                    tokens.append(
                        PhonemeToken(
                            text=character,
                            pinyin=reading,
                            initial=parsed.initial,
                            final=parsed.final,
                            tone=parsed.tone,
                            kind="syllable",
                        )
                    )
                    continue
                except ValueError:
                    pass

            if is_han(character):
                tokens.append(
                    PhonemeToken(
                        text=character,
                        pinyin=None,
                        final="e",
                        tone=5,
                        kind="raw",
                    )
                )
            else:
                tokens.append(PhonemeToken(character, None, kind="raw"))

        return tokens

    @staticmethod
    def _realign_by_character(text: str, readings: list[str | None]) -> list[str | None]:
        output: list[str | None] = []
        cursor = 0
        for character in text:
            if character in PAUSE_PUNCTUATION:
                output.append(None)
                continue
            if cursor < len(readings):
                output.append(readings[cursor])
                cursor += 1
            else:
                output.append(None)
        return output


def tokens_from_numbered_pinyin(text: str) -> list[PhonemeToken]:
    """Create tokens from whitespace-separated numbered Pinyin.

    Example: ``ni3 hao3 , wo3 shi4``. Punctuation tokens are preserved.
    This mode is useful for testing the articulatory backend without a G2P model.
    """

    tokens: list[PhonemeToken] = []
    for item in text.strip().split():
        if item in PAUSE_PUNCTUATION:
            tokens.append(PhonemeToken(item, None, kind="pause"))
            continue
        parsed = parse_pinyin_syllable(item)
        tokens.append(
            PhonemeToken(
                text=item,
                pinyin=item,
                initial=parsed.initial,
                final=parsed.final,
                tone=parsed.tone,
                kind="syllable",
            )
        )
    return tokens

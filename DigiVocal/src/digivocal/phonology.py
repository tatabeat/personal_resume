from __future__ import annotations

import re
from dataclasses import dataclass


INITIALS = (
    "zh",
    "ch",
    "sh",
    "b",
    "p",
    "m",
    "f",
    "d",
    "t",
    "n",
    "l",
    "g",
    "k",
    "h",
    "j",
    "q",
    "x",
    "r",
    "z",
    "c",
    "s",
)

TONE_RE = re.compile(r"([1-5])$")
PINYIN_RE = re.compile(r"^[a-zvü:]+[1-5]?$", re.IGNORECASE)


@dataclass(frozen=True, slots=True)
class ParsedSyllable:
    raw: str
    base: str
    initial: str
    final: str
    tone: int


_Y_MAP = {
    "yi": "i",
    "ya": "ia",
    "yo": "io",
    "ye": "ie",
    "yao": "iao",
    "you": "iou",
    "yan": "ian",
    "yin": "in",
    "yang": "iang",
    "ying": "ing",
    "yong": "iong",
    "yu": "v",
    "yue": "ve",
    "yuan": "van",
    "yun": "vn",
}

_W_MAP = {
    "wu": "u",
    "wa": "ua",
    "wo": "uo",
    "wai": "uai",
    "wei": "uei",
    "wan": "uan",
    "wen": "uen",
    "wang": "uang",
    "weng": "ueng",
}

_FINAL_ALIASES = {
    "iu": "iou",
    "ui": "uei",
    "un": "uen",
    "ue": "ve",
    "u:e": "ve",
    "u:an": "van",
    "u:n": "vn",
}


VALID_FINALS = {
    "a",
    "o",
    "e",
    "ai",
    "ei",
    "ao",
    "ou",
    "an",
    "en",
    "ang",
    "eng",
    "er",
    "i",
    "ia",
    "io",
    "ie",
    "iao",
    "iou",
    "ian",
    "in",
    "iang",
    "ing",
    "iong",
    "u",
    "ua",
    "uo",
    "uai",
    "uei",
    "uan",
    "uen",
    "uang",
    "ueng",
    "ong",
    "v",
    "ve",
    "van",
    "vn",
    "iz",
    "ir",
    "m",
    "n",
    "ng",
}


def _strip_tone(pinyin: str) -> tuple[str, int]:
    cleaned = pinyin.strip().lower().replace("ü", "v").replace("u:", "v")
    match = TONE_RE.search(cleaned)
    if match:
        tone = int(match.group(1))
        cleaned = cleaned[:-1]
    else:
        tone = 5
    return cleaned, tone


def _normalize_zero_initial(base: str) -> str:
    if base in _Y_MAP:
        return _Y_MAP[base]
    if base in _W_MAP:
        return _W_MAP[base]
    return base


def parse_pinyin_syllable(pinyin: str) -> ParsedSyllable:
    """Split numbered Hanyu Pinyin into initial/final/tone.

    The normalized final uses ``v`` for ü, expands iu/ui/un, and represents the
    two apical vowels as ``iz`` and ``ir`` so the synthesizer can give them
    different formants.
    """

    if not pinyin or not PINYIN_RE.match(pinyin.strip()):
        raise ValueError(f"Not a numbered pinyin syllable: {pinyin!r}")

    base, tone = _strip_tone(pinyin)
    base = _normalize_zero_initial(base)

    initial = ""
    for candidate in INITIALS:
        if base.startswith(candidate):
            initial = candidate
            break

    final = base[len(initial) :] if initial else base

    # In j/q/x spellings, written u represents the rounded front vowel ü.
    if initial in {"j", "q", "x"} and final.startswith("u"):
        final = "v" + final[1:]

    final = _FINAL_ALIASES.get(final, final)

    if final == "i" and initial in {"z", "c", "s"}:
        final = "iz"
    elif final == "i" and initial in {"zh", "ch", "sh", "r"}:
        final = "ir"

    # Interjections and syllabic nasals occur in conversational Mandarin.
    if not final and initial in {"m", "n"}:
        final, initial = initial, ""

    if final not in VALID_FINALS:
        # Keep the unknown final visible to callers while falling back later.
        final = final or "e"

    return ParsedSyllable(
        raw=pinyin,
        base=base,
        initial=initial,
        final=final,
        tone=tone,
    )

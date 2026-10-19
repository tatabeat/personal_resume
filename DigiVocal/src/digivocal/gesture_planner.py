from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.ndimage import gaussian_filter1d

from .models import GestureTimeline, PhonemeToken, SegmentSpan
from .voice_profile import VoiceProfile


@dataclass(frozen=True, slots=True)
class VowelTarget:
    f1: float
    f2: float
    f3: float
    b1: float
    b2: float
    b3: float
    jaw_open: float
    tongue_height: float
    tongue_front: float
    lip_round: float


# Editable starting targets. They are not a physiological standard for every
# Mandarin speaker. The purpose is to create a controllable first model.
VOWELS: dict[str, VowelTarget] = {
    "a": VowelTarget(850, 1220, 2810, 110, 140, 190, 0.92, 0.15, 0.48, 0.03),
    "o": VowelTarget(500, 900, 2600, 95, 125, 180, 0.58, 0.48, 0.27, 0.86),
    "e": VowelTarget(530, 1850, 2500, 90, 135, 180, 0.46, 0.58, 0.58, 0.03),
    "i": VowelTarget(300, 2400, 3100, 75, 145, 200, 0.20, 0.91, 0.91, 0.00),
    "u": VowelTarget(350, 800, 2200, 80, 115, 170, 0.25, 0.86, 0.16, 1.00),
    "v": VowelTarget(310, 2050, 2900, 80, 140, 190, 0.22, 0.90, 0.86, 0.82),
    "er": VowelTarget(470, 1370, 1750, 100, 150, 180, 0.44, 0.54, 0.48, 0.08),
    "iz": VowelTarget(360, 1650, 2650, 90, 150, 190, 0.24, 0.76, 0.77, 0.02),
    "ir": VowelTarget(380, 1400, 2100, 95, 155, 190, 0.28, 0.70, 0.62, 0.05),
}


# Digital Throat source-bank classes are intentionally coarser than Mandarin
# vowel acoustics.  These weights select the closest extracted source family;
# vocal-tract formants remain responsible for the actual vowel identity.
SOURCE_VOWEL_CLASSES = ("a", "e", "i", "o", "u")
SOURCE_VOWEL_MAP: dict[str, dict[str, float]] = {
    "a": {"a": 1.0},
    "e": {"e": 1.0},
    "i": {"i": 1.0},
    "o": {"o": 1.0},
    "u": {"u": 1.0},
    # /y/ is a high front rounded vowel: source behavior is represented as an
    # i/u blend while formants and lip rounding distinguish it acoustically.
    "v": {"i": 0.62, "u": 0.38},
    "er": {"e": 0.78, "a": 0.22},
    "iz": {"i": 0.62, "e": 0.38},
    "ir": {"e": 0.66, "i": 0.34},
}


def _source_vowel_weights(vowel_name: str) -> np.ndarray:
    mapping = SOURCE_VOWEL_MAP.get(vowel_name, SOURCE_VOWEL_MAP["e"])
    values = np.asarray([mapping.get(name, 0.0) for name in SOURCE_VOWEL_CLASSES], dtype=np.float64)
    total = float(np.sum(values))
    return values / total if total > 1e-12 else np.asarray([0.0, 1.0, 0.0, 0.0, 0.0])


@dataclass(frozen=True, slots=True)
class FinalPattern:
    anchors: tuple[tuple[float, str], ...]
    nasal: str | None = None
    nasal_fraction: float = 0.23


FINAL_PATTERNS: dict[str, FinalPattern] = {
    "a": FinalPattern(((0.0, "a"), (1.0, "a"))),
    "o": FinalPattern(((0.0, "o"), (1.0, "o"))),
    "e": FinalPattern(((0.0, "e"), (1.0, "e"))),
    "ai": FinalPattern(((0.0, "a"), (1.0, "i"))),
    "ei": FinalPattern(((0.0, "e"), (1.0, "i"))),
    "ao": FinalPattern(((0.0, "a"), (1.0, "o"))),
    "ou": FinalPattern(((0.0, "o"), (1.0, "u"))),
    "an": FinalPattern(((0.0, "a"), (1.0, "a")), nasal="n"),
    "en": FinalPattern(((0.0, "e"), (1.0, "e")), nasal="n"),
    "ang": FinalPattern(((0.0, "a"), (1.0, "a")), nasal="ng"),
    "eng": FinalPattern(((0.0, "e"), (1.0, "e")), nasal="ng"),
    "er": FinalPattern(((0.0, "er"), (1.0, "er"))),
    "i": FinalPattern(((0.0, "i"), (1.0, "i"))),
    "ia": FinalPattern(((0.0, "i"), (1.0, "a"))),
    "io": FinalPattern(((0.0, "i"), (1.0, "o"))),
    "ie": FinalPattern(((0.0, "i"), (1.0, "e"))),
    "iao": FinalPattern(((0.0, "i"), (0.52, "a"), (1.0, "o"))),
    "iou": FinalPattern(((0.0, "i"), (0.50, "o"), (1.0, "u"))),
    "ian": FinalPattern(((0.0, "i"), (1.0, "a")), nasal="n"),
    "in": FinalPattern(((0.0, "i"), (1.0, "i")), nasal="n"),
    "iang": FinalPattern(((0.0, "i"), (1.0, "a")), nasal="ng"),
    "ing": FinalPattern(((0.0, "i"), (1.0, "i")), nasal="ng"),
    "iong": FinalPattern(((0.0, "i"), (0.48, "o"), (1.0, "u")), nasal="ng"),
    "u": FinalPattern(((0.0, "u"), (1.0, "u"))),
    "ua": FinalPattern(((0.0, "u"), (1.0, "a"))),
    "uo": FinalPattern(((0.0, "u"), (1.0, "o"))),
    "uai": FinalPattern(((0.0, "u"), (0.52, "a"), (1.0, "i"))),
    "uei": FinalPattern(((0.0, "u"), (0.48, "e"), (1.0, "i"))),
    "uan": FinalPattern(((0.0, "u"), (1.0, "a")), nasal="n"),
    "uen": FinalPattern(((0.0, "u"), (1.0, "e")), nasal="n"),
    "uang": FinalPattern(((0.0, "u"), (1.0, "a")), nasal="ng"),
    "ueng": FinalPattern(((0.0, "u"), (1.0, "e")), nasal="ng"),
    "ong": FinalPattern(((0.0, "o"), (1.0, "u")), nasal="ng"),
    "v": FinalPattern(((0.0, "v"), (1.0, "v"))),
    "ve": FinalPattern(((0.0, "v"), (1.0, "e"))),
    "van": FinalPattern(((0.0, "v"), (1.0, "a")), nasal="n"),
    "vn": FinalPattern(((0.0, "v"), (1.0, "v")), nasal="n"),
    "iz": FinalPattern(((0.0, "iz"), (1.0, "iz"))),
    "ir": FinalPattern(((0.0, "ir"), (1.0, "ir"))),
    "m": FinalPattern(((0.0, "e"), (1.0, "e")), nasal="m", nasal_fraction=1.0),
    "n": FinalPattern(((0.0, "e"), (1.0, "e")), nasal="n", nasal_fraction=1.0),
    "ng": FinalPattern(((0.0, "e"), (1.0, "e")), nasal="ng", nasal_fraction=1.0),
}


@dataclass(frozen=True, slots=True)
class ConsonantSpec:
    manner: str
    place: str
    aspirated: bool = False
    voiced: bool = False
    duration_ratio: float = 0.26


CONSONANTS: dict[str, ConsonantSpec] = {
    "b": ConsonantSpec("stop", "labial", duration_ratio=0.24),
    "p": ConsonantSpec("stop", "labial", aspirated=True, duration_ratio=0.31),
    "m": ConsonantSpec("nasal", "labial", voiced=True, duration_ratio=0.27),
    "f": ConsonantSpec("fricative", "labial", duration_ratio=0.30),
    "d": ConsonantSpec("stop", "alveolar", duration_ratio=0.23),
    "t": ConsonantSpec("stop", "alveolar", aspirated=True, duration_ratio=0.31),
    "n": ConsonantSpec("nasal", "alveolar", voiced=True, duration_ratio=0.27),
    "l": ConsonantSpec("lateral", "alveolar", voiced=True, duration_ratio=0.24),
    "g": ConsonantSpec("stop", "velar", duration_ratio=0.24),
    "k": ConsonantSpec("stop", "velar", aspirated=True, duration_ratio=0.32),
    "h": ConsonantSpec("fricative", "glottal", duration_ratio=0.30),
    "j": ConsonantSpec("affricate", "palatal", duration_ratio=0.31),
    "q": ConsonantSpec("affricate", "palatal", aspirated=True, duration_ratio=0.36),
    "x": ConsonantSpec("fricative", "palatal", duration_ratio=0.32),
    "zh": ConsonantSpec("affricate", "postalveolar", duration_ratio=0.32),
    "ch": ConsonantSpec("affricate", "postalveolar", aspirated=True, duration_ratio=0.37),
    "sh": ConsonantSpec("fricative", "postalveolar", duration_ratio=0.33),
    "r": ConsonantSpec("approximant", "postalveolar", voiced=True, duration_ratio=0.25),
    "z": ConsonantSpec("affricate", "alveolar", duration_ratio=0.30),
    "c": ConsonantSpec("affricate", "alveolar", aspirated=True, duration_ratio=0.36),
    "s": ConsonantSpec("fricative", "alveolar", duration_ratio=0.32),
}


PLACE_TARGETS = {
    "labial": dict(tongue_height=0.52, tongue_front=0.50, lip_close=1.0, constriction=0.92),
    "alveolar": dict(tongue_height=0.82, tongue_front=0.88, lip_close=0.12, constriction=0.92),
    "postalveolar": dict(tongue_height=0.74, tongue_front=0.66, lip_close=0.10, constriction=0.88),
    "palatal": dict(tongue_height=0.90, tongue_front=0.91, lip_close=0.08, constriction=0.86),
    "velar": dict(tongue_height=0.82, tongue_front=0.24, lip_close=0.08, constriction=0.93),
    "glottal": dict(tongue_height=0.52, tongue_front=0.50, lip_close=0.05, constriction=0.20),
}

NOISE_BANDS = {
    # Place-dependent frication bands.  Keeping the alveolar band high and
    # narrow while lowering postalveolars prevents every consonant from
    # sounding like the same broadband white-noise switch.
    "labial": (4200.0, 3600.0),
    "alveolar": (7000.0, 2400.0),
    "postalveolar": (3400.0, 2100.0),
    "palatal": (5050.0, 2400.0),
    "velar": (2750.0, 2400.0),
    "glottal": (1800.0, 5000.0),
}


FRICATION_PEAKS = {
    "f": 0.56,
    "s": 0.82,
    "sh": 0.72,
    "x": 0.66,
    "h": 0.22,
    "z": 0.64,
    "c": 0.76,
    "zh": 0.58,
    "ch": 0.69,
    "j": 0.52,
    "q": 0.63,
}


@dataclass(slots=True)
class PlannerConfig:
    gesture_rate_hz: int = 200
    syllables_per_second: float = 3.7
    comma_pause_s: float = 0.17
    sentence_pause_s: float = 0.31
    question_pause_s: float = 0.27
    exclamation_pause_s: float = 0.22
    whitespace_pause_s: float = 0.045
    raw_symbol_pause_s: float = 0.07
    phrase_declination_semitones: float = 0.8
    smoothing_ms: float = 9.0

    def validate(self) -> None:
        if not 50 <= self.gesture_rate_hz <= 1000:
            raise ValueError("gesture_rate_hz must be between 50 and 1000")
        if not 1.0 <= self.syllables_per_second <= 12.0:
            raise ValueError("syllables_per_second must be between 1 and 12")


TRACK_DEFAULTS: dict[str, float] = {
    "f0_hz": 220.0,
    "amplitude": 0.0,
    "voicing": 0.0,
    "frication": 0.0,
    "aspiration": 0.0,
    "pressure": 0.0,
    "burst": 0.0,
    "f1_hz": VOWELS["e"].f1,
    "f2_hz": VOWELS["e"].f2,
    "f3_hz": VOWELS["e"].f3,
    "b1_hz": VOWELS["e"].b1,
    "b2_hz": VOWELS["e"].b2,
    "b3_hz": VOWELS["e"].b3,
    "jaw_open": VOWELS["e"].jaw_open,
    "tongue_height": VOWELS["e"].tongue_height,
    "tongue_front": VOWELS["e"].tongue_front,
    "lip_round": VOWELS["e"].lip_round,
    "lip_close": 0.0,
    "velum_open": 0.0,
    "constriction": 0.0,
    "noise_center_hz": 3500.0,
    "noise_bandwidth_hz": 3000.0,
    "vowel_a": 0.0,
    "vowel_e": 0.0,
    "vowel_i": 0.0,
    "vowel_o": 0.0,
    "vowel_u": 0.0,
}


def _pause_duration(text: str, config: PlannerConfig) -> float:
    if text in {"。", ".", "；", ";", "\n"}:
        return config.sentence_pause_s
    if text in {"？", "?"}:
        return config.question_pause_s
    if text in {"！", "!"}:
        return config.exclamation_pause_s
    if text in {"，", ",", "、", "：", ":", "…", "—", "-"}:
        return config.comma_pause_s
    if text.isspace():
        return config.whitespace_pause_s
    return config.raw_symbol_pause_s


def _syllable_duration(
    token: PhonemeToken,
    config: PlannerConfig,
    previous_token: PhonemeToken | None = None,
    next_token: PhonemeToken | None = None,
) -> float:
    duration = 1.0 / config.syllables_per_second
    if token.tone == 5:
        duration *= 0.74
    elif token.tone == 3:
        duration *= 1.08
    if token.final in {"iao", "iou", "uai", "uei", "iong"}:
        duration *= 1.07

    spec = CONSONANTS.get(token.initial)
    if spec is not None:
        if spec.manner == "affricate":
            duration *= 1.055
        elif spec.manner == "fricative":
            duration *= 1.025
        elif spec.manner == "stop" and not spec.aspirated:
            duration *= 0.97

    # Small phrase-boundary adjustments break the one-character/one-grid
    # rhythm while remaining fully deterministic and inspectable.
    if previous_token is None or previous_token.kind != "syllable":
        duration *= 1.025
    if next_token is None or next_token.kind != "syllable":
        duration *= 1.12
    return float(np.clip(duration, 0.11, 0.80))


def _tone_levels(tone: int) -> tuple[tuple[float, float], ...]:
    # Citation-like contours expressed with Chao-style relative levels.
    return {
        1: ((0.0, 5.0), (1.0, 5.0)),
        2: ((0.0, 2.0), (1.0, 5.0)),
        3: ((0.0, 3.0), (0.48, 1.0), (1.0, 3.2)),
        4: ((0.0, 5.0), (1.0, 1.0)),
        5: ((0.0, 3.1), (1.0, 2.7)),
    }.get(int(tone), ((0.0, 3.0), (1.0, 3.0)))


def _tone_f0(tone: int, frame_count: int, profile: VoiceProfile) -> np.ndarray:
    points = _tone_levels(tone)
    x = np.linspace(0.0, 1.0, max(frame_count, 1))
    positions = np.asarray([position for position, _ in points], dtype=np.float64)
    levels = np.asarray([level for _, level in points], dtype=np.float64)
    contour = np.interp(x, positions, levels)
    semitones = (contour - 3.0) / 4.0 * profile.f0_range_semitones
    return profile.base_f0_hz * np.power(2.0, semitones / 12.0)


def _final_pattern(final: str) -> FinalPattern:
    return FINAL_PATTERNS.get(final, FINAL_PATTERNS["e"])


def _vowel_tracks(pattern: FinalPattern, frame_count: int) -> dict[str, np.ndarray]:
    count = max(frame_count, 1)
    positions = np.asarray([anchor[0] for anchor in pattern.anchors], dtype=np.float64)
    targets = [VOWELS.get(anchor[1], VOWELS["e"]) for anchor in pattern.anchors]
    x = np.linspace(0.0, 1.0, count)

    output: dict[str, np.ndarray] = {}
    for field_name in (
        "f1",
        "f2",
        "f3",
        "b1",
        "b2",
        "b3",
        "jaw_open",
        "tongue_height",
        "tongue_front",
        "lip_round",
    ):
        values = np.asarray([getattr(target, field_name) for target in targets], dtype=np.float64)
        output[field_name] = np.interp(x, positions, values)

    source_anchors = np.vstack([_source_vowel_weights(anchor[1]) for anchor in pattern.anchors])
    for index, vowel_name in enumerate(SOURCE_VOWEL_CLASSES):
        output[f"vowel_{vowel_name}"] = np.interp(
            x,
            positions,
            source_anchors[:, index],
        )
    return output


def _empty_segment(frame_count: int, profile: VoiceProfile) -> dict[str, np.ndarray]:
    segment = {
        name: np.full(frame_count, value, dtype=np.float64)
        for name, value in TRACK_DEFAULTS.items()
    }
    segment["f0_hz"][:] = profile.base_f0_hz
    return segment


def _apply_envelope(values: np.ndarray, attack_fraction: float = 0.08, release_fraction: float = 0.10) -> np.ndarray:
    output = values.copy()
    count = len(output)
    attack = max(1, int(round(count * attack_fraction)))
    release = max(1, int(round(count * release_fraction)))
    output[:attack] *= np.sin(np.linspace(0.0, np.pi / 2.0, attack)) ** 2
    output[-release:] *= np.cos(np.linspace(0.0, np.pi / 2.0, release)) ** 2
    return output


def _plan_syllable(
    token: PhonemeToken,
    frame_count: int,
    profile: VoiceProfile,
) -> dict[str, np.ndarray]:
    segment = _empty_segment(frame_count, profile)
    spec = CONSONANTS.get(token.initial)
    initial_count = 0
    if spec is not None:
        initial_count = int(round(frame_count * spec.duration_ratio))
        initial_count = int(np.clip(initial_count, 1, max(1, frame_count - 2)))

    final_count = max(1, frame_count - initial_count)
    pattern = _final_pattern(token.final)
    vowel = _vowel_tracks(pattern, final_count)

    for short_name, track_name in (
        ("f1", "f1_hz"),
        ("f2", "f2_hz"),
        ("f3", "f3_hz"),
        ("b1", "b1_hz"),
        ("b2", "b2_hz"),
        ("b3", "b3_hz"),
        ("jaw_open", "jaw_open"),
        ("tongue_height", "tongue_height"),
        ("tongue_front", "tongue_front"),
        ("lip_round", "lip_round"),
    ):
        segment[track_name][initial_count:] = vowel[short_name]
        if initial_count:
            segment[track_name][:initial_count] = vowel[short_name][0]

    for vowel_name in SOURCE_VOWEL_CLASSES:
        track_name = f"vowel_{vowel_name}"
        segment[track_name][initial_count:] = vowel[track_name]
        if initial_count:
            segment[track_name][:initial_count] = vowel[track_name][0]

    f0 = _tone_f0(token.tone, final_count, profile)
    segment["f0_hz"][initial_count:] = f0
    if initial_count:
        segment["f0_hz"][:initial_count] = f0[0]

    segment["voicing"][initial_count:] = 1.0
    segment["amplitude"][initial_count:] = 0.90
    segment["pressure"][initial_count:] = 0.52

    if spec is not None and initial_count:
        place = PLACE_TARGETS[spec.place]
        noise_center, noise_width = NOISE_BANDS[spec.place]
        segment["noise_center_hz"][:initial_count] = noise_center
        segment["noise_bandwidth_hz"][:initial_count] = noise_width

        release_curve = np.linspace(0.0, 1.0, initial_count)
        hold = np.where(release_curve < 0.70, 0.0, (release_curve - 0.70) / 0.30)
        blend = np.clip(hold, 0.0, 1.0)

        for track_name in ("tongue_height", "tongue_front", "lip_close", "constriction"):
            target_value = place[track_name]
            vowel_value = float(segment[track_name][initial_count]) if initial_count < frame_count else target_value
            segment[track_name][:initial_count] = target_value * (1.0 - blend) + vowel_value * blend

        if spec.place == "labial":
            segment["lip_round"][:initial_count] *= 0.35

        if spec.manner == "stop":
            closure_end = max(1, int(round(initial_count * 0.74)))
            segment["amplitude"][:closure_end] = 0.015
            segment["voicing"][:closure_end] = 0.0
            segment["pressure"][:closure_end] = np.linspace(0.45, 0.95, closure_end)
            burst_index = min(frame_count - 1, closure_end)
            segment["burst"][burst_index] = 0.58 if spec.aspirated else 0.48
            release_length = initial_count - closure_end
            if release_length > 0:
                release = slice(closure_end, initial_count)
                release_shape = np.cos(np.linspace(0.0, np.pi / 2.0, release_length)) ** 2
                segment["frication"][release] = release_shape * (0.48 if spec.aspirated else 0.32)
                segment["aspiration"][release] = (
                    release_shape * 0.74 if spec.aspirated else release_shape * 0.16
                )
                segment["amplitude"][release] = np.linspace(0.12, 0.70, release_length)

        elif spec.manner == "affricate":
            closure_end = max(1, int(round(initial_count * 0.36)))
            segment["amplitude"][:closure_end] = 0.01
            segment["voicing"][:initial_count] = 0.0
            segment["pressure"][:closure_end] = np.linspace(0.50, 0.95, closure_end)
            segment["burst"][min(frame_count - 1, closure_end)] = 0.46
            fric_length = initial_count - closure_end
            if fric_length > 0:
                release = slice(closure_end, initial_count)
                fric_peak = FRICATION_PEAKS.get(token.initial, 0.62)
                release_shape = np.cos(np.linspace(0.0, np.pi / 2.0, fric_length)) ** 1.35
                segment["frication"][release] = 0.12 + release_shape * max(0.0, fric_peak - 0.12)
                segment["aspiration"][release] = (
                    release_shape * 0.62 if spec.aspirated else release_shape * 0.14
                )
                segment["amplitude"][release] = np.linspace(0.18, 0.58, fric_length)

        elif spec.manner == "fricative":
            segment["voicing"][:initial_count] = 0.0
            segment["amplitude"][:initial_count] = 0.16
            shape = np.sin(np.linspace(0.0, np.pi, initial_count)) ** 0.85
            segment["frication"][:initial_count] = shape * FRICATION_PEAKS.get(token.initial, 0.62)
            # /h/ is mainly a formant-shaped aspiration source rather than a
            # high-frequency sibilant.  Other fricatives retain only a trace.
            segment["aspiration"][:initial_count] = shape * (0.62 if spec.place == "glottal" else 0.018)
            segment["pressure"][:initial_count] = 0.58

        elif spec.manner == "nasal":
            segment["voicing"][:initial_count] = 0.88
            segment["amplitude"][:initial_count] = 0.58
            segment["velum_open"][:initial_count] = 1.0
            segment["pressure"][:initial_count] = 0.38

        elif spec.manner == "lateral":
            segment["voicing"][:initial_count] = 0.90
            segment["amplitude"][:initial_count] = 0.63
            segment["constriction"][:initial_count] *= 0.58
            segment["pressure"][:initial_count] = 0.40

        elif spec.manner == "approximant":
            segment["voicing"][:initial_count] = 0.88
            segment["amplitude"][:initial_count] = 0.62
            segment["frication"][:initial_count] = 0.10
            segment["constriction"][:initial_count] *= 0.55
            segment["pressure"][:initial_count] = 0.42

    # Add nasal codas after the vowel trajectory has been created.
    if pattern.nasal:
        coda_count = max(1, int(round(final_count * pattern.nasal_fraction)))
        coda_start = max(initial_count, frame_count - coda_count)
        ramp = np.linspace(0.0, 1.0, frame_count - coda_start)
        segment["velum_open"][coda_start:] = np.maximum(segment["velum_open"][coda_start:], ramp)
        segment["amplitude"][coda_start:] *= np.linspace(1.0, 0.66, frame_count - coda_start)
        segment["pressure"][coda_start:] *= 0.78
        if pattern.nasal == "m":
            segment["lip_close"][coda_start:] = np.maximum(segment["lip_close"][coda_start:], ramp)
            segment["constriction"][coda_start:] = np.maximum(segment["constriction"][coda_start:], 0.78 * ramp)
        elif pattern.nasal == "n":
            segment["tongue_front"][coda_start:] = 0.86
            segment["tongue_height"][coda_start:] = 0.80
            segment["constriction"][coda_start:] = np.maximum(segment["constriction"][coda_start:], 0.82 * ramp)
        else:  # ng
            segment["tongue_front"][coda_start:] = 0.22
            segment["tongue_height"][coda_start:] = 0.82
            segment["constriction"][coda_start:] = np.maximum(segment["constriction"][coda_start:], 0.82 * ramp)

    segment["amplitude"] = _apply_envelope(segment["amplitude"])
    return segment


def _append_segment(buffers: dict[str, list[np.ndarray]], segment: dict[str, np.ndarray]) -> None:
    for name in TRACK_DEFAULTS:
        buffers[name].append(segment[name])


def _smooth_tracks(tracks: dict[str, np.ndarray], sigma_frames: float) -> None:
    if sigma_frames <= 0.0:
        return
    continuous = {
        "f0_hz",
        "amplitude",
        "voicing",
        "frication",
        "aspiration",
        "pressure",
        "f1_hz",
        "f2_hz",
        "f3_hz",
        "b1_hz",
        "b2_hz",
        "b3_hz",
        "jaw_open",
        "tongue_height",
        "tongue_front",
        "lip_round",
        "lip_close",
        "velum_open",
        "constriction",
        "noise_center_hz",
        "noise_bandwidth_hz",
        "vowel_a",
        "vowel_e",
        "vowel_i",
        "vowel_o",
        "vowel_u",
    }
    multipliers = {
        "f0_hz": 2.2,
        "f1_hz": 2.1,
        "f2_hz": 2.1,
        "f3_hz": 2.1,
        "b1_hz": 1.5,
        "b2_hz": 1.5,
        "b3_hz": 1.5,
        "jaw_open": 1.8,
        "tongue_height": 1.8,
        "tongue_front": 1.8,
        "lip_round": 1.7,
        "lip_close": 1.0,
        "velum_open": 1.25,
        "constriction": 0.8,
        "amplitude": 0.78,
        "voicing": 0.72,
        "pressure": 0.78,
        "frication": 0.42,
        "aspiration": 0.42,
        "noise_center_hz": 1.15,
        "noise_bandwidth_hz": 1.15,
        "vowel_a": 1.45,
        "vowel_e": 1.45,
        "vowel_i": 1.45,
        "vowel_o": 1.45,
        "vowel_u": 1.45,
    }
    for name in continuous:
        track_sigma = max(0.35, sigma_frames * multipliers.get(name, 1.0))
        tracks[name] = gaussian_filter1d(tracks[name], sigma=track_sigma, mode="nearest")


def plan_gestures(
    tokens: Iterable[PhonemeToken],
    profile: VoiceProfile | None = None,
    config: PlannerConfig | None = None,
    *,
    metadata: dict | None = None,
) -> GestureTimeline:
    """Convert aligned Mandarin syllables into a controllable gesture timeline."""

    profile = profile or VoiceProfile()
    profile.validate()
    config = config or PlannerConfig()
    config.validate()

    token_list = list(tokens)
    buffers: dict[str, list[np.ndarray]] = {name: [] for name in TRACK_DEFAULTS}
    spans: list[SegmentSpan] = []
    current_frame = 0

    for token_index, token in enumerate(token_list):
        previous_token = token_list[token_index - 1] if token_index > 0 else None
        next_token = token_list[token_index + 1] if token_index + 1 < len(token_list) else None
        if token.kind == "syllable":
            duration = _syllable_duration(token, config, previous_token, next_token)
            frame_count = max(2, int(round(duration * config.gesture_rate_hz)))
            segment = _plan_syllable(token, frame_count, profile)
        else:
            duration = _pause_duration(token.text, config)
            frame_count = max(1, int(round(duration * config.gesture_rate_hz)))
            segment = _empty_segment(frame_count, profile)

        _append_segment(buffers, segment)
        start = current_frame / config.gesture_rate_hz
        current_frame += frame_count
        end = current_frame / config.gesture_rate_hz
        spans.append(
            SegmentSpan(
                text=token.text,
                pinyin=token.pinyin,
                initial=token.initial,
                final=token.final,
                tone=token.tone,
                kind=token.kind,
                start_s=start,
                end_s=end,
            )
        )

    if current_frame == 0:
        segment = _empty_segment(1, profile)
        _append_segment(buffers, segment)
        current_frame = 1

    tracks = {
        name: np.concatenate(parts) if parts else np.zeros(current_frame, dtype=np.float64)
        for name, parts in buffers.items()
    }

    sigma_frames = config.smoothing_ms / 1000.0 * config.gesture_rate_hz
    _smooth_tracks(tracks, sigma_frames)

    if tracks["f0_hz"].size and config.phrase_declination_semitones:
        declination = np.linspace(0.0, -config.phrase_declination_semitones, tracks["f0_hz"].size)
        tracks["f0_hz"] *= np.power(2.0, declination / 12.0)

    tracks["amplitude"] = np.clip(tracks["amplitude"], 0.0, 1.0)
    tracks["voicing"] = np.clip(tracks["voicing"], 0.0, 1.0)
    tracks["frication"] = np.clip(tracks["frication"], 0.0, 1.0)
    tracks["aspiration"] = np.clip(tracks["aspiration"], 0.0, 1.0)
    tracks["velum_open"] = np.clip(tracks["velum_open"], 0.0, 1.0)

    vowel_stack = np.vstack(
        [np.clip(tracks[f"vowel_{name}"], 0.0, 1.0) for name in SOURCE_VOWEL_CLASSES]
    )
    vowel_sum = np.sum(vowel_stack, axis=0)
    active = vowel_sum > 1e-9
    vowel_stack[:, active] /= vowel_sum[active][None, :]
    for index, name in enumerate(SOURCE_VOWEL_CLASSES):
        tracks[f"vowel_{name}"] = vowel_stack[index]

    timeline_metadata = {
        "voice_profile": profile.name,
        "base_f0_hz": profile.base_f0_hz,
        "f0_range_semitones": profile.f0_range_semitones,
        "syllables_per_second": config.syllables_per_second,
        "model": "digivocal-simplified-mandarin-gestures-v0.1",
        "vowel_source_tracks": list(SOURCE_VOWEL_CLASSES),
    }
    if metadata:
        timeline_metadata.update(metadata)

    return GestureTimeline(
        rate_hz=config.gesture_rate_hz,
        tracks=tracks,
        spans=spans,
        metadata=timeline_metadata,
    )

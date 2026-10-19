from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from scipy import signal

from .source_wave import (
    DEFAULT_BANK_NAMES,
    DEFAULT_BANK_WEIGHTS,
    DEFAULT_TABLE_POINTS,
    clean_inverse_filter_residual,
    generate_wave_bank,
    normalize_cycle,
    periodic_resample,
)


# Native 158-sample excitation period extracted by Digital Throat from the
# user-provided wave_analysis reference.  It is kept unmodified in every v2
# profile so future extraction/cleanup algorithms can be rerun without losing
# the original source estimate.
WAVE_ANALYSIS_NATIVE_CYCLE = (
    0.03829561567231915,
    0.09227393831718983,
    7.2482869176320789e-05,
    0.10053584716374449,
    0.12578359250403845,
    0.059827483752892416,
    0.046831113701479528,
    0.086766652248569429,
    0.026719202528395405,
    -0.029498672044035949,
    0.074608044503934975,
    0.067336513619502686,
    0.05127657390388022,
    0.079850160883526464,
    0.089890147762953468,
    0.041435569082606226,
    0.074678119828626405,
    0.05041573180137357,
    0.064885814507169445,
    -0.022072762821300416,
    0.0098886623329979603,
    -0.052697013267274329,
    0.035311161566538442,
    0.024031758653836076,
    -0.034321090698898271,
    -0.012272433794608061,
    0.011956896456851478,
    -0.050250307381448985,
    -0.070234481028888629,
    -0.063352163824249977,
    -0.0013691316818982696,
    -0.035787481933137105,
    0.032345303832790856,
    -0.01548685144735153,
    -0.031661614952936797,
    -0.011519902314511188,
    0.00095438861270774708,
    -0.056321470922105141,
    -0.093619119283829211,
    -0.033412508875822214,
    0.013877796203703011,
    -0.051325958705232308,
    -0.022002136315569878,
    -0.068365535518122911,
    -0.10500691139987145,
    -0.029104845152375856,
    -0.0087020107388499461,
    -0.065536030558306049,
    -0.084833317840656661,
    -0.10059553703137288,
    -0.028122233571415967,
    -0.061342725882486876,
    -0.033676484914400465,
    -0.1298601936127565,
    -0.12162650225013144,
    -0.046427213948097658,
    -0.073660166085726175,
    -0.019904101416649032,
    -0.024244707846332036,
    0.023785643212033903,
    -0.020065774595609258,
    -0.022343633625246891,
    -0.031389347657453458,
    -0.050426464651581224,
    -0.028945101487101312,
    -0.049377488251957591,
    -0.01090494933700421,
    -0.023424544341837068,
    -0.0061016022991566311,
    0.007199110998983611,
    0.017667648205848912,
    0.001071503223101865,
    0.020750588111507994,
    0.018315061416728673,
    -0.0028887949558368039,
    -0.0043693294358025892,
    0.015509135845338067,
    0.026559245867868352,
    0.044983539429280892,
    0.04312350619769529,
    0.020652237861433283,
    0.015420534564398816,
    -0.00050237707068518116,
    -0.0069015967298500401,
    -0.015063746244866408,
    0.036411416799816687,
    0.012116549787189805,
    0.026884163820570142,
    0.026755055183485404,
    0.045594145310573125,
    0.063715850730758819,
    0.09260196500430426,
    0.073820031531854596,
    0.060736801193963595,
    0.0294185213953779,
    0.0060382097430159171,
    0.0062871430816701327,
    -0.00019464433285179574,
    0.024290779254822859,
    -0.010625056306430986,
    -0.014779794228796867,
    -0.0061182574866294488,
    0.057789989700505237,
    0.083270521529269101,
    0.043878096313415722,
    0.018198399032042376,
    0.036544352436167687,
    0.065713736175308179,
    0.12962102645206708,
    0.057669051577640858,
    0.097497229836714633,
    0.11080290005250985,
    0.046784602794607666,
    0.075436937697909698,
    0.11297982280730665,
    0.05015393672015319,
    -0.0039981199003738307,
    0.049173378958050448,
    0.050681755766406308,
    0.079452386237779601,
    0.12717535330645882,
    0.17897881790996814,
    0.050426012793284576,
    0.0075492802774818979,
    0.00080170317769723883,
    -0.092949555043676357,
    -0.056271253070780458,
    -0.11911320100326471,
    -0.028597560938451443,
    -0.23442913483154798,
    -1,
    -0.13907158499132979,
    0.23200572951868545,
    -0.098927358096875348,
    0.16674463280687865,
    0.054946971198553421,
    -0.050702633869965333,
    -0.0066568898951628354,
    0.088216068515110102,
    -0.0075156577603541852,
    -0.07947576765369975,
    -0.033504521130434631,
    0.062169901721118623,
    -0.056835669388630862,
    -0.038312283356151271,
    -0.055230214628663198,
    -0.038473502341159427,
    -0.055351780512162004,
    0.091056580873083506,
    0.008641526987076124,
    -0.077201202734491978,
    0.023300705504842176,
    -0.043439070176814601,
    -0.096005822301957228,
    0.068183569515028977,
    0.11767999146765552,
    -0.0045539859890878403,
    -0.09382903802882174
)

WAVE_ANALYSIS_REFERENCE: dict[str, float | str] = {
    "name": "wave_analysis_clone_bright",
    "base_f0_hz": 304.132728,
    "f0_range_semitones": 9.8,
    "source_native_points": 158.0,
    "source_extraction_tilt_db_per_octave": -1.651,
    "source_hnr_db": 13.493,
    "spectral_tilt_bright_db_per_octave": -5.5,
    "spectral_tilt_balanced_db_per_octave": -7.0,
}


def _source_metadata_default() -> dict[str, Any]:
    return {
        "profile_schema": "digivocal-voice-profile-v2",
        "reference": "wave_analysis(3)",
        "native_cycle_preserved": True,
        "wave_points_are_cleaned": True,
        "source_extraction_tilt_db_per_octave": -1.651,
        "note": (
            "Wave bank variants are derived from the preserved 158-sample native cycle; "
            "the archive did not include all 101 accepted raw pitch cycles."
        ),
    }


def _wave_analysis_primary(point_count: int = DEFAULT_TABLE_POINTS) -> list[float]:
    high_resolution = periodic_resample(WAVE_ANALYSIS_NATIVE_CYCLE, point_count)
    cleaned, _metadata = clean_inverse_filter_residual(
        high_resolution,
        target_tilt_db_per_octave=-1.651,
        strength=0.82,
    )
    return [float(value) for value in cleaned]


def waveform_preset(name: str, point_count: int = DEFAULT_TABLE_POINTS) -> list[float]:
    """Return a high-resolution, periodic source table."""

    phase = np.linspace(0.0, 1.0, int(point_count), endpoint=False)
    key = name.strip().lower()
    if key in {"sine", "正弦"}:
        values = np.sin(2.0 * np.pi * phase)
    elif key in {"triangle", "三角"}:
        values = signal.sawtooth(2.0 * np.pi * phase, width=0.5)
    elif key in {"saw", "sawtooth", "鋸齒"}:
        values = signal.sawtooth(2.0 * np.pi * phase, width=0.72)
    elif key in {"square", "方波"}:
        values = signal.square(2.0 * np.pi * phase)
    elif key in {"pressed", "緊聲"}:
        values = 0.82 * signal.sawtooth(2.0 * np.pi * phase, width=0.38)
        values += 0.18 * np.sin(4.0 * np.pi * phase)
    elif key in {"breathy", "氣聲"}:
        values = 0.55 * signal.sawtooth(2.0 * np.pi * phase, width=0.82)
        values += 0.45 * np.sin(2.0 * np.pi * phase)
    elif key in {
        "captured",
        "captured_reference",
        "wave_analysis",
        "wave_analysis_bright",
        "截取聲線",
        "理想聲線",
    }:
        return _wave_analysis_primary(int(point_count))
    elif key in {"rosenberg", "rosenberg_glottal", "soft_glottal", "柔和聲門", "default"}:
        open_end = 0.56
        close_end = 0.84
        flow = np.zeros_like(phase)
        opening = phase < open_end
        closing = (phase >= open_end) & (phase < close_end)
        flow[opening] = 0.5 * (1.0 - np.cos(np.pi * phase[opening] / open_end))
        flow[closing] = np.cos(
            np.pi * (phase[closing] - open_end) / (2.0 * (close_end - open_end))
        )
        values = np.gradient(flow)
        window = 9 if point_count >= 256 else 7
        values = signal.savgol_filter(values, window, 2, mode="wrap")
        values += 0.06 * np.sin(2.0 * np.pi * phase)
    else:
        raise ValueError(f"Unknown waveform preset: {name}")
    return [float(value) for value in normalize_cycle(values)]


def captured_reference_settings() -> dict[str, float | str]:
    return dict(WAVE_ANALYSIS_REFERENCE)


@dataclass
class VoiceProfile:
    """Speaker source, dynamic wave bank, and independent vocal-tract controls."""

    profile_version: int = 2
    name: str = "wave_analysis_clone_bright"
    sample_rate: int = 48_000
    base_f0_hz: float = 304.132728
    f0_range_semitones: float = 9.8

    # Overall scale remains for backward compatibility.  The three per-formant
    # controls are applied in addition to it.
    formant_scale: float = 1.0
    f1_scale: float = 1.06
    f2_scale: float = 1.16
    f3_scale: float = 1.24
    f1_gain: float = 0.55
    f2_gain: float = 1.45
    f3_gain: float = 2.20
    f1_bandwidth_scale: float = 0.85
    f2_bandwidth_scale: float = 0.82
    f3_bandwidth_scale: float = 0.88

    output_gain: float = 0.55
    breathiness: float = 0.106337
    jitter_percent: float = 1.20
    shimmer_percent: float = 4.00
    spectral_tilt_db_per_octave: float = -5.5

    source_table_points: int = DEFAULT_TABLE_POINTS
    native_cycle: list[float] = field(default_factory=lambda: list(WAVE_ANALYSIS_NATIVE_CYCLE))
    wave_points: list[float] = field(default_factory=_wave_analysis_primary)
    wave_bank_names: list[str] = field(default_factory=lambda: list(DEFAULT_BANK_NAMES))
    wave_bank: list[list[float]] = field(default_factory=list)
    wave_bank_base_weights: list[float] = field(default_factory=lambda: list(DEFAULT_BANK_WEIGHTS))
    cycle_morph_percent: float = 26.0
    source_residual_cleanup_strength: float = 0.82
    source_target_tilt_db_per_octave: float = -1.651

    # Digital Throat profiles already contain an extracted, phase-aligned source
    # bank.  Preservation mode prevents DigiVocal from independently rotating,
    # re-cleaning, re-tilting or indiscriminately mixing its vowel tables.
    source_preservation: bool = False
    vowel_source_mix: float = 0.0
    source_direct_mix: float = 0.0
    source_metadata: dict[str, Any] = field(default_factory=_source_metadata_default)

    def validate(self) -> None:
        if not 8_000 <= int(self.sample_rate) <= 192_000:
            raise ValueError("sample_rate must be between 8000 and 192000")
        if not 35.0 <= float(self.base_f0_hz) <= 1200.0:
            raise ValueError("base_f0_hz must be between 35 and 1200")
        if not 0.0 <= float(self.f0_range_semitones) <= 36.0:
            raise ValueError("f0_range_semitones must be between 0 and 36")
        if not 0.55 <= float(self.formant_scale) <= 1.8:
            raise ValueError("formant_scale must be between 0.55 and 1.8")
        for label, value in (
            ("f1_scale", self.f1_scale),
            ("f2_scale", self.f2_scale),
            ("f3_scale", self.f3_scale),
        ):
            if not 0.60 <= float(value) <= 1.80:
                raise ValueError(f"{label} must be between 0.60 and 1.80")
        for label, value in (
            ("f1_gain", self.f1_gain),
            ("f2_gain", self.f2_gain),
            ("f3_gain", self.f3_gain),
        ):
            if not 0.10 <= float(value) <= 2.50:
                raise ValueError(f"{label} must be between 0.10 and 2.50")
        for label, value in (
            ("f1_bandwidth_scale", self.f1_bandwidth_scale),
            ("f2_bandwidth_scale", self.f2_bandwidth_scale),
            ("f3_bandwidth_scale", self.f3_bandwidth_scale),
        ):
            if not 0.45 <= float(value) <= 2.20:
                raise ValueError(f"{label} must be between 0.45 and 2.20")
        if not 0.01 <= float(self.output_gain) <= 0.95:
            raise ValueError("output_gain must be between 0.01 and 0.95")
        if not 0.0 <= float(self.breathiness) <= 1.0:
            raise ValueError("breathiness must be between 0 and 1")
        if not 0.0 <= float(self.jitter_percent) <= 5.0:
            raise ValueError("jitter_percent must be between 0 and 5")
        if not 0.0 <= float(self.shimmer_percent) <= 20.0:
            raise ValueError("shimmer_percent must be between 0 and 20")
        if not -24.0 <= float(self.spectral_tilt_db_per_octave) <= 6.0:
            raise ValueError("spectral_tilt_db_per_octave must be between -24 and 6")
        if not 128 <= int(self.source_table_points) <= 2048:
            raise ValueError("source_table_points must be between 128 and 2048")
        if not 0.0 <= float(self.cycle_morph_percent) <= 75.0:
            raise ValueError("cycle_morph_percent must be between 0 and 75")
        if not 0.0 <= float(self.source_residual_cleanup_strength) <= 1.0:
            raise ValueError("source_residual_cleanup_strength must be between 0 and 1")
        if not 0.0 <= float(self.vowel_source_mix) <= 0.80:
            raise ValueError("vowel_source_mix must be between 0 and 0.80")
        if not 0.0 <= float(self.source_direct_mix) <= 0.90:
            raise ValueError("source_direct_mix must be between 0 and 0.90")

        metadata = dict(self.source_metadata or {})
        input_schema = str(
            metadata.get("origin_profile_schema")
            or metadata.get("profile_schema")
            or ""
        )
        # ``from_dict`` enables preservation automatically for Digital Throat
        # producer schemas.  Once loaded, the field itself is authoritative so
        # the GUI/CLI can deliberately compare preserved and legacy rendering.
        digital_throat = bool(self.source_preservation)
        self.source_preservation = digital_throat
        if digital_throat:
            # Source extraction already happened in Digital Throat.  Keep its
            # stored harmonic tilt and phase; renderer controls may still be
            # explicitly adjusted by the user afterwards.
            self.source_residual_cleanup_strength = 0.0

        original_native = self.native_cycle if len(self.native_cycle) >= 8 else self.wave_points
        self.native_cycle = [float(value) for value in normalize_cycle(original_native)]

        table = periodic_resample(
            self.wave_points or self.native_cycle,
            int(self.source_table_points),
            preserve_phase_origin=digital_throat,
        )
        requested_bank_weights = list(self.wave_bank_base_weights)
        bank_is_valid = (
            bool(self.wave_bank)
            and len(self.wave_bank) == len(self.wave_bank_names)
            and all(len(table_values) >= 8 for table_values in self.wave_bank)
        )

        if not bank_is_valid:
            if (
                not digital_throat
                and not bool(metadata.get("wave_points_are_cleaned", False))
            ):
                table, cleanup = clean_inverse_filter_residual(
                    table,
                    target_tilt_db_per_octave=float(self.source_target_tilt_db_per_octave),
                    strength=float(self.source_residual_cleanup_strength),
                )
                metadata["residual_cleanup"] = cleanup
            names, bank, default_weights = generate_wave_bank(table)
            self.wave_bank_names = names
            self.wave_bank = bank
            requested = np.asarray(requested_bank_weights, dtype=np.float64)
            if (
                requested.size == len(bank)
                and np.all(np.isfinite(requested))
                and float(np.sum(np.clip(requested, 0.0, None))) > 1e-12
            ):
                self.wave_bank_base_weights = [float(value) for value in requested]
            else:
                self.wave_bank_base_weights = default_weights
            metadata["wave_points_are_cleaned"] = True
        else:
            normalized_bank: list[list[float]] = []
            for bank_table in self.wave_bank:
                normalized_bank.append(
                    [
                        float(value)
                        for value in periodic_resample(
                            bank_table,
                            int(self.source_table_points),
                            preserve_phase_origin=digital_throat,
                        )
                    ]
                )
            self.wave_bank = normalized_bank
            try:
                primary_index = self.wave_bank_names.index("native_clean")
            except ValueError:
                primary_index = 0
            table = np.asarray(self.wave_bank[primary_index], dtype=np.float64)

        if len(self.wave_bank_names) != len(self.wave_bank):
            self.wave_bank_names = [f"table_{index}" for index in range(len(self.wave_bank))]
        weights = np.asarray(self.wave_bank_base_weights, dtype=np.float64)
        if weights.size != len(self.wave_bank) or float(np.sum(weights)) <= 1e-12:
            weights = np.ones(len(self.wave_bank), dtype=np.float64)
        weights = np.clip(weights, 0.0, None)
        weights = weights / np.sum(weights)
        self.wave_bank_base_weights = [float(value) for value in weights]
        self.wave_points = [float(value) for value in table]

        if input_schema:
            metadata.setdefault("origin_profile_schema", input_schema)
        if not digital_throat:
            metadata["profile_schema"] = "digivocal-voice-profile-v2"
        else:
            # Keep the producer schema visible.  Older code overwrote this key,
            # making it impossible for the renderer to know the source had
            # already been extracted and validated by Digital Throat.
            metadata["profile_schema"] = input_schema or "digital-throat-source-bank-v2"
        metadata.update(
            {
                "renderer_schema": "digivocal-source-preserving-renderer-v3",
                "native_cycle_preserved": True,
                "native_cycle_points": len(self.native_cycle),
                "source_table_points": int(self.source_table_points),
                "wave_bank_tables": len(self.wave_bank),
                "wave_bank_names": list(self.wave_bank_names),
                "wave_points_are_cleaned": bool(
                    metadata.get("wave_points_are_cleaned", digital_throat)
                ),
                "source_preservation": bool(self.source_preservation),
                "vowel_source_mix": float(self.vowel_source_mix),
                "source_direct_mix": float(self.source_direct_mix),
            }
        )
        self.source_metadata = metadata

    @property
    def is_digital_throat_profile(self) -> bool:
        metadata = dict(self.source_metadata or {})
        schema = str(
            metadata.get("origin_profile_schema")
            or metadata.get("profile_schema")
            or ""
        )
        return bool(self.source_preservation)

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "VoiceProfile":
        payload = dict(data)
        legacy_wave = payload.get("wave_points")
        if "native_cycle" not in payload and isinstance(legacy_wave, list):
            payload["native_cycle"] = list(legacy_wave)
        payload.setdefault("source_table_points", DEFAULT_TABLE_POINTS)
        metadata = dict(payload.get("source_metadata") or {})
        if "profile_version" not in payload:
            metadata["imported_legacy_profile"] = True
            # Old JSON files had one table and one global formant scale. Preserve
            # their original timbre unless the user explicitly selects the new
            # wave_analysis clone profile.
            metadata.setdefault("wave_points_are_cleaned", True)
            payload.setdefault("f1_scale", 1.0)
            payload.setdefault("f2_scale", 1.0)
            payload.setdefault("f3_scale", 1.0)
            payload.setdefault("f1_gain", 1.0)
            payload.setdefault("f2_gain", 1.0)
            payload.setdefault("f3_gain", 1.0)
            payload.setdefault("f1_bandwidth_scale", 1.0)
            payload.setdefault("f2_bandwidth_scale", 1.0)
            payload.setdefault("f3_bandwidth_scale", 1.0)
            payload.setdefault("cycle_morph_percent", 10.0)
            payload.setdefault("source_residual_cleanup_strength", 0.0)
        producer_schema = str(metadata.get("profile_schema") or "")
        if producer_schema.startswith("digital-throat-"):
            metadata.setdefault("origin_profile_schema", producer_schema)
            payload.setdefault("source_preservation", True)
            payload.setdefault("vowel_source_mix", 0.30)
            payload.setdefault("source_direct_mix", 0.34)
            payload["source_residual_cleanup_strength"] = 0.0
            # Digital Throat stores the measured source tilt directly in the
            # waveform.  Keep the exported neutral value instead of silently
            # applying DigiVocal's old -5.5/-9 dB octave coloration.
            payload.setdefault("spectral_tilt_db_per_octave", 0.0)
        else:
            payload.setdefault("source_preservation", False)
            payload.setdefault("vowel_source_mix", 0.0)
            payload.setdefault("source_direct_mix", 0.0)
        payload["source_metadata"] = metadata

        allowed = {item.name for item in fields(cls)}
        unknown = {key: value for key, value in payload.items() if key not in allowed}
        filtered = {key: value for key, value in payload.items() if key in allowed}
        if unknown:
            filtered.setdefault("source_metadata", {})
            filtered["source_metadata"] = dict(filtered["source_metadata"])
            filtered["source_metadata"]["ignored_unknown_fields"] = sorted(unknown)
        profile = cls(**filtered)
        profile.validate()
        return profile

    @classmethod
    def load(cls, path: str | Path) -> "VoiceProfile":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8-sig")))

    @classmethod
    def wave_analysis_clone(cls, spectral_tilt_db_per_octave: float = -5.5) -> "VoiceProfile":
        native = list(WAVE_ANALYSIS_NATIVE_CYCLE)
        high_resolution = periodic_resample(native, DEFAULT_TABLE_POINTS)
        cleaned, cleanup = clean_inverse_filter_residual(
            high_resolution,
            target_tilt_db_per_octave=-1.651,
            strength=0.82,
        )
        metadata = _source_metadata_default()
        metadata["residual_cleanup"] = cleanup
        profile = cls(
            spectral_tilt_db_per_octave=float(spectral_tilt_db_per_octave),
            native_cycle=native,
            wave_points=[float(value) for value in cleaned],
            wave_bank=[],
            source_metadata=metadata,
        )
        if spectral_tilt_db_per_octave <= -6.2:
            profile.name = "wave_analysis_clone_balanced"
            profile.f1_scale = 1.03
            profile.f2_scale = 1.13
            profile.f3_scale = 1.20
            profile.f1_gain = 0.65
            profile.f2_gain = 1.35
            profile.f3_gain = 2.00
            profile.f1_bandwidth_scale = 0.90
            profile.f2_bandwidth_scale = 0.86
            profile.f3_bandwidth_scale = 0.90
            profile.cycle_morph_percent = 21.0
            profile.wave_bank_base_weights = [
                0.24, 0.015, 0.15, 0.10, 0.025, 0.035, 0.095, 0.33
            ]
        else:
            profile.name = "wave_analysis_clone_bright"
            profile.f1_scale = 1.06
            profile.f2_scale = 1.16
            profile.f3_scale = 1.24
            profile.f1_gain = 0.55
            profile.f2_gain = 1.45
            profile.f3_gain = 2.20
            profile.f1_bandwidth_scale = 0.85
            profile.f2_bandwidth_scale = 0.82
            profile.f3_bandwidth_scale = 0.88
            profile.cycle_morph_percent = 26.0
            profile.wave_bank_base_weights = list(DEFAULT_BANK_WEIGHTS)
        profile.validate()
        return profile

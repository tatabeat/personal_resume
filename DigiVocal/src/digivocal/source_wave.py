from __future__ import annotations

"""High-resolution periodic source-cycle processing and wavetable morphing.

The functions in this module are deliberately deterministic and inspectable.
They preserve the native extracted cycle, create a 512-sample periodic table by
Fourier resampling, suppress broad residual vocal-tract coloration, and build a
small bank of related glottal-source tables.  The renderer continuously morphs
between those tables instead of repeating one mathematically identical cycle.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np
from scipy import signal
from scipy.interpolate import CubicSpline
from scipy.ndimage import gaussian_filter1d


DEFAULT_TABLE_POINTS = 512
DEFAULT_BANK_NAMES = (
    "native_clean",
    "soft",
    "bright",
    "pressed",
    "breathy_open",
    "low_f0",
    "high_f0",
    "thin_young",
)
DEFAULT_BANK_WEIGHTS = (0.18, 0.00, 0.16, 0.11, 0.015, 0.025, 0.10, 0.41)


def normalize_cycle(values: Iterable[float]) -> np.ndarray:
    """Return a finite, zero-mean, peak-normalized one-dimensional cycle."""

    cycle = np.asarray(list(values), dtype=np.float64).reshape(-1)
    if cycle.size < 8:
        raise ValueError("A source cycle needs at least 8 samples")
    if not np.all(np.isfinite(cycle)):
        raise ValueError("Source cycle contains NaN or Infinity")
    cycle = cycle - float(np.mean(cycle))
    peak = float(np.max(np.abs(cycle)))
    if peak < 1e-12:
        phase = np.linspace(0.0, 2.0 * np.pi, cycle.size, endpoint=False)
        cycle = np.sin(phase)
        peak = 1.0
    return cycle / peak


def rotate_to_smooth_seam(values: Iterable[float]) -> tuple[np.ndarray, int, float]:
    """Rotate a periodic cycle so the wrap-around seam is minimally discontinuous.

    Rotation does not alter the spectrum.  It only moves phase zero to a region
    whose amplitude and slope agree well across the cycle boundary, which makes
    periodic FFT resampling and cubic table lookup substantially cleaner.
    """

    cycle = normalize_cycle(values)
    derivative = np.roll(cycle, -1) - cycle
    jump = np.abs(cycle - np.roll(cycle, 1))
    slope_mismatch = np.abs(derivative - np.roll(derivative, 1))
    local_amplitude = 0.5 * (np.abs(cycle) + np.abs(np.roll(cycle, 1)))
    cost = jump + 0.22 * slope_mismatch + 0.025 * local_amplitude
    seam = int(np.argmin(cost))
    return np.roll(cycle, -seam), seam, float(cost[seam])


def periodic_resample(
    values: Iterable[float],
    point_count: int = DEFAULT_TABLE_POINTS,
    *,
    preserve_phase_origin: bool = False,
) -> np.ndarray:
    """Resample one complete periodic cycle with a Fourier-domain method.

    ``preserve_phase_origin`` is important for Digital Throat source banks.
    Their tables have already been pitch-synchronously aligned, so rotating
    every table to an independently chosen seam would destroy the relative
    harmonic phase needed for clean interpolation between vowel/F0 variants.
    Legacy hand-drawn tables still use seam rotation by default.
    """

    point_count = int(point_count)
    if point_count < 32 or point_count > 4096:
        raise ValueError("point_count must be between 32 and 4096")
    if preserve_phase_origin:
        cycle = normalize_cycle(values)
    else:
        # Preserve the exact legacy path: one normalization inside seam search.
        # A second normalization can perturb tied seam costs and circularly shift
        # old profiles even though their spectrum is unchanged.
        cycle, _seam, _cost = rotate_to_smooth_seam(values)
    result = signal.resample(cycle, point_count, window=None)
    result = np.real_if_close(result).astype(np.float64, copy=False)
    return normalize_cycle(result)


def _harmonic_db(cycle: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    spectrum = np.fft.rfft(normalize_cycle(cycle))
    magnitude = np.abs(spectrum)
    db = 20.0 * np.log10(np.maximum(magnitude, 1e-12))
    harmonics = np.arange(db.size, dtype=np.float64)
    return spectrum, db, harmonics


def estimate_harmonic_tilt(values: Iterable[float], max_harmonic: int = 48) -> float:
    """Estimate the dB/octave slope of the stored source-cycle harmonics."""

    _spectrum, db, harmonic = _harmonic_db(np.asarray(list(values), dtype=np.float64))
    stop = min(db.size, max(4, int(max_harmonic) + 1))
    indices = np.arange(1, stop)
    if indices.size < 3:
        return 0.0
    slope, _intercept = np.polyfit(np.log2(harmonic[indices]), db[indices], 1)
    return float(slope)


def clean_inverse_filter_residual(
    values: Iterable[float],
    *,
    target_tilt_db_per_octave: float = -1.651,
    strength: float = 0.82,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Suppress residual vocal-tract peaks while preserving source phase.

    A one-period inverse-filter residual should mainly describe excitation.  In
    practice, running-speech LPC residuals can retain broad harmonic bulges from
    the vocal tract.  This routine compares the harmonic magnitudes with a smooth
    source tilt, keeps a bounded amount of speaker-specific fine structure, and
    reconstructs the cycle with its original phases.  The untouched native cycle
    remains available in the profile for later reprocessing.
    """

    source = normalize_cycle(values)
    strength = float(np.clip(strength, 0.0, 1.0))
    spectrum, observed_db, harmonic = _harmonic_db(source)
    phase = np.angle(spectrum)
    indices = np.arange(1, observed_db.size)
    if indices.size < 3 or strength <= 0.0:
        return source, {
            "enabled": False,
            "strength": strength,
            "target_tilt_db_per_octave": float(target_tilt_db_per_octave),
            "leakage_score_before_db": 0.0,
            "leakage_score_after_db": 0.0,
            "harmonics_adjusted": 0,
        }

    offsets = observed_db[indices] - float(target_tilt_db_per_octave) * np.log2(indices)
    intercept = float(np.median(offsets[: min(12, offsets.size)]))
    target_db = np.full_like(observed_db, -240.0)
    target_db[1:] = intercept + float(target_tilt_db_per_octave) * np.log2(indices)
    residual_db = observed_db - target_db

    # Retain the fundamental exactly. H2/H3 retain more identity than the upper
    # harmonics, while narrow upper-harmonic bulges are suppressed aggressively.
    general_preservation = max(0.10, 1.0 - 0.90 * strength)
    preservation = np.full_like(observed_db, general_preservation)
    preservation[1] = 1.0
    if preservation.size > 2:
        preservation[2] = max(preservation[2], 0.42)
    if preservation.size > 3:
        preservation[3] = max(preservation[3], 0.34)

    bounded_identity = np.clip(residual_db, -7.0, 6.0) * preservation
    cleaned_db = target_db + bounded_identity
    output_db = (1.0 - strength) * observed_db + strength * cleaned_db
    output_magnitude = np.power(10.0, output_db / 20.0)
    output_magnitude[0] = 0.0
    reconstructed = np.fft.irfft(output_magnitude * np.exp(1j * phase), n=source.size)
    reconstructed = normalize_cycle(reconstructed)

    # Running-speech inverse filtering often leaves one unrealistically narrow
    # closure spike.  A real glottal derivative is impulsive, but a one-sample
    # outlier produces the exact rubber-stamp/mechanical quality we are trying to
    # remove.  Softly compress the cycle itself (the untouched native cycle is
    # still stored separately) until its crest is in a useful source range.
    crest_before = float(
        np.max(np.abs(reconstructed))
        / (np.sqrt(np.mean(np.square(reconstructed)) + 1e-30))
    )
    crest_drive = 1.0 + 3.2 * strength
    if crest_before > 4.8 and strength > 0.0:
        reconstructed = np.tanh(reconstructed * crest_drive) / np.tanh(crest_drive)
        reconstructed = normalize_cycle(reconstructed)
    crest_after = float(
        np.max(np.abs(reconstructed))
        / (np.sqrt(np.mean(np.square(reconstructed)) + 1e-30))
    )

    before_positive = np.maximum(residual_db[2:], 0.0)
    after_positive = np.maximum((output_db - target_db)[2:], 0.0)
    threshold = 3.0
    adjusted = int(np.count_nonzero(np.abs(output_db[2:] - observed_db[2:]) > 0.5))
    metadata = {
        "enabled": True,
        "strength": strength,
        "target_tilt_db_per_octave": float(target_tilt_db_per_octave),
        "estimated_tilt_before_db_per_octave": estimate_harmonic_tilt(source),
        "estimated_tilt_after_db_per_octave": estimate_harmonic_tilt(reconstructed),
        "leakage_score_before_db": float(np.percentile(before_positive, 95))
        if before_positive.size
        else 0.0,
        "leakage_score_after_db": float(np.percentile(after_positive, 95))
        if after_positive.size
        else 0.0,
        "harmonics_above_3db_before": int(np.count_nonzero(before_positive > threshold)),
        "harmonics_above_3db_after": int(np.count_nonzero(after_positive > threshold)),
        "harmonics_adjusted": adjusted,
        "cycle_crest_before": crest_before,
        "cycle_crest_after": crest_after,
        "cycle_crest_softening_drive": crest_drive if crest_before > 4.8 else 0.0,
    }
    return reconstructed, metadata


def _apply_harmonic_shape(
    values: np.ndarray,
    *,
    delta_tilt_db_per_octave: float = 0.0,
    h1_db: float = 0.0,
    h2_db: float = 0.0,
    upper_taper_db: float = 0.0,
) -> np.ndarray:
    cycle = normalize_cycle(values)
    spectrum = np.fft.rfft(cycle)
    harmonic = np.arange(spectrum.size, dtype=np.float64)
    gain_db = np.zeros_like(harmonic)
    if harmonic.size > 1:
        gain_db[1:] = float(delta_tilt_db_per_octave) * np.log2(
            np.maximum(harmonic[1:], 1.0)
        )
        gain_db[1] += float(h1_db)
    if harmonic.size > 2:
        gain_db[2] += float(h2_db)
    if upper_taper_db > 0.0 and harmonic.size > 3:
        normalized = harmonic / max(float(harmonic[-1]), 1.0)
        gain_db -= float(upper_taper_db) * np.square(normalized)
    shaped = np.fft.irfft(spectrum * np.power(10.0, gain_db / 20.0), n=cycle.size)
    return normalize_cycle(shaped)


def _phase_warp(values: np.ndarray, amount: float, phase_offset: float) -> np.ndarray:
    cycle = normalize_cycle(values)
    phase = np.arange(cycle.size, dtype=np.float64) / cycle.size
    spline = CubicSpline(
        np.concatenate([phase, [1.0]]),
        np.concatenate([cycle, [cycle[0]]]),
        bc_type="periodic",
    )
    mapped = (
        phase
        + float(amount) * np.sin(2.0 * np.pi * (phase - float(phase_offset))) / (2.0 * np.pi)
    ) % 1.0
    return normalize_cycle(spline(mapped))


def _closure_emphasis(values: np.ndarray, amount: float, width_fraction: float = 0.018) -> np.ndarray:
    cycle = normalize_cycle(values)
    sample_count = cycle.size
    closure = int(np.argmin(cycle))
    indices = np.arange(sample_count)
    distance = np.minimum(np.abs(indices - closure), sample_count - np.abs(indices - closure))
    width = max(1.0, float(width_fraction) * sample_count)
    envelope = np.exp(-0.5 * np.square(distance / width))
    return normalize_cycle(cycle * (1.0 + float(amount) * envelope))


def _match_rms(values: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Match table energy while keeping closure impulses useful, not metallic.

    Bright/pressed variants can acquire a one-sample closure spike after harmonic
    shaping.  That spike lowers their usable RMS after peak normalization and makes
    the repeated source sound like a hard digital click.  A deliberately mild
    tanh pass is applied only when crest factor is above 4.8.  It preserves the
    bright upper-harmonic family but spreads the closure energy over a few samples.
    """

    values = normalize_cycle(values)
    reference = normalize_cycle(reference)
    crest = float(
        np.max(np.abs(values))
        / (np.sqrt(np.mean(np.square(values)) + 1e-30))
    )
    if crest > 4.8:
        drive = 0.80
        values = np.tanh(values * drive) / np.tanh(drive)
        values = normalize_cycle(values)

    source_rms = float(np.sqrt(np.mean(np.square(values)) + 1e-30))
    target_rms = float(np.sqrt(np.mean(np.square(reference)) + 1e-30))
    result = values * (target_rms / source_rms if source_rms > 1e-12 else 1.0)
    peak = float(np.max(np.abs(result)))
    if peak > 1.0:
        result = result / peak
    return result - float(np.mean(result))


def generate_wave_bank(values: Iterable[float]) -> tuple[list[str], list[list[float]], list[float]]:
    """Create a related eight-table bank from one cleaned reference cycle."""

    base = normalize_cycle(values)
    tables: list[np.ndarray] = []
    names: list[str] = []

    def add(name: str, table: np.ndarray) -> None:
        names.append(name)
        tables.append(_match_rms(table, base))

    add("native_clean", base)
    add(
        "soft",
        _apply_harmonic_shape(
            _phase_warp(base, 0.045, 0.25),
            delta_tilt_db_per_octave=-2.4,
            upper_taper_db=3.0,
        ),
    )
    add(
        "bright",
        _closure_emphasis(
            _apply_harmonic_shape(base, delta_tilt_db_per_octave=2.2, h1_db=-0.8),
            0.10,
        ),
    )
    add(
        "pressed",
        _closure_emphasis(
            _apply_harmonic_shape(
                _phase_warp(base, -0.075, 0.72),
                delta_tilt_db_per_octave=1.2,
                h1_db=-1.2,
            ),
            0.24,
            width_fraction=0.014,
        ),
    )
    smoothed = signal.savgol_filter(base, 15, 3, mode="wrap")
    add(
        "breathy_open",
        _apply_harmonic_shape(
            _phase_warp(smoothed, 0.075, 0.18),
            delta_tilt_db_per_octave=-1.3,
            h1_db=1.0,
            upper_taper_db=2.0,
        ),
    )
    add(
        "low_f0",
        _apply_harmonic_shape(
            _phase_warp(base, 0.035, 0.55),
            delta_tilt_db_per_octave=-0.8,
            h1_db=1.4,
            h2_db=0.6,
        ),
    )
    add(
        "high_f0",
        _closure_emphasis(
            _apply_harmonic_shape(
                _phase_warp(base, -0.035, 0.65),
                delta_tilt_db_per_octave=1.5,
                h1_db=-1.2,
            ),
            0.12,
        ),
    )
    add(
        "thin_young",
        _closure_emphasis(
            _apply_harmonic_shape(
                base,
                delta_tilt_db_per_octave=2.8,
                h1_db=-3.0,
                h2_db=-1.5,
            ),
            0.08,
        ),
    )
    return names, [[float(value) for value in table] for table in tables], list(DEFAULT_BANK_WEIGHTS)


def periodic_cubic_lookup(table: Sequence[float], phase_cycles: np.ndarray) -> np.ndarray:
    """Four-point cubic lookup into a periodic table."""

    values = np.asarray(table, dtype=np.float64).reshape(-1)
    if values.size < 8:
        raise ValueError("Wavetable requires at least 8 points")
    phase = np.mod(np.asarray(phase_cycles, dtype=np.float64), 1.0)
    position = phase * values.size
    i1 = np.floor(position).astype(np.int64)
    fraction = position - i1
    i0 = (i1 - 1) % values.size
    i2 = (i1 + 1) % values.size
    i3 = (i1 + 2) % values.size
    y0 = values[i0]
    y1 = values[i1]
    y2 = values[i2]
    y3 = values[i3]
    t = fraction
    # Catmull-Rom cubic interpolation with periodic indices.
    return 0.5 * (
        2.0 * y1
        + (-y0 + y2) * t
        + (2.0 * y0 - 5.0 * y1 + 4.0 * y2 - y3) * t * t
        + (-y0 + 3.0 * y1 - 3.0 * y2 + y3) * t * t * t
    )


def _cycle_random_track(
    phase_cycles: np.ndarray,
    rng: np.random.Generator,
    *,
    sigma_cycles: float,
) -> np.ndarray:
    cycle_number = np.floor(phase_cycles).astype(np.int64)
    count = int(np.max(cycle_number)) + 4 if cycle_number.size else 4
    values = rng.standard_normal(count)
    values = gaussian_filter1d(values, sigma=max(0.55, float(sigma_cycles)), mode="nearest")
    standard_deviation = float(np.std(values))
    if standard_deviation > 1e-12:
        values = values / standard_deviation
    fraction = phase_cycles - cycle_number
    smooth_fraction = fraction * fraction * (3.0 - 2.0 * fraction)
    left = values[np.clip(cycle_number, 0, count - 1)]
    right = values[np.clip(cycle_number + 1, 0, count - 1)]
    return left + (right - left) * smooth_fraction


def render_wave_bank(
    phase_cycles: np.ndarray,
    f0_hz: np.ndarray,
    pressure: np.ndarray,
    aspiration: np.ndarray,
    *,
    base_f0_hz: float,
    tables: Sequence[Sequence[float]],
    names: Sequence[str],
    base_weights: Sequence[float],
    morph_percent: float,
    breathiness: float,
    rng: np.random.Generator,
    vowel_weights: Mapping[str, np.ndarray] | None = None,
    vowel_source_mix: float = 0.0,
    source_preservation: bool = False,
    return_weights: bool = False,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Render a continuously morphing glottal-source wavetable bank.

    Digital Throat profiles contain ``vowel_a`` ... ``vowel_u`` source
    deviations.  In preservation mode those tables are *not* all mixed into
    the baseline.  Instead, the gesture timeline selects the current vowel
    (and smoothly crossfades diphthongs), while low/high-F0 and soft/pressed
    tables continue to describe phonation state.
    """

    if not tables:
        raise ValueError("Wave bank is empty")
    bank = [np.asarray(table, dtype=np.float64) for table in tables]
    sample_count = int(np.asarray(phase_cycles).size)
    if any(table.size != bank[0].size for table in bank):
        raise ValueError("All wave-bank tables must have the same length")
    if len(names) != len(bank):
        names = [f"table_{index}" for index in range(len(bank))]
    names = [str(name) for name in names]

    weights0 = np.asarray(base_weights, dtype=np.float64)
    if weights0.size != len(bank) or np.sum(weights0) <= 1e-12:
        weights0 = np.ones(len(bank), dtype=np.float64)
    weights0 = np.clip(weights0, 0.0, None)

    phase_cycles = np.asarray(phase_cycles, dtype=np.float64)
    f0_hz = np.asarray(f0_hz, dtype=np.float64)
    pressure = np.clip(np.asarray(pressure, dtype=np.float64), 0.0, 1.0)
    aspiration = np.clip(np.asarray(aspiration, dtype=np.float64), 0.0, 1.0)
    f0_octaves = np.clip(
        np.log2(np.maximum(f0_hz, 1.0) / max(float(base_f0_hz), 1.0)),
        -1.0,
        1.0,
    )
    morph_strength = float(np.clip(morph_percent / 100.0, 0.0, 0.75))
    micro_a = _cycle_random_track(phase_cycles, rng, sigma_cycles=0.85)
    micro_b = _cycle_random_track(phase_cycles, rng, sigma_cycles=1.35)

    name_to_index = {name: index for index, name in enumerate(names)}
    vowel_indices = {
        vowel: name_to_index.get(f"vowel_{vowel}") for vowel in ("a", "e", "i", "o", "u")
    }
    vowel_index_set = {index for index in vowel_indices.values() if index is not None}

    # Digital Throat already encoded vowel-specific deviations in dedicated
    # tables.  Keep those out of the neutral/core mix; otherwise every vowel is
    # permanently contaminated by all five source variants.
    if source_preservation and vowel_index_set:
        core_weights = weights0.copy()
        for index in vowel_index_set:
            core_weights[index] = 0.0
        if float(np.sum(core_weights)) <= 1e-12:
            native_index = name_to_index.get("native_clean", 0)
            core_weights[native_index] = 1.0
        core_weights /= np.sum(core_weights)
        full_weights = np.repeat(core_weights[:, None], sample_count, axis=1)
    else:
        weights0 = weights0 / np.sum(weights0)
        full_weights = np.repeat(weights0[:, None], sample_count, axis=1)

    def add(name: str, amount: np.ndarray | float) -> None:
        index = name_to_index.get(name)
        if index is not None and index not in vowel_index_set:
            full_weights[index] += np.asarray(amount, dtype=np.float64)

    add("high_f0", np.maximum(f0_octaves, 0.0) * 0.20)
    add("bright", np.maximum(f0_octaves, 0.0) * 0.08)
    add("thin_young", np.maximum(f0_octaves, 0.0) * 0.08)
    add("low_f0", np.maximum(-f0_octaves, 0.0) * 0.19)
    add("pressed", np.power(pressure, 1.55) * 0.18)
    add("soft", np.power(1.0 - pressure, 1.45) * 0.09)
    add("breathy_open", aspiration * 0.10 + float(breathiness) * 0.06)

    add("bright", np.maximum(micro_a, 0.0) * morph_strength * 0.18)
    add("soft", np.maximum(-micro_a, 0.0) * morph_strength * 0.15)
    add("pressed", np.maximum(micro_b, 0.0) * morph_strength * 0.14)
    add("breathy_open", np.maximum(-micro_b, 0.0) * morph_strength * 0.12)
    add("thin_young", np.maximum(micro_a + 0.35 * micro_b, 0.0) * morph_strength * 0.07)

    full_weights = np.clip(full_weights, 0.0, None)

    debug: dict[str, np.ndarray] = {
        "source_cycle_morph_a": micro_a,
        "source_cycle_morph_b": micro_b,
        "source_f0_octaves_from_base": f0_octaves,
        "source_preservation_mode": np.full(
            sample_count, 1.0 if source_preservation else 0.0, dtype=np.float64
        ),
    }

    if source_preservation and vowel_index_set:
        # Normalize the phonation-state/core bank first.
        core_sum = np.sum(full_weights, axis=0, keepdims=True)
        core = full_weights / np.maximum(core_sum, 1e-12)

        selected = np.zeros_like(core)
        target_sum = np.zeros(sample_count, dtype=np.float64)
        if vowel_weights is not None:
            for vowel, index in vowel_indices.items():
                if index is None:
                    continue
                values = np.clip(
                    np.asarray(vowel_weights.get(vowel, np.zeros(sample_count)), dtype=np.float64),
                    0.0,
                    1.0,
                )
                if values.size != sample_count:
                    raise ValueError(f"vowel weight {vowel!r} has the wrong length")
                selected[index] = values
                target_sum += values
                debug[f"source_vowel_target_{vowel}"] = values
        selected /= np.maximum(target_sum[None, :], 1e-12)
        active = np.clip(target_sum, 0.0, 1.0)
        mix = active * float(np.clip(vowel_source_mix, 0.0, 0.80))
        full_weights = core * (1.0 - mix[None, :]) + selected * mix[None, :]
        debug["source_vowel_mix_amount"] = mix
    else:
        weight_sum = np.sum(full_weights, axis=0, keepdims=True)
        full_weights = full_weights / np.maximum(weight_sum, 1e-12)
        debug["source_vowel_mix_amount"] = np.zeros(sample_count, dtype=np.float64)

    output = np.zeros(sample_count, dtype=np.float64)
    block_size = 65_536
    for start in range(0, sample_count, block_size):
        stop = min(sample_count, start + block_size)
        phase_block = phase_cycles[start:stop]
        for index, table in enumerate(bank):
            output[start:stop] += (
                periodic_cubic_lookup(table, phase_block)
                * full_weights[index, start:stop]
            )

    # Legacy generated banks used an energy correction to compensate for
    # unrelated table phases.  Digital Throat tables are phase-aligned; applying
    # that correction changes their intended cycle amplitude and obscures direct
    # waveform comparison, so preservation mode leaves the linear mixture alone.
    if not source_preservation:
        local_energy = np.sum(np.square(full_weights), axis=0)
        output = output / np.sqrt(np.maximum(local_energy, 0.55))
    output = np.clip(output, -1.35, 1.35)

    debug["source_wave_bank_dominant_index"] = np.argmax(
        full_weights, axis=0
    ).astype(np.float64)
    if return_weights:
        for index, name in enumerate(names):
            safe = "".join(character if character.isalnum() else "_" for character in name)
            debug[f"source_wave_bank_weight_{safe}"] = full_weights[index]
    return output, debug


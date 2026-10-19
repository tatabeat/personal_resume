from __future__ import annotations

import math
from collections.abc import Mapping, MutableMapping
from typing import Any

import numpy as np
from scipy import signal
from scipy.ndimage import gaussian_filter1d

from .models import GestureTimeline
from .precision_engine import apply_precision_guard, apply_precision_optimizer
from .source_wave import periodic_cubic_lookup, render_wave_bank
from .voice_profile import VoiceProfile


def _capture_debug(
    debug_trace: MutableMapping[str, np.ndarray] | None,
    name: str,
    values: np.ndarray,
) -> None:
    if debug_trace is None:
        return
    debug_trace[name] = np.asarray(values, dtype=np.float64).copy()


def _apply_component_gain(
    values: np.ndarray,
    name: str,
    component_gains: Mapping[str, float] | None,
) -> np.ndarray:
    if component_gains is None:
        return values
    gain = float(component_gains.get(name, 1.0))
    if not math.isfinite(gain) or gain < 0.0:
        raise ValueError(f"Invalid debug component gain for {name}: {gain}")
    if gain == 1.0:
        return values
    return values * gain


def _interp_track(
    timeline: GestureTimeline,
    name: str,
    target_times: np.ndarray,
    default: float = 0.0,
) -> np.ndarray:
    values = timeline.tracks.get(name)
    if values is None or len(values) == 0:
        return np.full_like(target_times, default, dtype=np.float64)
    source_times = timeline.times_s
    if len(values) == 1:
        return np.full_like(target_times, float(values[0]), dtype=np.float64)
    return np.interp(target_times, source_times, values, left=values[0], right=values[-1])


def _periodic_wave(phase_cycles: np.ndarray, points: list[float]) -> np.ndarray:
    # The profile already stores a Fourier-resampled periodic table.  Four-point
    # cubic lookup retains its 512-sample shape substantially better than the old
    # piecewise-linear lookup and does not blur the closure excitation.
    cycle = np.asarray(points, dtype=np.float64)
    cycle = cycle - np.mean(cycle)
    peak = float(np.max(np.abs(cycle))) or 1.0
    return periodic_cubic_lookup(cycle / peak, phase_cycles)


def _colored_random_track(
    frame_count: int,
    sigma_frames: float,
    rng: np.random.Generator,
) -> np.ndarray:
    if frame_count <= 1:
        return np.zeros(frame_count, dtype=np.float64)
    values = rng.standard_normal(frame_count)
    values = gaussian_filter1d(values, sigma=max(0.5, sigma_frames), mode="nearest")
    std = float(np.std(values))
    return values / std if std > 1e-9 else np.zeros_like(values)


def _spectral_tilt(
    frequencies: np.ndarray,
    db_per_octave: float,
    reference_hz: float = 200.0,
) -> np.ndarray:
    frequency = np.maximum(frequencies, reference_hz)
    octaves = np.log2(frequency / reference_hz)
    gain = np.power(10.0, db_per_octave * octaves / 20.0)
    gain[0] = gain[1] if gain.size > 1 else 1.0
    return gain


def _gaussian_peak(
    frequencies: np.ndarray,
    center: np.ndarray,
    bandwidth: np.ndarray,
) -> np.ndarray:
    sigma = np.maximum(20.0, bandwidth / 2.355)
    return np.exp(-0.5 * np.square((frequencies[:, None] - center[None, :]) / sigma[None, :]))


def _lorentzian_peak(
    frequencies: np.ndarray,
    center: np.ndarray,
    bandwidth: np.ndarray,
) -> np.ndarray:
    """Broad -3 dB resonator profile used by the source-preserving path.

    The old sum of narrow Gaussian peaks nearly zeroed all harmonics between
    formants.  A vocal tract is resonant, but it does not erase the excitation
    into three isolated spectral lines.  Lorentzian skirts retain substantially
    more of Digital Throat's measured harmonic magnitude and phase structure.
    """

    half_bandwidth = np.maximum(20.0, bandwidth * 0.5)
    distance = (frequencies[:, None] - center[None, :]) / half_bandwidth[None, :]
    return 1.0 / np.sqrt(1.0 + np.square(distance))


def _normalize_envelope_energy(
    envelope: np.ndarray,
    target_rms: float,
    energy_weight: np.ndarray | None = None,
) -> np.ndarray:
    """Normalize each spectral column by energy rather than its single peak.

    Max normalization made narrow formants much quieter than broad ones.  An
    RMS target keeps vowels and consonant places closer in perceived level
    while preserving their spectral shapes.
    """

    weighted = envelope
    if energy_weight is not None:
        weighted = envelope * energy_weight[:, None]
    rms = np.sqrt(np.mean(np.square(weighted), axis=0, keepdims=True) + 1e-12)
    normalized = envelope / rms * target_rms
    return np.clip(normalized, 0.0, 3.5)


def _shape_stft(
    frequencies: np.ndarray,
    frame_times: np.ndarray,
    voiced_stft: np.ndarray,
    noise_stft: np.ndarray,
    timeline: GestureTimeline,
    profile: VoiceProfile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply the time-varying vocal tract while retaining source detail.

    Legacy profiles keep the original Gaussian-envelope renderer.  Digital
    Throat profiles use a broader resonator response plus a controlled direct
    harmonic path.  The latter still colors vowels with F1/F2/F3, but no longer
    collapses a complex extracted glottal cycle into an almost sinusoidal wave.
    """

    columns = voiced_stft.shape[1]
    voiced_output = np.zeros_like(voiced_stft)
    noise_output = np.zeros_like(noise_stft)
    source_preserving = bool(profile.is_digital_throat_profile)
    applied_tilt_db = 0.0 if source_preserving else profile.spectral_tilt_db_per_octave
    tilt = _spectral_tilt(frequencies, applied_tilt_db)[:, None]

    # Block processing keeps memory predictable for longer utterances.
    block_size = 256
    for start in range(0, columns, block_size):
        stop = min(columns, start + block_size)
        times = frame_times[start:stop]

        f1 = (
            _interp_track(timeline, "f1_hz", times, 530.0)
            * profile.formant_scale
            * profile.f1_scale
        )
        f2 = (
            _interp_track(timeline, "f2_hz", times, 1850.0)
            * profile.formant_scale
            * profile.f2_scale
        )
        f3 = (
            _interp_track(timeline, "f3_hz", times, 2500.0)
            * profile.formant_scale
            * profile.f3_scale
        )
        b1 = _interp_track(timeline, "b1_hz", times, 90.0) * profile.f1_bandwidth_scale
        b2 = _interp_track(timeline, "b2_hz", times, 135.0) * profile.f2_bandwidth_scale
        b3 = _interp_track(timeline, "b3_hz", times, 180.0) * profile.f3_bandwidth_scale
        velum = np.clip(_interp_track(timeline, "velum_open", times), 0.0, 1.0)
        constriction = np.clip(_interp_track(timeline, "constriction", times), 0.0, 1.0)

        if source_preserving:
            direct = float(np.clip(profile.source_direct_mix, 0.0, 0.90))
            resonant = (
                0.12
                + (1.00 * profile.f1_gain) * _lorentzian_peak(frequencies, f1, b1)
                + (0.82 * profile.f2_gain) * _lorentzian_peak(frequencies, f2, b2)
                + (0.64 * profile.f3_gain) * _lorentzian_peak(frequencies, f3, b3)
            )
            source_energy_weight = 1.0 / (
                1.0 + np.power(np.maximum(frequencies, 1.0) / 1200.0, 0.75)
            )
            resonant = _normalize_envelope_energy(
                resonant,
                target_rms=1.0,
                energy_weight=source_energy_weight,
            )
            # The direct term is a phase-preserving path for harmonics not near a
            # formant.  It is not an unfiltered audio bypass; it lives inside the
            # same STFT path and is still modulated by velum/constriction below.
            voice_envelope = direct + (1.0 - direct) * resonant
        else:
            voice_envelope = (
                0.018
                + (1.00 * profile.f1_gain) * _gaussian_peak(frequencies, f1, b1)
                + (0.72 * profile.f2_gain) * _gaussian_peak(frequencies, f2, b2)
                + (0.42 * profile.f3_gain) * _gaussian_peak(frequencies, f3, b3)
            )
            source_energy_weight = 1.0 / (
                1.0 + np.power(np.maximum(frequencies, 1.0) / 850.0, 1.15)
            )

        nasal_peak = _lorentzian_peak if source_preserving else _gaussian_peak
        nasal_envelope = (
            0.02
            + 1.00
            * nasal_peak(
                frequencies,
                np.full_like(velum, 260.0),
                np.full_like(velum, 110.0),
            )
            + 0.32
            * nasal_peak(
                frequencies,
                np.full_like(velum, 1050.0),
                np.full_like(velum, 280.0),
            )
        )
        voice_envelope = voice_envelope * (1.0 - 0.72 * velum[None, :])
        voice_envelope += nasal_envelope * velum[None, :]

        # A near closure suppresses the resonated source without removing burst noise.
        voice_envelope *= 1.0 - 0.70 * np.power(constriction[None, :], 2.0)
        voice_envelope *= tilt
        voice_envelope = _normalize_envelope_energy(
            voice_envelope,
            target_rms=0.055,
            energy_weight=source_energy_weight,
        )

        noise_center = _interp_track(timeline, "noise_center_hz", times, 3500.0)
        noise_width = _interp_track(timeline, "noise_bandwidth_hz", times, 3000.0)
        focused_noise = _gaussian_peak(frequencies, noise_center, noise_width)
        diffuse_noise = _gaussian_peak(frequencies, noise_center, noise_width * 1.85)
        noise_envelope = 0.006 + 0.82 * focused_noise + 0.18 * diffuse_noise
        noise_envelope += 0.007 * np.sqrt(
            np.maximum(frequencies[:, None], 1.0) / 8000.0
        )
        noise_envelope = _normalize_envelope_energy(
            noise_envelope, target_rms=0.225
        )

        voiced_output[:, start:stop] = (
            voiced_stft[:, start:stop] * voice_envelope
        )
        noise_output[:, start:stop] = noise_stft[:, start:stop] * noise_envelope

    return voiced_output + noise_output, voiced_output, noise_output


def _balance_syllable_levels(
    audio: np.ndarray,
    timeline: GestureTimeline,
    intended_amplitude: np.ndarray,
    voicing: np.ndarray,
    sample_rate: int,
) -> np.ndarray:
    """Reduce vowel-to-vowel loudness jumps without flattening expression.

    One bounded gain is estimated from the voiced nucleus of each syllable.
    This avoids pumping inside diphthongs and does not normalize frication or
    pauses.  The resulting gain track is smoothed across segment boundaries.
    """

    if audio.size < max(64, int(0.06 * sample_rate)):
        return audio

    measurements: list[tuple[int, int, float]] = []
    for span in timeline.spans:
        if span.kind != "syllable":
            continue
        start = max(0, int(round(span.start_s * sample_rate)))
        stop = min(audio.size, int(round(span.end_s * sample_rate)))
        if stop - start < max(16, int(0.025 * sample_rate)):
            continue
        mask = (
            (intended_amplitude[start:stop] > 0.20)
            & (voicing[start:stop] > 0.55)
        )
        if np.count_nonzero(mask) < max(8, int(0.018 * sample_rate)):
            continue
        nucleus = audio[start:stop][mask]
        rms = float(np.sqrt(np.mean(np.square(nucleus)) + 1e-12))
        if rms > 1e-7:
            measurements.append((start, stop, rms))

    if len(measurements) < 2:
        return audio

    reference = float(np.median([item[2] for item in measurements]))
    gain_db_track = np.zeros(audio.size, dtype=np.float64)
    weight_track = np.zeros(audio.size, dtype=np.float64)
    for start, stop, rms in measurements:
        gain_db = float(np.clip(20.0 * np.log10(reference / rms), -8.0, 5.0))
        gain_db_track[start:stop] = gain_db
        weight_track[start:stop] = 1.0

    sigma = max(1.0, 0.012 * sample_rate)
    gain_db_track = gaussian_filter1d(gain_db_track, sigma=sigma, mode="nearest")
    weight_track = gaussian_filter1d(weight_track, sigma=sigma, mode="nearest")
    gain = np.power(10.0, gain_db_track * np.clip(weight_track, 0.0, 1.0) / 20.0)
    return audio * gain


def synthesize_timeline(
    timeline: GestureTimeline,
    profile: VoiceProfile | None = None,
    *,
    seed: int = 7,
    debug_trace: MutableMapping[str, np.ndarray] | None = None,
    component_gains: Mapping[str, float] | None = None,
    compute_backend: str = "auto",
    precision_guard: bool = True,
    precision_guard_report: MutableMapping[str, Any] | None = None,
    normalization_gain_override: float | None = None,
) -> np.ndarray:
    """Render a gesture timeline to mono float32 audio.

    This is a deliberately inspectable source-filter renderer. It is useful for
    experiments and control studies, but it is not expected to match a modern
    neural TTS system in naturalness.
    """

    profile = profile or VoiceProfile()
    profile.validate()
    sample_rate = int(profile.sample_rate)
    sample_count = max(1, int(round(timeline.duration_s * sample_rate)))
    sample_times = np.arange(sample_count, dtype=np.float64) / sample_rate
    rng = np.random.default_rng(seed)

    frame_count = max(1, timeline.frame_count)
    jitter_track = _colored_random_track(frame_count, timeline.rate_hz * 0.025, rng)
    shimmer_track = _colored_random_track(frame_count, timeline.rate_hz * 0.035, rng)
    jitter = np.interp(sample_times, timeline.times_s, jitter_track) if frame_count > 1 else np.zeros(sample_count)
    shimmer = np.interp(sample_times, timeline.times_s, shimmer_track) if frame_count > 1 else np.zeros(sample_count)

    f0 = _interp_track(timeline, "f0_hz", sample_times, profile.base_f0_hz)
    f0 *= 1.0 + jitter * profile.jitter_percent / 100.0
    f0 = np.clip(f0, 30.0, min(1800.0, sample_rate / 5.0))

    amplitude = np.clip(_interp_track(timeline, "amplitude", sample_times), 0.0, 1.0)
    voicing = np.clip(_interp_track(timeline, "voicing", sample_times), 0.0, 1.0)
    frication = np.clip(_interp_track(timeline, "frication", sample_times), 0.0, 1.0)
    aspiration = np.clip(_interp_track(timeline, "aspiration", sample_times), 0.0, 1.0)
    pressure = np.clip(_interp_track(timeline, "pressure", sample_times), 0.0, 1.0)
    burst = np.clip(_interp_track(timeline, "burst", sample_times), 0.0, 1.0)
    noise_center_hz = _interp_track(timeline, "noise_center_hz", sample_times, 3500.0)
    noise_bandwidth_hz = _interp_track(timeline, "noise_bandwidth_hz", sample_times, 3000.0)
    vowel_source_weights = {
        name: np.clip(
            _interp_track(timeline, f"vowel_{name}", sample_times, 0.0),
            0.0,
            1.0,
        )
        for name in ("a", "e", "i", "o", "u")
    }
    vowel_weight_sum = np.sum(np.vstack(list(vowel_source_weights.values())), axis=0)
    active_vowel = vowel_weight_sum > 1e-9
    for name, values in vowel_source_weights.items():
        normalized = values.copy()
        normalized[active_vowel] /= vowel_weight_sum[active_vowel]
        vowel_source_weights[name] = normalized

    _capture_debug(debug_trace, "sample_times_s", sample_times)
    _capture_debug(debug_trace, "f0_hz", f0)
    _capture_debug(debug_trace, "amplitude", amplitude)
    _capture_debug(debug_trace, "voicing", voicing)
    _capture_debug(debug_trace, "frication", frication)
    _capture_debug(debug_trace, "aspiration", aspiration)
    _capture_debug(debug_trace, "pressure", pressure)
    _capture_debug(debug_trace, "burst", burst)
    _capture_debug(debug_trace, "noise_center_hz", noise_center_hz)
    _capture_debug(debug_trace, "noise_bandwidth_hz", noise_bandwidth_hz)
    for vowel_name, values in vowel_source_weights.items():
        _capture_debug(debug_trace, f"vowel_source_{vowel_name}", values)

    shimmer_gain = np.clip(1.0 + shimmer * profile.shimmer_percent / 100.0, 0.65, 1.35)
    _capture_debug(debug_trace, "shimmer_gain", shimmer_gain)
    phase_cycles = np.cumsum(f0) / sample_rate
    if profile.wave_bank:
        periodic, source_morph_debug = render_wave_bank(
            phase_cycles,
            f0,
            pressure,
            aspiration,
            base_f0_hz=profile.base_f0_hz,
            tables=profile.wave_bank,
            names=profile.wave_bank_names,
            base_weights=profile.wave_bank_base_weights,
            morph_percent=profile.cycle_morph_percent,
            breathiness=profile.breathiness,
            rng=rng,
            vowel_weights=vowel_source_weights,
            vowel_source_mix=profile.vowel_source_mix,
            source_preservation=profile.is_digital_throat_profile,
            return_weights=debug_trace is not None,
        )
        for name, values in source_morph_debug.items():
            _capture_debug(debug_trace, name, values)
    else:
        periodic = _periodic_wave(phase_cycles, profile.wave_points)
    if debug_trace is not None:
        _capture_debug(debug_trace, "source_primary_table", np.asarray(profile.wave_points))
        _capture_debug(debug_trace, "source_native_cycle", np.asarray(profile.native_cycle))
        for index, table in enumerate(profile.wave_bank):
            _capture_debug(debug_trace, f"source_wave_bank_table_{index:02d}", np.asarray(table))
    _capture_debug(debug_trace, "source_rendered_unit", periodic)
    _capture_debug(debug_trace, "source_after_f0", periodic)
    voiced_source = periodic * amplitude * voicing * shimmer_gain * (0.35 + 0.65 * pressure)
    voiced_source = _apply_component_gain(voiced_source, "voiced_source", component_gains)
    _capture_debug(debug_trace, "periodic", periodic)
    _capture_debug(debug_trace, "source_after_amplitude_voicing", voiced_source)
    _capture_debug(debug_trace, "voiced_source", voiced_source)

    white = rng.standard_normal(sample_count)
    _capture_debug(debug_trace, "raw_aspiration_noise", white)
    highpass = signal.sosfilt(
        signal.butter(2, 500.0, btype="highpass", fs=sample_rate, output="sos"),
        white,
    )
    _capture_debug(debug_trace, "aspiration_highpass_noise", highpass)
    aspiration_gain_factor = np.full(
        sample_count,
        0.02 + 0.55 * profile.breathiness,
        dtype=np.float64,
    )
    _capture_debug(debug_trace, "aspiration_gain_factor", aspiration_gain_factor)
    aspiration_source = highpass * amplitude * aspiration * aspiration_gain_factor
    aspiration_source = _apply_component_gain(aspiration_source, "aspiration_source", component_gains)
    _capture_debug(debug_trace, "aspiration_source", aspiration_source)

    fric_noise = rng.standard_normal(sample_count)
    _capture_debug(debug_trace, "raw_frication_noise", fric_noise)
    fric_noise = signal.sosfilt(
        signal.butter(2, 900.0, btype="highpass", fs=sample_rate, output="sos"),
        fric_noise,
    )
    _capture_debug(debug_trace, "filtered_frication_noise", fric_noise)
    frication_pressure_factor = 0.13 + 0.34 * pressure
    _capture_debug(debug_trace, "frication_pressure_factor", frication_pressure_factor)
    _capture_debug(debug_trace, "frication_output_gain", np.full(sample_count, 0.74, dtype=np.float64))
    frication_source = fric_noise * frication_pressure_factor * frication * 0.74
    frication_source = _apply_component_gain(frication_source, "frication_source", component_gains)
    # Preserve a modest broadband component for /h/ and aspirated releases;
    # the main aspiration path remains formant-shaped with the vocal tract.
    aspiration_noise_source = highpass * amplitude * aspiration * 0.08
    aspiration_noise_source = _apply_component_gain(
        aspiration_noise_source,
        "aspiration_noise_source",
        component_gains,
    )
    burst_source = np.zeros(sample_count, dtype=np.float64)
    if debug_trace is not None:
        _capture_debug(debug_trace, "raw_burst_noise", np.zeros(sample_count, dtype=np.float64))
        _capture_debug(debug_trace, "burst_envelope", np.zeros(sample_count, dtype=np.float64))

    if np.max(burst) > 0.0:
        burst_noise = rng.standard_normal(sample_count)
        _capture_debug(debug_trace, "raw_burst_noise", burst_noise)
        # Widen each gesture-frame impulse into a short release transient.
        kernel_length = max(3, int(round(0.006 * sample_rate)))
        kernel = np.exp(-np.linspace(0.0, 5.0, kernel_length))
        burst_envelope = signal.fftconvolve(burst, kernel, mode="same")
        burst_envelope /= float(np.max(burst_envelope)) or 1.0
        _capture_debug(debug_trace, "burst_envelope", burst_envelope)
        _capture_debug(debug_trace, "burst_output_gain", np.full(sample_count, 0.19, dtype=np.float64))
        burst_source = burst_noise * burst_envelope * 0.19
    elif debug_trace is not None:
        _capture_debug(debug_trace, "burst_output_gain", np.full(sample_count, 0.19, dtype=np.float64))
    burst_source = _apply_component_gain(burst_source, "burst_source", component_gains)

    # Keep the exact pre-optimizer sources in Debug so every optimized sample can
    # be traced back to the original arithmetic.
    _capture_debug(debug_trace, "voiced_source_raw", voiced_source)
    _capture_debug(debug_trace, "aspiration_source_raw", aspiration_source)
    _capture_debug(debug_trace, "frication_source_raw", frication_source)
    _capture_debug(debug_trace, "aspiration_noise_source_raw", aspiration_noise_source)
    _capture_debug(debug_trace, "burst_source_raw", burst_source)

    optimizer_metadata: dict[str, Any] = {
        "enabled": bool(precision_guard),
        "mode": "disabled",
        "backend": {"requested": compute_backend, "active": "disabled"},
    }
    if precision_guard:
        optimizer_components = {
            "aspiration_source": aspiration_source,
            "frication_source": frication_source,
            "aspiration_noise_source": aspiration_noise_source,
            "burst_source": burst_source,
        }
        # A 2.5 ms sample-wise gain track can reshape one full high-pitched
        # glottal period.  Digital Throat source preservation therefore leaves
        # voiced morphology untouched while still optimizing turbulent sources.
        if not profile.is_digital_throat_profile:
            optimizer_components["voiced_source"] = voiced_source
        optimized_components, optimizer_metadata, optimizer_arrays = apply_precision_optimizer(
            optimizer_components,
            sample_rate,
            {
                "amplitude": amplitude,
                "voicing": voicing,
                "frication": frication,
                "aspiration": aspiration,
                "pressure": pressure,
                "burst": burst,
                "burst_envelope": burst_envelope if np.max(burst) > 0.0 else np.zeros(sample_count),
                "noise_center_hz": noise_center_hz,
                "noise_bandwidth_hz": noise_bandwidth_hz,
            },
            compute_backend=compute_backend,
            strength=1.0,
        )
        voiced_source = optimized_components.get("voiced_source", voiced_source)
        aspiration_source = optimized_components.get("aspiration_source", aspiration_source)
        frication_source = optimized_components.get("frication_source", frication_source)
        aspiration_noise_source = optimized_components.get(
            "aspiration_noise_source", aspiration_noise_source
        )
        burst_source = optimized_components.get("burst_source", burst_source)
        for name, values in optimizer_arrays.items():
            _capture_debug(debug_trace, name, values)
        if profile.is_digital_throat_profile:
            _capture_debug(
                debug_trace,
                "optimizer_voiced_gain",
                np.ones(sample_count, dtype=np.float64),
            )

    noise_source = frication_source + aspiration_noise_source
    noise_source += burst_source
    voiced_mix = voiced_source + aspiration_source

    _capture_debug(debug_trace, "voiced_source", voiced_source)
    _capture_debug(debug_trace, "source_after_optimizer", voiced_source)
    _capture_debug(debug_trace, "aspiration_source", aspiration_source)
    _capture_debug(debug_trace, "frication_source", frication_source)
    _capture_debug(debug_trace, "aspiration_noise_source", aspiration_noise_source)
    _capture_debug(debug_trace, "burst_source", burst_source)
    _capture_debug(debug_trace, "noise_source", noise_source)
    _capture_debug(debug_trace, "voiced_mix", voiced_mix)
    source_sum_pre_stft = voiced_mix + noise_source
    _capture_debug(debug_trace, "source_sum_pre_stft", source_sum_pre_stft)

    nperseg = 1024 if sample_count >= 1024 else max(64, 2 ** int(math.floor(math.log2(max(sample_count, 64)))))
    noverlap = int(nperseg * 0.75)
    frequencies, frame_times, voiced_stft = signal.stft(
        voiced_mix,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )
    _, _, noise_stft = signal.stft(
        noise_source,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )

    shaped, shaped_voiced, shaped_noise = _shape_stft(
        frequencies,
        frame_times,
        voiced_stft,
        noise_stft,
        timeline,
        profile,
    )

    def istft_to_length(values: np.ndarray) -> np.ndarray:
        _, rendered = signal.istft(
            values,
            fs=sample_rate,
            window="hann",
            nperseg=nperseg,
            noverlap=noverlap,
            input_onesided=True,
            boundary=True,
        )
        rendered = np.asarray(rendered[:sample_count], dtype=np.float64)
        if rendered.size < sample_count:
            rendered = np.pad(rendered, (0, sample_count - rendered.size))
        return rendered

    audio = istft_to_length(shaped)
    if debug_trace is not None:
        _capture_debug(
            debug_trace,
            "source_after_vocal_tract",
            istft_to_length(shaped_voiced),
        )
        _capture_debug(
            debug_trace,
            "noise_after_vocal_tract",
            istft_to_length(shaped_noise),
        )
    _capture_debug(debug_trace, "source_and_noise_after_vocal_tract", audio)
    _capture_debug(debug_trace, "istft_raw", audio)

    audio = _balance_syllable_levels(audio, timeline, amplitude, voicing, sample_rate)
    _capture_debug(debug_trace, "level_balanced", audio)

    # Remove very low-frequency drift and excess ultrasonic hiss.  Use SOS
    # forward/backward filtering when the utterance is long enough so phase
    # transitions stay aligned with the gesture timeline.
    if audio.size > 32:
        highpass_sos = signal.butter(2, 45.0, btype="highpass", fs=sample_rate, output="sos")
        lowpass_hz = min(14_500.0, sample_rate * 0.45)
        lowpass_sos = signal.butter(2, lowpass_hz, btype="lowpass", fs=sample_rate, output="sos")
        try:
            audio = signal.sosfiltfilt(highpass_sos, audio)
            audio = signal.sosfiltfilt(lowpass_sos, audio)
        except ValueError:
            audio = signal.sosfilt(highpass_sos, audio)
            audio = signal.sosfilt(lowpass_sos, audio)
    _capture_debug(debug_trace, "post_filter", audio)

    # The quality optimizer above is the primary precision stage.  Keep the old
    # post-mix guard only as an emergency fallback for truly pathological
    # transients; ordinary consonants should be shaped at their source instead.
    emergency_guard_metadata: dict[str, Any] = {
        "enabled": bool(precision_guard),
        "backend": {"requested": compute_backend, "active": "disabled"},
        "corrected_samples": 0,
        "max_attenuation_db": 0.0,
    }
    if precision_guard:
        guard_inputs: dict[str, np.ndarray] = {
            "f0_hz": f0,
            "frication": frication,
            "aspiration": aspiration,
            "burst": burst,
            "noise_center_hz": noise_center_hz,
            "noise_bandwidth_hz": noise_bandwidth_hz,
        }
        audio, emergency_guard_metadata, guard_arrays = apply_precision_guard(
            audio,
            sample_rate,
            guard_inputs,
            compute_backend=compute_backend,
            max_attenuation_db=3.0,
        )
        for name, values in guard_arrays.items():
            _capture_debug(debug_trace, name, values)
    else:
        _capture_debug(debug_trace, "precision_guard_gain", np.ones(audio.size, dtype=np.float64))
    combined_precision_report = {
        "enabled": bool(precision_guard),
        "mode": "always_on_quality_optimizer",
        "quality_optimizer": optimizer_metadata,
        "emergency_guard": emergency_guard_metadata,
    }
    if precision_guard_report is not None:
        precision_guard_report.clear()
        precision_guard_report.update(combined_precision_report)
    _capture_debug(debug_trace, "precision_guarded", audio)

    # A lighter saturator keeps source detail after the component optimizer;
    # loudness normalization later restores overall level.
    tanh_drive = 1.02 if profile.is_digital_throat_profile else 1.12
    tanh_input = audio * tanh_drive
    _capture_debug(
        debug_trace,
        "tanh_drive",
        np.asarray([tanh_drive], dtype=np.float64),
    )
    _capture_debug(debug_trace, "tanh_input", tanh_input)
    audio = np.tanh(tanh_input)
    _capture_debug(debug_trace, "post_tanh", audio)

    # Keep average loudness stable. output_gain is a peak ceiling, not a target
    # that every sentence must be pushed up to.
    if normalization_gain_override is None:
        rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-30)) if audio.size else 0.0
        target_rms = profile.output_gain / 4.82
        rms_gain = float(np.clip(target_rms / rms, 0.25, 4.0)) if rms > 1e-12 else 1.0
        provisional = audio * rms_gain
        peak = float(np.max(np.abs(provisional))) if provisional.size else 0.0
        ceiling_gain = min(1.0, profile.output_gain / peak) if peak > 1e-12 else 1.0
        normalization_gain = rms_gain * ceiling_gain
    else:
        normalization_gain = float(normalization_gain_override)
        if not math.isfinite(normalization_gain) or normalization_gain <= 0.0:
            raise ValueError("normalization_gain_override must be finite and positive")
    audio = audio * normalization_gain
    _capture_debug(debug_trace, "normalization_gain", np.asarray([normalization_gain], dtype=np.float64))
    _capture_debug(debug_trace, "normalized", audio)

    fade = min(int(0.012 * sample_rate), max(1, audio.size // 10))
    if fade > 1:
        ramp = np.sin(np.linspace(0.0, np.pi / 2.0, fade)) ** 2
        audio[:fade] *= ramp
        audio[-fade:] *= ramp[::-1]

    _capture_debug(debug_trace, "final_audio", audio)
    return audio.astype(np.float32)

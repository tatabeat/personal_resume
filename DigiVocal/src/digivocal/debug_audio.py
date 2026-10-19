from __future__ import annotations

import csv
import html
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
from scipy import signal
from scipy.io import wavfile
from scipy.ndimage import uniform_filter1d

from .audio_io import write_wav
from .models import GestureTimeline, SegmentSpan
from .precision_engine import LEDGER_TRACKS, build_precision_audit
from .voice_profile import VoiceProfile


TRACE_COMPONENTS = (
    "voiced_source",
    "aspiration_source",
    "frication_source",
    "aspiration_noise_source",
    "burst_source",
)

TRACE_STAGES = (
    "source_rendered_unit",
    "source_after_amplitude_voicing",
    "source_after_optimizer",
    "voiced_mix",
    "noise_source",
    "source_after_vocal_tract",
    "source_and_noise_after_vocal_tract",
    "istft_raw",
    "level_balanced",
    "post_filter",
    "tanh_input",
    "post_tanh",
    "normalized",
    "final_audio",
)

PLOT_TRACES = (
    "final_audio",
    "source_after_vocal_tract",
    "source_after_optimizer",
    "source_rendered_unit",
    "post_filter",
    "voiced_source",
    "frication_source",
    "aspiration_source",
    "burst_source",
)

SOURCE_STAGE_TRACES = (
    ("01_rendered_source", "source_rendered_unit", "Wave Bank + vowel/F0 morph, unit source"),
    ("02_after_amplitude_voicing", "source_after_amplitude_voicing", "F0 source after amplitude, voicing, pressure and shimmer"),
    ("03_after_source_optimizer", "source_after_optimizer", "Voiced source immediately before vocal tract"),
    ("04_after_vocal_tract", "source_after_vocal_tract", "Voiced source after F1/F2/F3, without consonant noise"),
    ("05_source_noise_mix", "source_and_noise_after_vocal_tract", "Voiced + turbulent sources after tract shaping"),
    ("06_post_filter", "post_filter", "After output high/low-pass stage"),
    ("07_final", "final_audio", "Final DigiVocal waveform"),
)

ABLATION_SPECS: dict[str, dict[str, float]] = {
    "no_voiced": {"voiced_source": 0.0},
    "no_aspiration": {
        "aspiration_source": 0.0,
        "aspiration_noise_source": 0.0,
    },
    "no_frication": {"frication_source": 0.0},
    "no_burst": {"burst_source": 0.0},
}


class DebugAnalysisError(RuntimeError):
    """Raised when an audio debug bundle cannot be generated."""


def _to_float_audio(data: np.ndarray) -> np.ndarray:
    array = np.asarray(data)
    if array.ndim == 2:
        array = np.mean(array.astype(np.float64), axis=1)
    if array.ndim != 1:
        raise DebugAnalysisError("Only mono or ordinary multi-channel WAV is supported")
    if np.issubdtype(array.dtype, np.floating):
        return np.asarray(array, dtype=np.float64)
    if np.issubdtype(array.dtype, np.signedinteger):
        scale = float(max(abs(np.iinfo(array.dtype).min), np.iinfo(array.dtype).max))
        return array.astype(np.float64) / scale
    if np.issubdtype(array.dtype, np.unsignedinteger):
        info = np.iinfo(array.dtype)
        midpoint = (info.max + 1.0) / 2.0
        return (array.astype(np.float64) - midpoint) / midpoint
    raise DebugAnalysisError(f"Unsupported WAV sample type: {array.dtype}")


def load_wav_float(path: str | Path) -> tuple[int, np.ndarray]:
    sample_rate, data = wavfile.read(Path(path))
    return int(sample_rate), _to_float_audio(data)


def _db(value: float, *, floor_db: float = -180.0) -> float:
    if not math.isfinite(value) or value <= 0.0:
        return floor_db
    return max(floor_db, 20.0 * math.log10(value))


def _safe_array(values: np.ndarray | list[float], length: int) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == length:
        return array
    if array.size == 0:
        return np.zeros(length, dtype=np.float64)
    source = np.linspace(0.0, 1.0, array.size)
    target = np.linspace(0.0, 1.0, length)
    return np.interp(target, source, array)


def _robust_positive_score(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return np.zeros_like(array)
    median = float(np.median(finite))
    mad = float(np.median(np.abs(finite - median))) * 1.4826
    percentile_span = float(np.percentile(finite, 90.0) - median)
    scale = max(mad, percentile_span / 1.2816, 1e-12)
    return np.maximum(0.0, (array - median) / scale)


def _short_rms(audio: np.ndarray, window_samples: int) -> np.ndarray:
    size = max(3, int(window_samples))
    power = uniform_filter1d(np.square(audio), size=size, mode="nearest")
    return np.sqrt(np.maximum(power, 0.0) + 1e-15)


def _spectral_flux_per_sample(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    if audio.size < 128:
        return np.zeros(audio.size, dtype=np.float64)
    nperseg = min(512, max(128, 2 ** int(math.floor(math.log2(audio.size)))))
    hop = max(16, nperseg // 8)
    noverlap = nperseg - hop
    _, frame_times, spectrum = signal.stft(
        audio,
        fs=sample_rate,
        window="hann",
        nperseg=nperseg,
        noverlap=noverlap,
        boundary="zeros",
        padded=True,
    )
    magnitude = np.log1p(np.abs(spectrum))
    delta = np.maximum(0.0, np.diff(magnitude, axis=1, prepend=magnitude[:, :1]))
    flux = np.mean(delta, axis=0)
    sample_times = np.arange(audio.size, dtype=np.float64) / float(sample_rate)
    return np.interp(sample_times, frame_times, flux, left=flux[0], right=flux[-1])


def _high_frequency_ratio(audio: np.ndarray, sample_rate: int) -> np.ndarray:
    cutoff = min(3000.0, sample_rate * 0.18)
    if cutoff <= 80.0 or audio.size < 32:
        return np.zeros(audio.size, dtype=np.float64)
    sos = signal.butter(2, cutoff, btype="highpass", fs=sample_rate, output="sos")
    try:
        high = signal.sosfiltfilt(sos, audio)
    except ValueError:
        high = signal.sosfilt(sos, audio)
    window = max(8, int(round(0.004 * sample_rate)))
    total_rms = _short_rms(audio, window)
    high_rms = _short_rms(high, window)
    return np.clip(high_rms / (total_rms + 1e-9), 0.0, 8.0)


def _true_peak(audio: np.ndarray, oversample: int = 4) -> float:
    if audio.size == 0:
        return 0.0
    if audio.size < 8:
        return float(np.max(np.abs(audio)))
    oversampled = signal.resample_poly(audio, oversample, 1, padtype="line")
    return float(np.max(np.abs(oversampled)))


def _span_for_time(timeline: GestureTimeline | None, time_s: float) -> SegmentSpan | None:
    if timeline is None:
        return None
    for span in timeline.spans:
        if span.start_s <= time_s < span.end_s:
            return span
    if timeline.spans and math.isclose(time_s, timeline.spans[-1].end_s, abs_tol=1e-6):
        return timeline.spans[-1]
    return None


def _track_at_time(
    timeline: GestureTimeline | None,
    name: str,
    time_s: float,
    default: float = 0.0,
) -> float:
    if timeline is None or name not in timeline.tracks or timeline.frame_count == 0:
        return default
    values = timeline.tracks[name]
    index = int(np.clip(round(time_s * timeline.rate_hz), 0, len(values) - 1))
    return float(values[index])


def _local_metrics(array: np.ndarray, center: int, radius: int) -> dict[str, float]:
    start = max(0, center - radius)
    stop = min(array.size, center + radius + 1)
    segment = array[start:stop]
    if segment.size == 0:
        return {"peak": 0.0, "rms": 0.0, "max_jump": 0.0}
    jumps = np.abs(np.diff(segment))
    return {
        "peak": float(np.max(np.abs(segment))),
        "rms": float(np.sqrt(np.mean(np.square(segment)) + 1e-15)),
        "max_jump": float(np.max(jumps)) if jumps.size else 0.0,
    }


def _classify_event(
    *,
    sample_peak: float,
    true_peak: float,
    component_metrics: Mapping[str, Mapping[str, float]],
    stage_metrics: Mapping[str, Mapping[str, float]],
    controls: Mapping[str, float],
    crest_db: float,
    hf_ratio: float,
) -> tuple[str, list[str]]:
    reasons: list[str] = []
    if sample_peak >= 0.999:
        reasons.append("樣本峰值已碰到數位滿刻度")
    if true_peak >= 1.0:
        reasons.append("4 倍過取樣估算的取樣間峰值超過 0 dBFS")
    if reasons:
        return "clipping_or_true_peak_overload", reasons

    ranked = sorted(
        component_metrics.items(),
        key=lambda item: item[1]["rms"],
        reverse=True,
    )
    top_name = ranked[0][0] if ranked else ""
    top_rms = ranked[0][1]["rms"] if ranked else 0.0
    second_rms = ranked[1][1]["rms"] if len(ranked) > 1 else 0.0

    if top_name == "burst_source" and controls.get("burst", 0.0) > 0.02:
        reasons.append("burst_source 是事件窗內最強的獨立來源")
        if top_rms > second_rms * 1.4:
            reasons.append("爆破來源明顯高於其他來源")
        return "burst_transient", reasons
    if top_name == "frication_source" and controls.get("frication", 0.0) > 0.15:
        reasons.append("frication_source 是事件窗內最強的獨立來源")
        if hf_ratio > 0.6:
            reasons.append("事件同時帶有高比例高頻能量")
        return "frication_transient", reasons
    if top_name in {"aspiration_source", "aspiration_noise_source"} and controls.get("aspiration", 0.0) > 0.10:
        reasons.append("送氣來源在事件窗內占主要能量")
        return "aspiration_transient", reasons

    raw = stage_metrics.get("istft_raw", {})
    pre_filter = stage_metrics.get("level_balanced", raw)
    post_filter = stage_metrics.get("post_filter", pre_filter)
    tanh_input = stage_metrics.get("tanh_input", post_filter)
    post_tanh = stage_metrics.get("post_tanh", tanh_input)

    if raw and max(
        stage_metrics.get("voiced_mix", {}).get("max_jump", 0.0),
        stage_metrics.get("noise_source", {}).get("max_jump", 0.0),
        1e-9,
    ) * 1.8 < raw.get("max_jump", 0.0):
        reasons.append("異常跳變在 STFT 塑形／ISTFT 後明顯放大")
        return "stft_or_formant_ringing", reasons
    if pre_filter and post_filter.get("max_jump", 0.0) > pre_filter.get("max_jump", 0.0) * 1.7 + 1e-6:
        reasons.append("後處理濾波後的局部跳變明顯增大")
        return "post_filter_ringing", reasons
    if tanh_input.get("peak", 0.0) > 1.0 and post_tanh.get("peak", 0.0) > 0.75:
        reasons.append("tanh 前輸入已進入明顯非線性區")
        return "nonlinear_saturation", reasons
    if crest_db >= 17.0:
        reasons.append("峰均比很高，事件能量集中在極短時間")
    if hf_ratio > 0.75:
        reasons.append("事件包含大量高頻瞬態")
    if not reasons:
        reasons.append("最終波形出現統計上異常的跳變，但沒有單一來源占主導")
    return "waveform_discontinuity_or_mixed_transient", reasons


def _consolidate_indices(indices: np.ndarray, score: np.ndarray, distance: int) -> list[int]:
    if indices.size == 0:
        return []
    ordered = sorted((int(index) for index in indices), key=lambda index: float(score[index]), reverse=True)
    accepted: list[int] = []
    for index in ordered:
        if all(abs(index - previous) >= distance for previous in accepted):
            accepted.append(index)
    return sorted(accepted)


def analyze_trace(
    trace: Mapping[str, np.ndarray],
    sample_rate: int,
    timeline: GestureTimeline | None = None,
    *,
    max_events: int = 24,
) -> dict[str, Any]:
    if "final_audio" not in trace:
        raise DebugAnalysisError("debug trace does not contain final_audio")
    audio = np.asarray(trace["final_audio"], dtype=np.float64).reshape(-1)
    if audio.size == 0:
        raise DebugAnalysisError("audio is empty")
    if not np.all(np.isfinite(audio)):
        raise DebugAnalysisError("audio contains NaN or Infinity")

    sample_count = audio.size
    absolute = np.abs(audio)
    first_difference = np.abs(np.diff(audio, prepend=audio[0]))
    second_difference = np.abs(np.diff(audio, n=2, prepend=[audio[0], audio[0]]))
    rms_4ms = _short_rms(audio, max(8, int(round(0.004 * sample_rate))))
    crest = absolute / (rms_4ms + 1e-9)
    crest_db_track = 20.0 * np.log10(np.maximum(crest, 1e-9))
    flux = _spectral_flux_per_sample(audio, sample_rate)
    hf_ratio_track = _high_frequency_ratio(audio, sample_rate)

    jump_score = _robust_positive_score(first_difference)
    curvature_score = _robust_positive_score(second_difference)
    flux_score = _robust_positive_score(flux)
    crest_score = np.maximum(0.0, (crest_db_track - 12.0) / 3.0)
    full_scale_score = np.maximum(0.0, (absolute - 0.92) / 0.02)
    combined_score = np.maximum.reduce(
        [jump_score / 3.0, curvature_score / 3.5, flux_score / 4.0, crest_score, full_scale_score]
    )

    edge_guard = min(sample_count // 4, max(1, int(round(0.025 * sample_rate))))
    combined_score[:edge_guard] = 0.0
    combined_score[-edge_guard:] = 0.0

    minimum_distance = max(1, int(round(0.025 * sample_rate)))
    threshold = max(2.8, float(np.percentile(combined_score, 99.75)))
    peaks, properties = signal.find_peaks(
        combined_score,
        height=threshold,
        prominence=0.35,
        distance=max(1, int(round(0.004 * sample_rate))),
    )
    clipping_indices = np.flatnonzero(absolute >= 0.999)
    all_indices = np.unique(np.concatenate([peaks, clipping_indices]))
    candidates = _consolidate_indices(all_indices, combined_score, minimum_distance)
    candidates = sorted(candidates, key=lambda index: float(combined_score[index]), reverse=True)[:max_events]
    candidates.sort()

    component_arrays = {
        name: _safe_array(trace.get(name, np.zeros(sample_count)), sample_count)
        for name in TRACE_COMPONENTS
    }
    stage_arrays = {
        name: _safe_array(trace.get(name, audio), sample_count)
        for name in TRACE_STAGES
    }

    event_radius = max(8, int(round(0.006 * sample_rate)))
    events: list[dict[str, Any]] = []
    for event_number, index in enumerate(candidates, start=1):
        start = max(0, index - event_radius)
        stop = min(sample_count, index + event_radius + 1)
        local_audio = audio[start:stop]
        sample_peak = float(np.max(np.abs(local_audio)))
        local_rms = float(np.sqrt(np.mean(np.square(local_audio)) + 1e-15))
        true_peak = _true_peak(local_audio, 4)
        crest_db = _db(sample_peak / max(local_rms, 1e-12))
        time_s = index / float(sample_rate)
        span = _span_for_time(timeline, time_s)

        component_metrics = {
            name: _local_metrics(values, index, event_radius)
            for name, values in component_arrays.items()
        }
        stage_metrics = {
            name: _local_metrics(values, index, event_radius)
            for name, values in stage_arrays.items()
        }
        controls = {
            name: _track_at_time(timeline, name, time_s)
            for name in ("amplitude", "voicing", "frication", "aspiration", "pressure", "burst")
        }
        cause, reasons = _classify_event(
            sample_peak=sample_peak,
            true_peak=true_peak,
            component_metrics=component_metrics,
            stage_metrics=stage_metrics,
            controls=controls,
            crest_db=crest_db,
            hf_ratio=float(hf_ratio_track[index]),
        )

        component_values = {name: float(values[index]) for name, values in component_arrays.items()}
        source_sum = float(sum(component_values.values()))
        first_stage = max(
            TRACE_STAGES,
            key=lambda name: stage_metrics[name]["max_jump"],
        )
        events.append(
            {
                "event_id": event_number,
                "sample": int(index),
                "time_s": round(time_s, 9),
                "score": float(combined_score[index]),
                "sample_peak": sample_peak,
                "sample_peak_dbfs": _db(sample_peak),
                "true_peak_4x": true_peak,
                "true_peak_4x_dbtp": _db(true_peak),
                "local_rms": local_rms,
                "local_rms_dbfs": _db(local_rms),
                "crest_factor_db": crest_db,
                "max_sample_jump": float(np.max(first_difference[start:stop])),
                "max_second_difference": float(np.max(second_difference[start:stop])),
                "spectral_flux": float(flux[index]),
                "high_frequency_ratio": float(hf_ratio_track[index]),
                "suspected_type": cause,
                "reasons": reasons,
                "text": span.text if span else None,
                "pinyin": span.pinyin if span else None,
                "segment_start_s": span.start_s if span else None,
                "segment_end_s": span.end_s if span else None,
                "controls": controls,
                "component_values_at_sample": component_values,
                "source_sum_at_sample": source_sum,
                "component_metrics": component_metrics,
                "stage_metrics": stage_metrics,
                "largest_jump_stage": first_stage,
            }
        )

    sample_peak = float(np.max(absolute))
    overall_true_peak = _true_peak(audio, 4)
    overall_rms = float(np.sqrt(np.mean(np.square(audio)) + 1e-15))
    return {
        "schema": "digivocal-audio-debug-v1",
        "sample_rate": int(sample_rate),
        "sample_count": int(sample_count),
        "duration_s": sample_count / float(sample_rate),
        "analysis": {
            "sample_peak": sample_peak,
            "sample_peak_dbfs": _db(sample_peak),
            "true_peak_4x": overall_true_peak,
            "true_peak_4x_dbtp": _db(overall_true_peak),
            "rms": overall_rms,
            "rms_dbfs": _db(overall_rms),
            "max_sample_jump": float(np.max(first_difference)),
            "max_second_difference": float(np.max(second_difference)),
            "candidate_threshold": threshold,
            "event_count": len(events),
            "true_peak_note": "4x polyphase oversampling estimate; useful for diagnosis, not a certified BS.1770 meter.",
        },
        "events": events,
    }


def _generate_counterfactual_ablations(
    timeline: GestureTimeline,
    profile: VoiceProfile,
    seed: int,
    *,
    normalization_gain: float | None = None,
    compute_backend: str = "auto",
    precision_guard: bool = True,
) -> dict[str, np.ndarray]:
    # Import locally to avoid a module-level cycle: synthesizer can emit a
    # trace, while this module can optionally rerun the same synthesis with
    # one source family removed.
    from .synthesizer import synthesize_timeline

    outputs: dict[str, np.ndarray] = {}
    for name, gains in ABLATION_SPECS.items():
        outputs[name] = synthesize_timeline(
            timeline,
            profile,
            seed=seed,
            component_gains=gains,
            compute_backend=compute_backend,
            precision_guard=precision_guard,
            normalization_gain_override=normalization_gain,
        )
    return outputs


def _attach_counterfactual_results(
    report: dict[str, Any],
    ablations: Mapping[str, np.ndarray],
    sample_rate: int,
) -> None:
    radius = max(8, int(round(0.006 * sample_rate)))
    label_to_type = {
        "no_voiced": "voiced_source_transient",
        "no_aspiration": "aspiration_transient",
        "no_frication": "frication_transient",
        "no_burst": "burst_transient",
    }
    for event in report.get("events", []):
        center = int(event["sample"])
        base_peak = max(float(event["sample_peak"]), 1e-12)
        base_jump = max(float(event["max_sample_jump"]), 1e-12)
        comparisons: dict[str, dict[str, float]] = {}
        for name, audio in ablations.items():
            array = np.asarray(audio, dtype=np.float64).reshape(-1)
            metrics = _local_metrics(array, center, radius)
            peak_reduction_db = 20.0 * math.log10(base_peak / max(metrics["peak"], 1e-12))
            jump_reduction_db = 20.0 * math.log10(base_jump / max(metrics["max_jump"], 1e-12))
            comparisons[name] = {
                "peak": metrics["peak"],
                "max_jump": metrics["max_jump"],
                "rms": metrics["rms"],
                "peak_reduction_db": peak_reduction_db,
                "jump_reduction_db": jump_reduction_db,
            }
        event["counterfactual_ablations"] = comparisons
        if comparisons:
            primary_name, primary = max(
                comparisons.items(),
                key=lambda item: max(
                    item[1]["jump_reduction_db"],
                    item[1]["peak_reduction_db"],
                ),
            )
            evidence_db = max(primary["jump_reduction_db"], primary["peak_reduction_db"])
            event["counterfactual_primary"] = primary_name
            event["counterfactual_evidence_db"] = evidence_db
            if evidence_db >= 2.0:
                event["reasons"].append(
                    f"反事實重跑移除 {primary_name} 後，事件峰值／跳變最多下降 {evidence_db:.2f} dB"
                )
                if primary_name != "no_voiced" or evidence_db >= 6.0:
                    event["suspected_type"] = label_to_type[primary_name]



def _format_number(value: Any, digits: int = 4) -> str:
    if value is None:
        return "-"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    if not math.isfinite(number):
        return str(number)
    return f"{number:.{digits}f}"


def _write_event_csv(
    path: Path,
    trace: Mapping[str, np.ndarray],
    sample_rate: int,
    center: int,
    radius: int,
) -> None:
    final = np.asarray(trace["final_audio"]).reshape(-1)
    start = max(0, center - radius)
    stop = min(final.size, center + radius + 1)
    names = [name for name in PLOT_TRACES if name in trace]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "time_s", *names])
        for index in range(start, stop):
            writer.writerow(
                [
                    index,
                    f"{index / sample_rate:.9f}",
                    *[f"{float(np.asarray(trace[name]).reshape(-1)[index]):.9g}" for name in names],
                ]
            )


def _write_precision_event_csv(
    path: Path,
    trace: Mapping[str, np.ndarray],
    precision_arrays: Mapping[str, np.ndarray],
    sample_rate: int,
    center: int,
    radius: int,
) -> None:
    final = np.asarray(trace["final_audio"], dtype=np.float64).reshape(-1)
    start = max(0, center - radius)
    stop = min(final.size, center + radius + 1)
    trace_names = [name for name in LEDGER_TRACKS if name in trace]
    audit_names = [name for name in precision_arrays if np.asarray(precision_arrays[name]).size == final.size]
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["sample", "time_s", *trace_names, *audit_names])
        for index in range(start, stop):
            row = [index, f"{index / sample_rate:.12f}"]
            for name in trace_names:
                values = np.asarray(trace[name], dtype=np.float64).reshape(-1)
                row.append(f"{float(values[index]):.17g}")
            for name in audit_names:
                values = np.asarray(precision_arrays[name], dtype=np.float64).reshape(-1)
                row.append(f"{float(values[index]):.17g}")
            writer.writerow(row)


def _precision_text_report(report: Mapping[str, Any]) -> str:
    summary = report["summary"]
    backend = report["backend"]
    lines = [
        "DigiVocal GPU/CPU 精算報告",
        "=" * 36,
        f"backend_requested: {backend.get('requested')}",
        f"backend_active: {backend.get('active')}",
        f"gpu_available: {backend.get('gpu_available')}",
        f"device_name: {backend.get('device_name')}",
        f"cupy_version: {backend.get('cupy_version')}",
        f"fallback_reason: {backend.get('reason')}",
        f"dtype: {report['precision']['internal_dtype']}",
        f"elapsed_s: {report['elapsed_s']:.6f}",
        f"sample_peak: {summary['sample_peak']:.12g}",
        f"true_peak_4x: {summary['true_peak_4x']:.12g}",
        f"rms: {summary['rms']:.12g}",
        f"max_first_difference: {summary['max_first_difference']:.12g}",
        f"max_second_difference: {summary['max_second_difference']:.12g}",
        f"max_jump_ratio: {summary['max_jump_ratio']:.12g}",
        f"max_crest_factor: {summary['max_crest_factor']:.12g}",
        f"source_sum_residual_max_abs: {summary['source_sum_residual_max_abs']:.12g}",
        f"event_count: {summary['event_count']}",
        "",
    ]
    for event in report.get("events", []):
        lines.extend([
            f"[{event['event_id']:03d}] {event['time_s']:.12f} s sample={event['sample']}",
            f"text={event.get('text')!r} pinyin={event.get('pinyin')!r}",
            f"risk={event['risk_score']:.8g} jump_ratio={event['sample_jump_ratio']:.8g} crest={event['crest_factor']:.8g}",
            f"dominant_component={event['dominant_component_by_local_rms']}",
            f"source_sum_math_fsum={event['source_sum_math_fsum']:.17g}",
            f"source_sum_pre_stft={event['source_sum_pre_stft']:.17g}",
            f"source_sum_residual={event['source_sum_residual']:.17g}",
            "component_values=" + json.dumps(event["component_values"], ensure_ascii=False),
            "",
        ])
    return "\n".join(lines)


def _svg_trace(
    trace: Mapping[str, np.ndarray],
    sample_rate: int,
    center: int,
    radius: int,
) -> str:
    final = np.asarray(trace["final_audio"]).reshape(-1)
    start = max(0, center - radius)
    stop = min(final.size, center + radius + 1)
    names = [name for name in PLOT_TRACES if name in trace]
    if not names or stop - start < 2:
        return ""
    width = 1000
    row_height = 88
    label_width = 145
    plot_width = width - label_width - 20
    height = row_height * len(names)
    parts = [f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="debug traces">']
    for row, name in enumerate(names):
        values = np.asarray(trace[name], dtype=np.float64).reshape(-1)[start:stop]
        if values.size > 600:
            indices = np.linspace(0, values.size - 1, 600).astype(int)
            values = values[indices]
        peak = float(np.max(np.abs(values))) or 1.0
        x = np.linspace(label_width, label_width + plot_width, values.size)
        center_y = row * row_height + row_height / 2.0
        y = center_y - values / peak * (row_height * 0.36)
        points = " ".join(f"{xv:.2f},{yv:.2f}" for xv, yv in zip(x, y, strict=False))
        parts.append(f'<rect x="0" y="{row * row_height}" width="{width}" height="{row_height}" fill="{("#f7f9fc" if row % 2 == 0 else "#eef3f8")}"/>')
        parts.append(f'<line x1="{label_width}" y1="{center_y:.2f}" x2="{label_width + plot_width}" y2="{center_y:.2f}" stroke="#bcc7d3" stroke-width="1"/>')
        parts.append(f'<text x="8" y="{center_y + 5:.2f}" font-size="14" fill="#263746">{html.escape(name)}</text>')
        parts.append(f'<polyline points="{points}" fill="none" stroke="#146c94" stroke-width="1.5" vector-effect="non-scaling-stroke"/>')
        marker_x = label_width + plot_width * ((center - start) / max(1, stop - start - 1))
        parts.append(f'<line x1="{marker_x:.2f}" y1="{row * row_height + 4}" x2="{marker_x:.2f}" y2="{(row + 1) * row_height - 4}" stroke="#b42318" stroke-width="1"/>')
    parts.append("</svg>")
    return "".join(parts)


def _html_report(report: Mapping[str, Any], trace: Mapping[str, np.ndarray]) -> str:
    analysis = report["analysis"]
    sample_rate = int(report["sample_rate"])
    precision = report.get("precision_audit", {})
    precision_summary = precision.get("summary", {})
    precision_backend = precision.get("backend", {})
    guard = report.get("precision_guard", {})
    optimizer = guard.get("quality_optimizer", {}) if isinstance(guard, Mapping) else {}
    optimizer_backend = optimizer.get("backend", {}) if isinstance(optimizer, Mapping) else {}
    emergency = guard.get("emergency_guard", {}) if isinstance(guard, Mapping) else {}
    component_reports = optimizer.get("component_reports", {}) if isinstance(optimizer, Mapping) else {}
    optimized_components = sum(
        1 for item in component_reports.values()
        if isinstance(item, Mapping) and int(item.get("changed_samples", 0)) > 0
    )
    precision_html = f"""
    <section class="summary">
      <h2>GPU／CPU 精算品質最佳化</h2>
      <table><tbody>
        <tr><th>要求後端</th><td>{html.escape(str(optimizer_backend.get('requested', precision_backend.get('requested', '-'))))}</td></tr>
        <tr><th>實際後端</th><td>{html.escape(str(optimizer_backend.get('active', precision_backend.get('active', '-'))))}</td></tr>
        <tr><th>GPU</th><td>{html.escape(str(optimizer_backend.get('device_name') or precision_backend.get('device_name') or '-'))}</td></tr>
        <tr><th>內部精度</th><td>{html.escape(str(precision.get('precision', {}).get('internal_dtype', '-')))}</td></tr>
        <tr><th>主動最佳化聲源</th><td>{optimized_components} / {len(component_reports)}</td></tr>
        <tr><th>最佳化 sample 計數</th><td>{optimizer.get('changed_component_samples', 0)}</td></tr>
        <tr><th>source sum 最大殘差</th><td>{_format_number(precision_summary.get('source_sum_residual_max_abs'), 12)}</td></tr>
        <tr><th>最大 jump ratio</th><td>{_format_number(precision_summary.get('max_jump_ratio'), 5)}</td></tr>
        <tr><th>精算事件</th><td>{precision_summary.get('event_count', 0)}</td></tr>
        <tr><th>緊急守門修正 samples</th><td>{emergency.get('corrected_samples', 0)}</td></tr>
      </tbody></table>
      <p class="links"><a href="precision_report.json">精算 JSON</a>｜
      <a href="precision_report.txt">精算文字</a>｜
      <a href="precision_trace.npz">全 sample 精算陣列</a></p>
    </section>
    """
    event_sections: list[str] = []
    radius = max(32, int(round(0.025 * sample_rate)))
    for event in report["events"]:
        reasons = "".join(f"<li>{html.escape(str(reason))}</li>" for reason in event["reasons"])
        controls = " ".join(
            f"<code>{html.escape(name)}={_format_number(value, 3)}</code>"
            for name, value in event["controls"].items()
        )
        components = "".join(
            f"<tr><td>{html.escape(name)}</td><td>{_format_number(value, 6)}</td>"
            f"<td>{_format_number(event['component_metrics'][name]['rms'], 6)}</td></tr>"
            for name, value in event["component_values_at_sample"].items()
        )
        counterfactual_rows = "".join(
            f"<tr><td>{html.escape(name)}</td>"
            f"<td>{_format_number(values['peak_reduction_db'], 2)} dB</td>"
            f"<td>{_format_number(values['jump_reduction_db'], 2)} dB</td></tr>"
            for name, values in event.get("counterfactual_ablations", {}).items()
        )
        counterfactual_html = (
            "<h3>反事實消融重跑</h3>"
            "<table><thead><tr><th>移除來源</th><th>峰值下降</th><th>跳變下降</th></tr></thead>"
            f"<tbody>{counterfactual_rows}</tbody></table>"
            if counterfactual_rows
            else ""
        )
        event_sections.append(
            f"""
            <section class="event">
              <h2>事件 {event['event_id']:03d}｜{event['time_s']:.6f} 秒｜{html.escape(event['suspected_type'])}</h2>
              <p><strong>對應文字：</strong>{html.escape(str(event.get('text') or '-'))}　
              <strong>拼音：</strong>{html.escape(str(event.get('pinyin') or '-'))}　
              <strong>sample：</strong>{event['sample']}</p>
              <div class="metrics">
                <span>sample peak {_format_number(event['sample_peak_dbfs'], 2)} dBFS</span>
                <span>true peak {_format_number(event['true_peak_4x_dbtp'], 2)} dBTP</span>
                <span>jump {_format_number(event['max_sample_jump'], 5)}</span>
                <span>crest {_format_number(event['crest_factor_db'], 2)} dB</span>
                <span>HF ratio {_format_number(event['high_frequency_ratio'], 3)}</span>
                <span>score {_format_number(event['score'], 2)}</span>
              </div>
              <p>{controls}</p>
              <ul>{reasons}</ul>
              {_svg_trace(trace, sample_rate, int(event['sample']), radius)}
              <details><summary>來源在事件 sample 的四則運算分解</summary>
                <p><code>source_sum = voiced_source + aspiration_source + frication_source + aspiration_noise_source + burst_source</code></p>
                <table><thead><tr><th>來源</th><th>當下值</th><th>事件窗 RMS</th></tr></thead><tbody>{components}</tbody></table>
                <p><strong>分量相加：</strong>{_format_number(event['source_sum_at_sample'], 8)}</p>
                <p><strong>局部最大跳變階段：</strong>{html.escape(event['largest_jump_stage'])}</p>
                {counterfactual_html}
              </details>
              <p class="links">檔案：<a href="events/event_{event['event_id']:03d}.wav">事件 WAV</a>｜
              <a href="events/event_{event['event_id']:03d}_window.csv">事件 CSV</a></p>
            </section>
            """
        )

    return f"""<!doctype html>
<html lang="zh-Hant">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>DigiVocal 爆音因果 Debug</title>
<style>
body{{font-family:"Microsoft JhengHei UI",system-ui,sans-serif;margin:0;background:#f4f6f8;color:#172b3a}}
main{{max-width:1180px;margin:24px auto;padding:0 18px}}
h1{{margin-bottom:6px}} .lead{{color:#536474}}
.summary,.event{{background:white;border:1px solid #d9e1e8;border-radius:12px;padding:18px;margin:16px 0;box-shadow:0 2px 8px rgba(19,38,55,.05)}}
.metrics{{display:flex;flex-wrap:wrap;gap:8px}} .metrics span,code{{background:#edf3f8;border-radius:6px;padding:4px 7px}}
table{{border-collapse:collapse;width:100%}} th,td{{border-bottom:1px solid #e0e6ec;text-align:left;padding:7px}}
svg{{width:100%;height:auto;border:1px solid #dbe3ea;border-radius:8px;background:#fff}}
a{{color:#0b6694}} .links{{font-size:.92rem;color:#536474}}
</style>
</head>
<body><main>
<h1>DigiVocal 爆音因果 Debug</h1>
<p class="lead">事件是統計異常候選；最終因果仍應配合聆聽、trace 與消融重跑確認。</p>
<section class="summary">
<h2>整體</h2>
<table><tbody>
<tr><th>取樣率</th><td>{report['sample_rate']} Hz</td></tr>
<tr><th>時間</th><td>{_format_number(report['duration_s'], 3)} 秒</td></tr>
<tr><th>Sample peak</th><td>{_format_number(analysis['sample_peak_dbfs'], 2)} dBFS</td></tr>
<tr><th>4x true peak estimate</th><td>{_format_number(analysis['true_peak_4x_dbtp'], 2)} dBTP</td></tr>
<tr><th>RMS</th><td>{_format_number(analysis['rms_dbfs'], 2)} dBFS</td></tr>
<tr><th>最大 sample jump</th><td>{_format_number(analysis['max_sample_jump'], 6)}</td></tr>
<tr><th>候選事件</th><td>{analysis['event_count']}</td></tr>
</tbody></table>
<p>{html.escape(analysis['true_peak_note'])}</p>
</section>
{precision_html}
{''.join(event_sections) if event_sections else '<section class="summary"><h2>沒有超過門檻的候選事件</h2><p>仍可從 debug_trace.npz 查看所有中間訊號。</p></section>'}
</main></body></html>"""


def _text_report(report: Mapping[str, Any]) -> str:
    analysis = report["analysis"]
    lines = [
        "DigiVocal 爆音因果 Debug",
        "=" * 34,
        f"sample_rate: {report['sample_rate']} Hz",
        f"duration: {report['duration_s']:.6f} s",
        f"sample_peak: {analysis['sample_peak_dbfs']:.3f} dBFS",
        f"true_peak_4x_estimate: {analysis['true_peak_4x_dbtp']:.3f} dBTP",
        f"rms: {analysis['rms_dbfs']:.3f} dBFS",
        f"max_sample_jump: {analysis['max_sample_jump']:.9g}",
        f"event_count: {analysis['event_count']}",
        "",
    ]
    for event in report["events"]:
        lines.extend(
            [
                f"[{event['event_id']:03d}] {event['time_s']:.9f} s | sample={event['sample']}",
                f"text={event.get('text')!r} pinyin={event.get('pinyin')!r}",
                f"type={event['suspected_type']} score={event['score']:.3f}",
                f"sample_peak={event['sample_peak_dbfs']:.3f} dBFS true_peak={event['true_peak_4x_dbtp']:.3f} dBTP",
                f"jump={event['max_sample_jump']:.9g} crest={event['crest_factor_db']:.3f} dB hf_ratio={event['high_frequency_ratio']:.5f}",
                "reasons=" + "；".join(event["reasons"]),
                "controls=" + json.dumps(event["controls"], ensure_ascii=False),
                "component_values=" + json.dumps(event["component_values_at_sample"], ensure_ascii=False),
                f"source_sum={event['source_sum_at_sample']:.9g}",
                f"largest_jump_stage={event['largest_jump_stage']}",
                "counterfactual_primary=" + str(event.get("counterfactual_primary", "")),
                "counterfactual=" + json.dumps(
                    event.get("counterfactual_ablations", {}),
                    ensure_ascii=False,
                ),
                "",
            ]
        )
    return "\n".join(lines)


def _write_float_wav(path: Path, values: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    wavfile.write(path, int(sample_rate), np.asarray(values, dtype=np.float32))


def _normalized_stage_copy(values: np.ndarray, peak_target: float = 0.82) -> tuple[np.ndarray, float]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    peak = float(np.max(np.abs(array))) if array.size else 0.0
    scale = float(peak_target / peak) if peak > 1e-12 else 1.0
    return array * scale, scale


def _write_source_stage_bundle(
    destination: Path,
    trace: Mapping[str, np.ndarray],
    sample_rate: int,
    profile: VoiceProfile | None,
) -> list[dict[str, Any]]:
    """Write exact float WAVs and normalized listening copies for each source stage."""

    stage_dir = destination / "source_stages"
    if stage_dir.exists():
        shutil.rmtree(stage_dir)
    stage_dir.mkdir(parents=True, exist_ok=True)
    manifest: list[dict[str, Any]] = []

    def add_file(stem: str, values: np.ndarray, description: str) -> None:
        array = np.asarray(values, dtype=np.float64).reshape(-1)
        if array.size == 0 or not np.all(np.isfinite(array)):
            return
        exact_name = f"{stem}.float32.wav"
        listen_name = f"{stem}.listen.wav"
        _write_float_wav(stage_dir / exact_name, array, sample_rate)
        listen, scale = _normalized_stage_copy(array)
        write_wav(stage_dir / listen_name, listen, sample_rate)
        manifest.append(
            {
                "stage": stem,
                "description": description,
                "exact_float_wav": exact_name,
                "normalized_listen_wav": listen_name,
                "sample_count": int(array.size),
                "peak": float(np.max(np.abs(array))),
                "rms": float(np.sqrt(np.mean(np.square(array)) + 1e-30)),
                "listen_scale": scale,
            }
        )

    if profile is not None:
        period_samples = max(16, int(round(sample_rate / max(profile.base_f0_hz, 1.0))))
        native = np.asarray(profile.native_cycle, dtype=np.float64)
        if native.size >= 8:
            add_file(
                "00a_profile_native_cycle",
                native,
                "Native cycle preserved from Digital Throat; one period at the extraction sample count",
            )
        primary = np.asarray(profile.wave_points, dtype=np.float64)
        if primary.size >= 8:
            primary_period = signal.resample(primary, period_samples)
            add_file(
                "00b_profile_primary_cycle_at_base_f0",
                primary_period,
                "Primary stored source table resampled to one real period at base F0",
            )
        for vowel in ("a", "e", "i", "o", "u"):
            bank_name = f"vowel_{vowel}"
            if bank_name in profile.wave_bank_names:
                index = profile.wave_bank_names.index(bank_name)
                cycle = signal.resample(
                    np.asarray(profile.wave_bank[index], dtype=np.float64),
                    period_samples,
                )
                add_file(
                    f"00c_profile_{bank_name}_cycle",
                    cycle,
                    f"Digital Throat {bank_name} source table at base F0",
                )

    for stem, trace_name, description in SOURCE_STAGE_TRACES:
        if trace_name in trace:
            add_file(stem, np.asarray(trace[trace_name]), description)

    (stage_dir / "source_stage_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (stage_dir / "README.txt").write_text(
        "float32.wav 保留原始數值，適合你的波形展開工具；listen.wav 只為方便播放，會各自正規化，因此不可拿來比較絕對音量。\n"
        "請依 00 → 07 順序觀察：哪一層第一次失去目標週期細節，那一層就是主要變形來源。\n",
        encoding="utf-8",
    )
    return manifest


def write_debug_bundle(
    output_dir: str | Path,
    trace: Mapping[str, np.ndarray],
    sample_rate: int,
    *,
    timeline: GestureTimeline | None = None,
    profile: VoiceProfile | None = None,
    seed: int | None = None,
    max_events: int = 24,
    run_ablations: bool = False,
    compute_backend: str = "auto",
    precision_guard: bool = True,
    precision_guard_report: Mapping[str, Any] | None = None,
) -> tuple[Path, dict[str, Any]]:
    destination = Path(output_dir)
    destination.mkdir(parents=True, exist_ok=True)
    events_dir = destination / "events"
    precision_events_dir = destination / "precision_events"
    for directory in (events_dir, precision_events_dir, destination / "ablations"):
        if directory.exists():
            shutil.rmtree(directory)
    events_dir.mkdir(parents=True, exist_ok=True)
    precision_events_dir.mkdir(parents=True, exist_ok=True)

    report = analyze_trace(trace, sample_rate, timeline, max_events=max_events)
    causal_trace_available = all(name in trace for name in TRACE_COMPONENTS)
    report["analysis_mode"] = "full_trace" if causal_trace_available else "wav_only"
    report["causal_trace_available"] = causal_trace_available
    precision_report, precision_arrays = build_precision_audit(
        trace,
        sample_rate,
        timeline,
        compute_backend=compute_backend,
        max_events=max_events,
        candidate_samples=[int(event["sample"]) for event in report.get("events", [])],
    )
    report["precision_audit"] = precision_report
    if precision_guard_report is not None:
        report["precision_guard"] = dict(precision_guard_report)

    ablations: dict[str, np.ndarray] = {}
    if run_ablations:
        if timeline is None or profile is None or seed is None:
            raise DebugAnalysisError(
                "run_ablations requires timeline, profile and seed"
            )
        normalization_gain = None
        if "normalization_gain" in trace:
            values = np.asarray(trace["normalization_gain"], dtype=np.float64).reshape(-1)
            if values.size:
                normalization_gain = float(values[0])
        ablations = _generate_counterfactual_ablations(
            timeline,
            profile,
            seed,
            normalization_gain=normalization_gain,
            compute_backend=compute_backend,
            precision_guard=precision_guard,
        )
        _attach_counterfactual_results(report, ablations, sample_rate)
        report["counterfactual_note"] = (
            "Each no_* result re-synthesizes with the same random seed while removing one source family."
        )

    if timeline is not None:
        report["timeline_metadata"] = timeline.metadata
    if profile is not None:
        report["voice_profile"] = profile.to_dict()
    if seed is not None:
        report["seed"] = int(seed)

    trace_payload: dict[str, np.ndarray] = {}
    for name, values in trace.items():
        array = np.asarray(values)
        if np.issubdtype(array.dtype, np.floating):
            array = array.astype(np.float64)
        trace_payload[name] = array
    trace_payload["sample_rate"] = np.asarray([sample_rate], dtype=np.int32)
    np.savez_compressed(destination / "debug_trace.npz", **trace_payload)
    precision_payload = {
        name: np.asarray(values, dtype=np.float64)
        for name, values in precision_arrays.items()
    }
    precision_payload["sample_rate"] = np.asarray([sample_rate], dtype=np.int32)
    np.savez_compressed(destination / "precision_trace.npz", **precision_payload)
    (destination / "precision_report.json").write_text(
        json.dumps(precision_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (destination / "precision_report.txt").write_text(
        _precision_text_report(precision_report),
        encoding="utf-8",
    )

    if timeline is not None:
        (destination / "gestures_debug.json").write_text(
            json.dumps(timeline.to_dict(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    audio = np.asarray(trace["final_audio"], dtype=np.float64).reshape(-1)
    write_wav(destination / "debug_audio.wav", audio, sample_rate)
    source_stage_manifest = _write_source_stage_bundle(
        destination,
        trace,
        sample_rate,
        profile,
    )
    report["source_stage_files"] = source_stage_manifest
    if ablations:
        ablation_dir = destination / "ablations"
        ablation_dir.mkdir(parents=True, exist_ok=True)
        for name, values in ablations.items():
            write_wav(ablation_dir / f"{name}.wav", values, sample_rate)

    clip_radius = max(128, int(round(0.090 * sample_rate)))
    csv_radius = max(64, int(round(0.020 * sample_rate)))
    for event in report["events"]:
        center = int(event["sample"])
        start = max(0, center - clip_radius)
        stop = min(audio.size, center + clip_radius + 1)
        write_wav(events_dir / f"event_{event['event_id']:03d}.wav", audio[start:stop], sample_rate)
        _write_event_csv(
            events_dir / f"event_{event['event_id']:03d}_window.csv",
            trace,
            sample_rate,
            center,
            csv_radius,
        )

    precision_radius = max(64, int(round(0.020 * sample_rate)))
    for event in precision_report.get("events", []):
        _write_precision_event_csv(
            precision_events_dir / f"precision_event_{event['event_id']:03d}_ledger.csv",
            trace,
            precision_arrays,
            sample_rate,
            int(event["sample"]),
            precision_radius,
        )

    report_path = destination / "debug_report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (destination / "debug_report.txt").write_text(_text_report(report), encoding="utf-8")
    html_path = destination / "debug_report.html"
    html_path.write_text(_html_report(report, trace), encoding="utf-8")
    return html_path, report


def debug_wav_file(
    wav_path: str | Path,
    output_dir: str | Path,
    *,
    max_events: int = 24,
    compute_backend: str = "auto",
) -> tuple[Path, dict[str, Any]]:
    sample_rate, audio = load_wav_float(wav_path)
    trace = {"final_audio": audio.astype(np.float64)}
    html_path, report = write_debug_bundle(
        output_dir,
        trace,
        sample_rate,
        max_events=max_events,
        compute_backend=compute_backend,
        precision_guard=False,
    )
    report["source_wav"] = str(Path(wav_path).resolve())
    (Path(output_dir) / "debug_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return html_path, report

from __future__ import annotations

import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
from scipy import signal as scipy_signal
from scipy.ndimage import (
    gaussian_filter1d as scipy_gaussian_filter1d,
    maximum_filter1d as scipy_maximum_filter1d,
    minimum_filter1d as scipy_minimum_filter1d,
    uniform_filter1d as scipy_uniform_filter1d,
)

from .models import GestureTimeline


COMPONENT_NAMES = (
    "voiced_source",
    "aspiration_source",
    "frication_source",
    "aspiration_noise_source",
    "burst_source",
)

LEDGER_TRACKS = (
    "raw_aspiration_noise",
    "aspiration_highpass_noise",
    "raw_frication_noise",
    "filtered_frication_noise",
    "raw_burst_noise",
    "burst_envelope",
    "periodic",
    "shimmer_gain",
    "aspiration_gain_factor",
    "frication_pressure_factor",
    "frication_output_gain",
    "burst_output_gain",
    "amplitude",
    "voicing",
    "frication",
    "aspiration",
    "pressure",
    "burst",
    "f0_hz",
    "noise_center_hz",
    "noise_bandwidth_hz",
    "voiced_source_raw",
    "aspiration_source_raw",
    "frication_source_raw",
    "aspiration_noise_source_raw",
    "burst_source_raw",
    "optimizer_voiced_gain",
    "optimizer_aspiration_gain",
    "optimizer_frication_gain",
    "optimizer_aspiration_noise_gain",
    "optimizer_burst_gain",
    "optimizer_voiced_local_rms",
    "optimizer_aspiration_local_rms",
    "optimizer_frication_local_rms",
    "optimizer_aspiration_noise_local_rms",
    "optimizer_burst_local_rms",
    "optimizer_voiced_target_rms",
    "optimizer_aspiration_target_rms",
    "optimizer_frication_target_rms",
    "optimizer_aspiration_noise_target_rms",
    "optimizer_burst_target_rms",
    "optimizer_voiced_crest",
    "optimizer_aspiration_crest",
    "optimizer_frication_crest",
    "optimizer_aspiration_noise_crest",
    "optimizer_burst_crest",
    *COMPONENT_NAMES,
    "voiced_mix",
    "noise_source",
    "source_sum_pre_stft",
    "istft_raw",
    "level_balanced",
    "post_filter",
    "precision_guard_gain",
    "precision_guarded",
    "tanh_input",
    "post_tanh",
    "normalized",
    "final_audio",
)


@dataclass(slots=True)
class BackendInfo:
    requested: str
    active: str
    gpu_available: bool
    device_name: str | None = None
    cupy_version: str | None = None
    reason: str | None = None
    float_dtype: str = "float64"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class BackendRuntime:
    info: BackendInfo
    xp: Any
    ndimage: Any
    gpu_signal: Any | None
    cupy: Any | None

    def to_numpy(self, values: Any) -> np.ndarray:
        if self.info.active == "gpu" and self.cupy is not None:
            return self.cupy.asnumpy(values)
        return np.asarray(values)

    def synchronize(self) -> None:
        if self.info.active == "gpu" and self.cupy is not None:
            self.cupy.cuda.get_current_stream().synchronize()


def _gpu_runtime(requested: str) -> BackendRuntime | None:
    try:
        import cupy as cp
        from cupyx.scipy import ndimage as cndimage
        from cupyx.scipy import signal as csignal
    except Exception as exc:
        return BackendRuntime(
            BackendInfo(
                requested=requested,
                active="cpu",
                gpu_available=False,
                reason=f"CuPy import failed: {exc}",
            ),
            np,
            _CpuNdimage,
            None,
            None,
        )

    try:
        count = int(cp.cuda.runtime.getDeviceCount())
        if count <= 0:
            raise RuntimeError("no CUDA device reported")
        device_id = int(cp.cuda.Device().id)
        props = cp.cuda.runtime.getDeviceProperties(device_id)
        raw_name = props.get("name", b"CUDA GPU")
        if isinstance(raw_name, bytes):
            name = raw_name.decode("utf-8", errors="replace")
        else:
            name = str(raw_name)
        # A tiny float64 calculation confirms that the selected device and
        # runtime can execute kernels before the long audit starts.
        probe = cp.asarray([1.0, 2.0, 3.0], dtype=cp.float64)
        value = float(cp.asnumpy(cp.sum(probe * probe)))
        if not math.isclose(value, 14.0, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(f"float64 CUDA probe returned {value!r}")
        return BackendRuntime(
            BackendInfo(
                requested=requested,
                active="gpu",
                gpu_available=True,
                device_name=name,
                cupy_version=str(cp.__version__),
            ),
            cp,
            cndimage,
            csignal,
            cp,
        )
    except Exception as exc:
        return BackendRuntime(
            BackendInfo(
                requested=requested,
                active="cpu",
                gpu_available=False,
                cupy_version=str(getattr(cp, "__version__", "unknown")),
                reason=f"CUDA probe failed: {exc}",
            ),
            np,
            _CpuNdimage,
            None,
            None,
        )


class _CpuNdimage:
    uniform_filter1d = staticmethod(scipy_uniform_filter1d)
    maximum_filter1d = staticmethod(scipy_maximum_filter1d)
    minimum_filter1d = staticmethod(scipy_minimum_filter1d)
    gaussian_filter1d = staticmethod(scipy_gaussian_filter1d)


def resolve_backend(
    requested: str = "auto",
    *,
    sample_count: int = 0,
    gpu_min_samples: int = 8192,
) -> BackendRuntime:
    requested = str(requested or "auto").strip().lower()
    if requested not in {"auto", "cpu", "gpu"}:
        raise ValueError("compute backend must be auto, cpu or gpu")
    if requested == "cpu":
        return BackendRuntime(
            BackendInfo(
                requested=requested,
                active="cpu",
                gpu_available=False,
                reason="CPU explicitly selected",
            ),
            np,
            _CpuNdimage,
            None,
            None,
        )

    runtime = _gpu_runtime(requested)
    assert runtime is not None
    if runtime.info.active != "gpu":
        return runtime
    if requested == "auto" and sample_count and sample_count < gpu_min_samples:
        return BackendRuntime(
            BackendInfo(
                requested=requested,
                active="cpu",
                gpu_available=True,
                device_name=runtime.info.device_name,
                cupy_version=runtime.info.cupy_version,
                reason=f"array shorter than GPU threshold ({sample_count} < {gpu_min_samples})",
            ),
            np,
            _CpuNdimage,
            None,
            None,
        )
    return runtime


def backend_status(requested: str = "auto") -> dict[str, Any]:
    runtime = resolve_backend(requested, sample_count=100_000, gpu_min_samples=0)
    return runtime.info.to_dict()


def _safe_numpy(values: Any, length: int, default: float = 0.0) -> np.ndarray:
    if values is None:
        return np.full(length, default, dtype=np.float64)
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size == length:
        return array
    if array.size == 0:
        return np.full(length, default, dtype=np.float64)
    if array.size == 1:
        return np.full(length, float(array[0]), dtype=np.float64)
    source = np.linspace(0.0, 1.0, array.size)
    target = np.linspace(0.0, 1.0, length)
    return np.interp(target, source, array)


def _xp_diff_abs(xp: Any, values: Any) -> tuple[Any, Any]:
    first = xp.zeros_like(values)
    if int(values.size) > 1:
        first[1:] = xp.abs(values[1:] - values[:-1])
    second = xp.zeros_like(values)
    if int(values.size) > 2:
        second[2:] = xp.abs(values[2:] - 2.0 * values[1:-1] + values[:-2])
    return first, second


def _fixed_source_sum_numpy(trace: Mapping[str, Any], length: int) -> np.ndarray:
    # Fixed left-to-right order makes the bookkeeping deterministic. Event
    # samples are also checked with math.fsum for compensated summation.
    total = np.zeros(length, dtype=np.float64)
    for name in COMPONENT_NAMES:
        total = total + _safe_numpy(trace.get(name), length)
    return total


def _timeline_span(timeline: GestureTimeline | None, time_s: float) -> dict[str, Any]:
    if timeline is None:
        return {
            "text": None,
            "pinyin": None,
            "initial": None,
            "final": None,
            "segment_start_s": None,
            "segment_end_s": None,
        }
    for span in timeline.spans:
        if span.start_s <= time_s < span.end_s or (
            span is timeline.spans[-1] and math.isclose(time_s, span.end_s, abs_tol=1e-9)
        ):
            return {
                "text": span.text,
                "pinyin": span.pinyin,
                "initial": span.initial,
                "final": span.final,
                "segment_start_s": span.start_s,
                "segment_end_s": span.end_s,
            }
    return {
        "text": None,
        "pinyin": None,
        "initial": None,
        "final": None,
        "segment_start_s": None,
        "segment_end_s": None,
    }


def _consolidate_candidates(score: np.ndarray, sample_rate: int, max_events: int) -> list[int]:
    finite = np.asarray(score, dtype=np.float64)
    if finite.size == 0 or not np.any(np.isfinite(finite)):
        return []
    threshold = max(1.0, float(np.percentile(finite[np.isfinite(finite)], 99.85)))
    candidates = np.flatnonzero(finite >= threshold)
    if candidates.size == 0:
        return []
    order = candidates[np.argsort(finite[candidates])[::-1]]
    distance = max(1, int(round(0.035 * sample_rate)))
    accepted: list[int] = []
    for index in order:
        index = int(index)
        if all(abs(index - old) >= distance for old in accepted):
            accepted.append(index)
            if len(accepted) >= max_events:
                break
    accepted.sort()
    return accepted


def _event_component_metrics(
    trace: Mapping[str, Any],
    center: int,
    radius: int,
    length: int,
) -> tuple[dict[str, float], dict[str, float], dict[str, float]]:
    start = max(0, center - radius)
    stop = min(length, center + radius + 1)
    values: dict[str, float] = {}
    rms: dict[str, float] = {}
    peak: dict[str, float] = {}
    for name in COMPONENT_NAMES:
        array = _safe_numpy(trace.get(name), length)
        values[name] = float(array[center])
        segment = array[start:stop]
        rms[name] = float(np.sqrt(np.mean(np.square(segment)) + 1e-30))
        peak[name] = float(np.max(np.abs(segment))) if segment.size else 0.0
    return values, rms, peak


def _formula_audit_at_sample(
    trace: Mapping[str, Any],
    index: int,
    length: int,
    component_values: Mapping[str, float],
) -> dict[str, dict[str, float | str | None]]:
    def value(name: str, default: float = 0.0) -> float:
        return float(_safe_numpy(trace.get(name), length, default)[index])

    specs: dict[str, tuple[tuple[str, ...], str, Any]] = {
        "voiced_source": (
            ("periodic", "amplitude", "voicing", "shimmer_gain", "pressure"),
            "periodic * amplitude * voicing * shimmer_gain * (0.35 + 0.65*pressure)",
            lambda: value("periodic")
            * value("amplitude")
            * value("voicing")
            * value("shimmer_gain", 1.0)
            * (0.35 + 0.65 * value("pressure")),
        ),
        "aspiration_source": (
            ("aspiration_highpass_noise", "amplitude", "aspiration", "aspiration_gain_factor"),
            "aspiration_highpass_noise * amplitude * aspiration * aspiration_gain_factor",
            lambda: value("aspiration_highpass_noise")
            * value("amplitude")
            * value("aspiration")
            * value("aspiration_gain_factor"),
        ),
        "frication_source": (
            ("filtered_frication_noise", "frication_pressure_factor", "frication", "frication_output_gain"),
            "filtered_frication_noise * frication_pressure_factor * frication * frication_output_gain",
            lambda: value("filtered_frication_noise")
            * value("frication_pressure_factor")
            * value("frication")
            * value("frication_output_gain", 0.74),
        ),
        "aspiration_noise_source": (
            ("aspiration_highpass_noise", "amplitude", "aspiration"),
            "aspiration_highpass_noise * amplitude * aspiration * 0.08",
            lambda: value("aspiration_highpass_noise")
            * value("amplitude")
            * value("aspiration")
            * 0.08,
        ),
        "burst_source": (
            ("raw_burst_noise", "burst_envelope", "burst_output_gain"),
            "raw_burst_noise * burst_envelope * burst_output_gain",
            lambda: value("raw_burst_noise")
            * value("burst_envelope")
            * value("burst_output_gain", 0.19),
        ),
    }
    gain_tracks = {
        "voiced_source": "optimizer_voiced_gain",
        "aspiration_source": "optimizer_aspiration_gain",
        "frication_source": "optimizer_frication_gain",
        "aspiration_noise_source": "optimizer_aspiration_noise_gain",
        "burst_source": "optimizer_burst_gain",
    }
    output: dict[str, dict[str, float | str | None]] = {}
    for name, (required, formula, calculator) in specs.items():
        actual = float(component_values.get(name, 0.0))
        if not all(track in trace for track in required):
            output[name] = {
                "formula": formula,
                "calculated": None,
                "actual": actual,
                "residual": None,
                "optimizer_gain": None,
                "missing_tracks": [track for track in required if track not in trace],
            }
            continue
        raw_calculated = float(calculator())
        gain_name = gain_tracks[name]
        optimizer_gain = value(gain_name, 1.0) if gain_name in trace else 1.0
        calculated = raw_calculated * optimizer_gain
        rendered_formula = formula
        if gain_name in trace:
            rendered_formula = f"({formula}) * {gain_name}"
        output[name] = {
            "formula": rendered_formula,
            "raw_calculated": raw_calculated,
            "optimizer_gain": optimizer_gain,
            "calculated": calculated,
            "actual": actual,
            "residual": actual - calculated,
            "missing_tracks": [],
        }
    return output


def build_precision_audit(
    trace: Mapping[str, Any],
    sample_rate: int,
    timeline: GestureTimeline | None = None,
    *,
    compute_backend: str = "auto",
    max_events: int = 32,
    candidate_samples: list[int] | tuple[int, ...] | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    if "final_audio" not in trace:
        raise ValueError("precision audit requires final_audio")
    audio_np = np.asarray(trace["final_audio"], dtype=np.float64).reshape(-1)
    if audio_np.size == 0:
        raise ValueError("precision audit received empty audio")
    if not np.all(np.isfinite(audio_np)):
        raise ValueError("precision audit received NaN or Infinity")

    length = int(audio_np.size)
    runtime = resolve_backend(compute_backend, sample_count=length)
    xp = runtime.xp
    ndimage = runtime.ndimage
    started = time.perf_counter()

    audio = xp.asarray(audio_np, dtype=xp.float64)
    absolute = xp.abs(audio)
    first, second = _xp_diff_abs(xp, audio)
    rms_window = max(3, int(round(0.004 * sample_rate)))
    peak_window = max(3, int(round(0.003 * sample_rate)))
    local_rms = xp.sqrt(
        xp.maximum(
            ndimage.uniform_filter1d(audio * audio, size=rms_window, mode="nearest"),
            0.0,
        )
        + 1e-30
    )
    local_peak = ndimage.maximum_filter1d(absolute, size=peak_window, mode="nearest")

    f0 = xp.asarray(_safe_numpy(trace.get("f0_hz"), length, 220.0), dtype=xp.float64)
    frication = xp.asarray(_safe_numpy(trace.get("frication"), length), dtype=xp.float64)
    aspiration = xp.asarray(_safe_numpy(trace.get("aspiration"), length), dtype=xp.float64)
    burst = xp.asarray(_safe_numpy(trace.get("burst"), length), dtype=xp.float64)
    noise_center = xp.asarray(
        _safe_numpy(trace.get("noise_center_hz"), length, 3500.0),
        dtype=xp.float64,
    )
    noise_width = xp.asarray(
        _safe_numpy(trace.get("noise_bandwidth_hz"), length, 3000.0),
        dtype=xp.float64,
    )
    noise_activity = xp.clip(xp.maximum(xp.maximum(frication, aspiration), burst), 0.0, 1.0)
    voiced_ceiling = xp.clip(f0 * 12.0, 300.0, min(6000.0, sample_rate * 0.45))
    noise_ceiling = xp.clip(
        noise_center + 0.5 * noise_width,
        500.0,
        sample_rate * 0.45,
    )
    expected_frequency = xp.where(noise_activity > 0.05, noise_ceiling, voiced_ceiling)

    # For a sinusoid, adjacent-sample difference is bounded by
    # 2*A*sin(pi*f/fs). Broadband stochastic sources receive an additional
    # RMS allowance so valid /s/, /sh/ and plosive releases are not flattened.
    sine_jump = 2.0 * local_peak * xp.sin(xp.pi * expected_frequency / float(sample_rate))
    stochastic_allowance = local_rms * (2.8 + 4.0 * noise_activity + 2.0 * burst)
    allowed_jump = xp.maximum(
        xp.maximum(sine_jump * (1.6 + 1.4 * noise_activity), stochastic_allowance),
        1e-7,
    )
    jump_ratio = first / (allowed_jump + 1e-30)
    crest_factor = absolute / (local_rms + 1e-30)
    curvature_ratio = second / (allowed_jump * 1.5 + 1e-30)
    risk_score = xp.maximum(
        xp.maximum(jump_ratio / 2.0, crest_factor / 12.0),
        curvature_ratio / 2.5,
    )
    # Near digital silence, dividing by a microscopic RMS can create enormous
    # but inaudible crest ratios. Keep the raw arrays for inspection while
    # excluding sub-audible numerical residue from risk ranking.
    audible_mask = (absolute > 1e-5) | (first > 1e-5) | (local_rms > 1e-6)
    risk_score = xp.where(audible_mask, risk_score, 0.0)

    component_gpu = [
        xp.asarray(_safe_numpy(trace.get(name), length), dtype=xp.float64)
        for name in COMPONENT_NAMES
    ]
    source_sum = xp.zeros(length, dtype=xp.float64)
    for values in component_gpu:
        source_sum = source_sum + values
    pre_stft = xp.asarray(
        _safe_numpy(trace.get("source_sum_pre_stft"), length),
        dtype=xp.float64,
    )
    if "source_sum_pre_stft" not in trace:
        pre_stft = xp.asarray(
            _safe_numpy(trace.get("voiced_mix"), length)
            + _safe_numpy(trace.get("noise_source"), length),
            dtype=xp.float64,
        )
    source_residual = pre_stft - source_sum

    # Transfer diagnostic arrays back once. GPU execution is asynchronous, so
    # synchronization is explicit before reporting elapsed time.
    runtime.synchronize()
    arrays = {
        "first_difference": runtime.to_numpy(first).astype(np.float64, copy=False),
        "second_difference": runtime.to_numpy(second).astype(np.float64, copy=False),
        "local_rms_4ms": runtime.to_numpy(local_rms).astype(np.float64, copy=False),
        "local_peak_3ms": runtime.to_numpy(local_peak).astype(np.float64, copy=False),
        "expected_frequency_hz": runtime.to_numpy(expected_frequency).astype(np.float64, copy=False),
        "allowed_sample_jump": runtime.to_numpy(allowed_jump).astype(np.float64, copy=False),
        "sample_jump_ratio": runtime.to_numpy(jump_ratio).astype(np.float64, copy=False),
        "crest_factor": runtime.to_numpy(crest_factor).astype(np.float64, copy=False),
        "curvature_ratio": runtime.to_numpy(curvature_ratio).astype(np.float64, copy=False),
        "risk_score": runtime.to_numpy(risk_score).astype(np.float64, copy=False),
        "audible_mask": runtime.to_numpy(audible_mask).astype(np.uint8, copy=False),
        "source_sum_fixed_order": runtime.to_numpy(source_sum).astype(np.float64, copy=False),
        "source_sum_pre_stft": runtime.to_numpy(pre_stft).astype(np.float64, copy=False),
        "source_sum_residual": runtime.to_numpy(source_residual).astype(np.float64, copy=False),
    }
    elapsed = time.perf_counter() - started

    risk_indices = _consolidate_candidates(arrays["risk_score"], sample_rate, max_events)
    forced_indices = {
        int(index)
        for index in (candidate_samples or ())
        if 0 <= int(index) < length
    }
    indices = sorted(set(risk_indices) | forced_indices)
    if len(indices) > max_events:
        forced_sorted = sorted(forced_indices)
        remaining = [index for index in risk_indices if index not in forced_indices]
        indices = sorted((forced_sorted + remaining)[:max_events])
    risk_index_set = set(risk_indices)
    event_radius = max(8, int(round(0.004 * sample_rate)))
    events: list[dict[str, Any]] = []
    for event_id, index in enumerate(indices, start=1):
        component_values, component_rms, component_peak = _event_component_metrics(
            trace,
            index,
            event_radius,
            length,
        )
        compensated_source_sum = math.fsum(component_values[name] for name in COMPONENT_NAMES)
        abs_total = math.fsum(abs(component_values[name]) for name in COMPONENT_NAMES)
        shares = {
            name: (abs(component_values[name]) / abs_total if abs_total > 1e-30 else 0.0)
            for name in COMPONENT_NAMES
        }
        dominant = max(component_rms, key=component_rms.get)
        time_s = index / float(sample_rate)
        span = _timeline_span(timeline, time_s)
        event = {
            "event_id": event_id,
            "sample": index,
            "selected_by": "precision_risk" if index in risk_index_set else "debug_detector",
            "time_s": time_s,
            **span,
            "final_sample": float(audio_np[index]),
            "first_difference": float(arrays["first_difference"][index]),
            "second_difference": float(arrays["second_difference"][index]),
            "local_rms_4ms": float(arrays["local_rms_4ms"][index]),
            "local_peak_3ms": float(arrays["local_peak_3ms"][index]),
            "expected_frequency_hz": float(arrays["expected_frequency_hz"][index]),
            "allowed_sample_jump": float(arrays["allowed_sample_jump"][index]),
            "sample_jump_ratio": float(arrays["sample_jump_ratio"][index]),
            "crest_factor": float(arrays["crest_factor"][index]),
            "curvature_ratio": float(arrays["curvature_ratio"][index]),
            "risk_score": float(arrays["risk_score"][index]),
            "dominant_component_by_local_rms": dominant,
            "component_values": component_values,
            "component_local_rms": component_rms,
            "component_local_peak": component_peak,
            "component_absolute_share": shares,
            "component_formula_audit": _formula_audit_at_sample(
                trace,
                index,
                length,
                component_values,
            ),
            "source_sum_math_fsum": compensated_source_sum,
            "source_sum_fixed_order": float(arrays["source_sum_fixed_order"][index]),
            "source_sum_pre_stft": float(arrays["source_sum_pre_stft"][index]),
            "source_sum_residual": float(arrays["source_sum_residual"][index]),
        }
        events.append(event)

    sample_peak = float(np.max(np.abs(audio_np)))
    if runtime.info.active == "gpu" and runtime.gpu_signal is not None and length >= 8:
        try:
            oversampled = runtime.gpu_signal.resample_poly(audio, 4, 1, padtype="line")
            true_peak = float(runtime.to_numpy(xp.max(xp.abs(oversampled))))
        except Exception:
            true_peak = float(np.max(np.abs(scipy_signal.resample_poly(audio_np, 4, 1, padtype="line"))))
    else:
        true_peak = (
            float(np.max(np.abs(scipy_signal.resample_poly(audio_np, 4, 1, padtype="line"))))
            if length >= 8
            else sample_peak
        )

    residual = arrays["source_sum_residual"]
    report = {
        "schema": "digivocal-precision-audit-v1",
        "analysis_mode": "full_trace" if all(name in trace for name in COMPONENT_NAMES) else "wav_only",
        "causal_trace_available": all(name in trace for name in COMPONENT_NAMES),
        "backend": runtime.info.to_dict(),
        "sample_rate": int(sample_rate),
        "sample_count": length,
        "duration_s": length / float(sample_rate),
        "elapsed_s": elapsed,
        "precision": {
            "internal_dtype": "float64",
            "event_sum_method": "math.fsum compensated summation",
            "array_sum_order": list(COMPONENT_NAMES),
            "note": (
                "GPU scans and guards use float64. Event source sums are recomputed on CPU "
                "with math.fsum; tiny CPU/GPU last-bit differences remain possible."
            ),
        },
        "summary": {
            "sample_peak": sample_peak,
            "true_peak_4x": true_peak,
            "rms": float(np.sqrt(np.mean(np.square(audio_np)) + 1e-30)),
            "max_first_difference": float(np.max(arrays["first_difference"])),
            "max_second_difference": float(np.max(arrays["second_difference"])),
            "max_jump_ratio": float(np.max(arrays["sample_jump_ratio"])),
            "max_crest_factor": float(
                np.max(arrays["crest_factor"][arrays["audible_mask"].astype(bool)])
                if np.any(arrays["audible_mask"])
                else 0.0
            ),
            "max_risk_score": float(np.max(arrays["risk_score"])),
            "source_sum_residual_max_abs": float(np.max(np.abs(residual))),
            "source_sum_residual_rms": float(np.sqrt(np.mean(np.square(residual)) + 1e-30)),
            "event_count": len(events),
        },
        "formulas": {
            "source_sum": "voiced_source + aspiration_source + frication_source + aspiration_noise_source + burst_source",
            "first_difference": "abs(x[n] - x[n-1])",
            "second_difference": "abs(x[n] - 2*x[n-1] + x[n-2])",
            "sine_jump_bound": "2*A*sin(pi*f/fs)",
            "allowed_jump": "max(sine_jump_bound*(1.6+1.4*noise_activity), local_rms*(2.8+4*noise_activity+2*burst))",
        },
        "events": events,
    }
    return report, arrays


def apply_precision_guard(
    audio: np.ndarray,
    sample_rate: int,
    trace: Mapping[str, Any] | None = None,
    *,
    compute_backend: str = "auto",
    max_attenuation_db: float = 6.0,
) -> tuple[np.ndarray, dict[str, Any], dict[str, np.ndarray]]:
    """Attenuate only statistically extreme, short transients.

    The guard is frequency-aware: valid high-frequency frication receives a
    larger adjacent-sample allowance than vowels. It never hard-clips samples;
    it applies a smoothed gain envelope bounded by ``max_attenuation_db``.
    """

    source = np.asarray(audio, dtype=np.float64).reshape(-1)
    if source.size < 16:
        info = BackendInfo("cpu", "cpu", False, reason="audio too short")
        gain = np.ones_like(source)
        return source.copy(), {
            "enabled": True,
            "backend": info.to_dict(),
            "corrected_samples": 0,
            "max_attenuation_db": 0.0,
        }, {"precision_guard_gain": gain}

    runtime = resolve_backend(compute_backend, sample_count=source.size)
    xp = runtime.xp
    ndimage = runtime.ndimage
    values = xp.asarray(source, dtype=xp.float64)
    absolute = xp.abs(values)
    first, second = _xp_diff_abs(xp, values)
    rms_window = max(3, int(round(0.004 * sample_rate)))
    peak_window = max(3, int(round(0.003 * sample_rate)))
    local_rms = xp.sqrt(
        xp.maximum(
            ndimage.uniform_filter1d(values * values, size=rms_window, mode="nearest"),
            0.0,
        )
        + 1e-30
    )
    local_peak = ndimage.maximum_filter1d(absolute, size=peak_window, mode="nearest")

    trace = trace or {}
    length = source.size
    f0 = xp.asarray(_safe_numpy(trace.get("f0_hz"), length, 220.0), dtype=xp.float64)
    frication = xp.asarray(_safe_numpy(trace.get("frication"), length), dtype=xp.float64)
    aspiration = xp.asarray(_safe_numpy(trace.get("aspiration"), length), dtype=xp.float64)
    burst = xp.asarray(_safe_numpy(trace.get("burst"), length), dtype=xp.float64)
    noise_center = xp.asarray(
        _safe_numpy(trace.get("noise_center_hz"), length, 3500.0),
        dtype=xp.float64,
    )
    noise_width = xp.asarray(
        _safe_numpy(trace.get("noise_bandwidth_hz"), length, 3000.0),
        dtype=xp.float64,
    )
    noise_activity = xp.clip(xp.maximum(xp.maximum(frication, aspiration), burst), 0.0, 1.0)
    voiced_ceiling = xp.clip(f0 * 12.0, 300.0, min(6000.0, sample_rate * 0.45))
    noise_ceiling = xp.clip(noise_center + 0.5 * noise_width, 500.0, sample_rate * 0.45)
    expected_frequency = xp.where(noise_activity > 0.05, noise_ceiling, voiced_ceiling)
    sine_jump = 2.0 * local_peak * xp.sin(xp.pi * expected_frequency / float(sample_rate))
    allowed_jump = xp.maximum(
        xp.maximum(
            sine_jump * (1.6 + 1.4 * noise_activity),
            local_rms * (2.8 + 4.0 * noise_activity + 2.0 * burst),
        ),
        1e-7,
    )

    jump_ratio = first / (allowed_jump + 1e-30)
    crest = absolute / (local_rms + 1e-30)
    curvature_ratio = second / (allowed_jump * 1.5 + 1e-30)

    # Absolute floors prevent inaudibly tiny numerical motion from triggering a
    # correction solely because the local RMS is close to zero.
    # Emergency-only thresholds. Ordinary consonant shaping is handled by the
    # always-on source optimizer; this stage intervenes only on clearly
    # pathological post-mix transients.
    jump_trigger = (jump_ratio > 4.50) & (first > 0.060)
    crest_trigger = (crest > 16.0) & (absolute > 0.100)
    curvature_trigger = (curvature_ratio > 5.0) & (second > 0.080)
    trigger = jump_trigger | crest_trigger | curvature_trigger

    minimum_gain = math.pow(10.0, -abs(max_attenuation_db) / 20.0)
    gain_from_jump = xp.clip((allowed_jump * 2.15) / (first + 1e-30), minimum_gain, 1.0)
    gain_from_crest = xp.clip((local_rms * 9.0) / (absolute + 1e-30), minimum_gain, 1.0)
    gain_from_curvature = xp.clip(
        (allowed_jump * 2.6) / (second + 1e-30),
        minimum_gain,
        1.0,
    )
    target_gain = xp.where(
        trigger,
        xp.minimum(xp.minimum(gain_from_jump, gain_from_crest), gain_from_curvature),
        1.0,
    )

    # Spread a correction over a very short neighborhood, then smooth it so the
    # gain itself cannot create a new click. Attack is short; release is about
    # 1.5 ms. This is not a hard limiter.
    spread = max(1, int(round(0.0006 * sample_rate)))
    if spread % 2 == 0:
        spread += 1
    gain = ndimage.minimum_filter1d(target_gain, size=spread, mode="nearest")
    gain = ndimage.gaussian_filter1d(
        gain,
        sigma=max(0.75, 0.0015 * sample_rate),
        mode="nearest",
    )
    # Keep the exact required attenuation at triggered samples; smoothing only
    # controls the neighborhood and cannot erase the correction itself.
    gain = xp.minimum(gain, target_gain)
    gain = xp.clip(gain, minimum_gain, 1.0)
    guarded = values * gain
    runtime.synchronize()

    gain_np = runtime.to_numpy(gain).astype(np.float64, copy=False)
    guarded_np = runtime.to_numpy(guarded).astype(np.float64, copy=False)
    trigger_np = runtime.to_numpy(trigger).astype(bool, copy=False)
    corrected = int(np.count_nonzero(gain_np < 0.999999))
    actual_max_attenuation = (
        float(-20.0 * math.log10(max(float(np.min(gain_np)), 1e-12)))
        if corrected
        else 0.0
    )
    report = {
        "enabled": True,
        "backend": runtime.info.to_dict(),
        "corrected_samples": corrected,
        "trigger_samples": int(np.count_nonzero(trigger_np)),
        "corrected_duration_ms": corrected / float(sample_rate) * 1000.0,
        "max_attenuation_db": actual_max_attenuation,
        "minimum_gain": float(np.min(gain_np)),
        "thresholds": {
            "jump_ratio": 4.50,
            "crest_factor": 16.0,
            "curvature_ratio": 5.0,
            "absolute_jump_floor": 0.060,
            "absolute_peak_floor": 0.100,
            "absolute_curvature_floor": 0.080,
        },
    }
    arrays = {
        "precision_guard_gain": gain_np,
        "precision_guard_allowed_jump": runtime.to_numpy(allowed_jump).astype(np.float64, copy=False),
        "precision_guard_jump_ratio": runtime.to_numpy(jump_ratio).astype(np.float64, copy=False),
        "precision_guard_crest_factor": runtime.to_numpy(crest).astype(np.float64, copy=False),
        "precision_guard_curvature_ratio": runtime.to_numpy(curvature_ratio).astype(np.float64, copy=False),
        "precision_guard_trigger": trigger_np.astype(np.uint8),
    }
    return guarded_np, report, arrays


def precision_environment() -> dict[str, Any]:
    requested = os.environ.get("DIGIVOCAL_COMPUTE_BACKEND", "auto")
    return {
        "requested_from_environment": requested,
        "status": backend_status(requested),
    }

# ---------------------------------------------------------------------------
# Always-on precision quality optimizer
# ---------------------------------------------------------------------------

_OPTIMIZER_SPECS: dict[str, dict[str, float | str]] = {
    # The targets are local source RMS values before the STFT vocal-tract stage.
    # They are intentionally different for each physical source family instead
    # of forcing the final waveform toward one generic limiter threshold.
    "voiced_source": {
        "control": "voiced_control",
        "target_rms": 0.285,
        "control_power": 0.58,
        "level_strength": 0.16,
        "crest_target": 2.70,
        "crest_strength": 0.22,
        "gain_min": 0.78,
        "gain_max": 1.16,
        "rms_ms": 6.0,
        "peak_ms": 2.0,
        "smooth_ms": 2.5,
    },
    "aspiration_source": {
        "control": "aspiration_control",
        "target_rms": 0.0090,
        "control_power": 0.72,
        "level_strength": 0.48,
        "crest_target": 4.80,
        "crest_strength": 0.32,
        "gain_min": 0.52,
        "gain_max": 1.90,
        "rms_ms": 4.0,
        "peak_ms": 1.5,
        "smooth_ms": 1.5,
    },
    "frication_source": {
        "control": "frication_control",
        "target_rms": 0.070,
        "control_power": 0.62,
        "level_strength": 0.52,
        "crest_target": 4.50,
        "crest_strength": 0.45,
        "gain_min": 0.42,
        "gain_max": 1.32,
        "rms_ms": 3.0,
        "peak_ms": 1.2,
        "smooth_ms": 1.15,
    },
    "aspiration_noise_source": {
        "control": "aspiration_control",
        "target_rms": 0.0065,
        "control_power": 0.72,
        "level_strength": 0.46,
        "crest_target": 4.80,
        "crest_strength": 0.30,
        "gain_min": 0.50,
        "gain_max": 1.70,
        "rms_ms": 3.5,
        "peak_ms": 1.5,
        "smooth_ms": 1.4,
    },
    "burst_source": {
        "control": "burst_control",
        "target_rms": 0.060,
        "control_power": 0.50,
        "level_strength": 0.58,
        "crest_target": 3.90,
        "crest_strength": 0.55,
        "gain_min": 0.34,
        "gain_max": 1.16,
        "rms_ms": 2.0,
        "peak_ms": 0.9,
        "smooth_ms": 0.70,
    },
}


def _optimizer_control_tracks(
    controls: Mapping[str, Any],
    length: int,
) -> dict[str, np.ndarray]:
    amplitude = np.clip(_safe_numpy(controls.get("amplitude"), length, 0.0), 0.0, 1.0)
    voicing = np.clip(_safe_numpy(controls.get("voicing"), length, 0.0), 0.0, 1.0)
    pressure = np.clip(_safe_numpy(controls.get("pressure"), length, 0.0), 0.0, 1.0)
    frication = np.clip(_safe_numpy(controls.get("frication"), length, 0.0), 0.0, 1.0)
    aspiration = np.clip(_safe_numpy(controls.get("aspiration"), length, 0.0), 0.0, 1.0)
    burst = np.clip(_safe_numpy(controls.get("burst"), length, 0.0), 0.0, 1.0)
    burst_envelope = np.clip(
        _safe_numpy(controls.get("burst_envelope"), length, 0.0),
        0.0,
        1.0,
    )
    return {
        "voiced_control": np.clip(amplitude * voicing * (0.55 + 0.45 * pressure), 0.0, 1.0),
        "aspiration_control": np.clip(amplitude * aspiration, 0.0, 1.0),
        "frication_control": np.clip(frication * (0.78 + 0.22 * pressure), 0.0, 1.0),
        "burst_control": np.maximum(burst, burst_envelope),
        "pressure": pressure,
    }


def apply_precision_optimizer(
    components: Mapping[str, np.ndarray],
    sample_rate: int,
    controls: Mapping[str, Any] | None = None,
    *,
    compute_backend: str = "auto",
    strength: float = 1.0,
) -> tuple[dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    """Continuously optimize every source family before source mixing.

    Unlike :func:`apply_precision_guard`, this is not an accident detector.  It
    is active on every voiced/noise source sample.  Each source gets its own
    rolling RMS target, crest target, smoothing time, and gain limits.  The
    result is a deterministic, component-aware level/crest normalization that
    keeps articulatory envelopes intact while reducing random noise spikes and
    bringing weak intended sources (notably aspiration) into a more useful
    range.

    The heavy rolling statistics and gain calculations use CuPy float64 when a
    CUDA backend is available; CPU NumPy/SciPy is an automatic fallback.
    """

    if not components:
        return {}, {
            "enabled": True,
            "mode": "quality_optimizer",
            "backend": BackendInfo("cpu", "cpu", False, reason="no components").to_dict(),
            "component_count": 0,
        }, {}

    first = next(iter(components.values()))
    length = int(np.asarray(first).size)
    if length <= 0:
        return {name: np.asarray(values, dtype=np.float64).copy() for name, values in components.items()}, {
            "enabled": True,
            "mode": "quality_optimizer",
            "backend": BackendInfo("cpu", "cpu", False, reason="empty components").to_dict(),
            "component_count": len(components),
        }, {}

    strength = float(np.clip(strength, 0.0, 1.6))
    controls = controls or {}
    control_np = _optimizer_control_tracks(controls, length)
    noise_center_np = np.clip(
        _safe_numpy(controls.get("noise_center_hz"), length, 3500.0),
        300.0,
        sample_rate * 0.45,
    )
    noise_width_np = np.clip(
        _safe_numpy(controls.get("noise_bandwidth_hz"), length, 3000.0),
        150.0,
        sample_rate * 0.9,
    )

    runtime = resolve_backend(compute_backend, sample_count=length)
    xp = runtime.xp
    ndimage = runtime.ndimage
    started = time.perf_counter()

    optimized: dict[str, np.ndarray] = {}
    audit_arrays: dict[str, np.ndarray] = {}
    component_reports: dict[str, Any] = {}

    for name in COMPONENT_NAMES:
        if name not in components:
            continue
        spec = _OPTIMIZER_SPECS[name]
        source_np = _safe_numpy(components.get(name), length)
        source = xp.asarray(source_np, dtype=xp.float64)
        control_name = str(spec["control"])
        control = xp.asarray(control_np[control_name], dtype=xp.float64)

        rms_window = max(3, int(round(float(spec["rms_ms"]) * 0.001 * sample_rate)))
        peak_window = max(3, int(round(float(spec["peak_ms"]) * 0.001 * sample_rate)))
        local_rms = xp.sqrt(
            xp.maximum(
                ndimage.uniform_filter1d(source * source, size=rms_window, mode="nearest"),
                0.0,
            )
            + 1e-30
        )
        local_peak = ndimage.maximum_filter1d(xp.abs(source), size=peak_window, mode="nearest")
        crest = local_peak / (local_rms + 1e-30)

        target = float(spec["target_rms"]) * xp.power(
            xp.clip(control, 0.0, 1.0),
            float(spec["control_power"]),
        )

        # High-frequency frication is perceptually more forward.  Reduce its
        # target gently as centre frequency rises and compensate very narrow
        # noise bands so /s/ does not turn into a whistle-like narrow peak.
        if name == "frication_source":
            center = xp.asarray(noise_center_np, dtype=xp.float64)
            width = xp.asarray(noise_width_np, dtype=xp.float64)
            frequency_factor = xp.clip(xp.power(3500.0 / xp.maximum(center, 1800.0), 0.18), 0.80, 1.10)
            bandwidth_factor = xp.clip(xp.sqrt(2200.0 / xp.maximum(width, 800.0)), 0.86, 1.12)
            target = target * frequency_factor * bandwidth_factor
        elif name == "burst_source":
            pressure = xp.asarray(control_np["pressure"], dtype=xp.float64)
            target = target * (0.88 + 0.12 * pressure)

        active = (control > 0.012) | (local_rms > 1e-7)
        raw_ratio = target / (local_rms + 1e-30)
        level_strength = float(spec["level_strength"]) * strength
        level_gain = xp.power(xp.clip(raw_ratio, 1e-5, 1e5), level_strength)

        crest_target = float(spec["crest_target"])
        crest_strength = float(spec["crest_strength"]) * strength
        crest_gain = xp.where(
            crest > crest_target,
            xp.power(crest_target / (crest + 1e-30), crest_strength),
            1.0,
        )
        target_gain = level_gain * crest_gain
        gain_min = float(spec["gain_min"])
        gain_max = float(spec["gain_max"])
        target_gain = xp.where(active, xp.clip(target_gain, gain_min, gain_max), 1.0)

        sigma = max(0.75, float(spec["smooth_ms"]) * 0.001 * sample_rate)
        gain = ndimage.gaussian_filter1d(target_gain, sigma=sigma, mode="nearest")
        gain = xp.where(active, xp.clip(gain, gain_min, gain_max), 1.0)
        result = source * gain

        runtime.synchronize()
        gain_np = runtime.to_numpy(gain).astype(np.float64, copy=False)
        result_np = runtime.to_numpy(result).astype(np.float64, copy=False)
        local_rms_np = runtime.to_numpy(local_rms).astype(np.float64, copy=False)
        target_np = runtime.to_numpy(target).astype(np.float64, copy=False)
        crest_np = runtime.to_numpy(crest).astype(np.float64, copy=False)

        optimized[name] = result_np
        short = name.removesuffix("_source")
        audit_arrays[f"optimizer_{short}_gain"] = gain_np
        audit_arrays[f"optimizer_{short}_local_rms"] = local_rms_np
        audit_arrays[f"optimizer_{short}_target_rms"] = target_np
        audit_arrays[f"optimizer_{short}_crest"] = crest_np

        active_np = control_np[control_name] > 0.012
        if np.any(active_np):
            before = source_np[active_np]
            after = result_np[active_np]
            active_gain = gain_np[active_np]
            component_reports[name] = {
                "active_samples": int(np.count_nonzero(active_np)),
                "rms_before": float(np.sqrt(np.mean(np.square(before)) + 1e-30)),
                "rms_after": float(np.sqrt(np.mean(np.square(after)) + 1e-30)),
                "peak_before": float(np.max(np.abs(before))),
                "peak_after": float(np.max(np.abs(after))),
                "gain_min": float(np.min(active_gain)),
                "gain_median": float(np.median(active_gain)),
                "gain_max": float(np.max(active_gain)),
                "changed_samples": int(np.count_nonzero(np.abs(active_gain - 1.0) > 1e-3)),
                "target_rms_base": float(spec["target_rms"]),
                "crest_target": crest_target,
            }
        else:
            component_reports[name] = {
                "active_samples": 0,
                "rms_before": 0.0,
                "rms_after": 0.0,
                "peak_before": 0.0,
                "peak_after": 0.0,
                "gain_min": 1.0,
                "gain_median": 1.0,
                "gain_max": 1.0,
                "changed_samples": 0,
                "target_rms_base": float(spec["target_rms"]),
                "crest_target": crest_target,
            }

    elapsed = time.perf_counter() - started
    all_gains = [values for key, values in audit_arrays.items() if key.endswith("_gain")]
    changed_total = int(
        sum(np.count_nonzero(np.abs(values - 1.0) > 1e-3) for values in all_gains)
    )
    report = {
        "enabled": True,
        "mode": "quality_optimizer",
        "backend": runtime.info.to_dict(),
        "strength": strength,
        "elapsed_s": elapsed,
        "component_count": len(optimized),
        "changed_component_samples": changed_total,
        "component_reports": component_reports,
        "principle": (
            "Always-on float64 component optimization: rolling RMS target + crest shaping + "
            "source-specific smoothing. This is quality shaping, not a final safety limiter."
        ),
    }
    return optimized, report, audit_arrays

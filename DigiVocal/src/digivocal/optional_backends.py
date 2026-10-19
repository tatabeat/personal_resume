from __future__ import annotations

import importlib.util
import json
from pathlib import Path
from typing import Any

import numpy as np

from .models import GestureTimeline


COMPONENTS = {
    "pypinyin": {
        "module": "pypinyin",
        "role": "Stable Mandarin text-to-Pinyin frontend",
        "license": "MIT",
    },
    "g2pW": {
        "module": "g2pw",
        "role": "Optional context-dependent polyphone disambiguation",
        "license": "Apache-2.0",
    },
    "SPARC": {
        "module": "sparc",
        "role": "Optional WAV articulatory coding and speaker embedding",
        "license": "MIT",
    },
    "VocalTractLab-Python": {
        "module": "vocaltractlab",
        "role": "Optional external physical articulatory synthesizer",
        "license": "GPL-3.0",
    },
    "VocalTractLab-Cython": {
        "module": "vocaltractlab_cython",
        "role": "Optional low-level VocalTractLab API",
        "license": "GPL-3.0",
    },
}


def component_status() -> dict[str, dict[str, Any]]:
    status: dict[str, dict[str, Any]] = {}
    for name, info in COMPONENTS.items():
        module = info["module"]
        installed = importlib.util.find_spec(module) is not None
        status[name] = {**info, "installed": installed}
    return status


def sparc_analyze(
    wav_path: str | Path,
    output_npz: str | Path,
    *,
    model: str = "multi",
    device: str = "cpu",
) -> Path:
    """Run SPARC on a WAV and save all returned arrays to NPZ.

    SPARC is not installed by the core setup because its neural dependencies and
    checkpoints are large. Install the ``sparc`` optional dependency first.
    """

    try:
        from sparc import load_model
    except ImportError as exc:
        raise RuntimeError(
            "SPARC is not installed. Run: python -m pip install -e '.[sparc]'"
        ) from exc

    coder = load_model(model, device=device)
    code = coder.encode(str(wav_path))
    payload: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"model": model, "device": device, "sample_rate": int(coder.sr)}
    for key, value in code.items():
        if isinstance(value, np.ndarray):
            payload[key] = value
        elif isinstance(value, (tuple, list)) and all(isinstance(item, (int, float, np.number)) for item in value):
            payload[key] = np.asarray(value)
        else:
            metadata[key] = value
    payload["metadata_json"] = np.asarray(json.dumps(metadata, ensure_ascii=False))
    destination = Path(output_npz)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination, **payload)
    return destination


def probe_vocaltractlab() -> dict[str, Any]:
    """Return installation/API information without assuming a stable VTL API."""

    result: dict[str, Any] = {
        "vocaltractlab_installed": importlib.util.find_spec("vocaltractlab") is not None,
        "vocaltractlab_cython_installed": importlib.util.find_spec("vocaltractlab_cython") is not None,
    }
    if result["vocaltractlab_cython_installed"]:
        try:
            from vocaltractlab_cython import get_constants

            constants = get_constants()
            result["constants"] = {
                key: value
                for key, value in constants.items()
                if isinstance(value, (str, int, float, bool))
            }
        except Exception as exc:  # alpha APIs can vary
            result["error"] = str(exc)
    return result


def export_vtl_bridge_payload(timeline: GestureTimeline, path: str | Path) -> Path:
    """Export a neutral bridge JSON for future Mandarin-to-VTL calibration.

    This is intentionally not presented as a .ges file. VocalTractLab's speaker
    shapes and gesture vocabulary must be calibrated before normalized DigiVocal
    tracks can be mapped safely to a physical model.
    """

    selected = {
        name: timeline.tracks[name].tolist()
        for name in (
            "f0_hz",
            "pressure",
            "jaw_open",
            "tongue_height",
            "tongue_front",
            "lip_round",
            "lip_close",
            "velum_open",
            "constriction",
        )
        if name in timeline.tracks
    }
    data = {
        "format": "digivocal-vtl-bridge-v0.1",
        "rate_hz": timeline.rate_hz,
        "tracks": selected,
        "spans": [span.to_dict() for span in timeline.spans],
        "note": "Normalized research controls; requires a speaker-specific VTL mapping.",
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
    return destination

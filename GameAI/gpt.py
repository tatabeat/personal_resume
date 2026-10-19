import base64
import io
import json
import os
import re
from pathlib import Path
from typing import Any

from openai import OpenAI


PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_PROFILE_DIR = PROJECT_DIR / "dataset" / "AOV"
LEGACY_PROFILE_DIR = PROJECT_DIR / "database" / "AOV"

_profile_dir = DEFAULT_PROFILE_DIR if DEFAULT_PROFILE_DIR.exists() else LEGACY_PROFILE_DIR


TRIPLE_QUOTE_PATTERN = re.compile(r'("system_prompt"\s*:\s*)"""(.*?)"""', re.DOTALL)


def get_default_profile_dir() -> Path:
    return _profile_dir


def set_profile_dir(path: str | Path) -> Path:
    global _profile_dir
    profile_dir = Path(path).expanduser().resolve()
    if not profile_dir.exists() or not profile_dir.is_dir():
        raise FileNotFoundError(f"Profile folder not found: {profile_dir}")
    _profile_dir = profile_dir
    return _profile_dir


def get_profile_dir() -> Path:
    return _profile_dir


def _read_text(path: Path) -> str:
    with path.open("r", encoding="utf-8") as f:
        return f.read()


def _load_json(path: Path) -> dict:
    return json.loads(_read_text(path))


def _load_config_like_json(path: Path) -> dict:
    text = _read_text(path)

    def replace_triple_quote(match: re.Match[str]) -> str:
        prefix = match.group(1)
        body = match.group(2).strip("\n")
        return f"{prefix}{json.dumps(body, ensure_ascii=False)}"

    normalized = TRIPLE_QUOTE_PATTERN.sub(replace_triple_quote, text)
    return json.loads(normalized)


def _candidate_config_paths(config_name: str, profile_dir: Path | None = None) -> list[Path]:
    target_dir = (profile_dir or _profile_dir).resolve()
    normalized = (config_name or "").strip()
    if not normalized:
        raise ValueError("config_name is empty")
    names = [normalized]
    if normalized.lower() != normalized:
        names.append(normalized.lower())
    if normalized.upper() != normalized:
        names.append(normalized.upper())
    seen = set()
    paths: list[Path] = []
    for name in names:
        path = target_dir / f"{name}.config"
        if str(path) not in seen:
            paths.append(path)
            seen.add(str(path))
    return paths


def resolve_config_path(config_name: str, profile_dir: Path | None = None) -> Path:
    for path in _candidate_config_paths(config_name, profile_dir=profile_dir):
        if path.exists():
            return path
    raise FileNotFoundError(f"Config file not found: {config_name}")


def load_config(config_name: str, profile_dir: Path | None = None) -> dict:
    return _load_config_like_json(resolve_config_path(config_name, profile_dir=profile_dir))


def load_api_key() -> str:
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError("OPENAI_API_KEY is not set; configure it in the environment before starting")
    return api_key


def get_client() -> OpenAI:
    return OpenAI(api_key=load_api_key())


def _clean_str(value: Any) -> str:
    return str(value).strip() if value is not None else ""


def _as_bool(value: Any, default: bool = False) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return default
    text = str(value).strip().lower()
    if text in {"on", "true", "1", "yes", "y"}:
        return True
    if text in {"off", "false", "0", "no", "n"}:
        return False
    return default


def _as_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _as_int(value: Any, default: int) -> int:
    try:
        return int(float(value))
    except Exception:
        return default


def build_system_prompt(settings_data: dict) -> str:
    system_prompt = settings_data.get("system_prompt", "")
    if isinstance(system_prompt, str):
        return system_prompt.strip()

    if isinstance(system_prompt, dict):
        task = _clean_str(system_prompt.get("task", ""))
        input_desc = _clean_str(system_prompt.get("input", ""))
        output_desc = _clean_str(system_prompt.get("output", ""))
        rules = system_prompt.get("rules", [])

        parts = []
        if task:
            parts.append(f"【任務】\n{task}")
        if input_desc:
            parts.append(f"【輸入】\n{input_desc}")
        if output_desc:
            parts.append(f"【輸出】\n{output_desc}")
        if isinstance(rules, list) and rules:
            lines = [f"- {_clean_str(rule)}" for rule in rules if _clean_str(rule)]
            if lines:
                parts.append("【限制】\n" + "\n".join(lines))
        return "\n\n".join(parts).strip()

    return ""


def get_stage_settings(config_name: str, profile_dir: Path | None = None) -> dict:
    raw = load_config(config_name, profile_dir=profile_dir)
    reasoning = raw.get("reasoning", {}) if isinstance(raw.get("reasoning", {}), dict) else {}
    web_search = raw.get("web_search", {}) if isinstance(raw.get("web_search", {}), dict) else {}
    runtime = raw.get("runtime", {}) if isinstance(raw.get("runtime", {}), dict) else {}
    plan = raw.get("plan", {}) if isinstance(raw.get("plan", {}), dict) else {}

    allowed_domains = web_search.get("allowed_domains", [])
    if not isinstance(allowed_domains, list):
        allowed_domains = []
    allowed_domains = [_clean_str(x) for x in allowed_domains if _clean_str(x)]

    settings = {
        "enabled": _as_bool(raw.get("enabled", True), True),
        "model": _clean_str(raw.get("model", "gpt-5.4-mini")) or "gpt-5.4-mini",
        "system_prompt": build_system_prompt(raw),
        "interval_seconds": _as_float(raw.get("interval_seconds", runtime.get("interval_seconds", 3.0)), 3.0),
        "reasoning": {
            "effort": _clean_str(reasoning.get("effort", "")),
        },
        "web_search": {
            "enabled": _as_bool(web_search.get("enabled", False), False),
            "external_web_access": _as_bool(web_search.get("external_web_access", False), False),
            "allowed_domains": allowed_domains,
        },
        "plan": {
            "list_length": max(1, _as_int(plan.get("list_length", raw.get("list_length", 1)), 1)),
            "step_seconds": max(0.0, _as_float(plan.get("step_seconds", 0.1), 0.1)),
            "start_offset_seconds": max(0.0, _as_float(plan.get("start_offset_seconds", 0.2), 0.2)),
            "refill_threshold_seconds": max(0.0, _as_float(plan.get("refill_threshold_seconds", 0.5), 0.5)),
        },
        "log_images": _as_bool(raw.get("log_images", True), True),
    }
    return settings


def image_to_data_url(image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/png;base64,{encoded}"


def _build_request_kwargs(input_items: list[dict], config_name: str, profile_dir: Path | None = None) -> dict:
    settings = get_stage_settings(config_name, profile_dir=profile_dir)
    kwargs = {
        "model": settings["model"],
        "instructions": settings["system_prompt"],
        "input": input_items,
    }
    effort = settings["reasoning"].get("effort", "")
    if effort:
        kwargs["reasoning"] = {"effort": effort}

    web_search = settings["web_search"]
    if web_search.get("enabled"):
        tool = {
            "type": "web_search",
            "external_web_access": web_search.get("external_web_access", False),
        }
        if web_search.get("allowed_domains"):
            tool["filters"] = {"allowed_domains": web_search["allowed_domains"]}
        kwargs["tools"] = [tool]
    return kwargs


def _create_response(input_items: list[dict], config_name: str, profile_dir: Path | None = None) -> str:
    client = get_client()
    response = client.responses.create(**_build_request_kwargs(input_items, config_name=config_name, profile_dir=profile_dir))
    return response.output_text.strip()


def call(text: str, config_name: str, profile_dir: Path | None = None) -> str:
    text = text.strip()
    if not text:
        return ""
    return _create_response(
        [{"role": "user", "content": [{"type": "input_text", "text": text}]}],
        config_name=config_name,
        profile_dir=profile_dir,
    )


def call_with_image(image, text: str, config_name: str, profile_dir: Path | None = None) -> str:
    image_url = image_to_data_url(image)
    content = []
    if text.strip():
        content.append({"type": "input_text", "text": text})
    content.append({"type": "input_image", "image_url": image_url, "detail": "auto"})
    return _create_response(
        [{"role": "user", "content": content}],
        config_name=config_name,
        profile_dir=profile_dir,
    )

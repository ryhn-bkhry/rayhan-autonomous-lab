"""Nightly dynamic model discovery, liveness canary, and embedding probe (Blueprint Sections 3.1 & 3.2)."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
import re
import time
from typing import Any, Mapping

import yaml

_FAMILY_RE = re.compile(r"^(gemini-\d+(?:\.\d+)?|gemma-\d+)")
_FENCED_JSON_RE = re.compile(r"^```(?:json)?\s*\n?(.*?)\n?```$", re.DOTALL | re.IGNORECASE)


@dataclass(frozen=True)
class LiveModelStatus:
    """Result of a fast 1-token strict-JSON liveness probe for a single model."""

    model_id: str
    family: str
    live: bool
    json_ok: bool
    latency_ms: int
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-serializable dictionary representation."""
        return asdict(self)


def infer_family(model_id: str) -> str:
    """Infer canonical model family (e.g., 'gemini-2.5', 'gemma-4') from model ID."""
    match = _FAMILY_RE.match(model_id.lower())
    return match.group(1) if match else model_id.split("-")[0].lower()


def load_policy(path: Path = Path("config/model_policy.yaml")) -> dict[str, Any]:
    """Load model isolation and embedding preference policy YAML."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in model policy {path}")
    return data


def is_excluded(model_id: str, policy: dict[str, Any]) -> bool:
    """Return True if `model_id` matches any frontier or user-reserved exclusion regex."""
    patterns = list(policy.get("frontier_exclude_patterns", [])) + list(
        policy.get("user_reserved_patterns", [])
    )
    return any(re.search(pat, model_id, re.IGNORECASE) for pat in patterns)


def _extract_actions(model_obj: Any) -> list[str]:
    actions = (
        getattr(model_obj, "supported_actions", None)
        or getattr(model_obj, "supported_generation_methods", None)
        or []
    )
    return [str(a) for a in actions]


def list_eligible_generate_models(client: Any, policy: dict[str, Any]) -> list[str]:
    """List non-excluded generateContent-capable model IDs from `client.models.list()`."""
    eligible: list[str] = []
    seen: set[str] = set()
    for model_obj in client.models.list():
        raw_name = str(getattr(model_obj, "name", "") or "")
        mid = raw_name.removeprefix("models/").strip()
        if not mid or mid in seen:
            continue
        actions = _extract_actions(model_obj)
        if "generateContent" not in actions or is_excluded(mid, policy):
            continue
        seen.add(mid)
        eligible.append(mid)
    return eligible


def supports_json_mime_type(model_id: str) -> bool:
    """Return False for model families (e.g. Gemma) that reject `response_mime_type='application/json'`."""
    normalized = model_id.strip().removeprefix("models/").lower()
    return not normalized.startswith("gemma-")


def is_json_mode_unsupported_error(exc: Exception) -> bool:
    """Detect API 400 INVALID_ARGUMENT errors caused by unsupported JSON mode or schema config."""
    msg = str(exc).lower()
    if "json mode is not enabled" in msg:
        return True
    has_json_param = "response_mime_type" in msg or "response_json_schema" in msg
    has_unsupported = any(
        tok in msg for tok in ("not enabled", "not supported", "unsupported", "invalid")
    )
    return has_json_param and has_unsupported


def parse_json_response(raw_response: str) -> Any:
    """Parse raw JSON or a single outer markdown-fenced JSON block."""
    text = (raw_response or "").strip()
    if not text:
        raise ValueError("Empty JSON response")
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        match = _FENCED_JSON_RE.match(text)
        if match is None:
            raise
        return json.loads(match.group(1).strip())


def generate_json_content(
    client: Any,
    model_id: str,
    contents: str,
    *,
    temperature: float = 0.0,
    max_output_tokens: int | None = None,
    response_json_schema: Mapping[str, Any] | None = None,
    allow_json_mime: bool = True,
) -> str:
    """Invoke `generate_content` with JSON mode when supported, falling back cleanly if rejected."""
    config: dict[str, Any] = {"temperature": temperature}
    if max_output_tokens is not None:
        config["max_output_tokens"] = max_output_tokens
    if allow_json_mime and supports_json_mime_type(model_id):
        config["response_mime_type"] = "application/json"
        if response_json_schema is not None:
            config["response_json_schema"] = dict(response_json_schema)

    while True:
        try:
            resp = client.models.generate_content(
                model=model_id,
                contents=contents,
                config=dict(config),
            )
            return (getattr(resp, "text", None) or "").strip()
        except Exception as exc:
            if not is_json_mode_unsupported_error(exc):
                raise
            if "response_json_schema" in config:
                config.pop("response_json_schema")
                continue
            if "response_mime_type" in config:
                config.pop("response_mime_type")
                continue
            raise


def _sanitize_error_message(exc: Exception, max_len: int = 200) -> str:
    collapsed = " ".join(f"{type(exc).__name__}: {exc}".split())
    if len(collapsed) <= max_len:
        return collapsed
    return collapsed[: max_len - 3] + "..."


def probe_generate_liveness(
    client: Any,
    model_id: str,
    timeout_s: int = 12,
) -> LiveModelStatus:
    """1-token strict-JSON liveness probe ONLY. Does NOT rank reasoning capability."""
    _ = timeout_s  # Retained for contract parity with Blueprint Section 3.2.
    t0 = time.monotonic()
    try:
        raw = generate_json_content(
            client,
            model_id,
            'Reply with exact JSON: {"ok":true}',
            temperature=0.0,
            max_output_tokens=1024,
        )
    except Exception as exc:
        return LiveModelStatus(
            model_id=model_id,
            family=infer_family(model_id),
            live=False,
            json_ok=False,
            latency_ms=int((time.monotonic() - t0) * 1000),
            error=_sanitize_error_message(exc),
        )

    latency_ms = int((time.monotonic() - t0) * 1000)
    if not raw:
        return LiveModelStatus(
            model_id=model_id,
            family=infer_family(model_id),
            live=True,
            json_ok=False,
            latency_ms=latency_ms,
            error="EMPTY_RESPONSE",
        )

    try:
        parsed = parse_json_response(raw)
    except Exception as exc:
        return LiveModelStatus(
            model_id=model_id,
            family=infer_family(model_id),
            live=True,
            json_ok=False,
            latency_ms=latency_ms,
            error=f"JSON_PARSE_ERROR: {type(exc).__name__}",
        )

    json_ok = isinstance(parsed, dict) and parsed.get("ok") is True
    return LiveModelStatus(
        model_id=model_id,
        family=infer_family(model_id),
        live=True,
        json_ok=json_ok,
        latency_ms=latency_ms,
        error=None if json_ok else "LIVENESS_JSON_MISMATCH",
    )


def discover_live_generate_models(
    client: Any,
    policy: dict[str, Any],
    *,
    timeout_s: int = 12,
) -> list[LiveModelStatus]:
    """Discover and probe all non-excluded text-generation models."""
    return [
        probe_generate_liveness(client, mid, timeout_s=timeout_s)
        for mid in list_eligible_generate_models(client, policy)
    ]


def discover_embedding_model(client: Any, policy: dict[str, Any]) -> dict[str, Any] | None:
    """Dynamically discover the highest-preference live free-tier embedding model."""
    available: list[str] = []
    for model_obj in client.models.list():
        raw_name = str(getattr(model_obj, "name", "") or "")
        mid = raw_name.removeprefix("models/").strip()
        if not mid:
            continue
        actions = _extract_actions(model_obj)
        if "embedContent" not in actions or is_excluded(mid, policy):
            continue
        available.append(mid)

    target_dim = int(policy["embedding_output_dim"])
    for pattern in policy.get("preferred_embedding_patterns", []):
        matches = sorted(
            [m for m in available if re.search(pattern, m, re.IGNORECASE)],
            reverse=True,
        )
        for mid in matches:
            try:
                # Single-string invocation (required for gemini-embedding-2 so list inputs aren't aggregated)
                res = client.models.embed_content(
                    model=mid,
                    contents="RAL embedding liveness canary",
                    config={"output_dimensionality": target_dim},
                )
                vec = res.embeddings[0].values
                if len(vec) == target_dim:
                    return {"model_id": mid, "dim": target_dim, "live": True}
            except Exception:
                continue
    return None


def run_preflight(
    client: Any,
    *,
    policy_path: Path = Path("config/model_policy.yaml"),
    drive_store: Any | None = None,
) -> dict[str, Any]:
    """
    Execute Milestone 2 preflight checks: optional Drive canary, model discovery/liveness,
    and dynamic 768-dim embedding discovery.
    """
    circuit_breaker: str | None = None
    drive_canary_dict: dict[str, Any] | None = None

    if drive_store is not None:
        canary = drive_store.run_preflight_canary()
        drive_canary_dict = asdict(canary)
        if not canary.ok:
            circuit_breaker = "DRIVE_AUTH_FAIL"

    policy = load_policy(policy_path)
    statuses = discover_live_generate_models(client, policy)
    embedding_info = discover_embedding_model(client, policy)

    return {
        "policy_version": str(policy.get("schema_version", "2.1")),
        "circuit_breaker": circuit_breaker,
        "drive_canary": drive_canary_dict,
        "live_models": [s.to_dict() for s in statuses],
        "embedding_model": embedding_info,
    }

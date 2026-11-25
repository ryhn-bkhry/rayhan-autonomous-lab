"""Stage 2 & 3 Formalizer: Structured Markdown + verify.py kernel + stderr repair loop (Sections 6.3 & 7.2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
from typing import Any, Sequence
import uuid

import yaml

from ci.validate_node import (
    BANNED_FILLER_PHRASES,
    TAXONOMY_PATTERN,
    split_frontmatter,
    validate_node_content,
)
from ci.validate_vault import scan_text_for_leaks
from ral.canonicalize import (
    compute_assumptions_hash,
    compute_oracle_script_sha256,
    compute_quote_sha256,
    normalize_newlines,
    sha256_bytes,
)
from ral.director import clamp_taxonomy_code, get_active_taxonomy_leaves
from ral.model_router import ModelRouter
from ral.pipeline.oracles.base import OraclePluginRegistry, OracleResult
from ral.pipeline.scout import (
    DOMAIN_ORACLE_ROUTING_RUBRIC,
    ORACLE_PLUGIN_ALIASES,
    VALID_ORACLE_PLUGINS,
    ScoutCandidate,
    normalize_oracle_alias,
)
from ral.preflight import generate_json_content, parse_json_response
from ral.sources import SourceRecord, load_license_allowlist
from ral.storage.drive_ore import build_ore_uri

VALID_NODE_TYPES: tuple[str, ...] = (
    "AXIOM",
    "THEOREM",
    "EMPIRICAL_LAW",
    "ALGORITHM",
    "BRIDGE_ISOMORPHISM",
    "NEGATIVE_RESULT",
)

SANDBOX_INSTALLED_PACKAGES: tuple[str, ...] = (
    "sympy",
    "scipy",
    "numpy",
    "z3",
    "hypothesis",
    "pint",
)

SANDBOX_ALLOWED_STDLIB: tuple[str, ...] = (
    "math",
    "statistics",
    "fractions",
    "itertools",
    "collections",
    "functools",
    "random",
    "cmath",
    "decimal",
    "hashlib",
)

SANDBOX_FORBIDDEN_MISSING_PACKAGES: tuple[str, ...] = (
    "networkx",
    "control",
    "cvxpy",
    "pandas",
    "statsmodels",
    "sklearn",
    "cryptography",
    "matplotlib",
)

SANDBOX_FORBIDDEN_BUILTINS_AND_MODULES: tuple[str, ...] = (
    "os",
    "sys",
    "io",
    "pathlib",
    "open",
    "getattr",
    "setattr",
    "delattr",
    "eval",
    "exec",
    "subprocess",
    "socket",
    "requests",
    "tempfile",
    "inspect",
)

_OUTER_FENCE_RE = re.compile(
    r"^```(?:[a-zA-Z0-9_+-]+)?[ \t]*\n([\s\S]*?)(?:\n)?```$"
)
_UNESCAPED_DOLLAR_RE = re.compile(r"(?<!\\)\$")

_STAGE3_REDACT_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (re.compile(r"[A-Za-z]:[/\\]Users[/\\][^\s\"']+"), "<REDACTED_LOCAL_PATH>"),
    (re.compile(r"(?<![A-Za-z0-9])/(?:Users|home)/[^\s\"']+"), "<REDACTED_LOCAL_PATH>"),
    (re.compile(r"\bAIza[0-9A-Za-z\-_]{10,}\b"), "<REDACTED_KEY>"),
    (re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}\b"), "<REDACTED_TOKEN>"),
    (re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b"), "<REDACTED_TOKEN>"),
    (re.compile(r"\bya29\.[0-9A-Za-z\-_]{12,}\b"), "<REDACTED_TOKEN>"),
    (re.compile(r"\b1//[0-9A-Za-z\-_]{16,}\b"), "<REDACTED_TOKEN>"),
    (re.compile(r"\bAKIA[0-9A-Z]{16}\b"), "<REDACTED_KEY>"),
    (re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"), "<REDACTED_KEY>"),
    (re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |)PRIVATE KEY-----"), "<REDACTED_PRIVATE_KEY>"),
    (
        re.compile(
            r'"(?:client_secret|refresh_token|private_key|private_key_id|access_token)"\s*:\s*"[^"]{8,}"'
        ),
        '"<REDACTED_OAUTH_FIELD>"',
    ),
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b"), "<REDACTED_EMAIL>"),
)


def _strip_code_fences(raw: str | None) -> str:
    """Strip outer Markdown code fences from LLM-returned Python snippets."""
    text = normalize_newlines(str(raw or "")).strip()
    while text:
        match = _OUTER_FENCE_RE.match(text)
        if not match:
            break
        text = match.group(1).strip()
    return text


def _strip_outer_math_delimiters(raw: str | None) -> str:
    """Strip outer display/inline LaTeX math delimiters from `formal_statement_latex`."""
    text = normalize_newlines(str(raw or "")).strip()
    while text:
        if text.startswith("$$") and text.endswith("$$") and len(text) >= 4:
            inner = text[2:-2].strip()
            if "$$" not in inner:
                text = inner
                continue
        if text.startswith("\\[") and text.endswith("\\]") and len(text) >= 4:
            inner = text[2:-2].strip()
            if "\\[" not in inner and "\\]" not in inner:
                text = inner
                continue
        if text.startswith("\\(") and text.endswith("\\)") and len(text) >= 4:
            inner = text[2:-2].strip()
            if "\\(" not in inner and "\\)" not in inner:
                text = inner
                continue
        if (
            text.startswith("$")
            and text.endswith("$")
            and not text.startswith("$$")
            and not text.endswith("$$")
            and len(text) >= 2
        ):
            inner = text[1:-1].strip()
            if not _UNESCAPED_DOLLAR_RE.search(inner):
                text = inner
                continue
        break
    return text


def sanitize_stage3_stderr(stderr: str | None, *, max_len: int = 400) -> str:
    """Redact local paths/secrets from Stage 3 stderr and keep at most `max_len` trailing chars."""
    msg = " ".join(str(stderr or "").split())
    for pat, repl in _STAGE3_REDACT_PATTERNS:
        msg = pat.sub(repl, msg)
    cap = max(0, int(max_len))
    if len(msg) <= cap:
        return msg
    if cap <= 3:
        return msg[-cap:] if cap > 0 else ""
    return "..." + msg[-(cap - 3) :]


def classify_stage3_error(stderr: str | None) -> str:
    """Classify Stage 3 failure stderr into distinct diagnostic error categories."""
    text = str(stderr or "")
    if "DISABLED_OR_UNKNOWN_ORACLE" in text:
        return "DISABLED_OR_UNKNOWN_ORACLE"
    if "AST_LINT_REJECTED" in text:
        return "AST_CHEAT_OR_ASSERT_FAIL"
    if "NODE_VALIDATION_FAILED" in text:
        return "NODE_SCHEMA_OR_PROSE_FAIL"
    return "STAGE3_ORACLE_EXEC_FAIL"


@dataclass(frozen=True)
class Stage3Diagnostics:
    """Structured Stage 3 execution and repair diagnostics."""

    oracle_plugin: str
    formalizer_model_id: str
    repair_attempts: int
    stage3_error_code: str | None
    stage3_stderr_tail: str

    def to_dict(self) -> dict[str, Any]:
        """Serialize diagnostics to a JSON-compatible dictionary."""
        return {
            "oracle_plugin": self.oracle_plugin,
            "formalizer_model_id": self.formalizer_model_id,
            "repair_attempts": int(self.repair_attempts),
            "stage3_error_code": self.stage3_error_code,
            "stage3_stderr_tail": self.stage3_stderr_tail,
        }


def normalize_taxonomy_code(raw: str | None, *, fallback: str = "math.analysis") -> str:
    """Normalize taxonomy codes by converting underscores/spaces to hyphens in lowercase."""
    text = (
        str(raw or "")
        .strip()
        .lower()
        .replace("_", "-")
        .replace("/", "-")
        .replace(" ", "-")
    )
    segments: list[str] = []
    for part in text.split("."):
        cleaned = re.sub(r"-{2,}", "-", part).strip("-")
        if cleaned:
            segments.append(cleaned)
    if not segments:
        return fallback
    return ".".join(segments)


def _schema_constraints_prompt() -> str:
    banned_list = ", ".join(repr(p) for p in BANNED_FILLER_PHRASES)
    return (
        "Strict Node Schema & Prose Constraints:\n"
        f"- `node_type` MUST be exactly one of: {', '.join(VALID_NODE_TYPES)}.\n"
        f"- `taxonomy_code` MUST match regex `{TAXONOMY_PATTERN}` (lowercase dot-separated segments "
        "using hyphens `-` only, NEVER underscores `_`).\n"
        "- Active natural-language prose outside LaTeX math, tables, and code MUST be <= 350 words.\n"
        f"- NEVER use any banned conversational filler phrases: {banned_list}.\n"
        "- `title` must be >= 3 characters; `formal_signature` must be a non-empty list of strings.\n"
        "- `formal_statement_latex` must contain raw LaTeX ONLY (NO outer `$$...$$` or `\\[...\\]` delimiters)."
    )


def _oracle_sandbox_contract_prompt() -> str:
    installed_pkgs = ", ".join(SANDBOX_INSTALLED_PACKAGES)
    allowed_stdlib = ", ".join(SANDBOX_ALLOWED_STDLIB)
    missing_pkgs = ", ".join(SANDBOX_FORBIDDEN_MISSING_PACKAGES)
    banned_apis = ", ".join(SANDBOX_FORBIDDEN_BUILTINS_AND_MODULES)
    return (
        "Oracle Plugin, Sandbox Package & AST Contract:\n"
        '- `oracle_plugin` MUST be strictly one of: "sympy_numeric" | "z3_smt" | "empirical_stats" '
        "(NEVER output aliases like sympy/z3/scipy or disabled names like lean4_external).\n"
        f"{DOMAIN_ORACLE_ROUTING_RUBRIC}\n"
        f"- Installed third-party sandbox packages ONLY: {installed_pkgs}.\n"
        f"- Allowed pure Python stdlib modules: {allowed_stdlib}.\n"
        "- Required plugin imports: `sympy_numeric` MUST import at least one of (sympy, numpy, scipy, pint, math); "
        "`z3_smt` MUST import at least one of (z3, hypothesis); "
        "`empirical_stats` MUST import at least one of (scipy, numpy, pint, statistics, math).\n"
        f"- FORBIDDEN missing packages (NOT installed in sandbox): {missing_pkgs}.\n"
        f"- FORBIDDEN builtins & modules (rejected by AST linter): {banned_apis}.\n"
        "- Verification kernel (`verify_py`) MUST contain >= 3 syntactically distinct `assert` statements "
        "over computed variables (NEVER constant literals like `assert True` or reflexive checks like `assert x == x`).\n"
        "- For SymPy symbolic identities, assert `sp.simplify(lhs - rhs) == 0`; for floating-point checks, "
        "assert `np.isclose(...)` or `math.isclose(...)`.\n"
        "- NEVER use bare `except:` blocks (or catch `AssertionError` / `BaseException`).\n"
        "- Output raw Python inside `verify_py` ONLY: NO markdown code fences (```) and NO `## ` or `### ` headers inside `verify_py`."
    )


def _synthetic_source_for_validation() -> SourceRecord:
    return SourceRecord(
        source_id="arxiv:0000.00000v1",
        source_license="CC-BY-4.0",
        source_format="arxiv_tex",
        title="Stage 3 Validation Context",
        pub_year=2024,
        peer_reviewed=True,
        retracted=False,
        canonical_doi=None,
        arxiv_base_id="0000.00000",
        arxiv_version=1,
        citation_count=0,
        raw_bytes=b"stage3-validation-placeholder",
        raw_ext="tar.gz",
        extracted_text="placeholder",
    )


@dataclass(frozen=True)
class FormalizedNodeDraft:
    """Verified or candidate formalization produced by Stage 2."""

    ral_id: str
    node_type: str
    taxonomy_code: str
    title: str
    formal_signature: tuple[str, ...]
    assumptions: tuple[str, ...]
    formal_statement_latex: str
    symbol_rows: tuple[dict[str, str], ...]
    failure_boundaries: tuple[str, ...]
    provenance_summary: str
    oracle_plugin: str
    verify_py: str
    formalizer_model: str
    model_capability_epoch: int
    oracle_timeout_s: int = 30


def generate_ral_id(taxonomy_code: str, *, entropy: str | None = None) -> str:
    """Generate an immutable ID matching `^RAL-[A-Z]{2,6}-[0-9A-Z]{10}$`."""
    macro = taxonomy_code.split(".")[0].upper()
    if not (2 <= len(macro) <= 6 and macro.isalpha()):
        macro = "GEN"
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    digest = (
        hashlib.sha256(entropy.encode("utf-8")).digest()
        if entropy is not None
        else uuid.uuid4().bytes
    )
    suffix = "".join(alphabet[b % len(alphabet)] for b in digest[:10])
    return f"RAL-{macro}-{suffix}"


def render_node_markdown(
    draft: FormalizedNodeDraft,
    *,
    epistemic_tier: str,
    oracle_status: str,
    skeptic_model: str,
    source: SourceRecord,
    quote_sha256: str,
    quote_char_len: int,
    source_locator: str,
    re_evaluate_after: str,
    consensus_version: int = 1,
    ore_uri: str | None = None,
    ore_sha256: str | None = None,
    verified_at: str | None = None,
    effective_date: str | None = None,
    active_challenges: Sequence[str] | None = None,
    archived_details: str = "",
) -> str:
    """Render a deterministic Section 7.2 canonical Markdown vault node."""
    now_dt = datetime.now(timezone.utc)
    ts_iso = verified_at or now_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    eff_date = effective_date or ts_iso[:10]
    ore_sha = ore_sha256 or sha256_bytes(source.raw_bytes)
    uri = ore_uri or build_ore_uri(ore_sha, source.raw_ext)

    frontmatter = {
        "ral_schema": "2.1",
        "ral_id": draft.ral_id,
        "node_type": draft.node_type,
        "epistemic_tier": epistemic_tier,
        "consensus_version": int(consensus_version),
        "taxonomy_code": draft.taxonomy_code,
        "title": draft.title,
        "formal_signature": list(draft.formal_signature),
        "assumptions": list(draft.assumptions),
        "assumptions_hash": compute_assumptions_hash(draft.assumptions),
        "verification": {
            "oracle_plugin": draft.oracle_plugin,
            "oracle_status": oracle_status,
            "oracle_script_sha256": compute_oracle_script_sha256(draft.verify_py),
            "verified_at": ts_iso,
            "formalizer_model": draft.formalizer_model,
            "skeptic_model": skeptic_model,
            "model_capability_epoch": int(draft.model_capability_epoch),
        },
        "provenance": [
            {
                "source_id": source.source_id,
                "source_license": source.source_license,
                "source_format": source.source_format,
                "source_locator": source_locator,
                "ore_uri": uri,
                "ore_sha256": ore_sha,
                "quote_sha256": quote_sha256,
                "quote_char_len": int(quote_char_len),
                "pub_year": int(source.pub_year),
                "peer_reviewed": bool(source.peer_reviewed),
                "retracted": bool(source.retracted),
            }
        ],
        "re_evaluate_after": re_evaluate_after,
    }
    yaml_block = yaml.safe_dump(frontmatter, sort_keys=False).strip()

    symbol_lines = ["| Symbol | Meaning | Domain | SI Units |", "| :--- | :--- | :--- | :--- |"]
    for row in draft.symbol_rows:
        symbol_lines.append(
            f"| {row.get('symbol', '-')} | {row.get('meaning', '-')} | "
            f"{row.get('domain', '-')} | {row.get('si_units', '-')} |"
        )
    if len(symbol_lines) == 2:
        symbol_lines.append("| $-$ | Formal invariant | $\\mathbb{R}$ | $-$ |")

    boundaries = "\n".join(f"- **Breaks when:** {b}" for b in draft.failure_boundaries)
    if not boundaries:
        boundaries = "- **Breaks when:** Stated frontmatter assumptions are violated."

    challenges_md = (
        "\n".join(f"- {c}" for c in active_challenges)
        if active_challenges
        else "- *(None)*"
    )
    archived_md = archived_details.strip() if archived_details.strip() else "*(None)*"

    return (
        f"---\n{yaml_block}\n---\n\n"
        f"## Active Consensus (v{consensus_version} — Effective {eff_date})\n\n"
        f"### Formal Statement\n"
        f"$${draft.formal_statement_latex.strip()}$$\n\n"
        + "\n".join(symbol_lines)
        + "\n\n"
        f"### Failure Boundaries & Known Pathologies\n"
        f"{boundaries}\n\n"
        f"### Executable Verification Kernel\n"
        f"```python\n{draft.verify_py.strip()}\n```\n\n"
        f"### Canonical Provenance Anchors\n"
        f"- `{source.source_id}` (`{source_locator}`): {draft.provenance_summary}\n\n"
        f"## Active Challenges & Open Anomalies\n"
        f"{challenges_md}\n\n"
        f"## Superseded & Archived\n"
        f"{archived_md}\n"
    )


def append_rejected_record(
    repo_root: Path,
    record: dict[str, Any],
    *,
    now: datetime | None = None,
) -> Path:
    """Append a rejected candidate entry to `ledger/rejected/YYYY-MM.ndjson` (Section 7.3)."""
    ts = now or datetime.now(timezone.utc)
    rejected_dir = Path(repo_root) / "ledger" / "rejected"
    rejected_dir.mkdir(parents=True, exist_ok=True)
    target = rejected_dir / f"{ts:%Y-%m}.ndjson"
    with target.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, sort_keys=False) + "\n")
    return target


class FormalizerStage:
    """Stage 2 & 3 Formalizer producing structured nodes + sandbox-verified Python kernels."""

    def __init__(
        self,
        client: Any,
        model_router: ModelRouter,
        registry: OraclePluginRegistry,
        *,
        oracle_config_path: Path = Path("config/oracle_plugins.yaml"),
    ) -> None:
        self._client = client
        self._router = model_router
        self._registry = registry
        self._oracle_config = (
            OraclePluginRegistry.load_config(oracle_config_path)
            if oracle_config_path.exists()
            else None
        )
        self.last_stage3_diagnostics: Stage3Diagnostics | None = None

    def _is_plugin_enabled(self, plugin_name: str) -> bool:
        if plugin_name not in self._registry.list_registered():
            return False
        if self._oracle_config is None:
            return True
        return OraclePluginRegistry.is_enabled(plugin_name, self._oracle_config)

    def _resolve_enabled_oracle(
        self,
        primary: str | None,
        fallback: str | None = None,
    ) -> str:
        """
        Map `primary` and `fallback` oracle identifiers (including common aliases) to an enabled
        canonical plugin (`sympy_numeric`, `empirical_stats`, `z3_smt`), or fall back to the
        first enabled plugin in priority order.
        """
        candidates: list[str] = []
        for raw in (primary, fallback):
            if raw is not None and str(raw).strip():
                mapped = normalize_oracle_alias(str(raw))
                if mapped and mapped not in candidates:
                    candidates.append(mapped)

        for cand in candidates:
            if self._is_plugin_enabled(cand):
                return cand

        for default_name in ("sympy_numeric", "empirical_stats", "z3_smt"):
            if self._is_plugin_enabled(default_name):
                return default_name

        for registered in self._registry.list_registered():
            if self._is_plugin_enabled(registered):
                return registered

        return candidates[0] if candidates else "sympy_numeric"

    def _draft_from_payload(
        self,
        payload: dict[str, Any],
        *,
        ral_id: str,
        formalizer_model: str,
        epoch: int,
        fallback_candidate: ScoutCandidate,
        active_leaves: set[str] | None = None,
        target_domain: str | None = None,
    ) -> FormalizedNodeDraft:
        oracle_name = self._resolve_enabled_oracle(
            str(payload.get("oracle_plugin") or ""),
            fallback_candidate.recommended_oracle,
        )

        assumptions = tuple(
            str(a).strip()
            for a in (payload.get("assumptions") or fallback_candidate.assumptions)
            if str(a).strip()
        )
        formal_sig = tuple(
            str(s).strip()
            for s in (payload.get("formal_signature") or fallback_candidate.formal_signature)
            if str(s).strip()
        )
        symbol_rows = tuple(
            {
                "symbol": str(r.get("symbol", "-")),
                "meaning": str(r.get("meaning", "-")),
                "domain": str(r.get("domain", "-")),
                "si_units": str(r.get("si_units", "-")),
            }
            for r in (payload.get("symbol_rows") or [])
            if isinstance(r, dict)
        )
        boundaries = tuple(
            str(b).strip() for b in (payload.get("failure_boundaries") or []) if str(b).strip()
        )
        raw_node_type = str(payload.get("node_type") or fallback_candidate.node_type).strip()
        norm_node_type = (
            raw_node_type.upper()
            if raw_node_type.upper() in VALID_NODE_TYPES
            else raw_node_type
        )
        fallback_tax = normalize_taxonomy_code(fallback_candidate.taxonomy_code)
        if active_leaves is not None:
            fallback_tax = clamp_taxonomy_code(
                fallback_tax,
                active_leaves=active_leaves,
                target_domain=target_domain,
            )
            norm_tax = clamp_taxonomy_code(
                payload.get("taxonomy_code"),
                active_leaves=active_leaves,
                target_domain=target_domain,
                fallback=fallback_tax,
            )
        else:
            norm_tax = normalize_taxonomy_code(payload.get("taxonomy_code"), fallback=fallback_tax)
        return FormalizedNodeDraft(
            ral_id=ral_id,
            node_type=norm_node_type,
            taxonomy_code=norm_tax,
            title=str(payload.get("title") or fallback_candidate.title).strip(),
            formal_signature=formal_sig or ("operator:identity",),
            assumptions=assumptions,
            formal_statement_latex=_strip_outer_math_delimiters(
                payload.get("formal_statement_latex")
            ),
            symbol_rows=symbol_rows,
            failure_boundaries=boundaries,
            provenance_summary=str(
                payload.get("provenance_summary") or fallback_candidate.claim_summary
            ).strip(),
            oracle_plugin=oracle_name,
            verify_py=_strip_code_fences(payload.get("verify_py")),
            formalizer_model=formalizer_model,
            model_capability_epoch=epoch,
            oracle_timeout_s=int(payload.get("oracle_timeout_s") or 30),
        )

    def _run_stage3_with_repairs(
        self,
        draft: FormalizedNodeDraft,
        *,
        max_repairs: int,
        repair_context: str,
        source: SourceRecord | None = None,
        quote_sha256: str | None = None,
        quote_char_len: int | None = None,
        source_locator: str = "tex:sec1",
        allowed_licenses: set[str] | None = None,
        active_leaves: set[str] | None = None,
        target_domain: str | None = None,
    ) -> tuple[FormalizedNodeDraft | None, OracleResult, Stage3Diagnostics]:
        current = FormalizedNodeDraft(
            ral_id=draft.ral_id,
            node_type=draft.node_type,
            taxonomy_code=draft.taxonomy_code,
            title=draft.title,
            formal_signature=draft.formal_signature,
            assumptions=draft.assumptions,
            formal_statement_latex=_strip_outer_math_delimiters(draft.formal_statement_latex),
            symbol_rows=draft.symbol_rows,
            failure_boundaries=draft.failure_boundaries,
            provenance_summary=draft.provenance_summary,
            oracle_plugin=draft.oracle_plugin,
            verify_py=_strip_code_fences(draft.verify_py),
            formalizer_model=draft.formalizer_model,
            model_capability_epoch=draft.model_capability_epoch,
            oracle_timeout_s=draft.oracle_timeout_s,
        )
        last_result = OracleResult(
            passed=False,
            plugin_name=current.oracle_plugin,
            stdout="",
            stderr="ORACLE_NOT_RUN",
            elapsed_ms=0,
        )
        val_src = source or _synthetic_source_for_validation()
        val_quote_sha = (
            quote_sha256
            if quote_sha256 and quote_sha256.startswith("sha256:")
            else compute_quote_sha256(val_src.title)
        )
        val_quote_len = max(1, int(quote_char_len if quote_char_len is not None else len(val_src.title)))

        attempts_used = 0
        for attempt in range(max_repairs + 1):
            attempts_used = attempt
            plugin_enabled = self._is_plugin_enabled(current.oracle_plugin)
            if not plugin_enabled:
                lint_errors = [f"DISABLED_OR_UNKNOWN_ORACLE: {current.oracle_plugin}"]
                plugin = None
            else:
                plugin = self._registry.get(current.oracle_plugin)
                lint_errors = plugin.lint(current.verify_py)

            preview_md = render_node_markdown(
                current,
                epistemic_tier="T2_ESTABLISHED",
                oracle_status="PASS",
                skeptic_model="gemma-4-27b-it",
                source=val_src,
                quote_sha256=val_quote_sha,
                quote_char_len=val_quote_len,
                source_locator=source_locator,
                re_evaluate_after="2028-01-01",
                consensus_version=1,
                verified_at="2026-11-24T00:00:00Z",
            )
            val_res = validate_node_content(
                preview_md,
                allowed_licenses=allowed_licenses,
                expected_filename_stem=current.ral_id,
            )
            node_errors = (
                [e for e in val_res.errors if not e.startswith("AST_LINT_ERROR:")]
                if lint_errors
                else list(val_res.errors)
            )
            leak_findings = scan_text_for_leaks(preview_md, f"{current.ral_id}.md")
            if leak_findings:
                node_errors.extend(leak_findings)

            if lint_errors or node_errors:
                diag_parts: list[str] = []
                if not plugin_enabled:
                    diag_parts.append(f"DISABLED_OR_UNKNOWN_ORACLE: {current.oracle_plugin}")
                elif lint_errors:
                    diag_parts.append("AST_LINT_REJECTED: " + "; ".join(lint_errors))
                if node_errors:
                    diag_parts.append("NODE_VALIDATION_FAILED: " + "; ".join(node_errors))
                last_result = OracleResult(
                    passed=False,
                    plugin_name=current.oracle_plugin,
                    stdout="",
                    stderr=" | ".join(diag_parts),
                    elapsed_ms=0,
                )
            elif plugin is not None:
                last_result = plugin.execute(current.verify_py, timeout_s=current.oracle_timeout_s)
                if last_result.passed:
                    diag = Stage3Diagnostics(
                        oracle_plugin=current.oracle_plugin,
                        formalizer_model_id=current.formalizer_model,
                        repair_attempts=attempt,
                        stage3_error_code=None,
                        stage3_stderr_tail=sanitize_stage3_stderr(last_result.stderr),
                    )
                    self.last_stage3_diagnostics = diag
                    return current, last_result, diag

            if attempt == max_repairs:
                break

            repair_prompt = (
                "Repair the executable verification script `verify_py` (and fix any schema, "
                "prose word-cap, banned-phrase, oracle selection, or assumption errors) to pass static "
                "anti-cheat AST linting, `validate_node_content()`, and the air-gapped Docker oracle.\n"
                f"Context: {repair_context}\n"
                f"{_schema_constraints_prompt()}\n"
                f"{_oracle_sandbox_contract_prompt()}\n"
                f"Current oracle_plugin: {current.oracle_plugin}\n"
                f"Current node_type: {current.node_type}\n"
                f"Current taxonomy_code: {current.taxonomy_code}\n"
                f"Current title: {current.title}\n"
                f"Current formal_signature: {json.dumps(list(current.formal_signature))}\n"
                f"Current assumptions: {json.dumps(list(current.assumptions))}\n"
                f"Current formal_statement_latex: {current.formal_statement_latex}\n"
                f"Current failure_boundaries: {json.dumps(list(current.failure_boundaries))}\n"
                f"Current provenance_summary: {current.provenance_summary}\n"
                f"Current verify_py:\n{current.verify_py}\n\n"
                f"Oracle / validation stdout diagnostic:\n{last_result.stdout}\n\n"
                f"Oracle / validation stderr diagnostic:\n{last_result.stderr}\n\n"
                'Return strict JSON: {"oracle_plugin": "sympy_numeric" | "z3_smt" | "empirical_stats", '
                '"verify_py": str, "assumptions": [str], '
                '"node_type": str, "taxonomy_code": str, "title": str, '
                '"formal_signature": [str], "formal_statement_latex": str, '
                '"failure_boundaries": [str], "provenance_summary": str}.'
            )

            def _repair_call(model_id: str) -> str:
                return generate_json_content(
                    self._client,
                    model_id,
                    repair_prompt,
                    temperature=0.0,
                )

            used_model, raw_repair = self._router.call_with_fallback("formalizer", _repair_call)
            repaired = parse_json_response(raw_repair)
            new_assump = repaired.get("assumptions")
            new_sig = repaired.get("formal_signature")
            new_bounds = repaired.get("failure_boundaries")
            new_rows = repaired.get("symbol_rows")
            rep_node_type_raw = str(repaired.get("node_type") or current.node_type).strip()
            rep_node_type = (
                rep_node_type_raw.upper()
                if rep_node_type_raw.upper() in VALID_NODE_TYPES
                else rep_node_type_raw
            )
            if active_leaves is not None:
                rep_tax = clamp_taxonomy_code(
                    repaired.get("taxonomy_code") or current.taxonomy_code,
                    active_leaves=active_leaves,
                    target_domain=target_domain,
                    fallback=current.taxonomy_code,
                )
            else:
                rep_tax = (
                    normalize_taxonomy_code(repaired.get("taxonomy_code"), fallback=current.taxonomy_code)
                    if repaired.get("taxonomy_code")
                    else current.taxonomy_code
                )
            rep_oracle_raw = repaired.get("oracle_plugin")
            rep_oracle = (
                self._resolve_enabled_oracle(str(rep_oracle_raw), current.oracle_plugin)
                if rep_oracle_raw is not None and str(rep_oracle_raw).strip()
                else current.oracle_plugin
            )
            parsed_rows = (
                tuple(
                    {
                        "symbol": str(r.get("symbol", "-")),
                        "meaning": str(r.get("meaning", "-")),
                        "domain": str(r.get("domain", "-")),
                        "si_units": str(r.get("si_units", "-")),
                    }
                    for r in new_rows
                    if isinstance(r, dict)
                )
                if isinstance(new_rows, list) and new_rows
                else current.symbol_rows
            )
            current = FormalizedNodeDraft(
                ral_id=current.ral_id,
                node_type=rep_node_type,
                taxonomy_code=rep_tax,
                title=str(repaired.get("title") or current.title).strip(),
                formal_signature=(
                    tuple(str(s).strip() for s in new_sig if str(s).strip())
                    if isinstance(new_sig, list) and new_sig
                    else current.formal_signature
                ),
                assumptions=(
                    tuple(str(a).strip() for a in new_assump if str(a).strip())
                    if isinstance(new_assump, list) and new_assump
                    else current.assumptions
                ),
                formal_statement_latex=_strip_outer_math_delimiters(
                    repaired.get("formal_statement_latex") or current.formal_statement_latex
                ),
                symbol_rows=parsed_rows,
                failure_boundaries=(
                    tuple(str(b).strip() for b in new_bounds if str(b).strip())
                    if isinstance(new_bounds, list) and new_bounds
                    else current.failure_boundaries
                ),
                provenance_summary=str(
                    repaired.get("provenance_summary") or current.provenance_summary
                ).strip(),
                oracle_plugin=rep_oracle,
                verify_py=_strip_code_fences(repaired.get("verify_py") or current.verify_py),
                formalizer_model=used_model,
                model_capability_epoch=current.model_capability_epoch,
                oracle_timeout_s=current.oracle_timeout_s,
            )

        fail_diag = Stage3Diagnostics(
            oracle_plugin=current.oracle_plugin,
            formalizer_model_id=current.formalizer_model,
            repair_attempts=attempts_used,
            stage3_error_code=classify_stage3_error(last_result.stderr),
            stage3_stderr_tail=sanitize_stage3_stderr(last_result.stderr),
        )
        self.last_stage3_diagnostics = fail_diag
        return None, last_result, fail_diag

    def formalize_and_verify(
        self,
        candidate: ScoutCandidate,
        source: SourceRecord,
        *,
        max_repairs: int = 2,
        repo_root: Path | None = None,
        current_epoch: int | None = None,
        ral_id: str | None = None,
        target_domain: str | None = None,
        return_diagnostics: bool = False,
    ) -> tuple[Any, ...]:
        """
        Run Stage 2 formalization and Stage 3 AST lint + `validate_node_content` + Docker oracle
        with up to `max_repairs` repairs.

        Logs persistent Stage 3 failures into `ledger/rejected/YYYY-MM.ndjson` (excluding
        configuration/plugin availability errors such as `DISABLED_OR_UNKNOWN_ORACLE`).
        """
        epoch = current_epoch if current_epoch is not None else self._router.capability_epoch
        suggested_oracle = self._resolve_enabled_oracle(candidate.recommended_oracle, "sympy_numeric")
        active_leaves = get_active_taxonomy_leaves(repo_root)
        suggested_tax = clamp_taxonomy_code(
            candidate.taxonomy_code,
            active_leaves=active_leaves,
            target_domain=target_domain,
        )
        target_hint = (
            f"Active Target Leaf Domain (default `taxonomy_code` here): {target_domain}\n"
            if target_domain
            else ""
        )
        prompt = (
            "Formalize this verified scientific quote into canonical v2.1 YAML metadata, "
            "concise LaTeX Formal Statement, Symbol table, Failure Boundaries, and a standalone "
            "executable Python verification script `verify_py` (>=3 distinct non-trivial assertions).\n"
            f"{_schema_constraints_prompt()}\n"
            f"{_oracle_sandbox_contract_prompt()}\n"
            f"Candidate Title: {candidate.title}\n"
            f"Claim Summary: {candidate.claim_summary}\n"
            f"Verbatim Quote: {candidate.quote_text}\n"
            f"Suggested Taxonomy (existing active leaf ONLY; NEVER invent sub-branches): {suggested_tax}\n"
            f"{target_hint}"
            f"Suggested Oracle: {suggested_oracle}\n"
            f"Seed Assumptions: {json.dumps(list(candidate.assumptions))}\n\n"
            'Return strict JSON: {"node_type": str, "taxonomy_code": str, "title": str, '
            '"formal_signature": [str], "assumptions": [str], "formal_statement_latex": str, '
            '"symbol_rows": [{"symbol": str, "meaning": str, "domain": str, "si_units": str}], '
            '"failure_boundaries": [str], "provenance_summary": str, '
            '"oracle_plugin": "sympy_numeric" | "z3_smt" | "empirical_stats", "verify_py": str}.'
        )

        def _formalize_call(model_id: str) -> str:
            return generate_json_content(
                self._client,
                model_id,
                prompt,
                temperature=0.0,
            )

        used_model, raw_json = self._router.call_with_fallback("formalizer", _formalize_call)
        payload = parse_json_response(raw_json)
        norm_tax_for_id = clamp_taxonomy_code(
            payload.get("taxonomy_code"),
            active_leaves=active_leaves,
            target_domain=target_domain,
            fallback=suggested_tax,
        )
        assigned_id = ral_id or generate_ral_id(
            norm_tax_for_id,
            entropy=f"{source.source_id}:{candidate.quote_sha256}",
        )
        draft = self._draft_from_payload(
            payload,
            ral_id=assigned_id,
            formalizer_model=used_model,
            epoch=epoch,
            fallback_candidate=candidate,
            active_leaves=active_leaves,
            target_domain=target_domain,
        )

        allowed_licenses: set[str] | None = None
        if repo_root is not None:
            allow_path = Path(repo_root) / "config" / "license_allowlist.yaml"
            if allow_path.exists():
                allowed_licenses = load_license_allowlist(allow_path)

        verified_draft, oracle_res, stage3_diag = self._run_stage3_with_repairs(
            draft,
            max_repairs=max_repairs,
            repair_context=f"{candidate.title} ({source.source_id})",
            source=source,
            quote_sha256=candidate.quote_sha256,
            quote_char_len=candidate.quote_char_len,
            source_locator=candidate.source_locator,
            allowed_licenses=allowed_licenses,
            active_leaves=active_leaves,
            target_domain=target_domain,
        )
        self.last_stage3_diagnostics = stage3_diag
        if verified_draft is None and repo_root is not None:
            err_code = stage3_diag.stage3_error_code or classify_stage3_error(oracle_res.stderr)
            if err_code != "DISABLED_OR_UNKNOWN_ORACLE":
                append_rejected_record(
                    repo_root,
                    {
                        "source_sha256": sha256_bytes(source.raw_bytes),
                        "canonical_doi": source.canonical_doi,
                        "arxiv_base_id": source.arxiv_base_id,
                        "stage_failed": "STAGE3_ORACLE",
                        "reason": err_code,
                        "oracle_plugin": stage3_diag.oracle_plugin,
                        "formalizer_model_id": stage3_diag.formalizer_model_id,
                        "repair_attempts": stage3_diag.repair_attempts,
                        "stage3_error_code": err_code,
                        "stage3_stderr_tail": stage3_diag.stage3_stderr_tail,
                        "retry_on_model_upgrade": True,
                        "model_capability_epoch": epoch,
                        "rejected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                    },
                )
        if return_diagnostics:
            return verified_draft, oracle_res, stage3_diag
        return verified_draft, oracle_res

    def generate_correspondence_script(
        self,
        existing_node_markdown: str,
        upgraded_draft: FormalizedNodeDraft,
    ) -> str:
        """
        Generate a standalone Python correspondence test verifying that `upgraded_draft`
        reproduces/generalizes the prior Active Consensus under its stated assumptions.
        """
        existing_fm, existing_body = split_frontmatter(existing_node_markdown)
        prompt = (
            "Write a standalone executable Python `correspondence_script` (>=1 non-trivial `assert`, "
            "using sympy/numpy/scipy/z3/math only, NO forbidden imports like os/sys/pathlib/open) "
            "demonstrating that the upgraded theorem reproduces or strictly generalizes the prior "
            "Active Consensus in the prior assumptions' limit.\n"
            f"Existing Node ID: {existing_fm.get('ral_id')}\n"
            f"Existing Title: {existing_fm.get('title')}\n"
            f"Existing Assumptions: {json.dumps(list(existing_fm.get('assumptions') or []))}\n"
            f"Existing Body Excerpt:\n{existing_body[:1600]}\n\n"
            f"Upgraded Title: {upgraded_draft.title}\n"
            f"Upgraded Formal Statement: {upgraded_draft.formal_statement_latex}\n"
            f"Upgraded Assumptions: {json.dumps(list(upgraded_draft.assumptions))}\n\n"
            'Return strict JSON: {"correspondence_script": str}.'
        )

        def _corr_call(model_id: str) -> str:
            return generate_json_content(
                self._client,
                model_id,
                prompt,
                temperature=0.0,
            )

        _, raw_json = self._router.call_with_fallback("formalizer", _corr_call)
        payload = parse_json_response(raw_json)
        return _strip_code_fences(payload.get("correspondence_script"))

    def tighten_assumptions(
        self,
        draft: FormalizedNodeDraft,
        counterexample_summary: str,
        missing_assumptions: Sequence[str],
        *,
        max_repairs: int = 1,
        source: SourceRecord | None = None,
        quote_sha256: str | None = None,
        quote_char_len: int | None = None,
        source_locator: str = "tex:sec1",
        allowed_licenses: set[str] | None = None,
    ) -> tuple[FormalizedNodeDraft | None, OracleResult]:
        """Tighten `assumptions[]` and update `verify_py` in response to a Skeptic counterexample."""
        prompt = (
            "An adversarial Skeptic produced a machine-verified counterexample against your formalization. "
            "Tighten `assumptions[]`, update `failure_boundaries` and `verify_py`, and keep >=3 distinct "
            "non-trivial assertions.\n"
            f"{_schema_constraints_prompt()}\n"
            f"{_oracle_sandbox_contract_prompt()}\n"
            f"Title: {draft.title}\n"
            f"Current Formal Statement: {draft.formal_statement_latex}\n"
            f"Current Assumptions: {json.dumps(list(draft.assumptions))}\n"
            f"Skeptic Counterexample Summary: {counterexample_summary}\n"
            f"Suggested Missing Assumptions: {json.dumps(list(missing_assumptions))}\n\n"
            'Return strict JSON: {"assumptions": [str], "failure_boundaries": [str], '
            '"formal_statement_latex": str, "verify_py": str}.'
        )

        def _tighten_call(model_id: str) -> str:
            return generate_json_content(
                self._client,
                model_id,
                prompt,
                temperature=0.0,
            )

        used_model, raw_json = self._router.call_with_fallback("formalizer", _tighten_call)
        payload = parse_json_response(raw_json)
        tightened = FormalizedNodeDraft(
            ral_id=draft.ral_id,
            node_type=draft.node_type,
            taxonomy_code=draft.taxonomy_code,
            title=draft.title,
            formal_signature=draft.formal_signature,
            assumptions=tuple(
                str(a).strip()
                for a in (payload.get("assumptions") or draft.assumptions)
                if str(a).strip()
            ),
            formal_statement_latex=_strip_outer_math_delimiters(
                payload.get("formal_statement_latex") or draft.formal_statement_latex
            ),
            symbol_rows=draft.symbol_rows,
            failure_boundaries=tuple(
                str(b).strip()
                for b in (payload.get("failure_boundaries") or draft.failure_boundaries)
                if str(b).strip()
            ),
            provenance_summary=draft.provenance_summary,
            oracle_plugin=draft.oracle_plugin,
            verify_py=_strip_code_fences(payload.get("verify_py") or draft.verify_py),
            formalizer_model=used_model,
            model_capability_epoch=draft.model_capability_epoch,
            oracle_timeout_s=draft.oracle_timeout_s,
        )
        verified, res, diag = self._run_stage3_with_repairs(
            tightened,
            max_repairs=max_repairs,
            repair_context=f"Assumption tightening for {draft.ral_id}",
            source=source,
            quote_sha256=quote_sha256,
            quote_char_len=quote_char_len,
            source_locator=source_locator,
            allowed_licenses=allowed_licenses,
        )
        self.last_stage3_diagnostics = diag
        return verified, res

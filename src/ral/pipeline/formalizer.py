"""Stage 2 & 3 Formalizer: Structured Markdown + verify.py kernel + stderr repair loop (Sections 6.3 & 7.2)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
from typing import Any, Sequence
import uuid

import yaml

from ral.canonicalize import (
    compute_assumptions_hash,
    compute_oracle_script_sha256,
    sha256_bytes,
)
from ral.model_router import ModelRouter
from ral.pipeline.oracles.base import OraclePluginRegistry, OracleResult
from ral.pipeline.scout import ScoutCandidate
from ral.preflight import generate_json_content, parse_json_response
from ral.sources import SourceRecord
from ral.storage.drive_ore import build_ore_uri


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

    def _is_plugin_enabled(self, plugin_name: str) -> bool:
        if plugin_name not in self._registry.list_registered():
            return False
        if self._oracle_config is None:
            return True
        return OraclePluginRegistry.is_enabled(plugin_name, self._oracle_config)

    def _draft_from_payload(
        self,
        payload: dict[str, Any],
        *,
        ral_id: str,
        formalizer_model: str,
        epoch: int,
        fallback_candidate: ScoutCandidate,
    ) -> FormalizedNodeDraft:
        oracle_name = str(payload.get("oracle_plugin") or fallback_candidate.recommended_oracle)
        if not self._is_plugin_enabled(oracle_name):
            oracle_name = fallback_candidate.recommended_oracle

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
        return FormalizedNodeDraft(
            ral_id=ral_id,
            node_type=str(payload.get("node_type") or fallback_candidate.node_type),
            taxonomy_code=str(payload.get("taxonomy_code") or fallback_candidate.taxonomy_code),
            title=str(payload.get("title") or fallback_candidate.title),
            formal_signature=formal_sig or ("operator:identity",),
            assumptions=assumptions,
            formal_statement_latex=str(payload.get("formal_statement_latex") or ""),
            symbol_rows=symbol_rows,
            failure_boundaries=boundaries,
            provenance_summary=str(
                payload.get("provenance_summary") or fallback_candidate.claim_summary
            ).strip(),
            oracle_plugin=oracle_name,
            verify_py=str(payload.get("verify_py") or ""),
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
    ) -> tuple[FormalizedNodeDraft | None, OracleResult]:
        current = draft
        last_result = OracleResult(
            passed=False,
            plugin_name=current.oracle_plugin,
            stdout="",
            stderr="ORACLE_NOT_RUN",
            elapsed_ms=0,
        )

        for attempt in range(max_repairs + 1):
            if not self._is_plugin_enabled(current.oracle_plugin):
                last_result = OracleResult(
                    passed=False,
                    plugin_name=current.oracle_plugin,
                    stdout="",
                    stderr=f"DISABLED_OR_UNKNOWN_ORACLE: {current.oracle_plugin}",
                    elapsed_ms=0,
                )
            else:
                plugin = self._registry.get(current.oracle_plugin)
                lint_errors = plugin.lint(current.verify_py)
                if lint_errors:
                    last_result = OracleResult(
                        passed=False,
                        plugin_name=current.oracle_plugin,
                        stdout="",
                        stderr="AST_LINT_REJECTED: " + "; ".join(lint_errors),
                        elapsed_ms=0,
                    )
                else:
                    last_result = plugin.execute(current.verify_py, timeout_s=current.oracle_timeout_s)
                    if last_result.passed:
                        return current, last_result

            if attempt == max_repairs:
                break

            repair_prompt = (
                "Repair the executable verification script `verify_py` (and tighten assumptions if needed) "
                "to pass static anti-cheat AST linting and the air-gapped Docker oracle.\n"
                f"Context: {repair_context}\n"
                f"Current assumptions: {json.dumps(list(current.assumptions))}\n"
                f"Current verify_py:\n{current.verify_py}\n\n"
                f"Oracle stderr diagnostic:\n{last_result.stderr}\n\n"
                'Return strict JSON: {"verify_py": str, "assumptions": [str]}.'
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
            current = FormalizedNodeDraft(
                ral_id=current.ral_id,
                node_type=current.node_type,
                taxonomy_code=current.taxonomy_code,
                title=current.title,
                formal_signature=current.formal_signature,
                assumptions=(
                    tuple(str(a).strip() for a in new_assump if str(a).strip())
                    if isinstance(new_assump, list) and new_assump
                    else current.assumptions
                ),
                formal_statement_latex=current.formal_statement_latex,
                symbol_rows=current.symbol_rows,
                failure_boundaries=current.failure_boundaries,
                provenance_summary=current.provenance_summary,
                oracle_plugin=current.oracle_plugin,
                verify_py=str(repaired.get("verify_py") or current.verify_py),
                formalizer_model=used_model,
                model_capability_epoch=current.model_capability_epoch,
                oracle_timeout_s=current.oracle_timeout_s,
            )

        return None, last_result

    def formalize_and_verify(
        self,
        candidate: ScoutCandidate,
        source: SourceRecord,
        *,
        max_repairs: int = 2,
        repo_root: Path | None = None,
        current_epoch: int | None = None,
        ral_id: str | None = None,
    ) -> tuple[FormalizedNodeDraft | None, OracleResult]:
        """
        Run Stage 2 formalization and Stage 3 AST lint + Docker oracle with up to 2 repairs.

        Logs persistent Stage 3 failures into `ledger/rejected/YYYY-MM.ndjson`.
        """
        epoch = current_epoch if current_epoch is not None else self._router.capability_epoch
        prompt = (
            "Formalize this verified scientific quote into canonical v2.1 YAML metadata, "
            "concise LaTeX Formal Statement, Symbol table, Failure Boundaries, and a standalone "
            "executable Python verification script `verify_py` (>=3 distinct non-trivial assertions).\n"
            f"Candidate Title: {candidate.title}\n"
            f"Claim Summary: {candidate.claim_summary}\n"
            f"Verbatim Quote: {candidate.quote_text}\n"
            f"Suggested Taxonomy: {candidate.taxonomy_code}\n"
            f"Suggested Oracle: {candidate.recommended_oracle}\n"
            f"Seed Assumptions: {json.dumps(list(candidate.assumptions))}\n\n"
            'Return strict JSON: {"node_type": str, "taxonomy_code": str, "title": str, '
            '"formal_signature": [str], "assumptions": [str], "formal_statement_latex": str, '
            '"symbol_rows": [{"symbol": str, "meaning": str, "domain": str, "si_units": str}], '
            '"failure_boundaries": [str], "provenance_summary": str, '
            '"oracle_plugin": str, "verify_py": str}.'
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
        assigned_id = ral_id or generate_ral_id(
            str(payload.get("taxonomy_code") or candidate.taxonomy_code),
            entropy=f"{source.source_id}:{candidate.quote_sha256}",
        )
        draft = self._draft_from_payload(
            payload,
            ral_id=assigned_id,
            formalizer_model=used_model,
            epoch=epoch,
            fallback_candidate=candidate,
        )

        verified_draft, oracle_res = self._run_stage3_with_repairs(
            draft,
            max_repairs=max_repairs,
            repair_context=f"{candidate.title} ({source.source_id})",
        )
        if verified_draft is None and repo_root is not None:
            reason = (
                "AST_CHEAT_OR_ASSERT_FAIL"
                if "AST_LINT_REJECTED" in oracle_res.stderr
                else "STAGE3_ORACLE_EXEC_FAIL"
            )
            append_rejected_record(
                repo_root,
                {
                    "source_sha256": sha256_bytes(source.raw_bytes),
                    "canonical_doi": source.canonical_doi,
                    "arxiv_base_id": source.arxiv_base_id,
                    "stage_failed": "STAGE3_ORACLE",
                    "reason": reason,
                    "retry_on_model_upgrade": True,
                    "model_capability_epoch": epoch,
                    "rejected_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                },
            )
        return verified_draft, oracle_res

    def tighten_assumptions(
        self,
        draft: FormalizedNodeDraft,
        counterexample_summary: str,
        missing_assumptions: Sequence[str],
        *,
        max_repairs: int = 1,
    ) -> tuple[FormalizedNodeDraft | None, OracleResult]:
        """Tighten `assumptions[]` and update `verify_py` in response to a Skeptic counterexample."""
        prompt = (
            "An adversarial Skeptic produced a machine-verified counterexample against your formalization. "
            "Tighten `assumptions[]`, update `failure_boundaries` and `verify_py`, and keep >=3 distinct "
            "non-trivial assertions.\n"
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
            formal_statement_latex=str(
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
            verify_py=str(payload.get("verify_py") or draft.verify_py),
            formalizer_model=used_model,
            model_capability_epoch=draft.model_capability_epoch,
            oracle_timeout_s=draft.oracle_timeout_s,
        )
        return self._run_stage3_with_repairs(
            tightened,
            max_repairs=max_repairs,
            repair_context=f"Assumption tightening for {draft.ral_id}",
        )

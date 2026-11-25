"""Stage 4: Blinded cross-family adversarial Skeptic & Epistemic Tier engine (Section 6.3)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime, timedelta, timezone
import json
from typing import Any

from ral.model_router import ModelRouter
from ral.pipeline.formalizer import FormalizedNodeDraft, FormalizerStage
from ral.pipeline.oracle_sandbox import OracleSandbox
from ral.pipeline.oracles.base import OracleResult
from ral.preflight import generate_json_content, infer_family, parse_json_response
from ral.sources import SourceRecord


@dataclass(frozen=True)
class SkepticVerdict:
    """Outcome of executing a blinded cross-family `falsify.py` adversarial attack."""

    survived: bool
    skeptic_model: str
    skeptic_family: str
    cross_family_verified: bool
    counterexample_verified: bool
    falsify_py: str
    sandbox_result: OracleResult | None
    missing_assumptions: tuple[str, ...]
    summary: str


def assign_epistemic_tier(
    *,
    oracle_plugin: str,
    oracle_passed: bool,
    skeptic_survived: bool,
    peer_reviewed: bool,
    retracted: bool,
    pub_year: int,
    citation_count: int = 0,
    source_format: str = "arxiv_tex",
    is_disproved_pathology: bool = False,
    today: date | None = None,
) -> tuple[str, str]:
    """
    Deterministic non-LLM Epistemic Tier rules engine (Blueprint Section 6.3).

    Returns `(epistemic_tier, re_evaluate_after_iso)`.
    """
    ref_date = today or datetime.now(timezone.utc).date()

    if retracted or is_disproved_pathology:
        return "T4_FALSIFIED", (ref_date + timedelta(days=3650)).isoformat()

    if not oracle_passed or not skeptic_survived:
        return "T3_SPECULATIVE", (ref_date + timedelta(days=180)).isoformat()

    age_years = ref_date.year - int(pub_year)
    canonical_backing = (source_format == "reference_db") or (
        peer_reviewed and (age_years >= 2 or int(citation_count) >= 50)
    )

    if oracle_plugin in ("sympy_numeric", "z3_smt") and canonical_backing:
        return "T1_CANONICAL", (ref_date + timedelta(days=730)).isoformat()

    if peer_reviewed:
        return "T2_ESTABLISHED", (ref_date + timedelta(days=365)).isoformat()

    return "T3_SPECULATIVE", (ref_date + timedelta(days=180)).isoformat()


class SkepticStage:
    """Blinded cross-family adversarial red-teamer attempting to falsify formalizations."""

    def __init__(
        self,
        client: Any,
        model_router: ModelRouter,
        sandbox: OracleSandbox | None = None,
    ) -> None:
        self._client = client
        self._router = model_router
        self._sandbox = sandbox or OracleSandbox()

    def challenge(
        self,
        draft: FormalizedNodeDraft,
        quote_text: str,
    ) -> SkepticVerdict:
        """
        Invoke the cross-family `skeptic` model blinded to formalizer metadata and execute `falsify.py`.

        Note: `falsify.py` passes through the forbidden-import AST linter before sandbox execution
        and succeeds (`returncode == 0`) ONLY when it constructs a concrete counterexample obeying
        every stated assumption while violating the Formal Statement.
        """
        # Blinded prompt: intentionally omits formalizer_model, verify_py, and ral_id.
        blinded_prompt = (
            "Assume this formalization has a hidden defect or unstated assumption. "
            "Write a standalone `falsify.py` script that obeys every item in `assumptions[]` "
            "while breaking the `Formal Statement` (exiting 0 ONLY if a concrete counterexample "
            "is verified; exiting non-zero if no counterexample exists), OR cite exact missing "
            "hypotheses from the primary quote.\n\n"
            f"Title: {draft.title}\n"
            f"Formal Statement (LaTeX): {draft.formal_statement_latex}\n"
            f"Assumptions: {json.dumps(list(draft.assumptions))}\n"
            f"Primary Quote: {quote_text}\n\n"
            'Return strict JSON: {"attack_found": bool, "falsify_py": str, '
            '"missing_assumptions": [str], "summary": str}.'
        )

        def _skeptic_call(model_id: str) -> str:
            return generate_json_content(
                self._client,
                model_id,
                blinded_prompt,
                temperature=0.0,
            )

        skeptic_model, raw_json = self._router.call_with_fallback("skeptic", _skeptic_call)
        skeptic_family = infer_family(skeptic_model)
        formalizer_family = infer_family(draft.formalizer_model)
        cross_family_ok = skeptic_family != formalizer_family

        payload = parse_json_response(raw_json)
        attack_found = bool(payload.get("attack_found", False))
        falsify_py = str(payload.get("falsify_py") or "").strip()
        missing_assumptions = tuple(
            str(a).strip() for a in (payload.get("missing_assumptions") or []) if str(a).strip()
        )
        summary = str(payload.get("summary") or "").strip()

        if not attack_found or not falsify_py:
            return SkepticVerdict(
                survived=True,
                skeptic_model=skeptic_model,
                skeptic_family=skeptic_family,
                cross_family_verified=cross_family_ok,
                counterexample_verified=False,
                falsify_py=falsify_py,
                sandbox_result=None,
                missing_assumptions=missing_assumptions,
                summary=summary or "No counterexample proposed",
            )

        sandbox_res = self._sandbox.run_script(
            falsify_py,
            plugin_name="skeptic_falsify",
            script_name="falsify.py",
            timeout_s=draft.oracle_timeout_s,
            enforce_ast_lint=True,
            min_assertions=0,
        )
        counterexample_verified = bool(sandbox_res.passed)
        return SkepticVerdict(
            survived=not counterexample_verified,
            skeptic_model=skeptic_model,
            skeptic_family=skeptic_family,
            cross_family_verified=cross_family_ok,
            counterexample_verified=counterexample_verified,
            falsify_py=falsify_py,
            sandbox_result=sandbox_res,
            missing_assumptions=missing_assumptions,
            summary=summary or sandbox_res.stdout or sandbox_res.stderr,
        )

    def run_adversarial_loop(
        self,
        draft: FormalizedNodeDraft,
        source: SourceRecord,
        formalizer: FormalizerStage,
        *,
        quote_text: str,
        max_tightening_loops: int = 1,
    ) -> tuple[FormalizedNodeDraft | None, SkepticVerdict]:
        """
        Execute Skeptic challenge and allow at most 1 assumption-tightening loop (Section 6.3).

        Returns `(surviving_draft, final_verdict)` or `(None, final_verdict)` if falsified.
        """
        _ = source  # Available for future quote context checks.
        verdict = self.challenge(draft, quote_text)
        if verdict.survived:
            return draft, verdict

        current_draft = draft
        current_verdict = verdict
        for _ in range(max_tightening_loops):
            tightened_draft, oracle_res = formalizer.tighten_assumptions(
                current_draft,
                current_verdict.summary,
                current_verdict.missing_assumptions,
            )
            if tightened_draft is None or not oracle_res.passed:
                return None, current_verdict

            re_verdict = self.challenge(tightened_draft, quote_text)
            if re_verdict.survived:
                return tightened_draft, re_verdict
            current_draft = tightened_draft
            current_verdict = re_verdict

        return None, current_verdict

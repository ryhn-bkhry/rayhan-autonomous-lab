"""Regression tests for Pass 2 post-Run-#6 forensic audit fixes (P1-1 through P1-5)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from ci.validate_node import extract_verification_kernel, split_frontmatter, validate_node_content
from ci.validate_vault import scan_text_for_leaks, validate_vault
from ral.canonicalize import compute_quote_sha256, sha256_bytes
from ral.director import clamp_taxonomy_code, get_active_taxonomy_leaves
from ral.github_client import GitHubClient
from ral.model_router import ModelRouter
from ral.orchestrator import execute_epoch_pipeline, run_consolidate_job
from ral.pipeline.architect import ArchitectStage, VaultCommitValidationError, find_matching_node
from ral.pipeline.formalizer import (
    FormalizedNodeDraft,
    FormalizerStage,
    _oracle_sandbox_contract_prompt,
    _schema_constraints_prompt,
    render_node_markdown,
)
from ral.pipeline.oracle_sandbox import OracleSandbox, SandboxConfig
from ral.pipeline.oracles.base import get_default_registry
from ral.pipeline.scout import ORACLE_PLUGIN_ALIASES, ScoutCandidate, ScoutStage, normalize_oracle_alias
from ral.pipeline.skeptic import SkepticStage
from ral.sources import SourceRecord
from ral.storage.drive_ore import DriveOreStore
from ral.storage.index_builder import (
    DEFAULT_EMBED_BACKOFF_SCHEDULE,
    build_or_update_index,
    embed_single_document,
    load_node_embeddings,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

CLEAN_KERNEL = """import numpy as np

### Verification Checks
x_vals = [1.0, 2.0, 3.0]
for x in x_vals:
    assert np.isclose((x + 1.0) ** 2, x ** 2 + 2.0 * x + 1.0)
assert np.isclose((5.0 + 1.0) ** 2, 36.0)
assert np.isclose((2.0 - 1.0) * (2.0 + 1.0), 3.0)
"""


def _init_repo(root: Path) -> None:
    cfg_dir = root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for fname in (
        "calibration_suite.json",
        "curriculum_params.yaml",
        "license_allowlist.yaml",
        "model_policy.yaml",
        "oracle_plugins.yaml",
        "taxonomy_seeds.yaml",
    ):
        (cfg_dir / fname).write_text(
            (REPO_ROOT / "config" / fname).read_text(encoding="utf-8"),
            encoding="utf-8",
        )


def _local_sandbox() -> OracleSandbox:
    def _runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        vol_idx = cmd.index("-v") + 1
        host_dir = Path(cmd[vol_idx].rsplit(":/work:ro", 1)[0])
        script_name = cmd[-1].rsplit("/", 1)[-1]
        script_path = host_dir / script_name
        return subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=kwargs.get("timeout", 20),
            env=kwargs.get("env"),
            check=False,
        )

    return OracleSandbox(SandboxConfig(), runner=_runner)


def _router() -> ModelRouter:
    return ModelRouter(
        {
            "status": "OK",
            "roles": {
                "scout": ["gemini-2.5-flash"],
                "formalizer": ["gemini-2.5-pro"],
                "architect": ["gemini-2.5-pro"],
                "skeptic": ["gemma-4-27b-it"],
            },
            "exhausted_models": [],
        }
    )


def _source(
    *,
    source_id: str = "arxiv:2404.00001v1",
    raw_bytes: bytes = b"pass2-source-bytes",
    source_license: str = "CC-BY-4.0",
) -> SourceRecord:
    text = "Theorem 1. For every real scalar x, (x + 1)^2 = x^2 + 2x + 1."
    return SourceRecord(
        source_id=source_id,
        source_license=source_license,
        source_format="arxiv_tex",
        title="Binomial Square Identity",
        pub_year=2024,
        peer_reviewed=True,
        retracted=False,
        canonical_doi="10.1000/pass2.1",
        arxiv_base_id="2404.00001",
        arxiv_version=1,
        citation_count=60,
        raw_bytes=raw_bytes,
        raw_ext="tar.gz",
        extracted_text=text,
    )


def _draft(
    *,
    ral_id: str = "RAL-MATH-01PASS2001",
    taxonomy_code: str = "math.algebra",
    title: str = "Binomial Square Identity",
    formal_signature: tuple[str, ...] = (
        "operator:binomial_expansion",
        "domain:real_scalars",
        "relation:polynomial_identity",
    ),
    formal_statement_latex: str = "(x + 1)^2 = x^2 + 2x + 1",
    provenance_summary: str = "Exact binomial square expansion over real scalars.",
    verify_py: str = CLEAN_KERNEL,
) -> FormalizedNodeDraft:
    return FormalizedNodeDraft(
        ral_id=ral_id,
        node_type="THEOREM",
        taxonomy_code=taxonomy_code,
        title=title,
        formal_signature=formal_signature,
        assumptions=("x is a real scalar",),
        formal_statement_latex=formal_statement_latex,
        symbol_rows=({"symbol": "x", "meaning": "real scalar", "domain": "\\mathbb{R}", "si_units": "1"},),
        failure_boundaries=("non-commutative rings",),
        provenance_summary=provenance_summary,
        oracle_plugin="sympy_numeric",
        verify_py=verify_py,
        formalizer_model="gemini-2.5-pro",
        model_capability_epoch=1,
    )


class _SpyDriveStore:
    def __init__(self, warehouse_name: str = "RAL_Ore_Warehouse") -> None:
        self.warehouse_name = warehouse_name
        self.upload_calls: list[bytes] = []

    def upload_ore_bytes(self, payload: bytes, ext: str, *, expected_sha256: str | None = None) -> Any:
        from ral.storage.drive_ore import OreUploadResult, build_ore_relative_path, build_ore_uri

        self.upload_calls.append(bytes(payload))
        sha = expected_sha256 or sha256_bytes(payload)
        return OreUploadResult(
            file_id=f"fake-{len(self.upload_calls)}",
            ore_sha256=sha,
            md5_checksum="0" * 32,
            size_bytes=len(payload),
            ore_uri=build_ore_uri(sha, ext, warehouse_name=self.warehouse_name),
            relative_path=build_ore_relative_path(sha, ext),
            deduplicated=False,
        )


class _FakeEmbedVec:
    def __init__(self, values: list[float]) -> None:
        self.values = values


# ---------------------------------------------------------------------------
# 1. Oracle alias resolution & ### comments inside verify_py
# ---------------------------------------------------------------------------


def test_oracle_alias_resolution_and_triple_hash_comments_in_verify_py(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    assert normalize_oracle_alias("SymPy") == "sympy_numeric"
    assert normalize_oracle_alias("z3") == "z3_smt"
    assert normalize_oracle_alias("scipy") == "empirical_stats"
    for alias, target in ORACLE_PLUGIN_ALIASES.items():
        assert normalize_oracle_alias(alias) == target

    reg = get_default_registry(sandbox=_local_sandbox())
    formalizer = FormalizerStage(
        SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: None)),
        _router(),
        reg,
        oracle_config_path=tmp_path / "config" / "oracle_plugins.yaml",
    )
    assert formalizer._resolve_enabled_oracle("sympy", None) == "sympy_numeric"
    assert formalizer._resolve_enabled_oracle("smt", None) == "z3_smt"
    assert formalizer._resolve_enabled_oracle("stats", None) == "empirical_stats"
    assert formalizer._resolve_enabled_oracle("unknown_tool", "scipy") == "empirical_stats"

    kernel_with_headers = (
        "import numpy as np\n"
        "### Matrix Setup\n"
        "A = np.array([[1.0, 2.0], [3.0, 4.0]])\n"
        "## Norm Check\n"
        "assert np.isclose(np.linalg.norm(A, 'fro') ** 2, 30.0)\n"
        "assert np.isclose(np.trace(A.T @ A), 30.0)\n"
        "assert np.isclose(np.trace(A @ A.T), 30.0)\n"
    )
    d = _draft(verify_py=kernel_with_headers)
    md = render_node_markdown(
        d,
        epistemic_tier="T1_CANONICAL",
        oracle_status="PASS",
        skeptic_model="gemma-4-27b-it",
        source=_source(),
        quote_sha256=compute_quote_sha256("quote"),
        quote_char_len=5,
        source_locator="tex:sec1",
        re_evaluate_after="2028-11-25",
    )
    val = validate_node_content(md, allowed_licenses={"CC-BY-4.0"}, expected_filename_stem=d.ral_id)
    assert val.ok is True, val.errors
    assert "### Matrix Setup" in (val.kernel_code or "")
    assert "## Norm Check" in (val.kernel_code or "")
    assert extract_verification_kernel(md.split("---\n", 2)[2]) == kernel_with_headers.strip()


# ---------------------------------------------------------------------------
# 2. P1-1: NumPy @ exemption, _TOKEN_SPLIT_RE, pre-upload gate & Stage 3 leak gate
# ---------------------------------------------------------------------------


def test_p1_1_numpy_matmul_exemption_and_pre_upload_leak_schema_gates(tmp_path: Path) -> None:
    _init_repo(tmp_path)

    # A@B.conj() and U@V.real inside ```python ... ``` must NOT trigger EMAIL_ADDRESS.
    clean_doc = (
        "# Node\n\n"
        "```python\n"
        "import numpy as np\n"
        "C = A@B.conj()\n"
        "D = U@V.real\n"
        "expr=x+y*z^2$w\n"
        "```\n"
    )
    assert scan_text_for_leaks(clean_doc, "node.md") == []

    # High-entropy tokens separated by =, +, *, ^, $ are split by _TOKEN_SPLIT_RE and not flagged
    # when each side is < 32 chars, whereas a real >=32-char high-entropy token IS caught.
    split_doc = "formula=AbCdEfGhIj123456+KlMnOpQrSt789012*UvWxYz345678^AbCdEfGh9012$ZxCvBn1234\n"
    assert scan_text_for_leaks(split_doc, "split.md") == []

    # Prefixed secrets and local paths inside ```python ... ``` are STILL flagged!
    leaky_py_doc = (
        "```python\n"
        "key = 'AIzaSyFakeSecretKey1234567890abcd'\n"
        "p = 'C:\\Users\\alice\\secret.txt'\n"
        "```\n"
        "Contact bob@example.com outside fence.\n"
    )
    findings = scan_text_for_leaks(leaky_py_doc, "leaky.md")
    assert any("GOOGLE_API_KEY" in f for f in findings)
    assert any("WIN_USERS_PATH" in f for f in findings)
    assert any("EMAIL_ADDRESS" in f for f in findings)

    # ArchitectStage pre-upload gates must abort BEFORE drive.upload_ore_bytes on schema or leak failure.
    spy_drive = _SpyDriveStore()
    arch = ArchitectStage(tmp_path, spy_drive, sandbox=_local_sandbox())  # type: ignore[arg-type]

    # 1) commit_new_node leak failure -> 0 uploads
    leaky_draft = _draft(
        ral_id="RAL-MATH-01LEAKNOD1",
        provenance_summary="Contact author@example.com for details.",
    )
    with pytest.raises(VaultCommitValidationError, match="leak"):
        arch.commit_new_node(
            leaky_draft,
            epistemic_tier="T1_CANONICAL",
            re_evaluate_after="2028-11-25",
            skeptic_model="gemma-4-27b-it",
            source=_source(),
            quote_sha256=compute_quote_sha256("quote"),
            quote_char_len=5,
            source_locator="tex:sec1",
        )
    assert len(spy_drive.upload_calls) == 0

    # Commit a valid base node first (1 upload).
    valid_draft = _draft(ral_id="RAL-MATH-01GOODNOD1")
    created = arch.commit_new_node(
        valid_draft,
        epistemic_tier="T1_CANONICAL",
        re_evaluate_after="2028-11-25",
        skeptic_model="gemma-4-27b-it",
        source=_source(raw_bytes=b"valid-ore-1"),
        quote_sha256=compute_quote_sha256("quote1"),
        quote_char_len=6,
        source_locator="tex:sec1",
    )
    assert created.node_path is not None
    assert len(spy_drive.upload_calls) == 1

    # 2) merge_into_existing_node leak failure -> still 1 upload (0 new uploads!)
    with pytest.raises(VaultCommitValidationError, match="leak"):
        arch.merge_into_existing_node(
            created.node_path,
            source=_source(source_id="arxiv:2404.00002v1", raw_bytes=b"bad-merge-ore"),
            quote_sha256=compute_quote_sha256("quote2"),
            quote_char_len=6,
            source_locator="tex:sec2",
            anchor_summary="Leaked email ops@example.com in anchor.",
        )
    assert len(spy_drive.upload_calls) == 1

    # 3) apply_theory_upgrade (CHALLENGED_ONLY) leak failure -> 0 new uploads
    bad_challenge = _draft(
        ral_id="RAL-MATH-01BADCHAL1",
        provenance_summary="Leaked /Users/alice/private.key in challenge.",
    )
    with pytest.raises(VaultCommitValidationError, match="leak"):
        arch.apply_theory_upgrade(
            created.node_path,
            bad_challenge,
            "T3_SPECULATIVE",
            re_evaluate_after="2027-06-01",
            skeptic_model="gemma-4-27b-it",
            source=_source(source_id="arxiv:2404.00003v1", raw_bytes=b"bad-chal-ore"),
            quote_sha256=compute_quote_sha256("quote3"),
            quote_char_len=6,
            source_locator="tex:sec3",
            reason="CHALLENGE_TEST",
        )
    assert len(spy_drive.upload_calls) == 1

    # 4) apply_theory_upgrade (SUPERSEDED) leak failure -> 0 new uploads
    bad_upgrade = _draft(
        ral_id=valid_draft.ral_id,
        provenance_summary="Leaked leakcheck@example.com in supersession.",
    )
    with pytest.raises(VaultCommitValidationError, match="leak"):
        arch.apply_theory_upgrade(
            created.node_path,
            bad_upgrade,
            "T1_CANONICAL",
            re_evaluate_after="2028-11-25",
            skeptic_model="gemma-4-27b-it",
            source=_source(source_id="arxiv:2404.00004v1", raw_bytes=b"bad-sup-ore"),
            quote_sha256=compute_quote_sha256("quote4"),
            quote_char_len=6,
            source_locator="tex:sec4",
            reason="GENERALIZATION",
            correspondence_script="import numpy as np\nx = 2.5\nassert np.isclose((x + 1.0) ** 2, x ** 2 + 2.0 * x + 1.0)\n",
        )
    assert len(spy_drive.upload_calls) == 1

    # Stage 3 FormalizerStage._run_stage3_with_repairs detects leaks and triggers repair.
    stage3_prompts: list[str] = []

    def _leak_repair_llm(**kw: Any) -> Any:
        prompt = kw["contents"]
        stage3_prompts.append(prompt)
        if "Repair the executable verification script" in prompt:
            assert "SECRET_LEAK(EMAIL_ADDRESS" in prompt
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "provenance_summary": "Clean summary without leaks.",
                        "assumptions": ["x is a real scalar"],
                        "verify_py": CLEAN_KERNEL,
                    }
                )
            )
        raise AssertionError("Unexpected formalize call")

    formalizer = FormalizerStage(
        SimpleNamespace(models=SimpleNamespace(generate_content=_leak_repair_llm)),
        _router(),
        get_default_registry(sandbox=_local_sandbox()),
        oracle_config_path=tmp_path / "config" / "oracle_plugins.yaml",
    )
    repaired, res, diag = formalizer._run_stage3_with_repairs(
        _draft(provenance_summary="Leaked info@example.com summary."),
        max_repairs=1,
        repair_context="leak-test",
        source=_source(),
        allowed_licenses={"CC-BY-4.0"},
    )
    assert repaired is not None and res.passed is True
    assert diag.repair_attempts == 1
    assert "info@example.com" not in repaired.provenance_summary


# ---------------------------------------------------------------------------
# 3. P1-2: Skeptic contract, min_assertions=2, COUNTEREXAMPLE_VERIFIED & tightening context
# ---------------------------------------------------------------------------


def test_p1_2_skeptic_hardening_and_tighten_assumptions_context(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    sbx = _local_sandbox()
    router = _router()
    draft = _draft()

    # 1) Single assertion falsify.py is rejected (min_assertions=2)
    prompts_seen: list[str] = []

    def _one_assert_llm(**kw: Any) -> Any:
        prompts_seen.append(kw["contents"])
        return SimpleNamespace(
            text=json.dumps(
                {
                    "attack_found": True,
                    "falsify_py": "x = 1.0\nassert x > 0.0\nprint('COUNTEREXAMPLE_VERIFIED: single')\n",
                    "missing_assumptions": ["x != 0"],
                    "summary": "Only one assert",
                }
            )
        )

    sk_one = SkepticStage(
        SimpleNamespace(models=SimpleNamespace(generate_content=_one_assert_llm)),
        router,
        sandbox=sbx,
    )
    v_one = sk_one.challenge(draft, "quote")
    assert _oracle_sandbox_contract_prompt() in prompts_seen[0]
    assert draft.formalizer_model not in prompts_seen[0]
    assert draft.ral_id not in prompts_seen[0]
    assert v_one.survived is True
    assert v_one.counterexample_verified is False
    assert v_one.sandbox_result is not None
    assert "INSUFFICIENT_ASSERTIONS" in v_one.sandbox_result.stderr

    # 2) Passing 2-assertion script WITHOUT COUNTEREXAMPLE_VERIFIED: marker does NOT falsify.
    sk_no_marker = SkepticStage(
        SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **kw: SimpleNamespace(
                    text=json.dumps(
                        {
                            "attack_found": True,
                            "falsify_py": (
                                "x = 0.0\n"
                                "y = 1.0\n"
                                "assert x == 0.0\n"
                                "assert y > 0.0\n"
                                "print('Found something')\n"
                            ),
                            "missing_assumptions": ["x > 0"],
                            "summary": "Missing marker",
                        }
                    )
                )
            )
        ),
        router,
        sandbox=sbx,
    )
    v_no_marker = sk_no_marker.challenge(draft, "quote")
    assert v_no_marker.sandbox_result is not None
    assert v_no_marker.sandbox_result.passed is True
    assert v_no_marker.counterexample_verified is False
    assert v_no_marker.survived is True

    # 3) Fenced 2-assertion script WITH COUNTEREXAMPLE_VERIFIED: strips fence & falsifies.
    fenced_falsify = (
        "```python\n"
        "x = 0.0\n"
        "eps = 1.0\n"
        "assert isinstance(x, float)\n"
        "assert (x + eps) > 0.0\n"
        "print('COUNTEREXAMPLE_VERIFIED: x=0 breaks x!=0 assumption')\n"
        "```"
    )
    sk_valid = SkepticStage(
        SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **kw: SimpleNamespace(
                    text=json.dumps(
                        {
                            "attack_found": True,
                            "falsify_py": fenced_falsify,
                            "missing_assumptions": ["x != 0"],
                            "summary": "Needs x != 0",
                        }
                    )
                )
            )
        ),
        router,
        sandbox=sbx,
    )
    v_valid = sk_valid.challenge(draft, "quote")
    assert not v_valid.falsify_py.startswith("```")
    assert v_valid.counterexample_verified is True
    assert v_valid.survived is False

    # 4) run_adversarial_loop passes source, quote_sha256, quote_char_len, source_locator, allowed_licenses
    #    into tighten_assumptions -> _run_stage3_with_repairs, and prompt includes _schema_constraints_prompt().
    tighten_prompts: list[str] = []
    stage3_calls: list[dict[str, Any]] = []

    class _SpyFormalizer(FormalizerStage):
        def _run_stage3_with_repairs(self, draft_in: FormalizedNodeDraft, **kwargs: Any) -> Any:
            stage3_calls.append(dict(kwargs))
            return super()._run_stage3_with_repairs(draft_in, **kwargs)

    def _loop_llm(**kw: Any) -> Any:
        prompt = kw["contents"]
        if "counterexample against your formalization" in prompt:
            tighten_prompts.append(prompt)
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "assumptions": ["x is a non-zero real scalar"],
                        "failure_boundaries": ["x = 0"],
                        "formal_statement_latex": "(x+1)^2 = x^2 + 2x + 1",
                        "verify_py": CLEAN_KERNEL,
                    }
                )
            )
        if "x is a non-zero real scalar" in prompt:
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "attack_found": False,
                        "falsify_py": "",
                        "missing_assumptions": [],
                        "summary": "Survived",
                    }
                )
            )
        return SimpleNamespace(
            text=json.dumps(
                {
                    "attack_found": True,
                    "falsify_py": fenced_falsify,
                    "missing_assumptions": ["x != 0"],
                    "summary": "x=0 counterexample",
                }
            )
        )

    loop_client = SimpleNamespace(models=SimpleNamespace(generate_content=_loop_llm))
    spy_formalizer = _SpyFormalizer(
        loop_client,
        router,
        get_default_registry(sandbox=sbx),
        oracle_config_path=tmp_path / "config" / "oracle_plugins.yaml",
    )
    sk_loop = SkepticStage(loop_client, router, sandbox=sbx)
    src = _source(source_license="CC-BY-4.0")
    q_text = "For every real scalar x, (x + 1)^2 = x^2 + 2x + 1."
    q_sha = compute_quote_sha256(q_text)

    surviving, final_v = sk_loop.run_adversarial_loop(
        draft,
        src,
        spy_formalizer,
        quote_text=q_text,
        quote_sha256=q_sha,
        quote_char_len=len(q_text),
        source_locator="tex:thm1",
        allowed_licenses={"CC-BY-4.0"},
        max_tightening_loops=1,
    )
    assert surviving is not None and final_v.survived is True
    assert len(tighten_prompts) == 1
    assert _schema_constraints_prompt() in tighten_prompts[0]
    assert len(stage3_calls) == 1
    assert stage3_calls[0]["source"] is src
    assert stage3_calls[0]["quote_sha256"] == q_sha
    assert stage3_calls[0]["quote_char_len"] == len(q_text)
    assert stage3_calls[0]["source_locator"] == "tex:thm1"
    assert stage3_calls[0]["allowed_licenses"] == {"CC-BY-4.0"}

    # Verify disallowed license in `source` is enforced when passed through run_adversarial_loop.
    bad_src = _source(source_license="PROPRIETARY-NONCOMMERCIAL")
    surviving_bad, _ = sk_loop.run_adversarial_loop(
        draft,
        bad_src,
        spy_formalizer,
        quote_text=q_text,
        quote_sha256=q_sha,
        quote_char_len=len(q_text),
        source_locator="tex:thm1",
        allowed_licenses={"CC-BY-4.0"},
        max_tightening_loops=1,
    )
    assert surviving_bad is None


# ---------------------------------------------------------------------------
# 4. P1-3: False-positive merge prevention & ral_id collision safety
# ---------------------------------------------------------------------------


def test_p1_3_find_matching_node_precision_and_commit_collision_handling(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    spy_drive = _SpyDriveStore()
    arch = ArchitectStage(tmp_path, spy_drive, sandbox=_local_sandbox())  # type: ignore[arg-type]

    # Node with fallback signature ("operator:identity",) must NEVER match another fallback node.
    d_fallback = _draft(
        ral_id="RAL-MATH-01IDENT001",
        title="Identity One",
        formal_signature=("operator:identity",),
        formal_statement_latex="a + b = b + a",
    )
    arch.commit_new_node(
        d_fallback,
        epistemic_tier="T1_CANONICAL",
        re_evaluate_after="2028-11-25",
        skeptic_model="gemma-4-27b-it",
        source=_source(source_id="arxiv:2404.10001v1", raw_bytes=b"id1"),
        quote_sha256=compute_quote_sha256("id1"),
        quote_char_len=3,
        source_locator="tex:sec1",
    )
    assert (
        find_matching_node(
            tmp_path,
            ("operator:identity",),
            candidate_title="Identity One",
            candidate_statement="a + b = b + a",
        )
        is None
    )

    # Node with 3 tags
    sig3 = (
        "operator:binomial_expansion",
        "domain:real_scalars",
        "relation:polynomial_identity",
    )
    d_real = _draft(
        ral_id="RAL-MATH-01REAL0001",
        title="Binomial Square Identity",
        formal_signature=sig3,
        formal_statement_latex="(x + 1)^2 = x^2 + 2x + 1",
    )
    res_real = arch.commit_new_node(
        d_real,
        epistemic_tier="T1_CANONICAL",
        re_evaluate_after="2028-11-25",
        skeptic_model="gemma-4-27b-it",
        source=_source(source_id="arxiv:2404.10002v1", raw_bytes=b"real1"),
        quote_sha256=compute_quote_sha256("real1"),
        quote_char_len=5,
        source_locator="tex:sec1",
    )
    assert res_real.node_path is not None

    # Only 2 matching tags (< 3) -> rejected even with identical title/statement.
    assert (
        find_matching_node(
            tmp_path,
            sig3[:2],
            candidate_title="Binomial Square Identity",
            candidate_statement="(x + 1)^2 = x^2 + 2x + 1",
        )
        is None
    )

    # All 3 tags (jac == 1.0) BUT completely different statement/title (< 0.85) -> rejected.
    assert (
        find_matching_node(
            tmp_path,
            sig3,
            candidate_title="Cauchy Schwarz Inequality",
            candidate_statement="(\\sum a_i b_i)^2 \\le (\\sum a_i^2)(\\sum b_i^2)",
        )
        is None
    )

    # All 3 tags (jac == 1.0) AND normalized statement similarity >= 0.85 -> matched.
    matched_sig = find_matching_node(
        tmp_path,
        sig3,
        candidate_title="Unrelated Alias",
        candidate_statement="$$(x+1)^2 = x^2 + 2x + 1$$",
    )
    assert matched_sig is not None and matched_sig[1].ral_id == "RAL-MATH-01REAL0001"

    # Embedding cosine >= 0.88 matches ONLY when embed_space_consistent=True.
    emb_map = {"RAL-MATH-01REAL0001": [1.0, 0.0, 0.0]}
    assert (
        find_matching_node(
            tmp_path,
            ("operator:different_tag",),
            candidate_embedding=[0.95, 0.3, 0.0],
            existing_embeddings=emb_map,
            embed_space_consistent=False,
        )
        is None
    )
    matched_emb = find_matching_node(
        tmp_path,
        ("operator:different_tag",),
        candidate_embedding=[0.95, 0.3, 0.0],
        existing_embeddings=emb_map,
        embed_space_consistent=True,
    )
    assert matched_emb is not None and matched_emb[1].ral_id == "RAL-MATH-01REAL0001"

    # commit_new_node on an existing node_path:
    # (a) Same statement -> delegates to merge_into_existing_node()
    dup_res = arch.commit_new_node(
        _draft(ral_id="RAL-MATH-01REAL0001"),
        epistemic_tier="T1_CANONICAL",
        re_evaluate_after="2028-11-25",
        skeptic_model="gemma-4-27b-it",
        source=_source(source_id="arxiv:2404.10003v1", raw_bytes=b"dup-ore"),
        quote_sha256=compute_quote_sha256("dup"),
        quote_char_len=3,
        source_locator="tex:sec2",
    )
    assert dup_res.action == "MERGED_PROVENANCE"
    assert dup_res.ral_id == "RAL-MATH-01REAL0001"
    fm_after_merge, _ = split_frontmatter(res_real.node_path.read_text(encoding="utf-8"))
    assert len(fm_after_merge["provenance"]) == 2

    # (b) Different statement -> mints non-colliding ral_id and preserves original file.
    diff_draft = _draft(
        ral_id="RAL-MATH-01REAL0001",
        title="Pythagorean Theorem",
        formal_statement_latex="a^2 + b^2 = c^2",
    )
    coll_res = arch.commit_new_node(
        diff_draft,
        epistemic_tier="T1_CANONICAL",
        re_evaluate_after="2028-11-25",
        skeptic_model="gemma-4-27b-it",
        source=_source(source_id="arxiv:2404.10004v1", raw_bytes=b"diff-ore"),
        quote_sha256=compute_quote_sha256("diff"),
        quote_char_len=4,
        source_locator="tex:sec3",
    )
    assert coll_res.action == "CREATED"
    assert coll_res.ral_id != "RAL-MATH-01REAL0001"
    assert coll_res.node_path is not None and coll_res.node_path != res_real.node_path
    assert res_real.node_path.exists()
    assert validate_vault(tmp_path).ok is True


# ---------------------------------------------------------------------------
# 5. P1-4: Index builder backoff (2s -> 6s -> 15s) & graceful Job 7 fallback
# ---------------------------------------------------------------------------


def test_p1_4_index_builder_429_503_backoff_and_graceful_job7_fallback(tmp_path: Path) -> None:
    assert DEFAULT_EMBED_BACKOFF_SCHEDULE == (2.0, 6.0, 15.0)
    sleeps: list[float] = []
    attempts = {"n": 0}

    class _TransientEmbedClient:
        class _Models:
            @staticmethod
            def embed_content(*, model: str, contents: str, config: dict[str, Any]) -> Any:
                attempts["n"] += 1
                if attempts["n"] == 1:
                    raise RuntimeError("HTTP 429 RESOURCE_EXHAUSTED")
                if attempts["n"] == 2:
                    raise RuntimeError("503 UNAVAILABLE")
                return SimpleNamespace(embeddings=[_FakeEmbedVec([1.0] + [0.0] * 767)])

        models = _Models()

    vec = embed_single_document(
        _TransientEmbedClient(),
        "gemini-embedding-2",
        "test doc",
        768,
        sleep_fn=sleeps.append,
    )
    assert len(vec) == 768
    assert sleeps == [2.0, 6.0]

    # Build initial index with 1 valid node and baseline vectors.
    _init_repo(tmp_path)
    spy_drive = _SpyDriveStore()
    arch = ArchitectStage(tmp_path, spy_drive, sandbox=_local_sandbox())  # type: ignore[arg-type]
    arch.commit_new_node(
        _draft(ral_id="RAL-MATH-01IDX00001"),
        epistemic_tier="T1_CANONICAL",
        re_evaluate_after="2028-11-25",
        skeptic_model="gemma-4-27b-it",
        source=_source(source_id="arxiv:2404.20001v1", raw_bytes=b"idx1"),
        quote_sha256=compute_quote_sha256("idx1"),
        quote_char_len=4,
        source_locator="tex:sec1",
    )
    db_path = tmp_path / "cache" / "ral-index.sqlite"
    ok_client = SimpleNamespace(
        models=SimpleNamespace(
            embed_content=lambda **kw: SimpleNamespace(
                embeddings=[_FakeEmbedVec([1.0] + [0.0] * 767)]
            )
        )
    )
    res0 = build_or_update_index(
        tmp_path,
        db_path,
        embed_client=ok_client,
        discovered_embed_model="gemini-embedding-2",
        sleep_fn=lambda _: None,
    )
    assert res0.vector_action == "INITIALIZED"
    assert res0.embed_space_consistent is True

    # Add a second node, then fail embeddings persistently with 429 -> prior vectors preserved!
    arch.commit_new_node(
        _draft(ral_id="RAL-MATH-01IDX00002", title="Second Node", formal_statement_latex="y^2 >= 0"),
        epistemic_tier="T1_CANONICAL",
        re_evaluate_after="2028-11-25",
        skeptic_model="gemma-4-27b-it",
        source=_source(source_id="arxiv:2404.20002v1", raw_bytes=b"idx2"),
        quote_sha256=compute_quote_sha256("idx2"),
        quote_char_len=4,
        source_locator="tex:sec1",
    )

    fail_sleeps: list[float] = []
    failing_embed_client = SimpleNamespace(
        models=SimpleNamespace(
            embed_content=lambda **kw: (_ for _ in ()).throw(RuntimeError("HTTP 429 rate limit exceeded"))
        )
    )
    res_err = build_or_update_index(
        tmp_path,
        db_path,
        embed_client=failing_embed_client,
        discovered_embed_model="gemini-embedding-2",
        sleep_fn=fail_sleeps.append,
    )
    assert fail_sleeps == [2.0, 6.0, 15.0]
    assert res_err.vector_action == "SKIPPED_EMBED_ERROR"
    assert res_err.embed_space_consistent is False
    assert res_err.node_count == 2

    # Prior vector for node 1 is intact and FTS5 contains both nodes!
    vecs_after, consistent_after, _ = load_node_embeddings(db_path)
    assert consistent_after is False
    assert "RAL-MATH-01IDX00001" in vecs_after
    conn = sqlite3.connect(str(db_path))
    try:
        fts_count = conn.execute("SELECT COUNT(*) FROM nodes_fts").fetchone()[0]
        assert fts_count == 2
    finally:
        conn.close()

    # End-to-end Job 7 consolidation finishes cleanly without crashing on embed 429!
    handoff = tmp_path / "handoff"
    handoff.mkdir(parents=True, exist_ok=True)
    (handoff / "circuit_breaker.json").write_text(
        json.dumps({"circuit_breaker": None, "drive_canary": {"ok": True}, "embedding_model": {"model_id": "gemini-embedding-2"}}),
        encoding="utf-8",
    )

    # Monkeypatch sleep inside index_builder for zero-delay Job 7 call.
    import ral.storage.index_builder as ib_mod

    orig_sleep = ib_mod.time.sleep
    ib_mod.time.sleep = lambda _: None
    try:
        stored_assets: dict[str, dict[str, Any]] = {}
        next_asset_id = {"val": 100}

        def _fake_gh_req(method: str, url: str, **kwargs: Any) -> Any:
            if f"/releases/tags/" in url:
                return SimpleNamespace(status_code=404, json=lambda: {})
            if url.endswith("/releases") and method == "GET":
                return SimpleNamespace(status_code=200, json=lambda: [])
            if url.endswith("/releases") and method == "POST":
                return SimpleNamespace(
                    status_code=201,
                    json=lambda: {
                        "id": 99,
                        "upload_url": "https://uploads.github.com/repos/rayhan/ral-test/releases/99/assets{?name}",
                    },
                )
            if "/releases/99/assets" in url and method == "GET":
                return SimpleNamespace(status_code=200, json=lambda: list(stored_assets.values()))
            if "uploads.github.com" in url and method == "POST":
                params = dict(kwargs.get("params") or {})
                name = str(params.get("name") or "asset")
                data = bytes(kwargs.get("data") or b"")
                aid = next_asset_id["val"]
                next_asset_id["val"] += 1
                dl_url = f"https://downloads.test/{name}"
                rec = {
                    "id": aid,
                    "name": name,
                    "size": len(data),
                    "browser_download_url": dl_url,
                    "payload": data,
                }
                stored_assets[dl_url] = rec
                return SimpleNamespace(status_code=201, json=lambda: {k: v for k, v in rec.items() if k != "payload"})
            if url in stored_assets:
                payload = stored_assets[url]["payload"]
                return SimpleNamespace(status_code=200, content=payload)
            return SimpleNamespace(status_code=200, json=lambda: {})

        gh = GitHubClient("rayhan", "ral-test", token="fake", request_fn=_fake_gh_req, sleep_fn=lambda _: None)
        con_res = run_consolidate_job(
            tmp_path,
            handoff_dir=handoff,
            cache_dir=tmp_path / "cache",
            run_date_iso="2026-11-25",
            run_id="embed429",
            dry_run=True,
            allow_release_in_dry_run=True,
            github_client=gh,
            embed_client=failing_embed_client,
            now=datetime(2026, 11, 25, 12, 0, tzinfo=timezone.utc),
        )
    finally:
        ib_mod.time.sleep = orig_sleep

    assert con_res.ok is True
    assert con_res.vector_action == "SKIPPED_EMBED_ERROR"
    assert con_res.embed_space_consistent is False
    assert con_res.blue_green_swap is not None
    assert (tmp_path / "INDEX.md").exists()
    assert (tmp_path / "llms.txt").exists()
    tel_rows = [
        json.loads(line)
        for line in con_res.telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert tel_rows[-1]["vector_action"] == "SKIPPED_EMBED_ERROR"


# ---------------------------------------------------------------------------
# 6. P1-5: Active-leaf taxonomy clamping & POSIX chmod on extra_files
# ---------------------------------------------------------------------------


def test_p1_5_taxonomy_leaf_clamping_and_extra_files_chmod(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _init_repo(tmp_path)
    leaves = get_active_taxonomy_leaves(tmp_path)
    assert "phys.statistical-mechanics" in leaves
    assert clamp_taxonomy_code(
        "phys.statistical-mechanics.ising-model",
        active_leaves=leaves,
        target_domain="phys.statistical-mechanics",
    ) == "phys.statistical-mechanics"
    assert clamp_taxonomy_code(
        "bio.invented_unknown",
        active_leaves=leaves,
        target_domain="phys.statistical-mechanics",
    ) == "phys.statistical-mechanics"

    # If a split leaf IS present in vault/, clamping preserves it.
    split_node_dir = tmp_path / "vault" / "phys" / "statistical-mechanics" / "ising-model"
    split_node_dir.mkdir(parents=True, exist_ok=True)
    split_draft = _draft(
        ral_id="RAL-PHYS-01SPLIT001",
        taxonomy_code="phys.statistical-mechanics.ising-model",
    )
    (split_node_dir / "RAL-PHYS-01SPLIT001.md").write_text(
        render_node_markdown(
            split_draft,
            epistemic_tier="T1_CANONICAL",
            oracle_status="PASS",
            skeptic_model="gemma-4-27b-it",
            source=_source(raw_bytes=b"split-seed"),
            quote_sha256=compute_quote_sha256("quote"),
            quote_char_len=5,
            source_locator="tex:sec1",
            re_evaluate_after="2028-11-25",
        ),
        encoding="utf-8",
    )
    leaves_after_split = get_active_taxonomy_leaves(tmp_path)
    assert "phys.statistical-mechanics.ising-model" in leaves_after_split
    assert (
        clamp_taxonomy_code(
            "phys.statistical-mechanics.ising-model",
            active_leaves=leaves_after_split,
            target_domain="phys.statistical-mechanics.ising-model",
        )
        == "phys.statistical-mechanics.ising-model"
    )

    # End-to-end execute_epoch_pipeline clamps invented sub-branches in Scout + Formalizer.
    pipe_repo = tmp_path / "pipe_repo"
    _init_repo(pipe_repo)
    src = _source(raw_bytes=b"pipe-clamp-ore")

    def _pipe_llm(**kw: Any) -> Any:
        prompt = kw["contents"]
        if "verbatim quotes" in prompt:
            assert "phys.statistical-mechanics" in prompt
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "candidates": [
                            {
                                "title": "Binomial Identity",
                                "claim_summary": "Exact polynomial square expansion.",
                                "quote_text": "For every real scalar x, (x + 1)^2 = x^2 + 2x + 1.",
                                "source_locator": "tex:sec1",
                                "node_type": "THEOREM",
                                "taxonomy_code": "phys.statistical-mechanics.invented-subbranch",
                                "formal_signature": [
                                    "operator:binomial_expansion",
                                    "domain:real_scalars",
                                    "relation:polynomial_identity",
                                ],
                                "assumptions": ["x is a real scalar"],
                                "recommended_oracle": "sympy_numeric",
                            }
                        ]
                    }
                )
            )
        if "Formalize this verified scientific quote" in prompt:
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "node_type": "THEOREM",
                        "taxonomy_code": "phys.nonexistent_branch",
                        "title": "Binomial Square Identity",
                        "formal_signature": [
                            "operator:binomial_expansion",
                            "domain:real_scalars",
                            "relation:polynomial_identity",
                        ],
                        "assumptions": ["x is a real scalar"],
                        "formal_statement_latex": "(x + 1)^2 = x^2 + 2x + 1",
                        "symbol_rows": [{"symbol": "x", "meaning": "real", "domain": "\\mathbb{R}", "si_units": "1"}],
                        "failure_boundaries": ["non-commutative rings"],
                        "provenance_summary": "Exact polynomial identity.",
                        "oracle_plugin": "sympy_numeric",
                        "verify_py": CLEAN_KERNEL,
                    }
                )
            )
        return SimpleNamespace(
            text=json.dumps(
                {
                    "attack_found": False,
                    "falsify_py": "",
                    "missing_assumptions": [],
                    "summary": "Survived",
                }
            )
        )

    sbx = _local_sandbox()
    ep_out = execute_epoch_pipeline(
        _router(),
        repo_root=pipe_repo,
        mode="explore",
        target="phys.statistical-mechanics",
        current_night=1,
        ore_cache_dir=pipe_repo / "ore_cache",
        gemini_client=SimpleNamespace(models=SimpleNamespace(generate_content=_pipe_llm)),
        drive_store=None,
        sandbox=sbx,
        registry=get_default_registry(sandbox=sbx),
        source_fetcher=lambda m, t, r: [src],
        dry_run=True,
        now=datetime(2026, 11, 25, 13, 0, tzinfo=timezone.utc),
    )
    assert len(ep_out["nodes_created"]) == 1
    committed_files = list((pipe_repo / "vault").rglob("*.md"))
    assert len(committed_files) == 1
    fm_committed, _ = split_frontmatter(committed_files[0].read_text(encoding="utf-8"))
    assert fm_committed["taxonomy_code"] == "phys.statistical-mechanics"
    assert committed_files[0].parent == pipe_repo / "vault" / "phys" / "statistical-mechanics"

    # OracleSandbox.run_script runs os.chmod(extra_path, 0o644) on POSIX for all extra_files.
    import ral.pipeline.oracle_sandbox as sandbox_mod

    chmod_calls: list[tuple[str, int]] = []
    posix_os_shim = SimpleNamespace(
        **{
            **os.__dict__,
            "name": "posix",
            "chmod": lambda path, mode: chmod_calls.append((Path(path).name, mode)),
        }
    )
    monkeypatch.setattr(sandbox_mod, "os", posix_os_shim)
    sbx_chmod = OracleSandbox(
        SandboxConfig(),
        runner=lambda cmd, **kw: subprocess.CompletedProcess(cmd, 0, stdout="OK", stderr=""),
    )
    sbx_res = sbx_chmod.run_script(
        "print('ok')\n",
        enforce_ast_lint=False,
        extra_files={"data.tar.gz": b"123", "helper.txt": b"abc"},
    )
    assert sbx_res.passed is True
    assert ("data.tar.gz", 0o644) in chmod_calls
    assert ("helper.txt", 0o644) in chmod_calls
    assert ("verify.py", 0o644) in chmod_calls

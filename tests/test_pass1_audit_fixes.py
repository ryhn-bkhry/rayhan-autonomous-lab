"""Regression tests for Pass 1 post-Run-#6 forensic audit fixes."""

from __future__ import annotations

from datetime import datetime, timezone
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from ci.validate_node import extract_verification_kernel, validate_node_content
from ral.canonicalize import (
    compute_assumptions_hash,
    compute_oracle_script_sha256,
    compute_quote_sha256,
)
from ral.model_router import ModelRouter
from ral.orchestrator import (
    _default_fetch_sources_for_epoch,
    run_consolidate_job,
    run_epoch_job,
)
from ral.pipeline.formalizer import (
    FormalizedNodeDraft,
    FormalizerStage,
    SANDBOX_ALLOWED_STDLIB,
    SANDBOX_FORBIDDEN_BUILTINS_AND_MODULES,
    SANDBOX_FORBIDDEN_MISSING_PACKAGES,
    SANDBOX_INSTALLED_PACKAGES,
    Stage3Diagnostics,
    _oracle_sandbox_contract_prompt,
    _strip_code_fences,
    _strip_outer_math_delimiters,
    classify_stage3_error,
    render_node_markdown,
    sanitize_stage3_stderr,
)
from ral.pipeline.oracle_sandbox import OracleSandbox, SandboxConfig
from ral.pipeline.oracles.base import OraclePluginRegistry, OracleResult, get_default_registry
from ral.pipeline.scout import (
    ORACLE_PLUGIN_ALIASES,
    VALID_ORACLE_PLUGINS,
    ScoutCandidate,
    ScoutStage,
    normalize_oracle_alias,
)
from ral.sources import SourceRecord
from ral.sources.openalex import search_oa_works
from ral.storage.drive_ore import build_ore_uri

REPO_ROOT = Path(__file__).resolve().parent.parent

CLEAN_KERNEL = """import numpy as np

x_vals = [1.0, 2.0, 3.0]
for x in x_vals:
    assert np.isclose((x + 1.0) ** 2, x ** 2 + 2.0 * x + 1.0)
assert np.isclose((5.0 + 1.0) ** 2, 36.0)
assert np.isclose((2.0 - 1.0) * (2.0 + 1.0), 3.0)
"""


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


def _source() -> SourceRecord:
    text = "Theorem 1. For every real scalar x, (x + 1)^2 = x^2 + 2x + 1."
    return SourceRecord(
        source_id="arxiv:2403.00001v1",
        source_license="CC-BY-4.0",
        source_format="arxiv_tex",
        title="Binomial Square Identity",
        pub_year=2024,
        peer_reviewed=True,
        retracted=False,
        canonical_doi="10.1000/pass1.1",
        arxiv_base_id="2403.00001",
        arxiv_version=1,
        citation_count=42,
        raw_bytes=b"pass1-source-bytes",
        raw_ext="tar.gz",
        extracted_text=text,
    )


def _candidate(recommended_oracle: str = "sympy_numeric") -> ScoutCandidate:
    quote = "For every real scalar x, (x + 1)^2 = x^2 + 2x + 1."
    return ScoutCandidate(
        source_id="arxiv:2403.00001v1",
        title="Binomial Square Identity",
        claim_summary="Exact binomial expansion identity for real scalars.",
        quote_text=quote,
        quote_sha256=compute_quote_sha256(quote),
        quote_char_len=len(quote),
        quote_match_ratio=1.0,
        source_locator="tex:sec1",
        node_type="THEOREM",
        taxonomy_code="math.algebra",
        formal_signature=("operator:binomial_expansion",),
        assumptions=("x is a real scalar",),
        recommended_oracle=recommended_oracle,
        dedup_action="PROCEED",
    )


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


def test_p0_1_strict_oracle_enum_aliases_and_repair_switching(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    src = _source()
    router = _router()

    expected_aliases = {
        "sympy": "sympy_numeric",
        "symbolic": "sympy_numeric",
        "numpy": "sympy_numeric",
        "pint": "sympy_numeric",
        "math": "sympy_numeric",
        "python": "sympy_numeric",
        "z3": "z3_smt",
        "smt": "z3_smt",
        "hypothesis": "z3_smt",
        "logic": "z3_smt",
        "scipy": "empirical_stats",
        "stats": "empirical_stats",
        "statistical": "empirical_stats",
        "empirical": "empirical_stats",
        "statistics": "empirical_stats",
    }
    for alias, canonical in expected_aliases.items():
        assert ORACLE_PLUGIN_ALIASES[alias] == canonical
        assert normalize_oracle_alias(alias) == canonical

    scout_prompts: list[str] = []

    def _scout_gen(**kw: Any) -> Any:
        scout_prompts.append(kw["contents"])
        return SimpleNamespace(
            text=json.dumps(
                {
                    "candidates": [
                        {
                            "title": "Binomial Identity",
                            "claim_summary": "Binomial expansion",
                            "quote_text": "For every real scalar x, (x + 1)^2 = x^2 + 2x + 1.",
                            "source_locator": "tex:sec1",
                            "node_type": "THEOREM",
                            "taxonomy_code": "math.algebra",
                            "formal_signature": ["operator:binomial"],
                            "assumptions": ["x is a real scalar"],
                            "recommended_oracle": "sympy",
                        },
                        {
                            "title": "Binomial Unknown Oracle",
                            "claim_summary": "Binomial expansion",
                            "quote_text": "For every real scalar x, (x + 1)^2 = x^2 + 2x + 1.",
                            "source_locator": "tex:sec1",
                            "node_type": "THEOREM",
                            "taxonomy_code": "math.algebra",
                            "formal_signature": ["operator:binomial"],
                            "assumptions": ["x is a real scalar"],
                            "recommended_oracle": "networkx_graph",
                        },
                    ]
                }
            )
        )

    scout = ScoutStage(
        SimpleNamespace(models=SimpleNamespace(generate_content=_scout_gen)),
        router,
        allowlist_path=tmp_path / "config" / "license_allowlist.yaml",
    )
    _, cands = scout.extract_candidates(src, repo_root=tmp_path)
    assert len(cands) == 2
    assert all(c.recommended_oracle in VALID_ORACLE_PLUGINS for c in cands)
    assert cands[0].recommended_oracle == "sympy_numeric"
    assert cands[1].recommended_oracle == "sympy_numeric"
    assert '"sympy_numeric" | "z3_smt" | "empirical_stats"' in scout_prompts[0]
    assert "Domain-to-Oracle Routing Rubric" in scout_prompts[0]

    sbx = _local_sandbox()
    reg = get_default_registry(sandbox=sbx)

    # Fallback priority order: sympy_numeric -> empirical_stats -> z3_smt
    no_sympy_cfg = tmp_path / "no_sympy_oracles.yaml"
    no_sympy_cfg.write_text(
        yaml.safe_dump(
            {
                "schema_version": "2.1",
                "plugins": {
                    "sympy_numeric": {"enabled": False},
                    "z3_smt": {"enabled": True},
                    "empirical_stats": {"enabled": True},
                },
            }
        ),
        encoding="utf-8",
    )
    f_no_sympy = FormalizerStage(
        SimpleNamespace(models=SimpleNamespace(generate_content=lambda **kw: None)),
        router,
        reg,
        oracle_config_path=no_sympy_cfg,
    )
    assert f_no_sympy._resolve_enabled_oracle("unknown_plugin", "also_unknown") == "empirical_stats"
    assert f_no_sympy._resolve_enabled_oracle("scipy", None) == "empirical_stats"
    assert f_no_sympy._resolve_enabled_oracle("smt", None) == "z3_smt"

    # Stage 3 repair can switch oracle_plugin instead of keeping current.oracle_plugin fixed.
    repair_calls = 0

    def _switch_oracle_gen(**kw: Any) -> Any:
        nonlocal repair_calls
        prompt = kw["contents"]
        if "Formalize this verified scientific quote" in prompt:
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "node_type": "THEOREM",
                        "taxonomy_code": "math.algebra",
                        "title": "Binomial Square Identity",
                        "formal_signature": ["operator:binomial_expansion"],
                        "assumptions": ["x is a real scalar"],
                        "formal_statement_latex": "(x+1)^2 = x^2 + 2x + 1",
                        "symbol_rows": [{"symbol": "x", "meaning": "real", "domain": "\\mathbb{R}", "si_units": "1"}],
                        "failure_boundaries": ["non-commutative rings"],
                        "provenance_summary": "Exact polynomial identity.",
                        "oracle_plugin": "z3_smt",
                        "verify_py": CLEAN_KERNEL,  # Uses numpy -> invalid for z3_smt
                    }
                )
            )
        repair_calls += 1
        return SimpleNamespace(
            text=json.dumps(
                {
                    "oracle_plugin": "sympy",  # Alias switched during repair!
                    "verify_py": CLEAN_KERNEL,
                    "assumptions": ["x is a real scalar"],
                }
            )
        )

    f_switch = FormalizerStage(
        SimpleNamespace(models=SimpleNamespace(generate_content=_switch_oracle_gen)),
        router,
        reg,
        oracle_config_path=tmp_path / "config" / "oracle_plugins.yaml",
    )
    draft, res, diag = f_switch.formalize_and_verify(
        _candidate("z3_smt"),
        src,
        max_repairs=1,
        repo_root=tmp_path,
        return_diagnostics=True,
    )
    assert repair_calls == 1
    assert draft is not None and res.passed is True
    assert draft.oracle_plugin == "sympy_numeric"
    assert diag.oracle_plugin == "sympy_numeric"
    assert diag.repair_attempts == 1
    assert diag.stage3_error_code is None

    # DISABLED_OR_UNKNOWN_ORACLE classified distinctly and NOT written to ledger/rejected/
    assert classify_stage3_error("DISABLED_OR_UNKNOWN_ORACLE: lean4_external") == "DISABLED_OR_UNKNOWN_ORACLE"
    assert classify_stage3_error("Traceback: ValueError") == "STAGE3_ORACLE_EXEC_FAIL"

    all_disabled_cfg = tmp_path / "all_disabled.yaml"
    all_disabled_cfg.write_text(
        yaml.safe_dump(
            {
                "schema_version": "2.1",
                "plugins": {
                    "sympy_numeric": {"enabled": False},
                    "z3_smt": {"enabled": False},
                    "empirical_stats": {"enabled": False},
                    "lean4_external": {"enabled": False},
                },
            }
        ),
        encoding="utf-8",
    )
    f_disabled = FormalizerStage(
        SimpleNamespace(
            models=SimpleNamespace(
                generate_content=lambda **kw: SimpleNamespace(
                    text=json.dumps(
                        {
                            "node_type": "THEOREM",
                            "taxonomy_code": "math.algebra",
                            "title": "Binomial Square Identity",
                            "formal_signature": ["operator:binomial_expansion"],
                            "assumptions": ["x is a real scalar"],
                            "formal_statement_latex": "(x+1)^2 = x^2 + 2x + 1",
                            "oracle_plugin": "lean4_external",
                            "verify_py": CLEAN_KERNEL,
                        }
                    )
                )
            )
        ),
        router,
        reg,
        oracle_config_path=all_disabled_cfg,
    )
    rej_draft, rej_res, rej_diag = f_disabled.formalize_and_verify(
        _candidate("lean4_external"),
        src,
        max_repairs=1,
        repo_root=tmp_path,
        return_diagnostics=True,
    )
    assert rej_draft is None and rej_res.passed is False
    assert rej_diag.stage3_error_code == "DISABLED_OR_UNKNOWN_ORACLE"
    assert not list((tmp_path / "ledger" / "rejected").glob("*.ndjson"))


def test_p0_2_sandbox_contract_prompt_and_repair_diagnostics(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    contract = _oracle_sandbox_contract_prompt()
    for pkg in SANDBOX_INSTALLED_PACKAGES:
        assert pkg in contract
    for mod in SANDBOX_ALLOWED_STDLIB:
        assert mod in contract
    for missing in SANDBOX_FORBIDDEN_MISSING_PACKAGES:
        assert missing in contract
    for banned in SANDBOX_FORBIDDEN_BUILTINS_AND_MODULES:
        assert banned in contract
    assert "sp.simplify(lhs - rhs) == 0" in contract
    assert "np.isclose" in contract
    assert "bare `except:`" in contract
    assert "NO markdown code fences" in contract

    prompts_seen: list[str] = []

    def _sandbox_runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        script_arg = cmd[-1]
        if len(prompts_seen) == 1:
            return subprocess.CompletedProcess(
                args=cmd,
                returncode=1,
                stdout="PARTIAL_STDOUT_TRACE=42",
                stderr="AssertionError at check 2",
            )
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="OK\n", stderr="")

    sbx = OracleSandbox(SandboxConfig(), runner=_sandbox_runner)
    reg = get_default_registry(sandbox=sbx)

    def _llm(**kw: Any) -> Any:
        prompt = kw["contents"]
        prompts_seen.append(prompt)
        return SimpleNamespace(
            text=json.dumps(
                {
                    "node_type": "THEOREM",
                    "taxonomy_code": "math.algebra",
                    "title": "Binomial Square Identity",
                    "formal_signature": ["operator:binomial_expansion"],
                    "assumptions": ["x is a real scalar"],
                    "formal_statement_latex": "(x+1)^2 = x^2 + 2x + 1",
                    "symbol_rows": [{"symbol": "x", "meaning": "real", "domain": "\\mathbb{R}", "si_units": "1"}],
                    "failure_boundaries": ["non-commutative rings"],
                    "provenance_summary": "Exact polynomial identity.",
                    "oracle_plugin": "sympy_numeric",
                    "verify_py": CLEAN_KERNEL,
                }
            )
        )

    formalizer = FormalizerStage(
        SimpleNamespace(models=SimpleNamespace(generate_content=_llm)),
        _router(),
        reg,
        oracle_config_path=tmp_path / "config" / "oracle_plugins.yaml",
    )
    draft, res = formalizer.formalize_and_verify(_candidate(), _source(), max_repairs=1, repo_root=tmp_path)
    assert draft is not None and res.passed is True
    assert len(prompts_seen) == 2

    formalize_prompt, repair_prompt = prompts_seen
    assert contract in formalize_prompt
    assert contract in repair_prompt
    assert "Current oracle_plugin: sympy_numeric" in repair_prompt
    assert "Current formal_statement_latex: (x+1)^2 = x^2 + 2x + 1" in repair_prompt
    assert "PARTIAL_STDOUT_TRACE=42" in repair_prompt


def test_p0_3_kernel_regex_and_fence_delimiter_stripping(tmp_path: Path) -> None:
    _init_repo(tmp_path)
    src = _source()

    # 1. Kernel containing `## ` and `### ` inside Python comments must not be truncated.
    commented_kernel = (
        "import numpy as np\n"
        "## Section comment\n"
        "### Subsection comment\n"
        "x = 2.0\n"
        "assert np.isclose((x + 1.0) ** 2, 9.0)\n"
        "assert np.isclose(x ** 2 + 2.0 * x + 1.0, 9.0)\n"
        "assert np.isclose((x - 1.0) * (x + 1.0), x ** 2 - 1.0)\n"
    )
    draft = FormalizedNodeDraft(
        ral_id="RAL-MATH-01KERNEL99",
        node_type="THEOREM",
        taxonomy_code="math.algebra",
        title="Commented Kernel Theorem",
        formal_signature=("operator:binomial_expansion",),
        assumptions=("x is real",),
        formal_statement_latex="(x+1)^2 = x^2 + 2x + 1",
        symbol_rows=({"symbol": "x", "meaning": "real", "domain": "\\mathbb{R}", "si_units": "1"},),
        failure_boundaries=("none",),
        provenance_summary="Verified identity.",
        oracle_plugin="sympy_numeric",
        verify_py=commented_kernel,
        formalizer_model="gemini-2.5-pro",
        model_capability_epoch=1,
    )
    md = render_node_markdown(
        draft,
        epistemic_tier="T1_CANONICAL",
        oracle_status="PASS",
        skeptic_model="gemma-4-27b-it",
        source=src,
        quote_sha256=compute_quote_sha256("quote"),
        quote_char_len=5,
        source_locator="tex:sec1",
        re_evaluate_after="2028-11-25",
    )
    val = validate_node_content(md, allowed_licenses={"CC-BY-4.0"}, expected_filename_stem=draft.ral_id)
    assert val.ok is True, val.errors
    assert val.kernel_code == commented_kernel.strip()

    # Missing kernel directly under heading should still fail even if a later section has a Python fence.
    missing_kernel_body = (
        "### Executable Verification Kernel\n"
        "No block here.\n\n"
        "### Canonical Provenance Anchors\n"
        "```python\n"
        "print('not the kernel')\n"
        "```\n"
    )
    assert extract_verification_kernel(missing_kernel_body) is None

    # 2. Stripper helpers
    assert _strip_code_fences("```python\nprint(1)\n```") == "print(1)"
    assert _strip_code_fences("```\n```py\nprint(2)\n```\n```") == "print(2)"
    assert _strip_outer_math_delimiters("$$ (x+1)^2 = x^2 + 2x + 1 $$") == "(x+1)^2 = x^2 + 2x + 1"
    assert _strip_outer_math_delimiters(r"\[E = mc^2\]") == "E = mc^2"
    assert _strip_outer_math_delimiters(r"\(a+b=c\)") == "a+b=c"
    assert _strip_outer_math_delimiters("$a = b$") == "a = b"
    assert _strip_outer_math_delimiters("$x$ and $y$") == "$x$ and $y$"

    # 3. Applied across _draft_from_payload, _run_stage3_with_repairs, and tighten_assumptions
    fenced_kernel = f"```python\n{CLEAN_KERNEL.strip()}\n```"

    def _strip_llm(**kw: Any) -> Any:
        prompt = kw["contents"]
        if "counterexample against your formalization" in prompt:
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "assumptions": ["x is a finite real scalar"],
                        "failure_boundaries": ["infinite x"],
                        "formal_statement_latex": r"\[(x+1)^2 = x^2 + 2x + 1\]",
                        "verify_py": fenced_kernel,
                    }
                )
            )
        if "Repair the executable verification script" in prompt:
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "formal_statement_latex": "$$(x+1)^2 = x^2 + 2x + 1$$",
                        "verify_py": fenced_kernel,
                        "assumptions": ["x is a real scalar"],
                    }
                )
            )
        return SimpleNamespace(
            text=json.dumps(
                {
                    "node_type": "THEOREM",
                    "taxonomy_code": "math.algebra",
                    "title": "Binomial Square Identity",
                    "formal_signature": ["operator:binomial_expansion"],
                    "assumptions": ["x is a real scalar"],
                    "formal_statement_latex": "$$(x+1)^2 = x^2 + 2x + 1$$",
                    "symbol_rows": [{"symbol": "x", "meaning": "real", "domain": "\\mathbb{R}", "si_units": "1"}],
                    "failure_boundaries": ["non-commutative rings"],
                    "provenance_summary": "Exact polynomial identity.",
                    "oracle_plugin": "sympy_numeric",
                    "verify_py": "```python\nimport numpy as np\nassert True\n```",
                }
            )
        )

    sbx = _local_sandbox()
    formalizer = FormalizerStage(
        SimpleNamespace(models=SimpleNamespace(generate_content=_strip_llm)),
        _router(),
        get_default_registry(sandbox=sbx),
        oracle_config_path=tmp_path / "config" / "oracle_plugins.yaml",
    )
    repaired_draft, res = formalizer.formalize_and_verify(_candidate(), src, max_repairs=1, repo_root=tmp_path)
    assert repaired_draft is not None and res.passed is True
    assert not repaired_draft.verify_py.startswith("```")
    assert not repaired_draft.formal_statement_latex.startswith("$")

    tightened, t_res = formalizer.tighten_assumptions(
        repaired_draft,
        "x cannot be infinite",
        ["x is finite"],
        max_repairs=0,
    )
    assert tightened is not None and t_res.passed is True
    assert not tightened.verify_py.startswith("```")
    assert tightened.formal_statement_latex == "(x+1)^2 = x^2 + 2x + 1"


def test_p2_diagnostics_domain_routing_and_ledger_cleanup(tmp_path: Path) -> None:
    # P2-3: 2026-11.ndjson cleared
    live_rejected = REPO_ROOT / "ledger" / "rejected" / "2026-11.ndjson"
    assert live_rejected.exists()
    assert live_rejected.read_text(encoding="utf-8") == ""

    # P2-1: stderr sanitization & tail <= 400 chars
    noisy = (
        "C:\\Users\\alice\\secret\\script.py /home/bob/work/run.py "
        "AIzaSyDummyKey1234567890 alice@example.com " + ("Z" * 500)
    )
    cleaned = sanitize_stage3_stderr(noisy, max_len=400)
    assert len(cleaned) <= 400
    assert cleaned.startswith("...")
    assert "alice" not in cleaned
    assert "AIza" not in cleaned
    assert "Users" not in cleaned

    # P2-2: OpenAlex sort="cited_by_count:desc" and taxonomy leaf query expansion
    _init_repo(tmp_path)
    seen_oa_params: list[dict[str, Any]] = []
    tar_buf = io.BytesIO()
    with tarfile.open(fileobj=tar_buf, mode="w:gz") as tf:
        tex = "\\documentclass{article}\\begin{document}\\section{Main}Theorem: E = mc^2.\\end{document}".encode("utf-8")
        info = tarfile.TarInfo("main.tex")
        info.size = len(tex)
        tf.addfile(info, io.BytesIO(tex))
    tar_bytes = tar_buf.getvalue()

    def _mock_http(url: str, **kwargs: Any) -> Any:
        params = dict(kwargs.get("params") or {})
        if "api.openalex.org" in url:
            seen_oa_params.append(params)
            return SimpleNamespace(
                raise_for_status=lambda: None,
                json=lambda: {
                    "results": [
                        {
                            "id": "https://openalex.org/W-TOP",
                            "title": "Statistical Mechanics Theorem",
                            "doi": "https://doi.org/10.48550/arXiv.2401.00001",
                            "publication_year": 2024,
                            "cited_by_count": 200,
                            "is_retracted": False,
                            "type": "article",
                            "best_oa_location": {
                                "license": "cc-by",
                                "pdf_url": "https://arxiv.org/pdf/2401.00001.pdf",
                            },
                            "primary_location": {"source": {"type": "repository"}},
                        }
                    ]
                },
            )
        if "api.crossref.org" in url:
            return SimpleNamespace(raise_for_status=lambda: None, json=lambda: {"message": {"update-to": []}})
        if "arxiv.org/e-print" in url:
            return SimpleNamespace(raise_for_status=lambda: None, content=tar_bytes)
        raise AssertionError(f"Unexpected URL {url}")

    recs = _default_fetch_sources_for_epoch(
        "explore",
        "phys.statistical-mechanics",
        tmp_path,
        http_get=_mock_http,
        max_sources=1,
        sandbox=_local_sandbox(),
    )
    assert len(recs) == 1
    assert seen_oa_params[0]["sort"] == "cited_by_count:desc"
    q_used = seen_oa_params[0]["search"]
    assert "Statistical Mechanics and Thermodynamics" in q_used
    for kw in ("theorem", "identity", "equation", "bound"):
        assert kw in q_used

    # P2-1: End-to-end Stage3Diagnostics -> ledger/rejected, epoch_{k}_summary.json, Job 7 telemetry
    handoff = tmp_path / "handoff"
    handoff.mkdir(parents=True, exist_ok=True)
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    (handoff / "plan.json").write_text(
        json.dumps(
            {
                "current_night": 1,
                "epoch_schedule": [{"epoch_index": 1, "mode": "explore", "target": "math.algebra"}],
            }
        ),
        encoding="utf-8",
    )
    (handoff / "router_state.json").write_text(
        json.dumps(_router().routing_table),
        encoding="utf-8",
    )
    (handoff / "circuit_breaker.json").write_text(
        json.dumps({"circuit_breaker": None, "drive_canary": {"ok": True}}),
        encoding="utf-8",
    )

    src = _source()

    def _always_fail_llm(**kw: Any) -> Any:
        prompt = kw["contents"]
        if "verbatim quotes" in prompt:
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "candidates": [
                            {
                                "title": "Binomial Identity",
                                "claim_summary": "Binomial expansion",
                                "quote_text": "For every real scalar x, (x + 1)^2 = x^2 + 2x + 1.",
                                "source_locator": "tex:sec1",
                                "node_type": "THEOREM",
                                "taxonomy_code": "math.algebra",
                                "formal_signature": ["operator:binomial"],
                                "assumptions": ["x is a real scalar"],
                                "recommended_oracle": "sympy_numeric",
                            }
                        ]
                    }
                )
            )
        return SimpleNamespace(
            text=json.dumps(
                {
                    "node_type": "THEOREM",
                    "taxonomy_code": "math.algebra",
                    "title": "Bad Kernel",
                    "formal_signature": ["operator:bad"],
                    "assumptions": ["x is real"],
                    "formal_statement_latex": "(x+1)^2 = x^2 + 2x + 1",
                    "oracle_plugin": "sympy_numeric",
                    "verify_py": "import sympy as sp\nassert True\n",
                }
            )
        )

    sbx = _local_sandbox()
    ep_res = run_epoch_job(
        tmp_path,
        handoff_dir=handoff,
        epoch_index=1,
        cache_dir=cache_dir,
        ore_cache_dir=tmp_path / "ore_cache",
        dry_run=True,
        gemini_client=SimpleNamespace(models=SimpleNamespace(generate_content=_always_fail_llm)),
        sandbox=sbx,
        registry=get_default_registry(sandbox=sbx),
        source_fetcher=lambda m, t, r: [src],
        now=datetime(2026, 11, 25, 10, 0, tzinfo=timezone.utc),
    )
    assert ep_res.rejected_count == 1
    assert ep_res.stage3_error_counts == {"AST_CHEAT_OR_ASSERT_FAIL": 1}
    assert len(ep_res.stage3_diagnostics) == 1
    assert ep_res.stage3_diagnostics[0]["repair_attempts"] == 2
    assert ep_res.stage3_diagnostics[0]["oracle_plugin"] == "sympy_numeric"

    summary_json = json.loads((handoff / "epoch_1_summary.json").read_text(encoding="utf-8"))
    assert summary_json["stage3_error_counts"] == {"AST_CHEAT_OR_ASSERT_FAIL": 1}
    assert len(summary_json["stage3_diagnostics"]) == 1

    rej_lines = [
        json.loads(line)
        for line in (tmp_path / "ledger" / "rejected" / "2026-11.ndjson").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rej_lines) == 1
    assert rej_lines[0]["oracle_plugin"] == "sympy_numeric"
    assert rej_lines[0]["formalizer_model_id"] == "gemini-2.5-pro"
    assert rej_lines[0]["repair_attempts"] == 2
    assert rej_lines[0]["stage3_error_code"] == "AST_CHEAT_OR_ASSERT_FAIL"
    assert len(rej_lines[0]["stage3_stderr_tail"]) <= 400

    con_res = run_consolidate_job(
        tmp_path,
        handoff_dir=handoff,
        cache_dir=cache_dir,
        run_date_iso="2026-11-25",
        run_id="pass1-test",
        dry_run=True,
        now=datetime(2026, 11, 25, 11, 0, tzinfo=timezone.utc),
    )
    tel_rows = [
        json.loads(line)
        for line in con_res.telemetry_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert tel_rows[-1]["stage3_error_counts"] == {"AST_CHEAT_OR_ASSERT_FAIL": 1}

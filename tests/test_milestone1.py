"""Comprehensive Milestone 1 acceptance and regression tests (RAL Blueprint v2.1)."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import tomllib
from typing import Any

import jsonschema
import pytest
import yaml

from ci.validate_node import (
    count_prose_words,
    validate_node_content,
    validate_node_file,
)
from ci.validate_vault import (
    check_tier_firewall,
    detect_depends_on_cycles,
    scan_text_for_leaks,
    validate_vault,
)
from ral.canonicalize import (
    canonical_text,
    compute_assumptions_hash,
    compute_oracle_script_sha256,
    compute_quote_sha256,
    quote_match_ratio,
)
from ral.pipeline.ast_linter import lint_verification_script
from ral.pipeline.oracle_sandbox import (
    OracleSandbox,
    SandboxConfig,
    scrubbed_host_env,
)
from ral.pipeline.oracles.base import (
    OraclePlugin,
    OraclePluginRegistry,
    OracleResult,
)
from ral.storage.drive_ore import (
    DRIVE_FILE_SCOPE,
    DRIVE_FOLDER_MIME,
    DriveAuthError,
    DriveOreStore,
    OreIntegrityError,
    build_drive_service_from_env,
    build_ore_relative_path,
    build_ore_uri,
)
from scripts import mint_gdrive_token

REPO_ROOT = Path(__file__).resolve().parent.parent

VALID_FDT_KERNEL = """# RAL_ORACLE: sympy_numeric | TIMEOUT: 15s
import sympy as sp
import numpy as np

w, kT, hbar = sp.symbols("w kT hbar", positive=True, real=True)
quantum_factor = hbar * sp.coth(hbar * w / (2 * kT))
classical_limit = sp.limit(quantum_factor, hbar, 0)
assert sp.simplify(classical_limit - (2 * kT / w)) == 0

for ratio in np.logspace(1, 5, 50):
    q_val = 1.0 / np.tanh(1.0 / (2.0 * ratio))
    c_val = 2.0 * ratio
    assert np.isclose(q_val, c_val, rtol=1e-2)

assert abs((1.0 / np.tanh(1.0 / (2.0 * 0.05))) - (2.0 * 0.05)) > 0.8
"""


def _build_node_markdown(
    *,
    ral_id: str = "RAL-PHYS-01J9X8K2M4",
    epistemic_tier: str = "T1_CANONICAL",
    node_type: str = "THEOREM",
    taxonomy_code: str = "phys.statistical-mechanics.thermodynamics",
    source_license: str = "arXiv-non-exclusive-1.0",
    ore_bytes: bytes = b"raw-arxiv-tarball-fixture",
    assumptions: list[str] | None = None,
    assumptions_hash_override: str | None = None,
    kernel_code: str = VALID_FDT_KERNEL,
    oracle_hash_override: str | None = None,
    extra_prose: str = "",
    archived_details: str = "",
) -> tuple[str, str]:
    assump = assumptions if assumptions is not None else [
        "system is in canonical thermal equilibrium with temperature T > 0",
        "external perturbation satisfies linear response |delta_H| << k_B * T",
        "classical high-temperature limit hbar * omega << k_B * T",
    ]
    ore_sha256 = hashlib.sha256(ore_bytes).hexdigest()
    ore_uri = build_ore_uri(ore_sha256, "tar.gz")
    quote_sha256 = compute_quote_sha256("Classical linear-response fluctuation-dissipation theorem.")
    a_hash = assumptions_hash_override or compute_assumptions_hash(assump)
    s_hash = oracle_hash_override or compute_oracle_script_sha256(kernel_code)

    fm = {
        "ral_schema": "2.1",
        "ral_id": ral_id,
        "node_type": node_type,
        "epistemic_tier": epistemic_tier,
        "consensus_version": 1,
        "taxonomy_code": taxonomy_code,
        "title": "Fluctuation-Dissipation Theorem (Classical Linear Response)",
        "formal_signature": [
            "operator:autocorrelation",
            "operator:fourier_transform",
            "symmetry:time_translation",
        ],
        "assumptions": assump,
        "assumptions_hash": a_hash,
        "verification": {
            "oracle_plugin": "sympy_numeric",
            "oracle_status": "PASS",
            "oracle_script_sha256": s_hash,
            "verified_at": "2026-11-24T20:14:02Z",
            "formalizer_model": "gemini-2.5-pro",
            "skeptic_model": "gemma-4-27b-it",
            "model_capability_epoch": 1,
        },
        "provenance": [
            {
                "source_id": "arxiv:2104.01234v2",
                "source_license": source_license,
                "source_format": "arxiv_tex",
                "source_locator": "tex:sec2.3:eq14",
                "ore_uri": ore_uri,
                "ore_sha256": ore_sha256,
                "quote_sha256": quote_sha256,
                "quote_char_len": 56,
                "pub_year": 2021,
                "peer_reviewed": True,
                "retracted": False,
            }
        ],
        "re_evaluate_after": "2028-11-24",
    }
    yaml_text = yaml.safe_dump(fm, sort_keys=False).strip()
    md = (
        f"---\n{yaml_text}\n---\n\n"
        "## Active Consensus (v1 — Effective 2026-11-24)\n\n"
        "### Formal Statement\n"
        "$$\\chi''(\\omega) = \\frac{\\omega}{2 k_B T} S_{AA}(\\omega)$$\n\n"
        "| Symbol | Meaning | Domain | SI Units |\n"
        "| :--- | :--- | :--- | :--- |\n"
        "| $\\chi''(\\omega)$ | Imaginary susceptibility | $\\mathbb{R}$ | $[A]^2 \\cdot \\mathrm{J}^{-1}$ |\n\n"
        "### Failure Boundaries & Known Pathologies\n"
        "- **Breaks when:** Driven non-equilibrium steady states or quantum regime $\\hbar\\omega \\gtrsim k_B T$.\n"
        f"{extra_prose}\n\n"
        "### Executable Verification Kernel\n"
        f"```python\n{kernel_code.strip()}\n```\n\n"
        "### Canonical Provenance Anchors\n"
        "- `arxiv:2104.01234v2` (`tex:sec2.3:eq14`): Classical linear-response FDT.\n\n"
        "## Active Challenges & Open Anomalies\n"
        "- *(None)*\n\n"
        "## Superseded & Archived\n"
        f"{archived_details}\n"
    )
    return md, ore_sha256


# ---------------------------------------------------------------------------
# 1. Packaging, Dockerfile, and Config Validation
# ---------------------------------------------------------------------------


def test_packaging_and_dockerfile_pins() -> None:
    pyproject = tomllib.loads((REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    reqs = (REPO_ROOT / "requirements.txt").read_text(encoding="utf-8")
    dockerfile = (REPO_ROOT / "ci" / "sandbox.Dockerfile").read_text(encoding="utf-8")

    expected_pkgs = [
        "pydantic",
        "pyyaml",
        "google-genai",
        "google-api-python-client",
        "google-auth-oauthlib",
        "sympy",
        "scipy",
        "numpy",
        "z3-solver",
        "hypothesis",
        "pint",
        "pytest",
    ]
    combined_deps = [d.lower() for d in pyproject["project"]["dependencies"]] + [
        d.lower() for d in pyproject["project"]["optional-dependencies"]["dev"]
    ]
    for pkg in expected_pkgs:
        assert any(dep.startswith(f"{pkg}==") for dep in combined_deps), f"{pkg} not pinned in pyproject.toml"
        assert re.search(rf"(?im)^{re.escape(pkg)}==", reqs), f"{pkg} not pinned in requirements.txt"

    assert "FROM python:3.12-slim" in dockerfile
    assert "USER 65534:65534" in dockerfile
    for pkg in ("sympy==1.13.3", "scipy==1.14.1", "numpy==2.1.3", "z3-solver==4.13.3.0", "hypothesis==6.119.0", "pint==0.24.4"):
        assert pkg in dockerfile


def test_config_yamls_match_blueprint_contracts() -> None:
    model_policy = yaml.safe_load((REPO_ROOT / "config" / "model_policy.yaml").read_text(encoding="utf-8"))
    deny_pats = model_policy["frontier_exclude_patterns"] + model_policy.get("user_reserved_patterns", [])
    assert any(re.search(p, "gemini-4-argon", re.I) for p in deny_pats)
    assert any(re.search(p, "gemini-3.5-neon", re.I) for p in deny_pats)
    assert not any(re.search(p, "gemini-2.5-pro", re.I) for p in deny_pats)
    assert not any(re.search(p, "gemma-4-27b-it", re.I) for p in deny_pats)
    assert model_policy["embedding_output_dim"] == 768

    allowlist = yaml.safe_load((REPO_ROOT / "config" / "license_allowlist.yaml").read_text(encoding="utf-8"))
    assert set(allowlist["allowed_licenses"]) >= {
        "CC-BY-4.0",
        "CC0-1.0",
        "arXiv-non-exclusive-1.0",
        "MIT",
        "Apache-2.0",
        "PMC-OA",
    }

    plugins_cfg = yaml.safe_load((REPO_ROOT / "config" / "oracle_plugins.yaml").read_text(encoding="utf-8"))
    assert plugins_cfg["plugins"]["sympy_numeric"]["enabled"] is True
    assert plugins_cfg["plugins"]["z3_smt"]["enabled"] is True
    assert plugins_cfg["plugins"]["empirical_stats"]["enabled"] is True
    assert plugins_cfg["plugins"]["lean4_external"]["enabled"] is False

    taxonomy = yaml.safe_load((REPO_ROOT / "config" / "taxonomy_seeds.yaml").read_text(encoding="utf-8"))
    macros = taxonomy["macro_disciplines"]
    assert set(macros.keys()) == {"math", "phys", "cs", "eng", "econ", "chem", "bio"}
    total_leaves = sum(len(m["leaves"]) for m in macros.values())
    assert total_leaves == 35
    for macro, info in macros.items():
        for leaf in info["leaves"]:
            assert leaf["code"].startswith(f"{macro}.")

    curriculum = yaml.safe_load((REPO_ROOT / "config" / "curriculum_params.yaml").read_text(encoding="utf-8"))
    assert curriculum["priority_score"]["alpha"] == 1.0
    assert curriculum["priority_score"]["beta"] == 0.85
    assert curriculum["prereq_backoff"]["quarantine_attempts"] == 5
    assert curriculum["portfolio_allocation"]["max_epochs_per_night"] == 6


def test_calibration_suite_20_problems_and_json_schemas() -> None:
    suite = json.loads((REPO_ROOT / "config" / "calibration_suite.json").read_text(encoding="utf-8"))
    problems = suite["problems"]
    assert len(problems) == 20
    assert len({p["id"] for p in problems}) == 20

    by_cat: dict[str, list[dict[str, Any]]] = {}
    for p in problems:
        by_cat.setdefault(p["category"], []).append(p)

    assert len(by_cat["sympy_math"]) == 8
    assert len(by_cat["counterexample_trap"]) == 6
    assert len(by_cat["strict_json"]) == 6

    for item in by_cat["strict_json"]:
        jsonschema.validate(instance=item["reference_output"], schema=item["json_schema"])

    # Numerically verify all 8 SymPy reference expressions on their deterministic test points.
    math_ns = {
        "exp": math.exp,
        "sqrt": math.sqrt,
        "pi": math.pi,
        "log": math.log,
    }
    for item in by_cat["sympy_math"]:
        expr = item["expected_expr"]
        for pt in item["numeric_test_points"]:
            val = eval(expr, {"__builtins__": {}}, {**math_ns, **pt["inputs"]})
            assert math.isclose(float(val), float(pt["expected"]), rel_tol=1e-9, abs_tol=1e-12)


# ---------------------------------------------------------------------------
# 2. Canonicalization & Hashing
# ---------------------------------------------------------------------------


def test_canonicalization_and_hashes() -> None:
    raw = "  \u212a_B * T  % inline LaTeX comment\n  \\&  50\\%   Yields\tResult  "
    assert canonical_text(raw) == "k_b * t \\& 50\\% yields result"

    h1 = compute_assumptions_hash(["B > 0   % positive\n", "A = 1"])
    h2 = compute_assumptions_hash(["a = 1", "b > 0"])
    assert h1 == h2 and h1.startswith("sha256:")

    assert compute_oracle_script_sha256("x = 1\r\ny = 2\r\n") == compute_oracle_script_sha256("x = 1\ny = 2\n")
    assert quote_match_ratio("Theorem 1:  % draft\nEvery norm is convex.", "EVERY NORM IS CONVEX.") == 1.0
    assert quote_match_ratio("Alpha beta gamma", "Completely unrelated text") < 0.5


# ---------------------------------------------------------------------------
# 3. Acceptance Criterion #2: AST Linter Rejects Cheats & Forbidden APIs
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "bad_snippet",
    [
        "assert True\nassert True\nassert True",
        "x = 5\nassert 1\nassert x > 0\nassert x < 10",
        "x = 5\nassert 1 == 1\nassert x > 0\nassert x < 10",
        "x = 5\nassert x == x\nassert x > 0\nassert x < 10",
        "x = 5\nassert x - x == 0\nassert x > 0\nassert x < 10",
        "import numpy as np\nx = 2.0\nassert np.isclose(x, x)\nassert x > 0\nassert x < 5",
        "x = 5\ntry:\n    assert x > 0\nexcept:\n    x = 1\nassert x > 0\nassert x < 10\nassert x != 0",
        "x = 5\ntry:\n    assert x > 0\nexcept ValueError:\n    pass\nassert x > 0\nassert x < 10\nassert x != 0",
        "x = 5\ntry:\n    assert x > 0\nexcept AssertionError:\n    x = 1\nassert x > 0\nassert x < 10\nassert x != 0",
        "x = 5\nassert x > 0\nassert x < 10",
        "x = 5\nassert x > 0\nassert x > 0\nassert x > 0",
    ],
)
def test_ast_linter_rejects_cheat_assertions_and_excepts(bad_snippet: str) -> None:
    violations = lint_verification_script(bad_snippet, min_assertions=3, check_anti_cheat=True)
    assert violations, f"Expected AST linter rejection for:\n{bad_snippet}"


@pytest.mark.parametrize(
    "forbidden_code",
    [
        "import os\nx = 1\nassert x > 0\nassert x < 2\nassert x != 3",
        "from sys import exit\nx = 1\nassert x > 0\nassert x < 2\nassert x != 3",
        "import subprocess\nx = 1\nassert x > 0\nassert x < 2\nassert x != 3",
        "import socket\nx = 1\nassert x > 0\nassert x < 2\nassert x != 3",
        "import urllib.request\nx = 1\nassert x > 0\nassert x < 2\nassert x != 3",
        "import requests\nx = 1\nassert x > 0\nassert x < 2\nassert x != 3",
        "import http.client\nx = 1\nassert x > 0\nassert x < 2\nassert x != 3",
        "from pathlib import Path\nx = 1\nassert x > 0\nassert x < 2\nassert x != 3",
        "f = open('/etc/passwd')\nx = 1\nassert x > 0\nassert x < 2\nassert x != 3",
        "().__class__.__mro__[1].__subclasses__()\nx = 1\nassert x > 0\nassert x < 2\nassert x != 3",
    ],
)
def test_ast_linter_rejects_forbidden_imports_and_escapes(forbidden_code: str) -> None:
    violations = lint_verification_script(forbidden_code, min_assertions=3, check_anti_cheat=True)
    assert any(
        v.startswith(("FORBIDDEN_IMPORT", "FORBIDDEN_BUILTIN", "FORBIDDEN_ATTRIBUTE"))
        for v in violations
    ), f"Expected forbidden API violation, got {violations}"


def test_ast_linter_accepts_legitimate_fdt_kernel() -> None:
    assert lint_verification_script(VALID_FDT_KERNEL, min_assertions=3, check_anti_cheat=True) == []


# ---------------------------------------------------------------------------
# 4. Acceptance Criterion #1: Air-Gapped Sandbox & OraclePlugin Registry
# ---------------------------------------------------------------------------


def test_sandbox_blocks_os_environ_and_network_sockets_and_scrubs_secrets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("RAL_GEMINI_FREE_KEY", "AIzaFakeSecretKeyForTesting")
    monkeypatch.setenv("RAL_GDRIVE_REFRESH_TOKEN", "1//FakeRefreshToken")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_FakeGitHubToken1234567890")

    scrubbed = scrubbed_host_env()
    assert "RAL_GEMINI_FREE_KEY" not in scrubbed
    assert "RAL_GDRIVE_REFRESH_TOKEN" not in scrubbed
    assert "GITHUB_TOKEN" not in scrubbed

    invoked_commands: list[list[str]] = []

    def fake_runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        invoked_commands.append(cmd)
        # Simulate air-gapped failure when container tries network/env if lint bypassed
        return subprocess.CompletedProcess(
            args=cmd,
            returncode=1,
            stdout="",
            stderr="OSError: [Errno 101] Network is unreachable",
        )

    sandbox = OracleSandbox(runner=fake_runner)

    # Pre-launch AST gate blocks `import os; os.environ` before spawning Docker.
    os_res = sandbox.run_script("import os\nprint(os.environ)\n")
    assert os_res.passed is False
    assert "AST_LINT_REJECTED" in os_res.stderr
    assert invoked_commands == []

    # Pre-launch AST gate blocks `import socket; socket.create_connection(...)` before spawning Docker.
    sock_res = sandbox.run_script('import socket\nsocket.create_connection(("1.1.1.1", 80), timeout=1)\n')
    assert sock_res.passed is False
    assert "AST_LINT_REJECTED" in sock_res.stderr
    assert invoked_commands == []

    # Even with enforce_ast_lint=False, container runs with `--network=none`, `--read-only`, `--cap-drop=ALL`, zero `-e` flags.
    container_res = sandbox.run_script(
        'import socket\nsocket.create_connection(("1.1.1.1", 80), timeout=1)\n',
        enforce_ast_lint=False,
    )
    assert container_res.passed is False
    assert "Network is unreachable" in container_res.stderr
    assert len(invoked_commands) == 1
    cmd = invoked_commands[0]
    for required_flag in (
        "--rm",
        "--network=none",
        "--read-only",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--memory=1g",
        "--memory-swap=1g",
        "--cpus=1",
        "--pids-limit=64",
        "--user=65534:65534",
        "/tmp:rw,noexec,nosuid,size=64m",
        "ral-sandbox:latest",
    ):
        assert required_flag in cmd
    assert "-e" not in cmd
    assert "--env" not in cmd


def test_sandbox_handles_success_timeout_and_oracle_plugin_registry(tmp_path: Path) -> None:
    observed_envs: list[dict[str, str]] = []

    def pass_runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        observed_envs.append(kwargs["env"])
        mount_arg = cmd[cmd.index("-v") + 1]
        host_dir = Path(mount_arg.split(":/work:ro")[0])
        assert (host_dir / "verify.py").exists()
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout="OK\n", stderr="")

    sandbox = OracleSandbox(config=SandboxConfig(default_timeout_s=15), runner=pass_runner)
    res = sandbox.run_script(VALID_FDT_KERNEL, min_assertions=3)
    assert res.passed is True
    assert res.stdout == "OK\n"
    assert len(observed_envs) == 1

    def timeout_runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        raise subprocess.TimeoutExpired(cmd=cmd, timeout=15)

    timeout_sandbox = OracleSandbox(runner=timeout_runner)
    t_res = timeout_sandbox.run_script(VALID_FDT_KERNEL, timeout_s=15, min_assertions=3)
    assert t_res.passed is False
    assert "SANDBOX_TIMEOUT(15s)" in t_res.stderr

    class DummyOracle(OraclePlugin):
        name = "sympy_numeric"

        def lint(self, source_code: str) -> list[str]:
            return lint_verification_script(source_code, min_assertions=3)

        def execute(self, source_code: str, timeout_s: int = 30) -> OracleResult:
            return sandbox.run_script(source_code, plugin_name=self.name, timeout_s=timeout_s, min_assertions=3)

    reg = OraclePluginRegistry()
    reg.register(DummyOracle())
    assert reg.list_registered() == ["sympy_numeric"]
    cfg = OraclePluginRegistry.load_config(REPO_ROOT / "config" / "oracle_plugins.yaml")
    assert OraclePluginRegistry.is_enabled("sympy_numeric", cfg) is True
    assert OraclePluginRegistry.is_enabled("lean4_external", cfg) is False
    assert reg.get("sympy_numeric").execute(VALID_FDT_KERNEL).passed is True


# ---------------------------------------------------------------------------
# 5. Acceptance Criterion #3: validate_node.py & validate_vault.py Gates
# ---------------------------------------------------------------------------


def _write_valid_mini_vault(root: Path) -> None:
    (root / "config").mkdir(parents=True, exist_ok=True)
    (root / "config" / "license_allowlist.yaml").write_text(
        (REPO_ROOT / "config" / "license_allowlist.yaml").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    node_md, ore_sha = _build_node_markdown(ral_id="RAL-PHYS-01J9X8K2M4")
    node_dir = root / "vault" / "phys" / "statistical-mechanics"
    node_dir.mkdir(parents=True, exist_ok=True)
    (node_dir / "RAL-PHYS-01J9X8K2M4.md").write_text(node_md, encoding="utf-8")

    sources_dir = root / "ledger" / "sources"
    sources_dir.mkdir(parents=True, exist_ok=True)
    (sources_dir / "2026-11.ndjson").write_text(
        json.dumps(
            {
                "ore_sha256": ore_sha,
                "canonical_doi": "10.1103/physrev.83.34",
                "arxiv_base_id": "2104.01234",
                "arxiv_version": 2,
                "source_license": "arXiv-non-exclusive-1.0",
                "ore_uri": build_ore_uri(ore_sha, "tar.gz"),
                "linked_ral_ids": ["RAL-PHYS-01J9X8K2M4"],
                "ingested_at": "2026-11-24T20:14:05Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    (root / "ledger" / "prereq_debt.ndjson").write_text(
        json.dumps(
            {
                "stub_id": "STUB-MATH-HAHN-BANACH-COMPLEX",
                "domain": "math.analysis.functional",
                "cited_by": ["RAL-PHYS-01J9X8K2M4"],
                "attempt_count": 1,
                "last_attempt_night": 1,
                "cooldown_until_night": 3,
                "status": "OPEN",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    edges_dir = root / "graph" / "edges"
    edges_dir.mkdir(parents=True, exist_ok=True)
    (edges_dir / "2026-11.ndjson").write_text(
        json.dumps(
            {
                "edge_id": "E-01J9X8K9ZZ",
                "src": "RAL-PHYS-01J9X8K2M4",
                "dst": "STUB-MATH-HAHN-BANACH-COMPLEX",
                "relation": "DEPENDS_ON",
                "created_at": "2026-11-24T20:15:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )


def test_validate_node_and_valid_mini_vault_pass(tmp_path: Path) -> None:
    _write_valid_mini_vault(tmp_path)
    report = validate_vault(tmp_path)
    assert report.ok is True, f"Expected valid vault, got errors: {report.errors}"
    assert report.node_count == 1
    assert report.edge_count == 1

    # Archived <details> block > 350 words does not violate the active 350-word prose cap.
    archived = "<details>\n<summary>v0</summary>\n\n" + ("word " * 500) + "\n</details>"
    md_with_archive, _ = _build_node_markdown(archived_details=archived)
    res = validate_node_content(md_with_archive, allowed_licenses={"arXiv-non-exclusive-1.0"})
    assert res.ok is True
    assert res.prose_word_count <= 350


def test_validate_node_rejects_hash_tamper_prose_cap_and_filler() -> None:
    bad_assump, _ = _build_node_markdown(assumptions_hash_override="sha256:" + "0" * 64)
    r1 = validate_node_content(bad_assump)
    assert not r1.ok and any("ASSUMPTIONS_HASH_MISMATCH" in e for e in r1.errors)

    bad_script_hash, _ = _build_node_markdown(oracle_hash_override="sha256:" + "f" * 64)
    r2 = validate_node_content(bad_script_hash)
    assert not r2.ok and any("ORACLE_SCRIPT_HASH_MISMATCH" in e for e in r2.errors)

    bloated_prose, _ = _build_node_markdown(extra_prose=" ".join(["hypothesis"] * 360))
    r3 = validate_node_content(bloated_prose)
    assert not r3.ok and any("PROSE_CAP_EXCEEDED" in e for e in r3.errors)
    assert count_prose_words(bloated_prose) > 350

    filler_md, _ = _build_node_markdown(extra_prose="In conclusion, it is worth noting this holds.")
    r4 = validate_node_content(filler_md)
    assert not r4.ok and any("BANNED_FILLER_PHRASE" in e for e in r4.errors)


def test_validate_vault_rejects_non_oa_license_private_paths_secrets_cycles_and_tier_firewall(
    tmp_path: Path,
) -> None:
    # 1. Non-OA license rejection
    vault_lic = tmp_path / "case_license"
    _write_valid_mini_vault(vault_lic)
    bad_lic_md, _ = _build_node_markdown(source_license="Elsevier-Proprietary-All-Rights-Reserved")
    (vault_lic / "vault" / "phys" / "statistical-mechanics" / "RAL-PHYS-01J9X8K2M4.md").write_text(
        bad_lic_md, encoding="utf-8"
    )
    rep_lic = validate_vault(vault_lic)
    assert not rep_lic.ok
    assert any("DISALLOWED_LICENSE" in e for e in rep_lic.errors)

    # 2. Private local path (/Users/, C:\Users\, /home/) and secret leaks
    vault_leak = tmp_path / "case_leak"
    _write_valid_mini_vault(vault_leak)
    leak_md, _ = _build_node_markdown(
        extra_prose="- Local debug path: /Users/rayhan/research/draft.tex and token AIzaSyD1234567890abcdef"
    )
    (vault_leak / "vault" / "phys" / "statistical-mechanics" / "RAL-PHYS-01J9X8K2M4.md").write_text(
        leak_md, encoding="utf-8"
    )
    rep_leak = validate_vault(vault_leak)
    assert not rep_leak.ok
    assert any("PRIVATE_PATH_LEAK" in e for e in rep_leak.errors)
    assert any("SECRET_LEAK" in e for e in rep_leak.errors)
    assert scan_text_for_leaks(r"path C:\Users\rayhan\keys.json", "test")

    # 3. DEPENDS_ON cycle detection (Kahn's algorithm) via full vault check
    vault_cycle = tmp_path / "case_cycle"
    _write_valid_mini_vault(vault_cycle)
    node_b, ore_b = _build_node_markdown(ral_id="RAL-MATH-01J9X8K2M9", ore_bytes=b"second-source-ore")
    node_b_dir = vault_cycle / "vault" / "math" / "analysis"
    node_b_dir.mkdir(parents=True, exist_ok=True)
    (node_b_dir / "RAL-MATH-01J9X8K2M9.md").write_text(node_b, encoding="utf-8")
    with (vault_cycle / "ledger" / "sources" / "2026-11.ndjson").open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "ore_sha256": ore_b,
                    "source_license": "arXiv-non-exclusive-1.0",
                    "ore_uri": build_ore_uri(ore_b, "tar.gz"),
                    "linked_ral_ids": ["RAL-MATH-01J9X8K2M9"],
                }
            )
            + "\n"
        )
    (vault_cycle / "graph" / "edges" / "2026-11.ndjson").write_text(
        json.dumps({"edge_id": "E-1", "src": "RAL-PHYS-01J9X8K2M4", "dst": "RAL-MATH-01J9X8K2M9", "relation": "DEPENDS_ON"})
        + "\n"
        + json.dumps({"edge_id": "E-2", "src": "RAL-MATH-01J9X8K2M9", "dst": "RAL-PHYS-01J9X8K2M4", "relation": "DEPENDS_ON"})
        + "\n",
        encoding="utf-8",
    )
    rep_cycle = validate_vault(vault_cycle)
    assert not rep_cycle.ok
    assert any("DEPENDS_ON_CYCLE" in e for e in rep_cycle.errors)
    assert detect_depends_on_cycles(
        [
            {"src": "A", "dst": "B", "relation": "DEPENDS_ON"},
            {"src": "B", "dst": "C", "relation": "DEPENDS_ON"},
            {"src": "C", "dst": "A", "relation": "DEPENDS_ON"},
        ]
    ) == ["A", "B", "C"]

    # 4. T1 -> T3/T4 dependency firewall
    vault_fw = tmp_path / "case_firewall"
    _write_valid_mini_vault(vault_fw)
    node_t3, ore_t3 = _build_node_markdown(
        ral_id="RAL-MATH-01J9X8K333",
        epistemic_tier="T3_SPECULATIVE",
        ore_bytes=b"preprint-t3-ore",
    )
    t3_dir = vault_fw / "vault" / "math" / "analysis"
    t3_dir.mkdir(parents=True, exist_ok=True)
    (t3_dir / "RAL-MATH-01J9X8K333.md").write_text(node_t3, encoding="utf-8")
    with (vault_fw / "ledger" / "sources" / "2026-11.ndjson").open("a", encoding="utf-8") as fh:
        fh.write(
            json.dumps(
                {
                    "ore_sha256": ore_t3,
                    "source_license": "arXiv-non-exclusive-1.0",
                    "ore_uri": build_ore_uri(ore_t3, "tar.gz"),
                    "linked_ral_ids": ["RAL-MATH-01J9X8K333"],
                }
            )
            + "\n"
        )
    (vault_fw / "graph" / "edges" / "2026-11.ndjson").write_text(
        json.dumps({"edge_id": "E-3", "src": "RAL-PHYS-01J9X8K2M4", "dst": "RAL-MATH-01J9X8K333", "relation": "DEPENDS_ON"})
        + "\n",
        encoding="utf-8",
    )
    rep_fw = validate_vault(vault_fw)
    assert not rep_fw.ok
    assert any("TIER_FIREWALL_VIOLATION" in e for e in rep_fw.errors)
    assert check_tier_firewall(
        [{"src": "N1", "dst": "N4", "relation": "DEPENDS_ON"}],
        {"N1": "T1_CANONICAL", "N4": "T4_FALSIFIED"},
    )


def test_validate_node_file_cli_and_real_repo_vault_pass(tmp_path: Path) -> None:
    _write_valid_mini_vault(tmp_path)
    node_file = tmp_path / "vault" / "phys" / "statistical-mechanics" / "RAL-PHYS-01J9X8K2M4.md"
    assert validate_node_file(node_file, check_filename=True).ok is True

    real_repo_report = validate_vault(REPO_ROOT)
    assert real_repo_report.ok is True, f"Real repo failed validation: {real_repo_report.errors}"


# ---------------------------------------------------------------------------
# 6. Acceptance Criterion #4: Mocked Google Drive 2PC Store & Canary
# ---------------------------------------------------------------------------


class _FakeExecute:
    def __init__(self, fn: Any) -> None:
        self._fn = fn

    def execute(self) -> Any:
        return self._fn()


class _FakeFilesResource:
    def __init__(self, service: FakeDriveService) -> None:
        self._svc = service

    def list(self, **kwargs: Any) -> _FakeExecute:
        q = kwargs.get("q", "")
        name_m = re.search(r"name = '([^']*)'", q)
        parent_m = re.search(r"'([^']*)' in parents", q)
        mime_m = re.search(r"mimeType = '([^']*)'", q)
        target_name = name_m.group(1) if name_m else None
        target_parent = parent_m.group(1) if parent_m else None
        target_mime = mime_m.group(1) if mime_m else None

        def _run() -> dict[str, Any]:
            matches = []
            for item in self._svc.items.values():
                if item.get("trashed"):
                    continue
                if target_name is not None and item["name"] != target_name:
                    continue
                if target_parent is not None and target_parent not in item["parents"]:
                    continue
                if target_mime is not None and item.get("mimeType") != target_mime:
                    continue
                matches.append(
                    {
                        "id": item["id"],
                        "name": item["name"],
                        "md5Checksum": item.get("md5Checksum"),
                        "size": str(len(item.get("data", b""))),
                    }
                )
            return {"files": matches[: kwargs.get("pageSize", 10)]}

        return _FakeExecute(_run)

    def create(self, **kwargs: Any) -> _FakeExecute:
        body = kwargs["body"]
        media_body = kwargs.get("media_body")

        def _run() -> dict[str, Any]:
            self._svc.counter += 1
            fid = f"fake-id-{self._svc.counter}"
            data = b""
            if media_body is not None:
                data = media_body.getbytes() if hasattr(media_body, "getbytes") else media_body.read()
            if self._svc.corrupt_remote_md5_on_create and body.get("mimeType") != DRIVE_FOLDER_MIME:
                md5 = "0" * 32
            else:
                md5 = hashlib.md5(data).hexdigest() if body.get("mimeType") != DRIVE_FOLDER_MIME else None
            self._svc.items[fid] = {
                "id": fid,
                "name": body["name"],
                "parents": list(body.get("parents", [])),
                "mimeType": body.get("mimeType", "application/octet-stream"),
                "data": data,
                "md5Checksum": md5,
                "trashed": False,
            }
            return {"id": fid, "md5Checksum": md5, "size": str(len(data))}

        return _FakeExecute(_run)

    def get(self, **kwargs: Any) -> _FakeExecute:
        fid = kwargs["fileId"]

        def _run() -> dict[str, Any]:
            item = self._svc.items[fid]
            return {
                "id": fid,
                "md5Checksum": item.get("md5Checksum"),
                "size": str(len(item.get("data", b""))),
            }

        return _FakeExecute(_run)

    def get_media(self, **kwargs: Any) -> _FakeExecute:
        fid = kwargs["fileId"]

        def _run() -> bytes:
            data = self._svc.items[fid]["data"]
            if self._svc.corrupt_readback:
                return b"corrupted-bytes!"
            return data

        return _FakeExecute(_run)

    def delete(self, **kwargs: Any) -> _FakeExecute:
        fid = kwargs["fileId"]

        def _run() -> None:
            self._svc.items.pop(fid, None)

        return _FakeExecute(_run)


class FakeDriveService:
    def __init__(self, root_id: str = "root-warehouse-id") -> None:
        self.counter = 0
        self.corrupt_remote_md5_on_create = False
        self.corrupt_readback = False
        self.items: dict[str, dict[str, Any]] = {
            root_id: {
                "id": root_id,
                "name": "RAL_Ore_Warehouse",
                "parents": [],
                "mimeType": DRIVE_FOLDER_MIME,
                "data": b"",
                "trashed": False,
            }
        }

    def files(self) -> _FakeFilesResource:
        return _FakeFilesResource(self)


def test_drive_ore_preflight_canary_and_content_addressed_2pc(tmp_path: Path) -> None:
    svc = FakeDriveService()
    store = DriveOreStore(service=svc, root_folder_id="root-warehouse-id")

    canary = store.run_preflight_canary()
    assert canary.ok is True
    assert canary.canary_size_bytes == 16
    assert canary.local_md5 == canary.remote_md5
    assert canary.cleaned_up is True
    assert list(svc.items.keys()) == ["root-warehouse-id"]

    # Canary detects read-back corruption AND still cleans up temporary file.
    svc.corrupt_readback = True
    bad_canary = store.run_preflight_canary()
    assert bad_canary.ok is False
    assert bad_canary.cleaned_up is True
    assert list(svc.items.keys()) == ["root-warehouse-id"]
    svc.corrupt_readback = False

    # Upload Ore from local file into by-sha256/{h[0:2]}/{h[2:4]}/{sha256}.{ext}
    ore_file = tmp_path / "paper_2104.01234.tar.gz"
    payload = b"%PDF-or-tar-ore-payload-bytes"
    ore_file.write_bytes(payload)
    expected_sha = hashlib.sha256(payload).hexdigest()

    res1 = store.upload_ore_file(ore_file)
    assert res1.ore_sha256 == expected_sha
    assert res1.relative_path == build_ore_relative_path(expected_sha, "tar.gz")
    assert res1.ore_uri == build_ore_uri(expected_sha, "tar.gz")
    assert res1.md5_checksum == hashlib.md5(payload).hexdigest()
    assert res1.deduplicated is False

    # Re-uploading identical Ore deduplicates without creating another remote copy.
    res2 = store.upload_ore_bytes(payload, "tar.gz")
    assert res2.deduplicated is True
    assert res2.file_id == res1.file_id

    # Remote MD5 mismatch triggers OreIntegrityError and deletes corrupted remote file.
    svc.corrupt_remote_md5_on_create = True
    corrupt_payload = b"corrupted-upload-candidate"
    items_before = len(svc.items)
    with pytest.raises(OreIntegrityError, match="Remote MD5 mismatch"):
        store.upload_ore_bytes(corrupt_payload, "pdf")
    # Only shard folders may have been created; the corrupted file itself was deleted.
    assert all(
        item["mimeType"] == DRIVE_FOLDER_MIME or item["id"] == res1.file_id
        for item in svc.items.values()
    )
    assert len(svc.items) >= items_before


def test_drive_env_error_and_mint_token_cli(tmp_path: Path) -> None:
    with pytest.raises(DriveAuthError, match="RAL_GDRIVE_ROOT_FOLDER_ID"):
        build_drive_service_from_env({})

    secrets_file = tmp_path / "client_secret_test.json"
    secrets_file.write_text(
        json.dumps(
            {
                "installed": {
                    "client_id": "12345.apps.googleusercontent.com",
                    "client_secret": "fake-secret-value",
                }
            }
        ),
        encoding="utf-8",
    )
    meta = mint_gdrive_token.load_client_secrets_metadata(secrets_file)
    assert meta["client_id"] == "12345.apps.googleusercontent.com"
    assert DRIVE_FILE_SCOPE == "https://www.googleapis.com/auth/drive.file"

    parsed = mint_gdrive_token.build_parser().parse_args(
        ["--client-secrets", str(secrets_file), "--port", "8080", "--root-folder-id", "folder123"]
    )
    assert parsed.port == 8080
    assert parsed.root_folder_id == "folder123"

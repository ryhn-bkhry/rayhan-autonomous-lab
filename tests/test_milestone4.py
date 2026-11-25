"""Comprehensive Milestone 4 acceptance and regression tests (RAL Blueprint v2.1)."""

from __future__ import annotations

from datetime import date, datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import sqlite3
import subprocess
import sys
from types import SimpleNamespace
from typing import Any

import pytest
import yaml

from ci.pre_push_gate import run_pre_push_gate
from ci.validate_node import split_frontmatter, validate_node_file
from ci.validate_vault import scan_text_for_leaks, validate_vault
from mcp.ral_mcp_server import (
    MCP_TOOLS_SCHEMA,
    RALMCPServer,
    serve_stdio,
    validate_domain_filter,
    validate_ral_id,
)
from ral.canonicalize import (
    compute_assumptions_hash,
    compute_oracle_script_sha256,
    compute_quote_sha256,
    sha256_bytes,
)
from ral.cli import main as cli_main
from ral.github_client import CRITICAL_ALERT_LABEL, GitHubClient, LATEST_POINTER_NAME
from ral.orchestrator import (
    compute_current_night,
    count_consecutive_zero_verified_nights,
    evaluate_dead_mans_switch,
    execute_epoch_pipeline,
    run_consolidate_job,
    run_epoch_job,
    run_full_nightly_pipeline,
    run_preflight_job,
)
from ral.model_router import AllModelsExhaustedError, ModelRouter, build_routing_table
from ral.pipeline.oracle_sandbox import OracleSandbox, SandboxConfig
from ral.sources import SourceRecord
from ral.storage.drive_ore import DRIVE_FOLDER_MIME, DriveOreStore, build_ore_uri
from ral.storage.index_builder import build_or_update_index, compress_sqlite_zst
from ral.vault_docs import estimate_tokens, regenerate_vault_docs

REPO_ROOT = Path(__file__).resolve().parent.parent

VALID_NUMPY_KERNEL = """# RAL_ORACLE: sympy_numeric | TIMEOUT: 15s
import numpy as np

# Verify Parseval / norm preservation & linearity across distinct orthogonal inputs
for scale in (0.5, 2.0, 5.0):
    vec = np.array([scale, -2.0 * scale, 3.0 * scale], dtype=float)
    assert np.isclose(np.linalg.norm(vec) ** 2, 14.0 * (scale ** 2), rtol=1e-9)

mat = np.array([[2.0, 0.0], [0.0, 3.0]])
vals = np.linalg.eigvalsh(mat)
assert np.allclose(vals, np.array([2.0, 3.0]))
assert np.isclose(np.trace(mat), float(np.sum(vals)))
"""

CHEAT_ASSERT_KERNEL = """assert True
assert True
assert True
"""


# ---------------------------------------------------------------------------
# Self-Contained Offline Test Fixtures & Mocks
# ---------------------------------------------------------------------------


def _init_mini_repo(root: Path) -> None:
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


def _write_node_fixture(
    repo_root: Path,
    *,
    ral_id: str,
    taxonomy_code: str,
    title: str,
    node_type: str = "THEOREM",
    epistemic_tier: str = "T1_CANONICAL",
    formal_signature: list[str] | None = None,
    formal_statement: str = r"$$\chi''(\omega) = \frac{\omega}{2 k_B T} S_{AA}(\omega)$$",
    source_id: str = "doi:10.1103/physrev.83.34",
    re_evaluate_after: str = "2028-11-24",
    record_source_ledger: bool = True,
) -> tuple[Path, str]:
    raw_bytes = f"ore:{ral_id}:{source_id}".encode("utf-8")
    ore_sha = sha256_bytes(raw_bytes)
    ore_uri = build_ore_uri(ore_sha, "tar.gz")
    assumptions = [
        "system is in canonical thermal equilibrium with T > 0",
        "perturbation lies strictly within linear-response regime",
    ]
    sig = formal_signature or ["operator:autocorrelation", "regime:linear_response"]
    fm = {
        "ral_schema": "2.1",
        "ral_id": ral_id,
        "node_type": node_type,
        "epistemic_tier": epistemic_tier,
        "consensus_version": 1,
        "taxonomy_code": taxonomy_code,
        "title": title,
        "formal_signature": sig,
        "assumptions": assumptions,
        "assumptions_hash": compute_assumptions_hash(assumptions),
        "verification": {
            "oracle_plugin": "sympy_numeric",
            "oracle_status": "PASS",
            "oracle_script_sha256": compute_oracle_script_sha256(VALID_NUMPY_KERNEL),
            "verified_at": "2026-11-24T20:14:02Z",
            "formalizer_model": "gemini-2.5-pro",
            "skeptic_model": "gemma-4-27b-it",
            "model_capability_epoch": 1,
        },
        "provenance": [
            {
                "source_id": source_id,
                "source_license": "CC-BY-4.0",
                "source_format": "arxiv_tex",
                "source_locator": "tex:sec2:eq1",
                "ore_uri": ore_uri,
                "ore_sha256": ore_sha,
                "quote_sha256": compute_quote_sha256(f"Verbatim quote for {title}"),
                "quote_char_len": 48,
                "pub_year": 2021,
                "peer_reviewed": True,
                "retracted": False,
            }
        ],
        "re_evaluate_after": re_evaluate_after,
    }
    md = (
        f"---\n{yaml.safe_dump(fm, sort_keys=False).strip()}\n---\n\n"
        "## Active Consensus (v1 — Effective 2026-11-24)\n\n"
        "### Formal Statement\n"
        f"{formal_statement}\n\n"
        "| Symbol | Meaning | Domain | SI Units |\n"
        "| :--- | :--- | :--- | :--- |\n"
        "| $x$ | Canonical variable | $\\mathbb{R}$ | $1$ |\n\n"
        "### Failure Boundaries & Known Pathologies\n"
        "- **Breaks when:** Stated assumptions are violated.\n\n"
        "### Executable Verification Kernel\n"
        f"```python\n{VALID_NUMPY_KERNEL.strip()}\n```\n\n"
        "### Canonical Provenance Anchors\n"
        f"- `{source_id}` (`tex:sec2:eq1`): Verified claim anchor.\n\n"
        "## Active Challenges & Open Anomalies\n"
        "- *(None)*\n\n"
        "## Superseded & Archived\n"
        "*(None)*\n"
    )
    parts = taxonomy_code.split(".")
    node_path = repo_root / "vault" / Path(*parts) / f"{ral_id}.md"
    node_path.parent.mkdir(parents=True, exist_ok=True)
    node_path.write_text(md, encoding="utf-8")

    if record_source_ledger:
        src_path = repo_root / "ledger" / "sources" / "2026-11.ndjson"
        src_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ore_sha256": ore_sha,
            "canonical_doi": source_id.removeprefix("doi:") if source_id.startswith("doi:") else None,
            "arxiv_base_id": None,
            "arxiv_version": None,
            "source_license": "CC-BY-4.0",
            "ore_uri": ore_uri,
            "linked_ral_ids": [ral_id],
            "ingested_at": "2026-11-24T20:14:02Z",
        }
        with src_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    return node_path, ore_sha


def _append_edge(repo_root: Path, edge: dict[str, Any]) -> None:
    edges_path = repo_root / "graph" / "edges" / "2026-11.ndjson"
    edges_path.parent.mkdir(parents=True, exist_ok=True)
    with edges_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(edge) + "\n")


class _FakeExecute:
    def __init__(self, fn: Any) -> None:
        self._fn = fn

    def execute(self) -> Any:
        return self._fn()


class _FakeFilesResource:
    def __init__(self, svc: _FakeDriveService) -> None:
        self._svc = svc

    def list(self, **kwargs: Any) -> _FakeExecute:
        q = kwargs.get("q", "")
        name_m = re.search(r"name = '([^']*)'", q)
        parent_m = re.search(r"'([^']*)' in parents", q)
        mime_m = re.search(r"mimeType = '([^']*)'", q)
        mime_ne_m = re.search(r"mimeType != '([^']*)'", q)
        t_name = name_m.group(1) if name_m else None
        t_parent = parent_m.group(1) if parent_m else None
        t_mime = mime_m.group(1) if mime_m else None
        t_not_mime = mime_ne_m.group(1) if mime_ne_m else None

        def _run() -> dict[str, Any]:
            out = []
            for item in self._svc.items.values():
                if t_name is not None and item["name"] != t_name:
                    continue
                if t_parent is not None and t_parent not in item["parents"]:
                    continue
                if t_mime is not None and item.get("mimeType") != t_mime:
                    continue
                if t_not_mime is not None and item.get("mimeType") == t_not_mime:
                    continue
                out.append(
                    {
                        "id": item["id"],
                        "name": item["name"],
                        "mimeType": item.get("mimeType"),
                        "parents": list(item.get("parents", [])),
                        "md5Checksum": item.get("md5Checksum"),
                        "size": str(len(item.get("data", b""))),
                    }
                )
            return {"files": out[: kwargs.get("pageSize", 10)]}

        return _FakeExecute(_run)

    def create(self, **kwargs: Any) -> _FakeExecute:
        body = kwargs["body"]
        media = kwargs.get("media_body")

        def _run() -> dict[str, Any]:
            self._svc.counter += 1
            fid = f"drive-{self._svc.counter}"
            data = (
                media.getbytes(0, media.size())
                if media is not None and hasattr(media, "size") and hasattr(media, "getbytes")
                else (media.getbytes() if media is not None and hasattr(media, "getbytes") else b"")
            )
            md5 = hashlib.md5(data).hexdigest() if body.get("mimeType") != DRIVE_FOLDER_MIME else None
            self._svc.items[fid] = {
                "id": fid,
                "name": body["name"],
                "parents": list(body.get("parents", [])),
                "mimeType": body.get("mimeType", "application/octet-stream"),
                "data": data,
                "md5Checksum": md5,
            }
            return {"id": fid, "md5Checksum": md5, "size": str(len(data))}

        return _FakeExecute(_run)

    def get(self, **kwargs: Any) -> _FakeExecute:
        fid = kwargs["fileId"]
        return _FakeExecute(lambda: {"id": fid, "md5Checksum": self._svc.items[fid].get("md5Checksum")})

    def get_media(self, **kwargs: Any) -> _FakeExecute:
        fid = kwargs["fileId"]
        return _FakeExecute(lambda: self._svc.items[fid]["data"])

    def delete(self, **kwargs: Any) -> _FakeExecute:
        fid = kwargs["fileId"]
        return _FakeExecute(lambda: self._svc.items.pop(fid, None))


class _FakeDriveService:
    def __init__(self, root_id: str = "root-warehouse-id") -> None:
        self.counter = 0
        self.items: dict[str, dict[str, Any]] = {
            root_id: {
                "id": root_id,
                "name": "RAL_Ore_Warehouse",
                "parents": [],
                "mimeType": DRIVE_FOLDER_MIME,
                "data": b"",
            }
        }

    def files(self) -> _FakeFilesResource:
        return _FakeFilesResource(self)


class _FakeResp:
    def __init__(
        self,
        status_code: int,
        *,
        json_data: Any = None,
        content: bytes = b"",
        headers: dict[str, str] | None = None,
        text: str = "",
    ) -> None:
        self.status_code = status_code
        self._json = json_data
        self.content = content if content else (text.encode("utf-8") if text else b"")
        self.headers = headers or {"X-RateLimit-Remaining": "950"}
        self.text = text or (json.dumps(json_data) if json_data is not None else "")

    def json(self) -> Any:
        if self._json is not None:
            return self._json
        return json.loads(self.content.decode("utf-8"))


class _FakeGitHubServer:
    """Stateful offline GitHub Releases & Issues simulator."""

    def __init__(self, owner: str = "rayhan", repo: str = "rayhans-autonomous-lab") -> None:
        self.owner = owner
        self.repo = repo
        self.next_id = 100
        self.clock = 0
        self.releases: dict[str, dict[str, Any]] = {}
        self.assets_by_id: dict[int, dict[str, Any]] = {}
        self.issues: dict[int, dict[str, Any]] = {}
        self.next_issue_num = 1
        self.requests_log: list[tuple[str, str]] = []

    def _serialize_asset(self, a: dict[str, Any]) -> dict[str, Any]:
        return {
            "id": a["id"],
            "name": a["name"],
            "size": len(a["bytes"]),
            "digest": f"sha256:{sha256_bytes(a['bytes'])}",
            "created_at": a["created_at"],
            "browser_download_url": a["browser_download_url"],
            "url": f"https://api.github.com/repos/{self.owner}/{self.repo}/releases/assets/{a['id']}",
        }

    def request(self, method: str, url: str, **kwargs: Any) -> _FakeResp:
        self.requests_log.append((method, url))
        prefix = f"https://api.github.com/repos/{self.owner}/{self.repo}"

        if method == "GET" and url.startswith(f"{prefix}/releases/tags/"):
            tag = url.rsplit("/", 1)[-1]
            rel = self.releases.get(tag)
            if rel is None:
                return _FakeResp(404, json_data={"message": "Not Found"})
            out = dict(rel)
            out["assets"] = [
                self._serialize_asset(self.assets_by_id[aid])
                for aid in rel["asset_ids"]
                if aid in self.assets_by_id
            ]
            return _FakeResp(200, json_data=out)

        if method == "POST" and url == f"{prefix}/releases":
            body = kwargs["json"]
            self.next_id += 1
            rid = self.next_id
            tag = body["tag_name"]
            rel = {
                "id": rid,
                "tag_name": tag,
                "name": body.get("name", tag),
                "upload_url": f"https://uploads.github.com/repos/{self.owner}/{self.repo}/releases/{rid}/assets{{?name,label}}",
                "asset_ids": [],
            }
            self.releases[tag] = rel
            return _FakeResp(201, json_data={**rel, "assets": []})

        m_list = re.match(rf"^{re.escape(prefix)}/releases/(\d+)/assets$", url)
        if method == "GET" and m_list:
            rid = int(m_list.group(1))
            rel = next((r for r in self.releases.values() if r["id"] == rid), None)
            if rel is None:
                return _FakeResp(404, json_data={"message": "Not Found"})
            return _FakeResp(
                200,
                json_data=[
                    self._serialize_asset(self.assets_by_id[aid])
                    for aid in rel["asset_ids"]
                    if aid in self.assets_by_id
                ],
            )

        m_up = re.match(
            rf"^https://uploads\.github\.com/repos/{re.escape(self.owner)}/{re.escape(self.repo)}/releases/(\d+)/assets$",
            url,
        )
        if method == "POST" and m_up:
            rid = int(m_up.group(1))
            name = (kwargs.get("params") or {})["name"]
            payload = bytes(kwargs.get("data") or b"")
            rel = next((r for r in self.releases.values() if r["id"] == rid), None)
            if rel is None:
                return _FakeResp(404, json_data={"message": "Not Found"})
            self.next_id += 1
            self.clock += 1
            aid = self.next_id
            asset = {
                "id": aid,
                "release_id": rid,
                "name": name,
                "bytes": payload,
                "created_at": f"2026-11-24T00:00:{self.clock:02d}Z",
                "browser_download_url": f"https://github.com/{self.owner}/{self.repo}/releases/download/{rel['tag_name']}/{name}?id={aid}",
            }
            self.assets_by_id[aid] = asset
            rel["asset_ids"].append(aid)
            return _FakeResp(201, json_data=self._serialize_asset(asset))

        if method == "GET" and "/releases/download/" in url:
            aid_m = re.search(r"id=(\d+)", url)
            if aid_m and int(aid_m.group(1)) in self.assets_by_id:
                return _FakeResp(200, content=self.assets_by_id[int(aid_m.group(1))]["bytes"])
            return _FakeResp(404)

        m_del = re.match(rf"^{re.escape(prefix)}/releases/assets/(\d+)$", url)
        if method == "DELETE" and m_del:
            aid = int(m_del.group(1))
            if aid in self.assets_by_id:
                del self.assets_by_id[aid]
                for rel in self.releases.values():
                    if aid in rel["asset_ids"]:
                        rel["asset_ids"].remove(aid)
                return _FakeResp(204)
            return _FakeResp(404)

        if method == "GET" and url == f"{prefix}/issues":
            label = (kwargs.get("params") or {}).get("labels")
            state = (kwargs.get("params") or {}).get("state", "open")
            matched = [
                iss
                for iss in self.issues.values()
                if iss["state"] == state and (not label or label in iss["labels"])
            ]
            return _FakeResp(200, json_data=matched)

        if method == "POST" and url == f"{prefix}/issues":
            body = kwargs["json"]
            num = self.next_issue_num
            self.next_issue_num += 1
            issue = {
                "number": num,
                "title": body["title"],
                "body": body["body"],
                "labels": list(body.get("labels", [])),
                "state": "open",
            }
            self.issues[num] = issue
            return _FakeResp(201, json_data=issue)

        m_patch = re.match(rf"^{re.escape(prefix)}/issues/(\d+)$", url)
        if method == "PATCH" and m_patch:
            num = int(m_patch.group(1))
            body = kwargs["json"]
            issue = self.issues[num]
            issue["title"] = body.get("title", issue["title"])
            issue["body"] = body.get("body", issue["body"])
            return _FakeResp(200, json_data=issue)

        return _FakeResp(404, json_data={"message": f"Unhandled {method} {url}"})


def _make_fake_gemini_client(*, fail_429_in_stage: bool = False) -> Any:
    suite = json.loads((REPO_ROOT / "config" / "calibration_suite.json").read_text(encoding="utf-8"))
    by_prompt: dict[str, str] = {}
    for p in suite["problems"]:
        cat = p["category"]
        prompt = p["prompt"]
        if cat == "strict_json":
            prompt = f"{prompt}\n\nJSON Schema:\n{json.dumps(p['json_schema'])}"
            by_prompt[prompt] = json.dumps(p["reference_output"])
        elif cat == "sympy_math":
            by_prompt[prompt] = json.dumps({"expr": p["expected_expr"]})
        elif cat == "counterexample_trap":
            by_prompt[prompt] = json.dumps(
                {
                    "is_valid": False,
                    "counterexample_witness": p["reference_counterexample"],
                    "missing_hypothesis": " ".join(p["required_missing_assumption_keywords"]),
                }
            )

    models = [
        SimpleNamespace(name="models/gemini-4-argon", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-2.5-pro", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemma-4-27b-it", supported_actions=["generateContent"]),
        SimpleNamespace(name="models/gemini-embedding-2", supported_actions=["embedContent"]),
    ]

    class _RateLimitError(RuntimeError):
        def __init__(self) -> None:
            super().__init__(
                "429 RESOURCE_EXHAUSTED: GenerateRequestsPerDayPerProjectPerModel daily RPD limit exceeded"
            )
            self.status_code = 429

    def _generate_content(*, model: str, contents: str, config: dict[str, Any]) -> Any:
        _ = config
        if contents == 'Reply with exact JSON: {"ok":true}':
            return SimpleNamespace(text='{"ok":true}')
        if contents in by_prompt:
            if model == "gemma-4-27b-it" and contents == suite["problems"][7]["prompt"]:
                return SimpleNamespace(text=json.dumps({"expr": "0"}))
            return SimpleNamespace(text=by_prompt[contents])

        if fail_429_in_stage:
            raise _RateLimitError()

        if "zero-tool primary-source scientific claim extractor" in contents:
            if "BAD_SOURCE" in contents:
                return SimpleNamespace(
                    text=json.dumps(
                        {
                            "candidates": [
                                {
                                    "title": "Rejected Cheat Claim",
                                    "claim_summary": "Tautological claim that fails AST lint.",
                                    "quote_text": "Every tautology fails static AST verification.",
                                    "source_locator": "tex:sec1:eq1",
                                    "node_type": "THEOREM",
                                    "taxonomy_code": "math.analysis",
                                    "formal_signature": ["operator:cheat"],
                                    "assumptions": ["x is real"],
                                    "recommended_oracle": "sympy_numeric",
                                }
                            ]
                        }
                    )
                )
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "candidates": [
                            {
                                "title": "Parseval Spectral Norm Identity",
                                "claim_summary": "Euclidean norm scales quadratically with positive scalar factor.",
                                "quote_text": "Euclidean norm squared scales quadratically and matrix trace equals eigenvalue sum.",
                                "source_locator": "tex:sec2:eq4",
                                "node_type": "THEOREM",
                                "taxonomy_code": "math.analysis",
                                "formal_signature": ["operator:norm", "operator:trace", "operator:eigenvalues"],
                                "assumptions": ["x is finite-dimensional real vector", "A is real symmetric matrix"],
                                "recommended_oracle": "sympy_numeric",
                            }
                        ]
                    }
                )
            )

        if "Formalize this verified scientific quote" in contents:
            if "Rejected Cheat Claim" in contents:
                return SimpleNamespace(
                    text=json.dumps(
                        {
                            "node_type": "THEOREM",
                            "taxonomy_code": "math.analysis",
                            "title": "Rejected Cheat Claim",
                            "formal_signature": ["operator:cheat"],
                            "assumptions": ["x is real"],
                            "formal_statement_latex": "x = x",
                            "symbol_rows": [{"symbol": "x", "meaning": "scalar", "domain": "\\mathbb{R}", "si_units": "1"}],
                            "failure_boundaries": ["fails AST"],
                            "provenance_summary": "Bad kernel.",
                            "oracle_plugin": "sympy_numeric",
                            "verify_py": CHEAT_ASSERT_KERNEL,
                        }
                    )
                )
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "node_type": "THEOREM",
                        "taxonomy_code": "math.analysis",
                        "title": "Parseval Spectral Norm Identity",
                        "formal_signature": ["operator:norm", "operator:trace", "operator:eigenvalues"],
                        "assumptions": ["x is finite-dimensional real vector", "A is real symmetric matrix"],
                        "formal_statement_latex": "\\|c v\\|_2^2 = c^2 \\|v\\|_2^2 \\land \\operatorname{tr}(A) = \\sum_i \\lambda_i(A)",
                        "symbol_rows": [
                            {"symbol": "v", "meaning": "real vector", "domain": "\\mathbb{R}^n", "si_units": "1"},
                            {"symbol": "A", "meaning": "symmetric matrix", "domain": "\\mathbb{R}^{n\\times n}", "si_units": "1"},
                        ],
                        "failure_boundaries": ["non-Hermitian matrices with non-diagonalizable blocks"],
                        "provenance_summary": "Verified Euclidean norm scaling and symmetric trace identity.",
                        "oracle_plugin": "sympy_numeric",
                        "verify_py": VALID_NUMPY_KERNEL,
                    }
                )
            )

        if "Repair the executable verification script" in contents:
            return SimpleNamespace(
                text=json.dumps({"verify_py": CHEAT_ASSERT_KERNEL, "assumptions": ["x is real"]})
            )

        if "Assume this formalization has a hidden defect" in contents:
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "attack_found": False,
                        "falsify_py": "",
                        "missing_assumptions": [],
                        "summary": "No counterexample found under real symmetric assumptions.",
                    }
                )
            )

        raise AssertionError(f"Unexpected prompt in FakeGenAIClient: {contents[:120]}")

    def _embed_content(*, model: str, contents: str, config: dict[str, Any]) -> Any:
        assert isinstance(contents, str)
        dim = int(config.get("output_dimensionality", 768))
        vec = [0.0] * dim
        vec[0] = 1.0
        if "eigenvalues" in contents:
            vec[1] = 0.5
        return SimpleNamespace(embeddings=[SimpleNamespace(values=vec)])

    models_ns = SimpleNamespace(
        list=lambda: list(models),
        generate_content=_generate_content,
        embed_content=_embed_content,
    )
    return SimpleNamespace(models=models_ns)


def _make_local_exec_sandbox() -> OracleSandbox:
    """Sandbox runner checking air-gapped Docker CLI flags & env scrubbing while running local numpy scripts."""

    def _runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert "--network=none" in cmd
        assert "--read-only" in cmd
        assert "--cap-drop=ALL" in cmd
        assert "--security-opt=no-new-privileges" in cmd
        assert "--user=65534:65534" in cmd
        env = kwargs.get("env") or {}
        assert "RAL_GEMINI_FREE_KEY" not in env
        assert "GITHUB_TOKEN" not in env
        assert "RAL_GDRIVE_REFRESH_TOKEN" not in env

        vol_idx = cmd.index("-v") + 1
        host_dir = cmd[vol_idx].rsplit(":/work:ro", 1)[0]
        script_name = cmd[-1].rsplit("/", 1)[-1]
        script_path = Path(host_dir) / script_name
        return subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=kwargs.get("timeout", 20),
            env=env,
            check=False,
        )

    return OracleSandbox(SandboxConfig(), runner=_runner)


# ---------------------------------------------------------------------------
# 1. Workflow YAML Schema, Secrets, Docker Image & Artifact Retention
# ---------------------------------------------------------------------------


def test_workflow_yaml_schema_secrets_jobs_docker_and_retention() -> None:
    wf_path = REPO_ROOT / ".github" / "workflows" / "ral_nightly.yml"
    assert wf_path.is_file()
    raw_text = wf_path.read_text(encoding="utf-8")
    data = yaml.safe_load(raw_text)
    assert isinstance(data, dict)

    # PyYAML parses top-level `on:` as boolean `True` under YAML 1.1.
    triggers = data.get("on") or data.get(True)
    assert isinstance(triggers, dict)
    assert triggers["schedule"] == [{"cron": "13 18 * * *"}]
    wd_inputs = triggers["workflow_dispatch"]["inputs"]
    for key in ("dry_run", "available_epochs", "simulate_drive_fail", "run_date"):
        assert key in wd_inputs

    assert data["concurrency"] == {
        "group": "ral-nightly-pipeline",
        "cancel-in-progress": False,
    }
    assert data["permissions"]["contents"] == "write"
    assert data["permissions"]["issues"] == "write"

    expected_jobs = [
        "preflight_and_plan",
        "epoch_1",
        "epoch_2",
        "epoch_3",
        "epoch_4",
        "epoch_5",
        "epoch_6",
        "consolidate_and_watchdog",
    ]
    jobs = data["jobs"]
    assert list(jobs.keys()) == expected_jobs

    assert jobs["preflight_and_plan"]["timeout-minutes"] == 25
    assert jobs["consolidate_and_watchdog"]["timeout-minutes"] == 15
    assert jobs["consolidate_and_watchdog"]["needs"] == ["preflight_and_plan", "epoch_6"]
    assert jobs["consolidate_and_watchdog"]["if"] == "always() && !cancelled()"
    assert ".ral_handoff" not in raw_text
    assert "ral_handoff" in raw_text

    prev_job = "preflight_and_plan"
    for idx in range(1, 7):
        jname = f"epoch_{idx}"
        job = jobs[jname]
        assert job["runs-on"] == "ubuntu-latest"
        assert job["timeout-minutes"] == 55
        assert job["needs"] == [prev_job]
        assert job["if"] == "always() && !cancelled()"
        prev_job = jname

    # All 5 required repository secrets wired cleanly and never passed to Docker commands.
    for secret_name in (
        "RAL_GEMINI_FREE_KEY",
        "RAL_GDRIVE_CLIENT_ID",
        "RAL_GDRIVE_CLIENT_SECRET",
        "RAL_GDRIVE_REFRESH_TOKEN",
        "RAL_GDRIVE_ROOT_FOLDER_ID",
    ):
        assert f"${{{{ secrets.{secret_name} }}}}" in raw_text

    # Docker build & air-gapped verification + artifact retention-days: 1 + exact 20 artifact ops.
    upload_count = 0
    download_count = 0
    docker_build_found = False
    airgapped_smoke_found = False
    for job_name, job in jobs.items():
        assert job["runs-on"] == "ubuntu-latest"
        upload_indices: list[int] = []
        push_indices: list[int] = []
        job_downloads: list[dict[str, Any]] = []
        for idx, step in enumerate(job["steps"]):
            uses = str(step.get("uses") or "")
            if uses.startswith("actions/upload-artifact@v4"):
                upload_count += 1
                upload_indices.append(idx)
                assert step["with"]["retention-days"] == 1
                assert step["with"]["include-hidden-files"] is True
                assert step["with"]["if-no-files-found"] == "error"
                assert step["with"]["path"] == "ral_handoff/"
            elif uses.startswith("actions/download-artifact@v4"):
                download_count += 1
                job_downloads.append(step)
            elif uses.startswith("docker/build-push-action@v6"):
                docker_build_found = True
                assert step["with"]["file"] == "ci/sandbox.Dockerfile"
                assert step["with"]["tags"] == "ral-sandbox:latest"
                assert step["with"]["load"] is True
                assert "env" not in step  # Zero secrets passed to Docker build

            run_cmd = str(step.get("run") or "")
            assert "git add metrics/model_benchmarks.json vault/" not in run_cmd
            assert "git add vault/ ledger/ graph/ metrics/" not in run_cmd
            assert "git add INDEX.md llms.txt metrics/telemetry/" not in run_cmd
            if "docker run --rm" in run_cmd:
                airgapped_smoke_found = True
                assert "--network=none" in run_cmd
                assert "--read-only" in run_cmd
                assert "--cap-drop=ALL" in run_cmd
                assert "--security-opt=no-new-privileges" in run_cmd
                assert "--user=65534:65534" in run_cmd
                assert "env" not in step
            if "ral.cli run-epoch" in run_cmd:
                guard_cmd = "ls -la ral_handoff && test -s ral_handoff/plan.json"
                assert guard_cmd in run_cmd
                assert run_cmd.index(guard_cmd) < run_cmd.index("python -m ral.cli run-epoch")
            if "ral.cli consolidate" in run_cmd:
                assert step["env"]["RAL_GEMINI_FREE_KEY"] == "${{ secrets.RAL_GEMINI_FREE_KEY }}"
            if "git push origin main" in run_cmd:
                push_indices.append(idx)
                assert "ci/validate_vault.py --root ." in run_cmd
                assert "ci/pre_push_gate.py" in run_cmd
                assert run_cmd.index("ci/validate_vault.py") < run_cmd.index("git push origin main")
                assert run_cmd.index("ci/pre_push_gate.py") < run_cmd.index("git push origin main")

        if job_name == "epoch_1":
            assert len(job_downloads) == 1
            assert job_downloads[0]["with"]["name"] == "ral-handoff-0"
            assert job_downloads[0]["with"]["path"] == "ral_handoff"
            assert "continue-on-error" not in job_downloads[0]
        elif job_name.startswith("epoch_"):
            epoch_k = int(job_name.split("_")[1])
            assert len(job_downloads) == 2
            assert job_downloads[0]["with"]["name"] == "ral-handoff-0"
            assert job_downloads[0]["with"]["path"] == "ral_handoff"
            assert "continue-on-error" not in job_downloads[0]
            assert job_downloads[1]["with"]["name"] == f"ral-handoff-{epoch_k - 1}"
            assert job_downloads[1]["with"]["path"] == "ral_handoff"
            assert job_downloads[1]["continue-on-error"] is True
        elif job_name == "consolidate_and_watchdog":
            assert len(job_downloads) == 2
            assert job_downloads[0]["with"]["name"] == "ral-handoff-0"
            assert job_downloads[0]["with"]["path"] == "ral_handoff"
            assert "continue-on-error" not in job_downloads[0]
            assert job_downloads[1]["with"]["name"] == "ral-handoff-6"
            assert job_downloads[1]["with"]["path"] == "ral_handoff"
            assert job_downloads[1]["continue-on-error"] is True

        if job_name == "preflight_and_plan" or job_name.startswith("epoch_"):
            assert len(upload_indices) == 1
            assert len(push_indices) == 1
            assert upload_indices[0] < push_indices[0]

    assert docker_build_found is True
    assert airgapped_smoke_found is True
    assert upload_count == 7
    assert download_count == 13
    assert upload_count + download_count == 20
    assert "--clobber" not in raw_text


def _git_commit_count(repo: Path) -> int:
    proc = subprocess.run(
        ["git", "-C", str(repo), "rev-list", "--count", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return int(proc.stdout.strip())


def _git_head_files(repo: Path) -> set[str]:
    proc = subprocess.run(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", "HEAD"],
        capture_output=True,
        text=True,
        check=True,
    )
    return {line.strip() for line in proc.stdout.splitlines() if line.strip()}


def test_cold_start_missing_vault_job0_handoff_and_pre_push_gate(tmp_path: Path) -> None:
    repo = tmp_path / "repo"

    subprocess.run(["git", "init", "-b", "main", str(repo)], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.name", "ral-test"], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "config", "user.email", "ral-test@example.invalid"], check=True, capture_output=True)

    _init_mini_repo(repo)
    for fname in (".gitignore", "INDEX.md", "llms.txt"):
        (repo / fname).write_text((REPO_ROOT / fname).read_text(encoding="utf-8"), encoding="utf-8")

    subprocess.run(["git", "-C", str(repo), "add", "."], check=True, capture_output=True)
    subprocess.run(["git", "-C", str(repo), "commit", "-m", "chore: initial cold-start repo"], check=True, capture_output=True)

    for optional_dir in ("vault", "ledger", "graph", "metrics"):
        assert not (repo / optional_dir).exists()

    fake_drive_svc = _FakeDriveService()
    drive_store = DriveOreStore(fake_drive_svc, "root-warehouse-id")
    fake_gh_server = _FakeGitHubServer()
    gh_client = GitHubClient(
        "rayhan",
        "rayhans-autonomous-lab",
        token="test-token",
        request_fn=fake_gh_server.request,
    )
    gemini_client = _make_fake_gemini_client()

    handoff_dir = repo / "ral_handoff"
    cache_dir = repo / ".ral_cache"
    pre = run_preflight_job(
        repo,
        handoff_dir=handoff_dir,
        cache_dir=cache_dir,
        run_date_iso="2026-11-24",
        current_night=1,
        available_epochs=6,
        run_id="coldstart1",
        dry_run=False,
        gemini_client=gemini_client,
        drive_store=drive_store,
        github_client=gh_client,
        now=datetime(2026, 11, 24, 18, 20, 0, tzinfo=timezone.utc),
    )

    assert pre.ok is True
    assert pre.circuit_breaker is None
    assert pre.available_epochs == 6
    plan_file = handoff_dir / "plan.json"
    assert plan_file.is_file() and plan_file.stat().st_size > 0
    plan_payload = json.loads(plan_file.read_text(encoding="utf-8"))
    assert len(plan_payload["epoch_schedule"]) == 6
    assert (handoff_dir / "router_state.json").is_file()
    assert (handoff_dir / "circuit_breaker.json").is_file()
    assert (repo / "metrics" / "model_benchmarks.json").is_file()
    assert not (repo / "vault").exists()

    # 1. Job 0 gate via real CLI subprocess MUST succeed (no exit 128) when vault/ does not exist yet.
    commits_before = _git_commit_count(repo)
    gate_script = str(REPO_ROOT / "ci" / "pre_push_gate.py")
    proc_job0 = subprocess.run(
        [
            sys.executable,
            gate_script,
            "--root",
            str(repo),
            "--no-push",
            "--message",
            "chore(ral): job0 cold start",
            "--paths",
            "metrics/model_benchmarks.json",
            "vault/",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_job0.returncode == 0, f"stderr:\n{proc_job0.stderr}\nstdout:\n{proc_job0.stdout}"
    job0_out = json.loads(proc_job0.stdout)
    assert job0_out["ok"] is True
    assert job0_out["committed"] is True
    assert job0_out["staged_paths"] == ["metrics/model_benchmarks.json"]
    assert not (repo / "vault").exists()
    assert _git_commit_count(repo) == commits_before + 1
    assert "metrics/model_benchmarks.json" in _git_head_files(repo)

    # 2. Job 0 gate idempotency when clean tree and vault/ still missing.
    res_job0_noop = run_pre_push_gate(
        repo,
        paths=["metrics/model_benchmarks.json", "vault/"],
        commit_message="chore(ral): job0 noop",
        push=False,
    )
    assert res_job0_noop.ok is True
    assert res_job0_noop.committed is False
    assert _git_commit_count(repo) == commits_before + 1

    # 3. Epoch 1 consumes ral_handoff/plan.json without FileNotFoundError.
    ep1 = run_epoch_job(
        repo,
        handoff_dir=handoff_dir,
        epoch_index=1,
        cache_dir=cache_dir,
        ore_cache_dir=tmp_path / "ore_cache",
        dry_run=False,
        gemini_client=gemini_client,
        drive_store=drive_store,
        sandbox=_make_local_exec_sandbox(),
        source_fetcher=lambda _m, _t, _r: [],
        now=datetime(2026, 11, 24, 18, 25, 0, tzinfo=timezone.utc),
    )
    assert ep1.summary_path.is_file()

    # 3a. Epoch 1 gate with vault/, ledger/, graph/ all missing and metrics/ clean.
    res_ep1_empty = run_pre_push_gate(
        repo,
        paths=["vault/", "ledger/", "graph/", "metrics/"],
        commit_message="feat(ral): epoch 1 empty",
        push=False,
    )
    assert res_ep1_empty.ok is True
    assert res_ep1_empty.committed is False
    assert _git_commit_count(repo) == commits_before + 1

    # 3b. Epoch 1 gate with only ledger/ created (vault/ and graph/ still missing).
    rej_path = repo / "ledger" / "rejected" / "2026-11.ndjson"
    rej_path.parent.mkdir(parents=True, exist_ok=True)
    rej_path.write_text(
        json.dumps({"source_sha256": "0" * 64, "stage_failed": "STAGE3_ORACLE", "reason": "TEST"}) + "\n",
        encoding="utf-8",
    )
    res_ep1_ledger_only = run_pre_push_gate(
        repo,
        paths=["vault/", "ledger/", "graph/", "metrics/"],
        commit_message="feat(ral): epoch 1 rejected ledger",
        push=False,
    )
    assert res_ep1_ledger_only.ok is True
    assert res_ep1_ledger_only.committed is True
    assert res_ep1_ledger_only.staged_paths == ("ledger/", "metrics/")
    assert _git_commit_count(repo) == commits_before + 2
    assert "ledger/rejected/2026-11.ndjson" in _git_head_files(repo)
    assert not (repo / "vault").exists()
    assert not (repo / "graph").exists()

    # 3c. Epoch 1 gate when first valid node in vault/ and ledger/sources/ are created, while graph/ STILL missing.
    node_path, _ = _write_node_fixture(
        repo,
        ral_id="RAL-MATH-0000000001",
        taxonomy_code="math.analysis",
        title="Cold Start Verified Spectral Norm Identity",
        record_source_ledger=True,
    )
    assert not (repo / "graph").exists()
    res_ep1_first_node = run_pre_push_gate(
        repo,
        paths=["vault/", "ledger/", "graph/", "metrics/"],
        commit_message="feat(ral): epoch 1 first node",
        push=False,
    )
    assert res_ep1_first_node.ok is True, res_ep1_first_node.errors
    assert res_ep1_first_node.committed is True
    rel_node = node_path.relative_to(repo).as_posix()
    assert rel_node in res_ep1_first_node.validated_nodes
    assert _git_commit_count(repo) == commits_before + 3
    head_files = _git_head_files(repo)
    assert rel_node in head_files
    assert "ledger/sources/2026-11.ndjson" in head_files
    assert not (repo / "graph").exists()

    # 3d. Tampered node in vault/ MUST block validation & commit.
    node_path.write_text("---\nral_schema: '999'\n---\nbroken\n", encoding="utf-8")
    res_ep1_tampered = run_pre_push_gate(
        repo,
        paths=["vault/", "ledger/", "graph/", "metrics/"],
        commit_message="feat(ral): should block",
        push=False,
    )
    assert res_ep1_tampered.ok is False
    assert res_ep1_tampered.committed is False
    assert res_ep1_tampered.vault_ok is False
    assert _git_commit_count(repo) == commits_before + 3
    subprocess.run(["git", "-C", str(repo), "checkout", "--", rel_node], check=True, capture_output=True)

    # 4. Job 7 consolidate + gate via real CLI commits INDEX.md, llms.txt, and telemetry.
    con = run_consolidate_job(
        repo,
        handoff_dir=handoff_dir,
        cache_dir=cache_dir,
        run_date_iso="2026-11-24",
        run_id="coldstart1",
        dry_run=False,
        github_client=gh_client,
        embed_client=gemini_client,
        now=datetime(2026, 11, 24, 18, 30, 0, tzinfo=timezone.utc),
    )
    assert con.ok is True
    proc_job7 = subprocess.run(
        [
            sys.executable,
            gate_script,
            "--root",
            str(repo),
            "--no-push",
            "--always-validate-vault",
            "--message",
            "chore(ral): job7 consolidate",
            "--paths",
            "INDEX.md",
            "llms.txt",
            "metrics/telemetry/",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc_job7.returncode == 0, f"stderr:\n{proc_job7.stderr}\nstdout:\n{proc_job7.stdout}"
    job7_out = json.loads(proc_job7.stdout)
    assert job7_out["ok"] is True
    assert job7_out["committed"] is True
    assert _git_commit_count(repo) == commits_before + 4
    assert "metrics/telemetry/2026-11.ndjson" in _git_head_files(repo)

    # 5. Mocked push=True ordering check: commit -> git pull --rebase origin main -> git push origin main.
    llms_path = repo / "llms.txt"
    llms_path.write_text(llms_path.read_text(encoding="utf-8") + "\n<!-- cold-start-push-check -->\n", encoding="utf-8")
    recorded_cmds: list[list[str]] = []

    def _spy_run(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        recorded_cmds.append(list(cmd))
        if len(cmd) >= 4 and cmd[0] == "git" and cmd[3] in ("pull", "push"):
            return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")
        return subprocess.run(cmd, **kwargs)

    res_push = run_pre_push_gate(
        repo,
        paths=["INDEX.md", "llms.txt", "metrics/telemetry/"],
        commit_message="chore(ral): verify push ordering",
        push=True,
        run_cmd=_spy_run,
    )
    assert res_push.ok is True
    assert res_push.committed is True
    assert res_push.pushed is True
    assert _git_commit_count(repo) == commits_before + 5
    git_subcmds = [c[3] if c[3] != "-c" else c[5] for c in recorded_cmds if c[0] == "git"]
    assert "commit" in git_subcmds
    assert "pull" in git_subcmds
    assert "push" in git_subcmds
    assert git_subcmds.index("commit") < git_subcmds.index("pull") < git_subcmds.index("push")
    pull_cmd = next(c for c in recorded_cmds if len(c) >= 4 and c[3] == "pull")
    push_cmd = next(c for c in recorded_cmds if len(c) >= 4 and c[3] == "push")
    assert pull_cmd[3:] == ["pull", "--rebase", "origin", "main"]
    assert push_cmd[3:] == ["push", "origin", "main"]


# ---------------------------------------------------------------------------
# 2. Full 8-Job Mocked Dry-Run Pipeline (< 40 GitHub API Calls & 2PC Ore)
# ---------------------------------------------------------------------------


def test_full_8_job_mocked_nightly_dry_run_under_40_api_calls(tmp_path: Path) -> None:
    _init_mini_repo(tmp_path)
    handoff_dir = tmp_path / "ral_handoff"
    cache_dir = tmp_path / ".ral_cache"
    ore_cache_dir = tmp_path / "tmp_ore_cache"

    fake_drive_svc = _FakeDriveService()
    drive_store = DriveOreStore(fake_drive_svc, "root-warehouse-id")
    fake_gh_server = _FakeGitHubServer()
    gh_client = GitHubClient(
        "rayhan",
        "rayhans-autonomous-lab",
        token="test-token",
        request_fn=fake_gh_server.request,
        max_calls_budget=40,
    )
    gemini_client = _make_fake_gemini_client()
    sandbox = _make_local_exec_sandbox()

    good_ore = b"%PDF-or-tex-good-source-payload"
    bad_ore = b"%tex-bad-cheat-source-payload"
    bad_sha = sha256_bytes(bad_ore)

    def _source_fetcher(mode: str, target: Any, repo_root: Path) -> list[SourceRecord]:
        _ = (mode, target, repo_root)
        # Only yield candidates in Epoch 1; return empty in Epochs 2..6 (or dedup protects).
        if not (tmp_path / "vault").exists():
            return [
                SourceRecord(
                    source_id="arxiv:2105.00001v1",
                    source_license="CC-BY-4.0",
                    source_format="arxiv_tex",
                    title="Bad Cheat Paper",
                    pub_year=2022,
                    peer_reviewed=True,
                    retracted=False,
                    canonical_doi="10.1000/bad.cheat",
                    arxiv_base_id="2105.00001",
                    arxiv_version=1,
                    citation_count=10,
                    raw_bytes=bad_ore,
                    raw_ext="tar.gz",
                    extracted_text="BAD_SOURCE: Every tautology fails static AST verification.",
                ),
                SourceRecord(
                    source_id="arxiv:2105.00002v2",
                    source_license="CC-BY-4.0",
                    source_format="arxiv_tex",
                    title="Parseval Spectral Norm Identity",
                    pub_year=2021,
                    peer_reviewed=True,
                    retracted=False,
                    canonical_doi="10.1000/spectral.norm",
                    arxiv_base_id="2105.00002",
                    arxiv_version=2,
                    citation_count=95,
                    raw_bytes=good_ore,
                    raw_ext="tar.gz",
                    extracted_text=(
                        "GOOD_SOURCE: Euclidean norm squared scales quadratically "
                        "and matrix trace equals eigenvalue sum."
                    ),
                ),
            ]
        return []

    summary = run_full_nightly_pipeline(
        tmp_path,
        handoff_dir=handoff_dir,
        cache_dir=cache_dir,
        ore_cache_dir=ore_cache_dir,
        run_date_iso="2026-11-24",
        current_night=1,
        available_epochs=6,
        run_id="dryrun42",
        dry_run=True,
        allow_release_in_dry_run=True,
        gemini_client=gemini_client,
        drive_store=drive_store,
        github_client=gh_client,
        sandbox=sandbox,
        source_fetcher=_source_fetcher,
        now=datetime(2026, 11, 24, 18, 20, 0, tzinfo=timezone.utc),
    )

    assert summary.preflight.ok is True
    assert summary.preflight.circuit_breaker is None
    assert summary.preflight.available_epochs == 6
    assert (handoff_dir / "plan.json").is_file()
    assert (handoff_dir / "router_state.json").is_file()
    assert (handoff_dir / "circuit_breaker.json").is_file()
    for k in range(1, 7):
        assert (handoff_dir / f"epoch_{k}_summary.json").is_file()

    # Epoch 1 verified 1 node and rejected 1 AST-cheating candidate; staged /tmp/ore_cache cleaned up.
    assert summary.epochs[0].nodes_verified_tonight == 1
    assert summary.epochs[0].rejected_count == 1
    assert summary.epochs[0].ore_uploaded_count == 1
    assert list(ore_cache_dir.iterdir()) == []

    # Verify-Before-Archive 2PC invariant: bad_ore MUST NOT exist in Google Drive!
    stored_hashes = {obj["ore_sha256"] for obj in drive_store.list_ore_objects()}
    assert sha256_bytes(good_ore) in stored_hashes
    assert bad_sha not in stored_hashes

    # Consolidated Job 7 checks: vault valid, Blue-Green swapped, docs & telemetry updated, <40 GitHub API calls.
    assert summary.consolidate.ok is True
    assert summary.consolidate.vault_ok is True
    assert summary.consolidate.node_count == 1
    assert summary.consolidate.nodes_verified_tonight == 1
    assert summary.consolidate.embed_space_consistent is True
    assert summary.consolidate.blue_green_swap is not None
    assert summary.consolidate.blue_green_swap.snapshot_name == "ral-index-20261124-dryrun42.sqlite.zst"
    assert summary.consolidate.dead_mans_switch.triggered is False
    assert len(fake_gh_server.issues) == 0

    assert (tmp_path / "INDEX.md").is_file()
    assert (tmp_path / "llms.txt").is_file()
    assert estimate_tokens((tmp_path / "INDEX.md").read_text(encoding="utf-8")) < 15000
    assert summary.consolidate.telemetry_path.is_file()
    assert summary.estimated_total_github_api_calls < 40
    assert validate_vault(tmp_path).ok is True


# ---------------------------------------------------------------------------
# 3. Dead-Man's Switch: Drive Canary Fail, .git > 700 MB, 3 Zero Nights
# ---------------------------------------------------------------------------


def test_dead_mans_switch_drive_fail_git_size_and_three_zero_nights(tmp_path: Path) -> None:
    _init_mini_repo(tmp_path)
    fake_gh_server = _FakeGitHubServer()
    gh_client = GitHubClient(
        "rayhan",
        "rayhans-autonomous-lab",
        token="test-token",
        request_fn=fake_gh_server.request,
    )
    drive_store = DriveOreStore(_FakeDriveService(), "root-warehouse-id")

    # (a) Simulated Drive canary failure trips circuit_breaker="DRIVE_AUTH_FAIL", K=0, zero Drive/LLM calls.
    llm_called = {"count": 0}

    def _boom_if_called(*args: Any, **kwargs: Any) -> Any:
        llm_called["count"] += 1
        raise AssertionError("LLM must not be called when DRIVE_AUTH_FAIL is tripped")

    bad_gemini = SimpleNamespace(
        models=SimpleNamespace(
            list=lambda: [],
            generate_content=_boom_if_called,
            embed_content=_boom_if_called,
        )
    )

    summary = run_full_nightly_pipeline(
        tmp_path,
        handoff_dir=tmp_path / "ral_handoff",
        cache_dir=tmp_path / ".ral_cache",
        ore_cache_dir=tmp_path / "ore_cache",
        run_date_iso="2026-11-24",
        available_epochs=6,
        run_id="drivefail1",
        dry_run=True,
        simulate_drive_fail=True,
        gemini_client=bad_gemini,
        drive_store=drive_store,
        github_client=gh_client,
        sandbox=_make_local_exec_sandbox(),
        now=datetime(2026, 11, 24, 18, 25, 0, tzinfo=timezone.utc),
    )

    assert llm_called["count"] == 0
    assert summary.preflight.circuit_breaker == "DRIVE_AUTH_FAIL"
    assert summary.preflight.available_epochs == 0
    assert summary.preflight.alert_issue_opened is True
    assert all(e.status == "CIRCUIT_BREAKER_MAINTENANCE" for e in summary.epochs)
    assert drive_store.list_ore_objects() == []
    assert summary.consolidate.dead_mans_switch.triggered is True
    assert "DRIVE_CANARY_FAILED" in summary.consolidate.dead_mans_switch.reasons
    # Both Job 0 and Job 7 upsert into ONE single issue labeled ral-critical-alert (no duplicate issues).
    assert len(fake_gh_server.issues) == 1
    issue = next(iter(fake_gh_server.issues.values()))
    assert CRITICAL_ALERT_LABEL in issue["labels"]

    # (b) .git > 700 MB threshold triggers Dead-Man's Switch cleanly.
    dms_git = evaluate_dead_mans_switch(
        tmp_path,
        circuit_breaker=None,
        drive_canary={"ok": True},
        git_size_mb=725.4,
        max_git_mb=700.0,
        current_run_date="2026-11-24",
        current_verified_count=1,
        github_client=gh_client,
    )
    assert dms_git.triggered is True
    assert any("GIT_SIZE_EXCEEDED" in r for r in dms_git.reasons)
    assert len(fake_gh_server.issues) == 1

    # (c) 3 consecutive nights with 0 verified nodes triggers switch (while 2 nights does NOT).
    repo2 = tmp_path / "zero_streak_repo"
    _init_mini_repo(repo2)
    tel_file = repo2 / "metrics" / "telemetry" / "2026-11.ndjson"
    tel_file.parent.mkdir(parents=True, exist_ok=True)
    tel_file.write_text(
        "\n".join(
            [
                json.dumps({"record_type": "nightly_summary", "run_date": "2026-11-21", "nodes_verified_tonight": 2}),
                json.dumps({"record_type": "nightly_summary", "run_date": "2026-11-22", "nodes_verified_tonight": 0}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    assert count_consecutive_zero_verified_nights(repo2, current_run_date="2026-11-23", current_verified_count=0) == 2
    dms_two = evaluate_dead_mans_switch(
        repo2,
        circuit_breaker=None,
        drive_canary={"ok": True},
        git_size_mb=10.0,
        current_run_date="2026-11-23",
        current_verified_count=0,
    )
    assert dms_two.triggered is False

    with tel_file.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps({"record_type": "nightly_summary", "run_date": "2026-11-23", "nodes_verified_tonight": 0}) + "\n")
    dms_three = evaluate_dead_mans_switch(
        repo2,
        circuit_breaker=None,
        drive_canary={"ok": True},
        git_size_mb=10.0,
        current_run_date="2026-11-24",
        current_verified_count=0,
        github_client=gh_client,
    )
    assert dms_three.triggered is True
    assert dms_three.consecutive_zero_verified_nights == 3
    assert "ZERO_VERIFIED_NODES_3_CONSECUTIVE_NIGHTS" in dms_three.reasons


# ---------------------------------------------------------------------------
# 4. Orchestrator 429 RPD Pivot & CLI Entrypoints
# ---------------------------------------------------------------------------


def test_orchestrator_429_rpd_pivot_and_cli_entrypoints(tmp_path: Path) -> None:
    _init_mini_repo(tmp_path)
    handoff = tmp_path / "ral_handoff"
    cache = tmp_path / ".ral_cache"
    ore_cache = tmp_path / "ore_cache"

    # Populate benchmark cache via live client, then inject 429 on epoch calls.
    good_client = _make_fake_gemini_client(fail_429_in_stage=False)
    drive_store = DriveOreStore(_FakeDriveService(), "root-warehouse-id")

    pre = run_preflight_job(
        tmp_path,
        handoff_dir=handoff,
        cache_dir=cache,
        run_date_iso="2026-11-24",
        available_epochs=6,
        dry_run=True,
        gemini_client=good_client,
        drive_store=drive_store,
    )
    assert pre.available_epochs == 6

    client_429 = _make_fake_gemini_client(fail_429_in_stage=True)
    src = SourceRecord(
        source_id="arxiv:2105.00003v1",
        source_license="CC-BY-4.0",
        source_format="arxiv_tex",
        title="RPD Exhaustion Test Paper",
        pub_year=2022,
        peer_reviewed=False,
        retracted=False,
        canonical_doi=None,
        arxiv_base_id="2105.00003",
        arxiv_version=1,
        citation_count=0,
        raw_bytes=b"rpd-test",
        raw_ext="tar.gz",
        extracted_text="Euclidean norm squared scales quadratically.",
    )
    ep1 = run_epoch_job(
        tmp_path,
        handoff_dir=handoff,
        epoch_index=1,
        cache_dir=cache,
        ore_cache_dir=ore_cache,
        dry_run=True,
        gemini_client=client_429,
        drive_store=drive_store,
        sandbox=_make_local_exec_sandbox(),
        source_fetcher=lambda _m, _t, _r: [src],
    )
    assert ep1.status == "PIVOTED_TO_MAINTENANCE"
    assert ep1.pivoted_to_maintenance is True
    assert set(ep1.exhausted_models) == {"gemini-2.5-pro", "gemma-4-27b-it"}
    assert list(ore_cache.iterdir()) == []

    # Subsequent epoch reads persisted router_state.json and pivots immediately with 0 LLM calls.
    ep2 = run_epoch_job(
        tmp_path,
        handoff_dir=handoff,
        epoch_index=2,
        cache_dir=cache,
        ore_cache_dir=ore_cache,
        dry_run=True,
        gemini_client=client_429,
        drive_store=drive_store,
        sandbox=_make_local_exec_sandbox(),
        source_fetcher=lambda _m, _t, _r: [src],
    )
    assert ep2.status == "PIVOTED_TO_MAINTENANCE"
    assert ep2.pivot_reason == "ALL_MODELS_EXHAUSTED_BEFORE_EPOCH"

    # CLI subcommands smoke check.
    cli_repo = tmp_path / "cli_repo"
    _init_mini_repo(cli_repo)
    assert cli_main(["render-docs", "--repo-root", str(cli_repo)]) == 0
    assert (cli_repo / "INDEX.md").is_file()
    assert (cli_repo / "llms.txt").is_file()
    assert cli_main(["validate", "--repo-root", str(cli_repo)]) == 0
    assert (
        cli_main(
            [
                "preflight",
                "--repo-root",
                str(cli_repo),
                "--handoff-dir",
                str(cli_repo / "ral_handoff"),
                "--cache-dir",
                str(cli_repo / ".ral_cache"),
                "--dry-run",
            ]
        )
        == 0
    )


# ---------------------------------------------------------------------------
# 5 & 6. Hardened Read-Only MCP Server: Security, Cone, Failure Modes, Sync
# ---------------------------------------------------------------------------


def _build_mcp_fixture_vault(root: Path) -> tuple[RALMCPServer, Path]:
    _init_mini_repo(root)
    # T1 Axiom -> T1 Lemma -> T1 Main Theorem; plus T3 preprint, T4 Negative Result, and Econ Isomorphism
    _write_node_fixture(
        root,
        ral_id="RAL-MATH-0000000001",
        taxonomy_code="math.analysis",
        title="Hilbert Space Inner Product Axiom",
        node_type="AXIOM",
        epistemic_tier="T1_CANONICAL",
        formal_signature=["space:hilbert", "operator:inner_product"],
        formal_statement=r"$$\langle x, x \rangle \ge 0$$",
        source_id="doi:10.1000/axiom.1",
    )
    _write_node_fixture(
        root,
        ral_id="RAL-MATH-0000000002",
        taxonomy_code="math.analysis",
        title="Plancherel Fourier Isometry",
        node_type="THEOREM",
        epistemic_tier="T1_CANONICAL",
        formal_signature=["operator:fourier_transform", "space:hilbert"],
        formal_statement=r"$$\|\hat{f}\|_2 = \|f\|_2$$",
        source_id="doi:10.1000/lemma.2",
    )
    _write_node_fixture(
        root,
        ral_id="RAL-PHYS-0000000003",
        taxonomy_code="phys.statistical-mechanics",
        title="Classical Fluctuation-Dissipation Theorem",
        node_type="THEOREM",
        epistemic_tier="T1_CANONICAL",
        formal_signature=["operator:autocorrelation", "operator:fourier_transform", "regime:linear_response"],
        formal_statement=r"$$\chi''(\omega) = \frac{\omega}{2 k_B T} S_{AA}(\omega)$$",
        source_id="doi:10.1000/fdt.3",
    )
    _write_node_fixture(
        root,
        ral_id="RAL-ECON-0000000004",
        taxonomy_code="econ.financial-economics",
        title="Market Liquidity Impact Response Kernel",
        node_type="THEOREM",
        epistemic_tier="T2_ESTABLISHED",
        formal_signature=["operator:autocorrelation", "operator:fourier_transform", "regime:linear_response"],
        formal_statement=r"$$G''(\omega) = \frac{\omega}{2 \Theta_{\mathrm{liq}}} S_{QQ}(\omega)$$",
        source_id="doi:10.1000/econ.4",
    )
    _write_node_fixture(
        root,
        ral_id="RAL-PHYS-0000000005",
        taxonomy_code="phys.statistical-mechanics",
        title="Driven Non-Equilibrium FDT Violation Counterexample",
        node_type="NEGATIVE_RESULT",
        epistemic_tier="T4_FALSIFIED",
        formal_signature=["regime:non_equilibrium", "pathology:detailed_balance_violation"],
        formal_statement=r"$$\chi''(\omega) \neq \frac{\omega}{2 k_B T} S_{AA}(\omega)$$",
        source_id="doi:10.1000/neg.5",
        re_evaluate_after="2036-11-24",
    )
    _write_node_fixture(
        root,
        ral_id="RAL-PHYS-0000000006",
        taxonomy_code="phys.quantum-mechanics",
        title="Speculative Quantum Thermalization Preprint",
        node_type="THEOREM",
        epistemic_tier="T3_SPECULATIVE",
        formal_signature=["regime:quantum_chaos"],
        formal_statement=r"$$\tau_{\mathrm{eq}} \sim \hbar / (k_B T)$$",
        source_id="doi:10.1000/spec.6",
        re_evaluate_after="2027-05-24",
    )

    # Prereq stub
    debt_path = root / "ledger" / "prereq_debt.ndjson"
    debt_path.parent.mkdir(parents=True, exist_ok=True)
    debt_path.write_text(
        json.dumps(
            {
                "stub_id": "STUB-MATH-BOCHNER-THEOREM",
                "domain": "math.analysis",
                "cited_by": ["RAL-MATH-0000000002"],
                "attempt_count": 1,
                "last_attempt_night": 1,
                "cooldown_until_night": 3,
                "status": "OPEN",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    _append_edge(
        root,
        {
            "edge_id": "E-DEP0000001",
            "src": "RAL-PHYS-0000000003",
            "dst": "RAL-MATH-0000000002",
            "relation": "DEPENDS_ON",
            "confidence": 0.99,
            "evidence_sha256": "a" * 64,
            "created_at": "2026-11-24T20:15:00Z",
            "created_by_model": "gemini-2.5-pro",
        },
    )
    _append_edge(
        root,
        {
            "edge_id": "E-DEP0000002",
            "src": "RAL-MATH-0000000002",
            "dst": "RAL-MATH-0000000001",
            "relation": "DEPENDS_ON",
            "confidence": 0.99,
            "evidence_sha256": "b" * 64,
            "created_at": "2026-11-24T20:15:00Z",
            "created_by_model": "gemini-2.5-pro",
        },
    )
    _append_edge(
        root,
        {
            "edge_id": "E-DEP0000003",
            "src": "RAL-MATH-0000000002",
            "dst": "STUB-MATH-BOCHNER-THEOREM",
            "relation": "DEPENDS_ON",
            "confidence": 0.90,
            "evidence_sha256": "c" * 64,
            "created_at": "2026-11-24T20:15:00Z",
            "created_by_model": "gemini-2.5-pro",
        },
    )
    _append_edge(
        root,
        {
            "edge_id": "E-BRK0000004",
            "src": "RAL-PHYS-0000000005",
            "dst": "RAL-PHYS-0000000003",
            "relation": "BREAKS_WITHOUT_ASSUMPTION",
            "confidence": 1.0,
            "evidence_sha256": "d" * 64,
            "created_at": "2026-11-24T20:15:00Z",
            "created_by_model": "gemma-4-27b-it",
        },
    )
    _append_edge(
        root,
        {
            "edge_id": "E-ISO0000005",
            "src": "RAL-PHYS-0000000003",
            "dst": "RAL-ECON-0000000004",
            "relation": "ISOMORPHIC_TO",
            "formal_mapping": {"k_B*T": "Theta_liq", "chi''(omega)": "G''(omega)"},
            "confidence": 0.89,
            "evidence_sha256": "e" * 64,
            "created_at": "2026-11-24T20:15:00Z",
            "created_by_model": "gemini-2.5-pro",
        },
    )

    assert validate_vault(root).ok is True
    db_path = root / ".ral_cache" / "ral-index.sqlite"
    build_or_update_index(root, db_path)
    return RALMCPServer(root, db_path=db_path, cache_dir=root / ".ral_cache"), db_path


def test_mcp_server_security_readonly_injection_and_path_traversal(tmp_path: Path) -> None:
    server, db_path = _build_mcp_fixture_vault(tmp_path)

    # 1. Read-only mode enforcement: attempting writes via MCP's read-only URI connection MUST fail.
    from mcp.ral_mcp_server import _connect_readonly, _resolve_vault_node_path

    ro_conn = _connect_readonly(db_path)
    try:
        with pytest.raises(sqlite3.OperationalError):
            ro_conn.execute("DELETE FROM nodes")
        with pytest.raises(sqlite3.OperationalError):
            ro_conn.execute("CREATE TABLE hacked (x TEXT)")
    finally:
        ro_conn.close()

    # 2. Strict `ral_id` regex guard rejects SQL injection, traversal, and malformed IDs.
    bad_ids = [
        "RAL-PHYS-0000000003' OR '1'='1",
        "RAL-PHYS-0000000003; DROP TABLE nodes; --",
        "../../config/model_policy.yaml",
        "RAL-PHYS-0000000003\n",
        "ral-phys-0000000003",
        "RAL-TOO-SHORT",
        "",
        None,
    ]
    for bad in bad_ids:
        with pytest.raises(ValueError):
            validate_ral_id(bad)
        if isinstance(bad, str):
            with pytest.raises(ValueError):
                server.ral_get_node(bad)
            with pytest.raises(ValueError):
                server.ral_get_dependency_cone(bad)

    # 3. Domain filter injection guard.
    for bad_dom in ("phys'; DROP TABLE edges;--", "phys%", "../vault", "PHYS"):
        with pytest.raises(ValueError):
            validate_domain_filter(bad_dom)
        with pytest.raises(ValueError):
            server.ral_find_isomorphisms(domain_a=bad_dom)

    # 4. Path confinement guard blocks escaping `vault/` even if SQLite `rel_path` was tampered!
    with pytest.raises(PermissionError):
        _resolve_vault_node_path(tmp_path, "../config/model_policy.yaml")
    with pytest.raises(PermissionError):
        _resolve_vault_node_path(tmp_path, "/etc/passwd")

    tamper_conn = sqlite3.connect(str(db_path))
    try:
        with tamper_conn:
            tamper_conn.execute(
                "UPDATE nodes SET rel_path = ? WHERE ral_id = ?",
                ("../config/model_policy.yaml", "RAL-PHYS-0000000006"),
            )
    finally:
        tamper_conn.close()

    with pytest.raises(PermissionError):
        server.ral_get_node("RAL-PHYS-0000000006")


def test_mcp_server_tools_topological_cone_failure_modes_warnings_and_sync(tmp_path: Path) -> None:
    server, db_path = _build_mcp_fixture_vault(tmp_path)

    # 1. ral_search respects max_tier and returns required epistemic fields.
    search_t2 = server.ral_search(
        "Fluctuation Dissipation Quantum",
        formal_signature_tags=["operator:autocorrelation"],
        max_tier="T2_ESTABLISHED",
        limit=10,
    )
    ids_t2 = {r["ral_id"] for r in search_t2["results"]}
    assert "RAL-PHYS-0000000003" in ids_t2
    assert "RAL-PHYS-0000000006" not in ids_t2  # T3 excluded when max_tier=T2_ESTABLISHED
    assert "RAL-PHYS-0000000005" not in ids_t2  # T4 excluded

    search_t4 = server.ral_search("Quantum Fluctuation", max_tier="T4_FALSIFIED", limit=10)
    ids_t4 = {r["ral_id"] for r in search_t4["results"]}
    assert "RAL-PHYS-0000000006" in ids_t4
    t3_hit = next(r for r in search_t4["results"] if r["ral_id"] == "RAL-PHYS-0000000006")
    assert "warning" in t3_hit and "T3_SPECULATIVE" in t3_hit["warning"]
    for item in search_t4["results"]:
        assert {"ral_id", "epistemic_tier", "taxonomy_code", "consensus_version"} <= set(item.keys())

    # 2. ral_get_node attaches linked BREAKS_WITHOUT_ASSUMPTION failure modes & T3/T4 warning banners.
    fdt_node = server.ral_get_node("RAL-PHYS-0000000003")
    assert fdt_node["warning_banner"] is None
    assert len(fdt_node["failure_modes"]) == 1
    assert fdt_node["failure_modes"][0]["ral_id"] == "RAL-PHYS-0000000005"
    assert fdt_node["failure_modes"][0]["node_type"] == "NEGATIVE_RESULT"
    assert "BREAKS_WITHOUT_ASSUMPTION" in fdt_node["annotated_markdown"]

    t3_node = server.ral_get_node("RAL-PHYS-0000000006")
    assert t3_node["warning_banner"] is not None
    assert "T3_SPECULATIVE" in t3_node["warning_banner"]
    assert "WARNING [T3_SPECULATIVE]" in t3_node["annotated_markdown"]

    t4_node = server.ral_get_node("RAL-PHYS-0000000005")
    assert t4_node["warning_banner"] is not None
    assert "T4_FALSIFIED" in t4_node["warning_banner"]

    # 3. ral_get_dependency_cone returns acyclic prerequisite DAG in topological order (foundations first).
    cone = server.ral_get_dependency_cone("RAL-PHYS-0000000003", max_depth=5)
    topo_ids = [n["ral_id"] for n in cone["topological_order"]]
    assert topo_ids == [
        "RAL-MATH-0000000001",  # Axiom foundation first
        "RAL-MATH-0000000002",  # Intermediate lemma
        "RAL-PHYS-0000000003",  # Target theorem last
    ]
    assert cone["unresolved_stubs"] == ["STUB-MATH-BOCHNER-THEOREM"]

    cone_depth1 = server.ral_get_dependency_cone("RAL-PHYS-0000000003", max_depth=1)
    assert [n["ral_id"] for n in cone_depth1["topological_order"]] == [
        "RAL-MATH-0000000002",
        "RAL-PHYS-0000000003",
    ]

    # 4. ral_find_isomorphisms returns structured formal_mapping dictionaries across sciences.
    isos = server.ral_find_isomorphisms(domain_a="econ", domain_b="phys")
    assert isos["count"] == 1
    assert isos["isomorphisms"][0]["formal_mapping"] == {
        "k_B*T": "Theta_liq",
        "chi''(omega)": "G''(omega)",
    }

    # 5. Index sync verifies SHA-256 against ral-index-latest.json BEFORE decompressing.
    release_dir = tmp_path / "release_stage"
    zst_path, valid_sha, _ = compress_sqlite_zst(
        db_path, release_dir / "ral-index-20261124-sync1.sqlite.zst"
    )
    ptr_path = release_dir / LATEST_POINTER_NAME
    ptr_path.write_text(
        json.dumps(
            {
                "schema_version": "2.1",
                "asset_name": zst_path.name,
                "sha256": valid_sha,
            }
        ),
        encoding="utf-8",
    )
    synced_db = tmp_path / "mcp_client" / "ral-index.sqlite"
    client_server = RALMCPServer(
        tmp_path, db_path=synced_db, cache_dir=tmp_path / "mcp_client"
    )
    sync_ok = client_server.sync_index(local_pointer_path=ptr_path)
    assert sync_ok["synced"] is True
    assert synced_db.is_file()

    # Tampered archive must be rejected BEFORE overwriting active DB.
    zst_path.write_bytes(b"TAMPERED_CORRUPT_ARCHIVE")
    ptr_path.write_text(
        json.dumps(
            {
                "schema_version": "2.1",
                "asset_name": zst_path.name,
                "sha256": "f" * 64,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        client_server.sync_index(local_pointer_path=ptr_path)
    assert client_server.ral_get_node("RAL-MATH-0000000001")["ral_id"] == "RAL-MATH-0000000001"

    # 6. Stdio JSON-RPC 2.0 end-to-end transport test.
    requests_jsonl = "\n".join(
        [
            json.dumps({"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}),
            json.dumps({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {
                        "name": "ral_get_dependency_cone",
                        "arguments": {"ral_id": "RAL-PHYS-0000000003", "max_depth": 5},
                    },
                }
            ),
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "tools/call",
                    "params": {
                        "name": "ral_get_node",
                        "arguments": {"ral_id": "RAL-PHYS-0000000003; DROP TABLE nodes;"},
                    },
                }
            ),
        ]
    ) + "\n"
    out_buf = io.StringIO()
    assert serve_stdio(server, stdin=io.StringIO(requests_jsonl), stdout=out_buf) == 0
    responses = [json.loads(line) for line in out_buf.getvalue().splitlines() if line.strip()]
    assert len(responses) == 4
    assert responses[0]["result"]["serverInfo"]["name"] == "ral-mcp-server"
    assert [t["name"] for t in responses[1]["result"]["tools"]] == [t["name"] for t in MCP_TOOLS_SCHEMA]
    assert responses[2]["result"]["isError"] is False
    cone_payload = json.loads(responses[2]["result"]["content"][0]["text"])
    assert [n["ral_id"] for n in cone_payload["topological_order"]] == [
        "RAL-MATH-0000000001",
        "RAL-MATH-0000000002",
        "RAL-PHYS-0000000003",
    ]
    assert responses[3]["result"]["isError"] is True


# ---------------------------------------------------------------------------
# 7. Root /INDEX.md (<15k Token Cap) & /llms.txt Verification
# ---------------------------------------------------------------------------


def test_root_index_md_token_budget_and_llms_txt_hygiene(tmp_path: Path) -> None:
    index_path = REPO_ROOT / "INDEX.md"
    llms_path = REPO_ROOT / "llms.txt"
    assert index_path.is_file()
    assert llms_path.is_file()

    index_text = index_path.read_text(encoding="utf-8")
    llms_text = llms_path.read_text(encoding="utf-8")

    assert estimate_tokens(index_text) < 15000
    assert scan_text_for_leaks(index_text, "INDEX.md") == []
    assert scan_text_for_leaks(llms_text, "llms.txt") == []

    for macro in ("math", "phys", "cs", "eng", "econ", "chem", "bio"):
        assert f"`{macro}`" in index_text
        assert f"`{macro}`" in llms_text
    for tool_name in (
        "ral_search",
        "ral_get_node",
        "ral_get_dependency_cone",
        "ral_find_isomorphisms",
    ):
        assert tool_name in llms_text

    # Stress test: 600 synthetic nodes MUST automatically compact INDEX.md strictly below 15,000 tokens.
    stress_repo = tmp_path / "stress_repo"
    _init_mini_repo(stress_repo)
    for i in range(600):
        ral_id = f"RAL-MATH-{i:010d}"
        _write_node_fixture(
            stress_repo,
            ral_id=ral_id,
            taxonomy_code="math.analysis" if i % 2 == 0 else "phys.statistical-mechanics",
            title=f"Synthetic Verified Scaling Theorem Variant Number {i} With Extended Descriptors",
            source_id=f"doi:10.1000/stress.{i}",
            record_source_ledger=False,
        )
    res = regenerate_vault_docs(stress_repo, max_index_tokens=15000)
    assert res["index_tokens"] < 15000
    assert res["node_count"] == 600
    stress_index = (stress_repo / "INDEX.md").read_text(encoding="utf-8")
    assert estimate_tokens(stress_index) < 15000
    assert scan_text_for_leaks(stress_index, "INDEX.md") == []


# ---------------------------------------------------------------------------
# 8. Batch A Regression Tests: Cross-Job Handoff, Monotonic Night,
#    Job 7 Index Restore + Embedder, & Per-Candidate Fault Isolation
# ---------------------------------------------------------------------------


def test_cross_job_handoff_defaults_monotonic_night_and_job7_restore_embedder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    gitignore_text = (REPO_ROOT / ".gitignore").read_text(encoding="utf-8")
    assert ".ral_handoff" not in gitignore_text
    assert "ral_handoff/" in gitignore_text

    # 1. Monotonic night computation when --night is omitted in CLI preflight.
    cli_repo = tmp_path / "cli_monotonic"
    _init_mini_repo(cli_repo)
    assert compute_current_night("2026-11-24") == (date(2026, 11, 24) - date(2026, 1, 1)).days
    assert (
        cli_main(
            [
                "preflight",
                "--repo-root",
                str(cli_repo),
                "--date",
                "2026-11-24",
                "--dry-run",
            ]
        )
        == 0
    )
    default_handoff = Path("ral_handoff")
    # Pass explicit repo-local path check: CLI default is relative `ral_handoff` unless passed; clean up CWD if created.
    cli_plan_path = default_handoff / "plan.json"
    if cli_plan_path.is_file():
        plan_obj = json.loads(cli_plan_path.read_text(encoding="utf-8"))
        assert plan_obj["current_night"] == compute_current_night("2026-11-24")
        import shutil

        shutil.rmtree(default_handoff, ignore_errors=True)
        shutil.rmtree(Path(".ral_cache"), ignore_errors=True)

    # Verify explicit --handoff-dir ral_handoff without --night inside tmp_path.
    repo = tmp_path / "restore_repo"
    _init_mini_repo(repo)
    handoff_dir = repo / "ral_handoff"
    cache_dir = repo / ".ral_cache"
    assert (
        cli_main(
            [
                "preflight",
                "--repo-root",
                str(repo),
                "--handoff-dir",
                str(handoff_dir),
                "--cache-dir",
                str(cache_dir),
                "--date",
                "2026-11-25",
                "--dry-run",
            ]
        )
        == 0
    )
    assert json.loads((handoff_dir / "plan.json").read_text(encoding="utf-8"))["current_night"] == (
        date(2026, 11, 25) - date(2026, 1, 1)
    ).days

    # 2. Job 7 downloads prior SQLite index before build_or_update_index() & CLI instantiates embed_client.
    _write_node_fixture(
        repo,
        ral_id="RAL-MATH-0000000077",
        taxonomy_code="math.analysis",
        title="Existing Indexed Theorem",
        source_id="doi:10.1000/existing.77",
    )
    seed_db = tmp_path / "seed_index.sqlite"
    fake_gemini = _make_fake_gemini_client()
    build_or_update_index(
        repo,
        seed_db,
        embed_client=fake_gemini,
        discovered_embed_model="gemini-embedding-2",
        now=datetime(2026, 11, 24, 12, 0, 0, tzinfo=timezone.utc),
    )
    seed_zst, _, _ = compress_sqlite_zst(seed_db, tmp_path / "seed_index.sqlite.zst")

    fake_gh_server = _FakeGitHubServer()
    gh_client = GitHubClient(
        "rayhan",
        "rayhans-autonomous-lab",
        token="test-token",
        request_fn=fake_gh_server.request,
    )
    gh_client.blue_green_swap_index(
        seed_zst,
        run_date="2026-11-24",
        run_id="seed1",
        embedding_model_id="gemini-embedding-2",
        embedding_dim=768,
        node_count=1,
    )

    # Stamp handoff circuit_breaker.json as Job 0 would, with NO local ral-index.sqlite in fresh Job 7 cache.
    handoff_dir.mkdir(parents=True, exist_ok=True)
    (handoff_dir / "circuit_breaker.json").write_text(
        json.dumps(
            {
                "circuit_breaker": None,
                "drive_canary": {"ok": True},
                "embedding_model": {"model_id": "gemini-embedding-2", "dim": 768, "live": True},
            }
        ),
        encoding="utf-8",
    )
    embed_calls: list[str] = []

    class _CountingEmbedClient:
        class _Models:
            def embed_content(self, *, model: str, contents: str, config: dict[str, Any]) -> Any:
                embed_calls.append(contents)
                return fake_gemini.models.embed_content(model=model, contents=contents, config=config)

        models = _Models()

    fresh_cache = repo / "job7_fresh_cache"
    con = run_consolidate_job(
        repo,
        handoff_dir=handoff_dir,
        cache_dir=fresh_cache,
        run_date_iso="2026-11-25",
        run_id="job7restore",
        dry_run=False,
        github_client=gh_client,
        embed_client=_CountingEmbedClient(),
        now=datetime(2026, 11, 25, 1, 0, 0, tzinfo=timezone.utc),
    )
    assert con.ok is True
    assert con.vector_action == "INCREMENTAL_UPDATED"
    assert con.embed_space_consistent is True
    assert embed_calls == []  # Zero re-embeddings because prior SQLite index was restored first!

    # CLI consolidate instantiates embed_client when RAL_GEMINI_FREE_KEY is present.
    captured_embed_clients: list[Any] = []
    sentinel_client = object()
    monkeypatch.setenv("RAL_GEMINI_FREE_KEY", "test-fake-key-for-cli")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr("ral.cli.build_gemini_client_from_env", lambda: sentinel_client)

    def _spy_consolidate(repo_root: Path, **kwargs: Any) -> Any:
        captured_embed_clients.append(kwargs.get("embed_client"))
        return con

    monkeypatch.setattr("ral.cli.run_consolidate_job", _spy_consolidate)
    assert (
        cli_main(
            [
                "consolidate",
                "--repo-root",
                str(repo),
                "--handoff-dir",
                str(handoff_dir),
                "--cache-dir",
                str(fresh_cache),
                "--date",
                "2026-11-25",
                "--dry-run",
            ]
        )
        == 0
    )
    assert captured_embed_clients == [sentinel_client]


def test_per_candidate_fault_isolation_and_quota_re_raise(tmp_path: Path) -> None:
    _init_mini_repo(tmp_path)
    good_ore = b"%PDF-or-tex-isolated-candidate-test"
    src = SourceRecord(
        source_id="arxiv:2106.00001v1",
        source_license="CC-BY-4.0",
        source_format="arxiv_tex",
        title="Multi-Candidate Fault Isolation Paper",
        pub_year=2021,
        peer_reviewed=True,
        retracted=False,
        canonical_doi="10.1000/fault.iso",
        arxiv_base_id="2106.00001",
        arxiv_version=1,
        citation_count=80,
        raw_bytes=good_ore,
        raw_ext="tar.gz",
        extracted_text=(
            "Broken candidate quote for fault isolation test. "
            "Euclidean norm squared scales quadratically and matrix trace equals eigenvalue sum."
        ),
    )

    def _two_candidate_gen(*, model: str, contents: str, config: dict[str, Any]) -> Any:
        _ = (model, config)
        if "zero-tool primary-source scientific claim extractor" in contents:
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "candidates": [
                            {
                                "title": "Broken Candidate Claim",
                                "claim_summary": "Triggers malformed JSON in Stage 2.",
                                "quote_text": "Broken candidate quote for fault isolation test.",
                                "source_locator": "tex:sec1:eq1",
                                "node_type": "THEOREM",
                                "taxonomy_code": "math.analysis",
                                "formal_signature": ["operator:broken"],
                                "assumptions": ["x is real"],
                                "recommended_oracle": "sympy_numeric",
                            },
                            {
                                "title": "Parseval Spectral Norm Identity",
                                "claim_summary": "Euclidean norm scales quadratically.",
                                "quote_text": "Euclidean norm squared scales quadratically and matrix trace equals eigenvalue sum.",
                                "source_locator": "tex:sec2:eq4",
                                "node_type": "THEOREM",
                                "taxonomy_code": "math.analysis",
                                "formal_signature": ["operator:norm", "operator:trace", "operator:eigenvalues"],
                                "assumptions": ["x is finite-dimensional real vector", "A is real symmetric matrix"],
                                "recommended_oracle": "sympy_numeric",
                            },
                        ]
                    }
                )
            )
        if "Formalize this verified scientific quote" in contents:
            if "Broken Candidate Claim" in contents:
                return SimpleNamespace(text="not json at all /Users/alice/secret_path")
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "node_type": "THEOREM",
                        "taxonomy_code": "math.analysis",
                        "title": "Parseval Spectral Norm Identity",
                        "formal_signature": ["operator:norm", "operator:trace", "operator:eigenvalues"],
                        "assumptions": ["x is finite-dimensional real vector", "A is real symmetric matrix"],
                        "formal_statement_latex": "\\|c v\\|_2^2 = c^2 \\|v\\|_2^2",
                        "symbol_rows": [{"symbol": "v", "meaning": "real vector", "domain": "\\mathbb{R}^n", "si_units": "1"}],
                        "failure_boundaries": ["non-real spaces"],
                        "provenance_summary": "Verified Euclidean norm identity.",
                        "oracle_plugin": "sympy_numeric",
                        "verify_py": VALID_NUMPY_KERNEL,
                    }
                )
            )
        if "Assume this formalization has a hidden defect" in contents:
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "attack_found": False,
                        "falsify_py": "",
                        "missing_assumptions": [],
                        "summary": "Verified",
                    }
                )
            )
        raise AssertionError(f"Unexpected prompt: {contents[:80]}")

    client = SimpleNamespace(models=SimpleNamespace(generate_content=_two_candidate_gen))
    router = ModelRouter(
        {
            "status": "OK",
            "model_capability_epoch": 1,
            "roles": {
                "formalizer": ["gemini-2.5-pro"],
                "architect": ["gemini-2.5-pro"],
                "skeptic": ["gemma-4-27b-it"],
                "scout": ["gemini-2.5-flash"],
            },
            "exhausted_models": [],
        }
    )
    drive_store = DriveOreStore(_FakeDriveService(), "root-warehouse-id")
    out = execute_epoch_pipeline(
        router,
        repo_root=tmp_path,
        mode="explore",
        target="math.analysis",
        current_night=1,
        ore_cache_dir=tmp_path / "ore_cache",
        gemini_client=client,
        drive_store=drive_store,
        sandbox=_make_local_exec_sandbox(),
        source_fetcher=lambda _m, _t, _r: [src],
        dry_run=False,
        now=datetime(2026, 11, 25, 2, 0, 0, tzinfo=timezone.utc),
    )

    assert out["rejected_count"] == 1
    assert len(out["nodes_created"]) == 1
    assert out["nodes_verified_tonight"] == 1
    rej_rows = [
        json.loads(line)
        for line in (tmp_path / "ledger" / "rejected" / "2026-11.ndjson").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(rej_rows) == 1
    assert rej_rows[0]["stage_failed"] == "STAGE2_FORMALIZER"
    assert "JSONDecodeError" in rej_rows[0]["reason"]
    assert "/Users/alice" not in rej_rows[0]["error_detail"]
    assert validate_vault(tmp_path).ok is True

    # Quota exhaustion error MUST propagate out of execute_epoch_pipeline rather than being swallowed.
    class _HardQuotaError(RuntimeError):
        status_code = 429

    def _quota_boom_gen(*, model: str, contents: str, config: dict[str, Any]) -> Any:
        _ = (model, config)
        if "zero-tool primary-source scientific claim extractor" in contents:
            return _two_candidate_gen(model=model, contents=contents, config=config)
        raise _HardQuotaError("429 RESOURCE_EXHAUSTED: GenerateRequestsPerDay daily quota limit: 0")

    quota_repo = tmp_path / "quota_prop_repo"
    _init_mini_repo(quota_repo)
    quota_router = ModelRouter(
        {
            "status": "OK",
            "model_capability_epoch": 1,
            "roles": {
                "formalizer": ["gemini-2.5-pro"],
                "architect": ["gemini-2.5-pro"],
                "skeptic": ["gemma-4-27b-it"],
                "scout": ["gemini-2.5-flash"],
            },
            "exhausted_models": [],
        }
    )
    with pytest.raises(AllModelsExhaustedError):
        execute_epoch_pipeline(
            quota_router,
            repo_root=quota_repo,
            mode="explore",
            target="math.analysis",
            current_night=1,
            ore_cache_dir=tmp_path / "ore_cache_q",
            gemini_client=SimpleNamespace(models=SimpleNamespace(generate_content=_quota_boom_gen)),
            drive_store=drive_store,
            sandbox=_make_local_exec_sandbox(),
            source_fetcher=lambda _m, _t, _r: [src],
            dry_run=True,
            now=datetime(2026, 11, 25, 2, 0, 0, tzinfo=timezone.utc),
        )

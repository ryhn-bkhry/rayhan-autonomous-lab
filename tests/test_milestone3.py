"""Comprehensive Milestone 3 acceptance and regression tests (RAL Blueprint v2.1)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
import subprocess
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import yaml

from ci.validate_node import split_frontmatter, validate_node_file
from ci.validate_vault import validate_vault
from ral.canonicalize import (
    compute_assumptions_hash,
    compute_oracle_script_sha256,
    compute_quote_sha256,
    sha256_bytes,
)
from ral.director import (
    DomainState,
    MACRO_DISCIPLINES,
    allocate_epochs,
    build_nightly_plan,
    compute_coverage_entropy,
    compute_cross_domain_pagerank,
    compute_domain_priorities,
    compute_epistemic_tension,
    compute_portfolio_weights,
    compute_prereq_cooldown,
    execute_leaf_split,
    is_eligible_for_passive_recheck,
    load_curriculum_params,
    load_prereq_debt,
    load_taxonomy_seeds,
    main as director_main,
    plan_leaf_split,
    record_prereq_attempt_failure,
    resolve_prereq_stub,
    score_synthesis_pairs,
    whiten_embeddings,
)
from ral.github_client import (
    CRITICAL_ALERT_LABEL,
    GitHubBudgetExceededError,
    GitHubClient,
    LATEST_POINTER_NAME,
    ReleaseIntegrityError,
)
from ral.maintenance import (
    main as maintenance_main,
    run_epoch_or_pivot,
    run_zero_llm_maintenance,
)
from ral.model_router import ModelRouter
from ral.pipeline.oracle_sandbox import OracleSandbox
from ral.storage.drive_ore import DRIVE_FOLDER_MIME, DriveOreStore, build_ore_uri
from ral.storage.index_builder import (
    build_or_update_index,
    compress_sqlite_zst,
    decompress_sqlite_zst,
    hybrid_search,
    load_node_embeddings,
    main as index_builder_main,
    read_index_meta,
)

REPO_ROOT = Path(__file__).resolve().parent.parent

VALID_SYMPY_KERNEL = """# RAL_ORACLE: sympy_numeric | TIMEOUT: 15s
import sympy as sp
import numpy as np

w, kT, hbar = sp.symbols("w kT hbar", positive=True, real=True)
quantum_factor = hbar * sp.coth(hbar * w / (2 * kT))
classical_limit = sp.limit(quantum_factor, hbar, 0)
assert sp.simplify(classical_limit - (2 * kT / w)) == 0

for ratio in np.logspace(1, 5, 40):
    q_val = 1.0 / np.tanh(1.0 / (2.0 * ratio))
    c_val = 2.0 * ratio
    assert np.isclose(q_val, c_val, rtol=1e-2)

assert abs((1.0 / np.tanh(1.0 / (2.0 * 0.05))) - (2.0 * 0.05)) > 0.8
"""


def _init_mini_repo(root: Path) -> None:
    """Create minimal config files in `tmp_path` so `validate_vault` & `director` run offline."""
    cfg_dir = root / "config"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    for fname in ("license_allowlist.yaml", "curriculum_params.yaml", "taxonomy_seeds.yaml", "model_policy.yaml", "oracle_plugins.yaml"):
        (cfg_dir / fname).write_text((REPO_ROOT / "config" / fname).read_text(encoding="utf-8"), encoding="utf-8")


def _write_valid_node(
    repo_root: Path,
    *,
    ral_id: str,
    taxonomy_code: str,
    title: str,
    epistemic_tier: str = "T1_CANONICAL",
    formal_signature: list[str] | None = None,
    formal_statement: str = r"$$\chi''(\omega) = \frac{\omega}{2 k_B T} S_{AA}(\omega)$$",
    source_id: str = "doi:10.1103/physrev.83.34",
    canonical_doi: str = "10.1103/physrev.83.34",
    ore_payload: bytes | None = None,
    verified_at: str = "2026-11-24T20:14:02Z",
    re_evaluate_after: str = "2028-11-24",
    record_source_ledger: bool = True,
) -> tuple[Path, str]:
    raw_bytes = ore_payload if ore_payload is not None else f"ore:{ral_id}:{source_id}".encode("utf-8")
    ore_sha = hashlib.sha256(raw_bytes).hexdigest()
    ore_uri = build_ore_uri(ore_sha, "tar.gz")
    assumptions = [
        "system is in canonical thermal equilibrium with temperature T > 0",
        "external perturbation satisfies linear response |delta_H| << k_B * T",
    ]
    sigs = formal_signature or [
        "operator:autocorrelation",
        "operator:fourier_transform",
        "symmetry:time_translation",
    ]
    fm = {
        "ral_schema": "2.1",
        "ral_id": ral_id,
        "node_type": "THEOREM",
        "epistemic_tier": epistemic_tier,
        "consensus_version": 1,
        "taxonomy_code": taxonomy_code,
        "title": title,
        "formal_signature": sigs,
        "assumptions": assumptions,
        "assumptions_hash": compute_assumptions_hash(assumptions),
        "verification": {
            "oracle_plugin": "sympy_numeric",
            "oracle_status": "PASS",
            "oracle_script_sha256": compute_oracle_script_sha256(VALID_SYMPY_KERNEL),
            "verified_at": verified_at,
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
                "quote_sha256": compute_quote_sha256(title),
                "quote_char_len": max(10, len(title)),
                "pub_year": 2021,
                "peer_reviewed": True,
                "retracted": False,
            }
        ],
        "re_evaluate_after": re_evaluate_after,
    }
    md = (
        f"---\n{yaml.safe_dump(fm, sort_keys=False).strip()}\n---\n\n"
        f"## Active Consensus (v1 — Effective {verified_at[:10]})\n\n"
        "### Formal Statement\n"
        f"{formal_statement}\n\n"
        "### Failure Boundaries & Known Pathologies\n"
        "- **Breaks when:** Non-equilibrium driving violates detailed balance.\n\n"
        "### Executable Verification Kernel\n"
        f"```python\n{VALID_SYMPY_KERNEL.strip()}\n```\n\n"
        "### Canonical Provenance Anchors\n"
        f"- `{source_id}` (`tex:sec2:eq1`): Verified formal claim.\n\n"
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
        src_path = repo_root / "ledger" / "sources" / f"{verified_at[:7]}.ndjson"
        src_path.parent.mkdir(parents=True, exist_ok=True)
        row = {
            "ore_sha256": ore_sha,
            "canonical_doi": canonical_doi,
            "arxiv_base_id": None,
            "arxiv_version": None,
            "source_license": "CC-BY-4.0",
            "ore_uri": ore_uri,
            "linked_ral_ids": [ral_id],
            "ingested_at": verified_at,
        }
        with src_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    return node_path, ore_sha


# ---------------------------------------------------------------------------
# Offline Fakes (Embedder, GitHub REST Server, Drive Service, Sandbox)
# ---------------------------------------------------------------------------


class _FakeEmbedClient:
    def __init__(self, model_vectors: dict[str, dict[str, list[float]]], dim: int = 768) -> None:
        self.model_vectors = model_vectors
        self.dim = dim
        self.calls: list[tuple[str, Any, dict[str, Any]]] = []
        self.models = self

    def embed_content(self, *, model: str, contents: Any, config: dict[str, Any]) -> Any:
        # Enforce Section 3.3 invariant: single-string invocation only (never list).
        assert isinstance(contents, str), f"embed_content must receive str, got {type(contents)}"
        self.calls.append((model, contents, config))
        by_text = self.model_vectors.get(model, {})
        for key, vec in by_text.items():
            if key in contents:
                return SimpleNamespace(embeddings=[SimpleNamespace(values=list(vec))])
        # Deterministic fallback hash vector
        seed = int(hashlib.sha256(f"{model}:{contents}".encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        v = rng.normal(size=self.dim).astype(np.float64)
        v /= max(float(np.linalg.norm(v)), 1e-8)
        return SimpleNamespace(embeddings=[SimpleNamespace(values=v.tolist())])


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

    def raise_for_status(self) -> None:
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeGitHubServer:
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
        self.inject_429_once = False
        self.corrupt_next_upload = False
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
        if self.inject_429_once:
            self.inject_429_once = False
            return _FakeResp(
                429,
                headers={"X-RateLimit-Remaining": "0", "Retry-After": "2"},
                text="secondary rate limit exceeded",
            )

        prefix = f"https://api.github.com/repos/{self.owner}/{self.repo}"

        # Release by tag
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

        # Create release
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

        # List release assets
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

        # Upload release asset
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
            if any(self.assets_by_id[aid]["name"] == name for aid in rel["asset_ids"] if aid in self.assets_by_id):
                return _FakeResp(422, json_data={"message": "already_exists"})
            self.next_id += 1
            self.clock += 1
            aid = self.next_id
            stored = b"CORRUPTED_TRANSFER" if self.corrupt_next_upload else payload
            if self.corrupt_next_upload:
                self.corrupt_next_upload = False
            asset = {
                "id": aid,
                "release_id": rid,
                "name": name,
                "bytes": stored,
                "created_at": f"2026-11-24T00:00:{self.clock:02d}Z",
                "browser_download_url": f"https://github.com/{self.owner}/{self.repo}/releases/download/{rel['tag_name']}/{name}?id={aid}",
            }
            self.assets_by_id[aid] = asset
            rel["asset_ids"].append(aid)
            # Report original size/digest if simulating download-time corruption
            return _FakeResp(201, json_data=self._serialize_asset(asset))

        # Download asset
        if method == "GET" and "/releases/download/" in url:
            aid_m = re.search(r"id=(\d+)$", url)
            if aid_m and int(aid_m.group(1)) in self.assets_by_id:
                asset = self.assets_by_id[int(aid_m.group(1))]
                return _FakeResp(200, content=asset["bytes"])
            return _FakeResp(404)

        # Delete asset
        m_del = re.match(rf"^{re.escape(prefix)}/releases/assets/(\d+)$", url)
        if method == "DELETE" and m_del:
            aid = int(m_del.group(1))
            asset = self.assets_by_id.pop(aid, None)
            if asset is None:
                return _FakeResp(404)
            for rel in self.releases.values():
                if aid in rel["asset_ids"]:
                    rel["asset_ids"].remove(aid)
            return _FakeResp(204)

        # Issues list / create / patch
        if method == "GET" and url == f"{prefix}/issues":
            params = kwargs.get("params") or {}
            req_label = params.get("labels")
            req_state = params.get("state", "open")
            matches = [
                iss
                for iss in self.issues.values()
                if iss["state"] == req_state
                and (not req_label or req_label in iss["labels"])
            ]
            return _FakeResp(200, json_data=matches)

        if method == "POST" and url == f"{prefix}/issues":
            body = kwargs["json"]
            num = self.next_issue_num
            self.next_issue_num += 1
            issue = {
                "number": num,
                "title": body["title"],
                "body": body["body"],
                "labels": list(body.get("labels") or []),
                "state": "open",
            }
            self.issues[num] = issue
            return _FakeResp(201, json_data=issue)

        m_patch = re.match(rf"^{re.escape(prefix)}/issues/(\d+)$", url)
        if method == "PATCH" and m_patch:
            num = int(m_patch.group(1))
            issue = self.issues[num]
            body = kwargs["json"]
            if "title" in body:
                issue["title"] = body["title"]
            if "body" in body:
                issue["body"] = body["body"]
            return _FakeResp(200, json_data=issue)

        raise AssertionError(f"Unhandled fake GitHub call: {method} {url}")


class _FakeDriveExec:
    def __init__(self, fn: Any) -> None:
        self._fn = fn

    def execute(self) -> Any:
        return self._fn()


class _FakeDriveFiles:
    def __init__(self, svc: FakeDriveGCService) -> None:
        self._svc = svc

    def list(self, **kwargs: Any) -> _FakeDriveExec:
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
                        "createdTime": item.get("createdTime"),
                        "modifiedTime": item.get("modifiedTime"),
                    }
                )
            return {"files": out[: kwargs.get("pageSize", 100)]}

        return _FakeDriveExec(_run)

    def create(self, **kwargs: Any) -> _FakeDriveExec:
        body = kwargs["body"]
        media = kwargs.get("media_body")

        def _run() -> dict[str, Any]:
            self._svc.counter += 1
            fid = f"drive-{self._svc.counter}"
            data = (
                media.getbytes(0, media.size())
                if media is not None and hasattr(media, "size") and hasattr(media, "getbytes")
                else b""
            )
            md5 = hashlib.md5(data).hexdigest() if body.get("mimeType") != DRIVE_FOLDER_MIME else None
            ts = "2026-11-24T00:00:00Z"
            self._svc.items[fid] = {
                "id": fid,
                "name": body["name"],
                "parents": list(body.get("parents", [])),
                "mimeType": body.get("mimeType", "application/octet-stream"),
                "data": data,
                "md5Checksum": md5,
                "createdTime": ts,
                "modifiedTime": ts,
            }
            return {"id": fid, "md5Checksum": md5, "size": str(len(data))}

        return _FakeDriveExec(_run)

    def update(self, **kwargs: Any) -> _FakeDriveExec:
        fid = kwargs["fileId"]
        add_p = kwargs.get("addParents", "")
        rem_p = kwargs.get("removeParents", "")

        def _run() -> dict[str, Any]:
            item = self._svc.items[fid]
            parents = list(item["parents"])
            for p in rem_p.split(","):
                if p and p in parents:
                    parents.remove(p)
            for p in add_p.split(","):
                if p and p not in parents:
                    parents.append(p)
            item["parents"] = parents
            return {"id": fid, "parents": parents}

        return _FakeDriveExec(_run)

    def get(self, **kwargs: Any) -> _FakeDriveExec:
        fid = kwargs["fileId"]
        return _FakeDriveExec(lambda: {"id": fid, "md5Checksum": self._svc.items[fid].get("md5Checksum")})

    def get_media(self, **kwargs: Any) -> _FakeDriveExec:
        fid = kwargs["fileId"]
        return _FakeDriveExec(lambda: self._svc.items[fid]["data"])

    def delete(self, **kwargs: Any) -> _FakeDriveExec:
        fid = kwargs["fileId"]
        return _FakeDriveExec(lambda: self._svc.items.pop(fid, None))


class FakeDriveGCService:
    def __init__(self, root_id: str = "root-warehouse-id") -> None:
        self.counter = 0
        self.items: dict[str, dict[str, Any]] = {
            root_id: {
                "id": root_id,
                "name": "RAL_Ore_Warehouse",
                "parents": [],
                "mimeType": DRIVE_FOLDER_MIME,
                "data": b"",
                "createdTime": "2026-10-01T00:00:00Z",
                "modifiedTime": "2026-10-01T00:00:00Z",
            }
        }

    def files(self) -> _FakeDriveFiles:
        return _FakeDriveFiles(self)


# ---------------------------------------------------------------------------
# Acceptance Test 1: allocate_epochs for K in 0..6 & Day-0 (|V_t|=0 -> 6 explore)
# ---------------------------------------------------------------------------


def test_allocate_epochs_all_k_and_day0_invariants(tmp_path: Path) -> None:
    params = load_curriculum_params()
    # K == 0 and negative K must return all zeros safely without ZeroDivisionError.
    assert allocate_epochs({"exploit": 0.5, "explore": 0.5, "audit": 0.2, "synth": 0.1}, 0) == {
        "exploit": 0,
        "explore": 0,
        "audit": 0,
        "synth": 0,
    }
    assert allocate_epochs({}, -1) == {"exploit": 0, "explore": 0, "audit": 0, "synth": 0}

    # Degenerate zero weights safely fall back to explore when K > 0.
    for k in range(1, 7):
        alloc_zero = allocate_epochs({"exploit": 0.0, "explore": 0.0, "audit": 0.0, "synth": 0.0}, k)
        assert alloc_zero == {"exploit": 0, "explore": k, "audit": 0, "synth": 0}

    # Non-degenerate proportional Hamilton allocation across all K in {0..6}.
    sample_weights = {"exploit": 0.35, "explore": 0.30, "audit": 0.20, "synth": 0.15}
    for k in range(0, 7):
        alloc = allocate_epochs(sample_weights, k)
        assert set(alloc.keys()) == {"exploit", "explore", "audit", "synth"}
        assert all(isinstance(v, int) and v >= 0 for v in alloc.values())
        assert sum(alloc.values()) == k

    # Day-0 invariant: |V_t| == 0, K == 6 -> 100% explore ({"explore": 6, ...}).
    day0_weights = compute_portfolio_weights(
        total_verified_nodes=0,
        macro_counts={},
        total_sigma=0.0,
        global_oracle_pass_rate=1.0,
        params=params,
    )
    assert day0_weights["exploit"] == 0.0
    assert day0_weights["audit"] == 0.0
    assert day0_weights["synth"] == 0.0
    assert day0_weights["explore"] == pytest.approx(0.60)
    assert allocate_epochs(day0_weights, 6) == {
        "exploit": 0,
        "explore": 6,
        "audit": 0,
        "synth": 0,
    }

    # Empty repo plan on Day 0 also yields exact 6 explore epochs.
    _init_mini_repo(tmp_path)
    day0_plan = build_nightly_plan(tmp_path, run_date_iso="2026-11-24", available_epochs=6)
    assert day0_plan["total_verified_nodes"] == 0
    assert day0_plan["epoch_allocation"] == {"exploit": 0, "explore": 6, "audit": 0, "synth": 0}
    assert len(day0_plan["epoch_schedule"]) == 6
    assert all(item["mode"] == "explore" for item in day0_plan["epoch_schedule"])

    # Threshold activation checks: |V_t| >= 10 -> exploit, >= 25 -> audit, >= 150 + consistent -> synth.
    w_10 = compute_portfolio_weights(
        total_verified_nodes=10,
        macro_counts={"math": 10},
        total_sigma=4.0,
        global_oracle_pass_rate=1.0,
        params=params,
    )
    assert w_10["exploit"] > 0.0
    assert w_10["audit"] == 0.0
    assert w_10["synth"] == 0.0

    w_25 = compute_portfolio_weights(
        total_verified_nodes=25,
        macro_counts={"math": 25},
        total_sigma=4.0,
        global_oracle_pass_rate=1.0,
        epoch_upgraded=True,
        params=params,
    )
    assert w_25["audit"] == pytest.approx(0.35)
    assert w_25["synth"] == 0.0

    balanced_macros = {m: 25 for m in MACRO_DISCIPLINES}  # 175 nodes across 7 macros
    assert compute_coverage_entropy(balanced_macros) == pytest.approx(1.0)
    w_175_inconsistent = compute_portfolio_weights(
        total_verified_nodes=175,
        macro_counts=balanced_macros,
        total_sigma=0.0,
        global_oracle_pass_rate=0.9,
        embed_space_consistent=False,
        params=params,
    )
    assert w_175_inconsistent["synth"] == 0.0

    w_175_consistent = compute_portfolio_weights(
        total_verified_nodes=175,
        macro_counts=balanced_macros,
        total_sigma=0.0,
        global_oracle_pass_rate=0.9,
        embed_space_consistent=True,
        params=params,
    )
    assert w_175_consistent["synth"] == pytest.approx(0.25 * 1.0 * 0.9)


# ---------------------------------------------------------------------------
# Director Math: Day-0 Tie-Break, Balance Guard, Whitening & Ward Split
# ---------------------------------------------------------------------------


def test_director_tie_break_balance_guard_pagerank_whitening_and_ward_split(tmp_path: Path) -> None:
    params = load_curriculum_params()

    # 1. Day-0 tau_d = +inf => exploration term non-zero; unbiased sha256(date:d) tie-breaking.
    equal_states = [
        DomainState(domain="math.algebra", macro="math", seed_prior=1.0, nights_since_last_work=None),
        DomainState(domain="phys.quantum", macro="phys", seed_prior=1.0, nights_since_last_work=None),
        DomainState(domain="bio.genetics", macro="bio", seed_prior=1.0, nights_since_last_work=None),
        DomainState(domain="chem.kinetics", macro="chem", seed_prior=1.0, nights_since_last_work=None),
    ]
    ranked_a1 = compute_domain_priorities(equal_states, run_date_iso="2026-11-24", params=params)
    ranked_a2 = compute_domain_priorities(equal_states, run_date_iso="2026-11-24", params=params)
    ranked_b = compute_domain_priorities(equal_states, run_date_iso="2026-11-25", params=params)
    assert [p.domain for p in ranked_a1] == [p.domain for p in ranked_a2]
    assert all(p.exploration > 0.0 for p in ranked_a1)
    expected_order_b = sorted(
        [s.domain for s in equal_states],
        key=lambda d: hashlib.sha256(f"2026-11-25:{d}".encode("utf-8")).hexdigest()[:8],
    )
    assert [p.domain for p in ranked_b] == expected_order_b

    # 2. Macro-discipline Balance Guard (>=20% masked, <4% boosted 2x; all-masked fallback).
    bal_states = [
        DomainState(domain="math.analysis", macro="math", trailing_epochs_30d=25.0, half_life_years=20.0),
        DomainState(domain="phys.statmech", macro="phys", trailing_epochs_30d=15.0, half_life_years=20.0),
        DomainState(domain="cs.logic", macro="cs", trailing_epochs_30d=58.0, half_life_years=20.0),
        DomainState(domain="bio.ecology", macro="bio", trailing_epochs_30d=2.0, half_life_years=20.0),
    ]
    bal_ranked = compute_domain_priorities(bal_states, run_date_iso="2026-11-24", params=params)
    by_dom = {p.domain: p for p in bal_ranked}
    assert by_dom["math.analysis"].masked is True
    assert by_dom["cs.logic"].masked is True
    assert by_dom["phys.statmech"].masked is False
    assert by_dom["phys.statmech"].boosted is False
    assert by_dom["bio.ecology"].masked is False
    assert by_dom["bio.ecology"].boosted is True
    assert by_dom["bio.ecology"].effective_score == pytest.approx(by_dom["bio.ecology"].raw_score * 2.0)
    assert bal_ranked[0].domain == "bio.ecology"
    assert [p.masked for p in bal_ranked] == [False, False, True, True]

    # Single-macro all-masked fallback unmasks rather than masking everything.
    single_macro = [DomainState(domain="math.analysis", macro="math", trailing_epochs_30d=10.0)]
    assert compute_domain_priorities(single_macro, run_date_iso="2026-11-24", params=params)[0].masked is False

    # 3. Cross-domain reverse PageRank boosts foundational domains.
    pr = compute_cross_domain_pagerank(
        ["math.analysis", "phys.quantum", "econ.finance"],
        [
            {"src": "N_P1", "dst": "N_M1", "relation": "DEPENDS_ON"},
            {"src": "N_E1", "dst": "N_M1", "relation": "DEPENDS_ON"},
        ],
        {"N_P1": "phys.quantum", "N_E1": "econ.finance", "N_M1": "math.analysis"},
    )
    assert pr["math.analysis"] == pytest.approx(1.0)
    assert pr["phys.quantum"] < pr["math.analysis"]

    # 4. Isotropic All-but-the-Top whitening strips shared 2-PC anisotropy before S(d_i, d_j, t).
    rng = np.random.default_rng(7)
    bias1 = np.zeros(16)
    bias1[0] = 1.0
    bias2 = np.zeros(16)
    bias2[1] = 1.0
    struct_a = np.zeros(16)
    struct_a[2] = 1.0
    struct_b = np.zeros(16)
    struct_b[3] = 1.0
    struct_c = np.zeros(16)
    struct_c[4] = 1.0
    struct_d = np.zeros(16)
    struct_d[5] = 1.0

    # Two shared nuisance/anisotropy directions (dim 0, dim 1) give all pairs raw cosine > 0.70.
    # Whitening projects out those top 2 PCs, isolating true structural isomorphism (math.a <-> phys.a).
    raw_vecs = {}
    node_doms = {}
    specs = [
        ("M", "math.analysis", struct_a),
        ("P", "phys.statmech", struct_a),
        ("C", "cs.algorithms", struct_b),
        ("E", "eng.control", struct_c),
        ("B", "bio.genetics", struct_d),
    ]
    for prefix_ch, dom_name, s_vec in specs:
        for i in range(4):
            w1 = 4.0 + 1.5 * ((i % 2) * 2 - 1)
            w2 = 2.5 + 1.0 * (((i // 2) % 2) * 2 - 1)
            noise = rng.normal(scale=0.01, size=16)
            raw_vecs[f"{prefix_ch}{i}"] = w1 * bias1 + w2 * bias2 + 1.2 * s_vec + noise
            node_doms[f"{prefix_ch}{i}"] = dom_name

    whitened = whiten_embeddings(raw_vecs, top_pcs=2)
    assert len(whitened) == 20
    synth_cands = score_synthesis_pairs(
        raw_vecs,
        node_doms,
        {
            "math.analysis": 0.9,
            "phys.statmech": 0.9,
            "cs.algorithms": 0.9,
            "eng.control": 0.9,
            "bio.genetics": 0.9,
        },
        bridge_edge_counts={("math.analysis", "phys.statmech"): 0},
        params=params,
    )
    assert len(synth_cands) == 1
    assert synth_cands[0].domain_a == "math.analysis"
    assert synth_cands[0].domain_b == "phys.statmech"
    assert synth_cands[0].score >= 0.35

    # Adding bridge edges reduces unbridged structural-hole factor; low rigor drops candidate.
    bridged = score_synthesis_pairs(
        raw_vecs,
        node_doms,
        {"math.analysis": 0.9, "phys.statmech": 0.9, "cs.algorithms": 0.9, "eng.control": 0.9, "bio.genetics": 0.9},
        bridge_edge_counts={("math.analysis", "phys.statmech"): 4},
        params=params,
    )
    assert all((c.domain_a, c.domain_b) != ("math.analysis", "phys.statmech") for c in bridged)
    low_rigor = score_synthesis_pairs(
        raw_vecs,
        node_doms,
        {"math.analysis": 0.9, "phys.statmech": 0.50, "cs.algorithms": 0.9, "eng.control": 0.9, "bio.genetics": 0.9},
        params=params,
    )
    assert all("phys.statmech" not in (c.domain_a, c.domain_b) for c in low_rigor)

    # 5. Deterministic Ward leaf split when n_d = 42 > 40 & consistent embeddings.
    _init_mini_repo(tmp_path)
    split_ids: list[str] = []
    split_paths: dict[str, Path] = {}
    split_sigs: dict[str, list[str]] = {}
    split_vecs: dict[str, np.ndarray] = {}
    for idx in range(42):
        rid = f"RAL-MATH-{idx:010d}"
        cluster = 0 if idx < 21 else 1
        sig = ["operator:measure", "domain:lebesgue"] if cluster == 0 else ["operator:banach", "domain:hilbert"]
        p, _ = _write_valid_node(
            tmp_path,
            ral_id=rid,
            taxonomy_code="math.analysis",
            title=f"Analysis Theorem {idx}",
            formal_signature=sig,
            source_id=f"doi:10.1000/analysis.{idx}",
            canonical_doi=f"10.1000/analysis.{idx}",
        )
        vec = np.zeros(16, dtype=np.float64)
        vec[cluster] = 1.0
        vec[4 + (idx % 4)] = 0.05
        vec /= np.linalg.norm(vec)
        split_ids.append(rid)
        split_paths[rid] = p
        split_sigs[rid] = sig
        split_vecs[rid] = vec

    # Gate: inconsistent vector space blocks splitting.
    assert (
        plan_leaf_split(
            "math.analysis",
            split_ids,
            split_vecs,
            split_paths,
            split_sigs,
            embed_space_consistent=False,
            params=params,
        )
        is None
    )

    split_plan = plan_leaf_split(
        "math.analysis",
        split_ids,
        split_vecs,
        split_paths,
        split_sigs,
        embed_space_consistent=True,
        params=params,
    )
    assert split_plan is not None
    assert split_plan.k == 2
    assert split_plan.silhouette_score > 0.25

    moved = execute_leaf_split(tmp_path, split_plan, use_git_mv=False)
    assert len(moved) == 42
    for _, dst in moved:
        assert dst.exists()
        res = validate_node_file(dst, check_filename=True)
        assert res.ok, res.errors
        assert res.frontmatter is not None
        assert res.frontmatter.taxonomy_code.startswith("math.analysis.")
    assert validate_vault(tmp_path).ok


# ---------------------------------------------------------------------------
# Acceptance Test 2: Prerequisite Exponential Backoff & Quarantine at k_p = 5
# ---------------------------------------------------------------------------


def test_prereq_backoff_exponential_cooldown_and_quarantine_at_kp5(tmp_path: Path) -> None:
    _init_mini_repo(tmp_path)
    params = load_curriculum_params()
    stub_id = "STUB-MATH-HAHN-BANACH-COMPLEX"
    domain = "math.analysis"

    # Cooldown formula check including 90-night cap at large k_p.
    assert compute_prereq_cooldown(10, 1, params=params) == 12
    assert compute_prereq_cooldown(10, 5, params=params) == 42
    assert compute_prereq_cooldown(10, 7, params=params) == 100  # min(90, 128) == 90

    night = 10
    expected_steps = [(1, 2, "OPEN"), (2, 4, "OPEN"), (3, 8, "OPEN"), (4, 16, "OPEN"), (5, 32, "UNRESOLVABLE_PAYWALLED_OR_MISSING")]
    for exp_k, exp_delta, exp_status in expected_steps:
        stub = record_prereq_attempt_failure(
            tmp_path,
            stub_id,
            domain,
            night,
            cited_by=["RAL-PHYS-01J9X8K2M4"],
            params=params,
        )
        assert stub.attempt_count == exp_k
        assert stub.cooldown_until_night == night + exp_delta
        assert stub.status == exp_status

        if exp_status == "OPEN":
            # Suppressed while inside cooldown window.
            sigma_during_cooldown = compute_epistemic_tension(
                domain, stubs=[stub], current_night=stub.cooldown_until_night - 1, params=params
            )
            assert sigma_during_cooldown == 0.0
            # Decaying positive tension once cooldown expires.
            sigma_after_cooldown = compute_epistemic_tension(
                domain, stubs=[stub], current_night=stub.cooldown_until_night, params=params
            )
            assert sigma_after_cooldown == pytest.approx(math.exp(-0.5 * exp_k))
        else:
            # Quarantined at k_p == 5 => tension contribution drops strictly to 0 forever.
            assert (
                compute_epistemic_tension(
                    domain, stubs=[stub], current_night=stub.cooldown_until_night + 500, params=params
                )
                == 0.0
            )
        night = stub.cooldown_until_night

    # Verify persisted NDJSON state & 180-night passive recheck gate.
    loaded = load_prereq_debt(tmp_path / "ledger" / "prereq_debt.ndjson")
    assert loaded[stub_id].status == "UNRESOLVABLE_PAYWALLED_OR_MISSING"
    assert loaded[stub_id].attempt_count == 5
    last_n = int(loaded[stub_id].last_attempt_night or 0)
    assert is_eligible_for_passive_recheck(loaded[stub_id], last_n + 179, params=params) is False
    assert is_eligible_for_passive_recheck(loaded[stub_id], last_n + 180, params=params) is True

    # Opportunistic resolution transitions status -> RESOLVED and vault CI validates cleanly.
    resolved = resolve_prereq_stub(tmp_path, stub_id, last_n + 180)
    assert resolved is not None and resolved.status == "RESOLVED"
    assert validate_vault(tmp_path).ok


# ---------------------------------------------------------------------------
# Acceptance Test 3: Embedding Mismatch Halts Overwrite & Staged vectors_next Swap
# ---------------------------------------------------------------------------


def test_index_builder_vector_space_guard_and_atomic_vectors_next_migration(tmp_path: Path) -> None:
    _init_mini_repo(tmp_path)
    _write_valid_node(
        tmp_path,
        ral_id="RAL-PHYS-0000000001",
        taxonomy_code="phys.statistical-mechanics",
        title="Classical Fluctuation-Dissipation Theorem",
        formal_signature=["operator:autocorrelation", "regime:linear_response"],
        source_id="doi:10.1103/fdt.1",
        canonical_doi="10.1103/fdt.1",
    )
    _write_valid_node(
        tmp_path,
        ral_id="RAL-MATH-0000000002",
        taxonomy_code="math.analysis",
        title="Spectral Measure Decomposition",
        formal_signature=["operator:fourier_transform", "space:hilbert"],
        formal_statement=r"$$S_{AA}(\omega) = \int e^{-i\omega t} C_{AA}(t)\, dt$$",
        source_id="doi:10.1103/spec.2",
        canonical_doi="10.1103/spec.2",
    )
    edges_path = tmp_path / "graph" / "edges" / "2026-11.ndjson"
    edges_path.parent.mkdir(parents=True, exist_ok=True)
    edges_path.write_text(
        json.dumps(
            {
                "edge_id": "E-01TEST0001",
                "src": "RAL-PHYS-0000000001",
                "dst": "RAL-MATH-0000000002",
                "relation": "DEPENDS_ON",
                "confidence": 0.95,
                "evidence_sha256": "a" * 64,
                "created_at": "2026-11-24T20:15:00Z",
                "created_by_model": "gemini-2.5-pro",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert validate_vault(tmp_path).ok

    v_fdt_m1 = [1.0] + [0.0] * 767
    v_spec_m1 = [0.0, 1.0] + [0.0] * 766
    v_fdt_m2 = [0.0, 0.0, 1.0] + [0.0] * 765
    v_spec_m2 = [0.0, 0.0, 0.0, 1.0] + [0.0] * 764

    client = _FakeEmbedClient(
        {
            "gemini-embedding-2": {
                "autocorrelation": v_fdt_m1,
                "hilbert": v_spec_m1,
            },
            "gemini-embedding-001": {
                "autocorrelation": v_fdt_m2,
                "hilbert": v_spec_m2,
            },
        },
        dim=768,
    )
    db_path = tmp_path / ".ral_cache" / "index.sqlite"

    # Initial compile with gemini-embedding-2.
    res0 = build_or_update_index(
        tmp_path,
        db_path,
        embed_client=client,
        discovered_embed_model="gemini-embedding-2",
    )
    assert res0.vector_action == "INITIALIZED"
    assert res0.embed_space_consistent is True
    assert res0.embedding_model_id == "gemini-embedding-2"
    assert res0.node_count == 2
    assert res0.edge_count == 1
    assert len(client.calls) == 2

    def _snapshot_vectors() -> list[tuple[str, str, bytes]]:
        conn = sqlite3.connect(str(db_path))
        try:
            return conn.execute(
                "SELECT ral_id, embedding_model_id, vec_int8 FROM vectors ORDER BY ral_id"
            ).fetchall()
        finally:
            conn.close()

    baseline_vectors = _snapshot_vectors()

    # Nights 1 & 2: mismatched model (gemini-embedding-001) halts vector overwrites!
    calls_before = len(client.calls)
    for night_idx in (1, 2):
        res_grace = build_or_update_index(
            tmp_path,
            db_path,
            embed_client=client,
            discovered_embed_model="gemini-embedding-001",
        )
        assert res_grace.vector_action == "HALTED_MISMATCH_GRACE"
        assert res_grace.mismatch_nights == night_idx
        assert res_grace.embed_space_consistent is False
        assert res_grace.vectors_written == 0
        assert len(client.calls) == calls_before  # Zero embed calls made during grace!
        assert _snapshot_vectors() == baseline_vectors  # Existing vectors completely untouched!
        assert read_index_meta(db_path)["embedding_model_id"] == "gemini-embedding-2"

    # During mismatch grace window, hybrid search seamlessly falls back to FTS5 BM25 + Jaccard.
    hits = hybrid_search(
        db_path,
        "Fluctuation-Dissipation",
        formal_signature_tags=["operator:autocorrelation"],
        query_vector=v_fdt_m2,
        query_embed_model_id="gemini-embedding-001",
    )
    assert hits and hits[0]["ral_id"] == "RAL-PHYS-0000000001"
    assert hits[0]["used_vector"] is False

    # Night 3 (grace == 3): batched staged migration via vectors_next (batch size = 1).
    res_mig_1 = build_or_update_index(
        tmp_path,
        db_path,
        embed_client=client,
        discovered_embed_model="gemini-embedding-001",
        migration_batch_size=1,
    )
    assert res_mig_1.vector_action == "MIGRATING_STAGED"
    assert res_mig_1.staged_count == 1
    assert res_mig_1.embed_space_consistent is False
    assert _snapshot_vectors() == baseline_vectors  # Still untouched while 1/2 staged!

    # Next batch completes 2/2 and atomically swaps vectors_next -> vectors.
    res_mig_2 = build_or_update_index(
        tmp_path,
        db_path,
        embed_client=client,
        discovered_embed_model="gemini-embedding-001",
        migration_batch_size=1,
    )
    assert res_mig_2.vector_action == "MIGRATION_SWAPPED"
    assert res_mig_2.embed_space_consistent is True
    assert res_mig_2.embedding_model_id == "gemini-embedding-001"

    vecs, consistent, model_id = load_node_embeddings(db_path)
    assert consistent is True
    assert model_id == "gemini-embedding-001"
    assert float(np.dot(vecs["RAL-PHYS-0000000001"], np.asarray(v_fdt_m2))) > 0.999

    conn = sqlite3.connect(str(db_path))
    try:
        has_next = conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='vectors_next'"
        ).fetchone()
        assert has_next is None
    finally:
        conn.close()

    # Zstd compression/decompression & SHA-256 tamper guard.
    zst_path, digest, _ = compress_sqlite_zst(db_path, tmp_path / "out" / "ral-index.sqlite.zst")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        decompress_sqlite_zst(zst_path, tmp_path / "out" / "bad.sqlite", expected_sha256="0" * 64)
    restored = decompress_sqlite_zst(zst_path, tmp_path / "out" / "ok.sqlite", expected_sha256=digest)
    assert read_index_meta(restored)["embedding_model_id"] == "gemini-embedding-001"

    # Smoke test index_builder CLI.
    assert (
        index_builder_main(
            [
                "--repo-root",
                str(tmp_path),
                "--db",
                str(tmp_path / "cli.sqlite"),
                "--compress-out",
                str(tmp_path / "cli.sqlite.zst"),
            ]
        )
        == 0
    )


# ---------------------------------------------------------------------------
# Acceptance Test 4: Blue-Green Release Swap Keeps 2 Snapshots & Backoff
# ---------------------------------------------------------------------------


def test_github_blue_green_release_swap_keeps_two_snapshots_and_rate_limit_backoff(tmp_path: Path) -> None:
    _init_mini_repo(tmp_path)
    _write_valid_node(
        tmp_path,
        ral_id="RAL-PHYS-0000000009",
        taxonomy_code="phys.quantum-mechanics",
        title="Canonical Commutation Relation",
    )
    db_path = tmp_path / "idx.sqlite"
    build_or_update_index(tmp_path, db_path)
    zst_path, zst_sha, _ = compress_sqlite_zst(db_path, tmp_path / "idx.sqlite.zst")

    server = FakeGitHubServer()
    sleeps: list[float] = []
    gh = GitHubClient(
        "rayhan",
        "rayhans-autonomous-lab",
        token="fake-token",
        request_fn=server.request,
        sleep_fn=sleeps.append,
        max_calls_budget=40,
    )

    # Inject a 429 rate-limit response on first swap to verify Retry-After backoff.
    server.inject_429_once = True
    swap1 = gh.blue_green_swap_index(
        zst_path, run_date="2026-11-24", run_id="run1", node_count=1, keep_snapshots=2
    )
    assert 2.0 in sleeps
    assert swap1.kept_snapshots == ("ral-index-20261124-run1.sqlite.zst",)
    assert swap1.pruned_snapshots == ()

    swap2 = gh.blue_green_swap_index(
        zst_path, run_date="2026-11-25", run_id="run2", node_count=1, keep_snapshots=2
    )
    assert len(swap2.kept_snapshots) == 2

    swap3 = gh.blue_green_swap_index(
        zst_path, run_date="2026-11-26", run_id="run3", node_count=1, keep_snapshots=2
    )
    assert swap3.kept_snapshots == (
        "ral-index-20261126-run3.sqlite.zst",
        "ral-index-20261125-run2.sqlite.zst",
    )
    assert swap3.pruned_snapshots == ("ral-index-20261124-run1.sqlite.zst",)
    assert gh.api_call_count < 40  # Well within nightly <40 GitHub API call budget

    rel_assets = [
        server.assets_by_id[aid]["name"]
        for aid in server.releases["nightly-index"]["asset_ids"]
    ]
    assert sorted(rel_assets) == [
        "ral-index-20261125-run2.sqlite.zst",
        "ral-index-20261126-run3.sqlite.zst",
        LATEST_POINTER_NAME,
    ]
    pointer_asset = next(
        a for a in server.assets_by_id.values() if a["name"] == LATEST_POINTER_NAME
    )
    assert len(pointer_asset["bytes"]) <= 512

    # Corrupted upload on run4 MUST abort before updating ral-index-latest.json or deleting run2/run3!
    server.corrupt_next_upload = True
    with pytest.raises(ReleaseIntegrityError):
        gh.blue_green_swap_index(
            zst_path, run_date="2026-11-27", run_id="run4", node_count=1, keep_snapshots=2
        )
    latest_ptr = gh.fetch_latest_index_pointer()
    assert latest_ptr is not None
    assert latest_ptr["asset_name"] == "ral-index-20261126-run3.sqlite.zst"
    assert latest_ptr["sha256"] == zst_sha

    # Download & verify SHA-256 before decompressing.
    dl_db, dl_ptr = gh.download_latest_index(tmp_path / "downloaded.sqlite", cache_dir=tmp_path / "dl")
    assert dl_ptr["run_id"] == "run3"
    assert read_index_meta(dl_db)["compiled_at"] != ""

    # Single Dead-Man's Switch alert issue upsert (never creates duplicate issues).
    i1 = gh.upsert_alert_issue("RAL Alert Day 1", "Drive canary failed")
    i2 = gh.upsert_alert_issue("RAL Alert Day 2", "Still failing")
    assert i1["number"] == i2["number"]
    assert len(server.issues) == 1
    assert CRITICAL_ALERT_LABEL in server.issues[i1["number"]]["labels"]

    # Budget exhaustion guard raises cleanly.
    tight_gh = GitHubClient(
        "rayhan", "rayhans-autonomous-lab", request_fn=server.request, max_calls_budget=1
    )
    tight_gh.get_or_create_release("nightly-index")
    with pytest.raises(GitHubBudgetExceededError):
        tight_gh.get_or_create_release("nightly-index")


# ---------------------------------------------------------------------------
# Acceptance Test 5: 429 RPD Triggers Graceful Zero-LLM Maintenance Pivot
# ---------------------------------------------------------------------------


def test_maintenance_rpd_429_graceful_zero_llm_pivot_retraction_regression_and_drive_gc(
    tmp_path: Path,
) -> None:
    _init_mini_repo(tmp_path)
    now = datetime(2026, 11, 24, 22, 30, tzinfo=timezone.utc)

    # N2 (foundation) & N1 (T1_CANONICAL DEPENDS_ON N2).
    n2_path, ore_n2 = _write_valid_node(
        tmp_path,
        ral_id="RAL-PHYS-0000000022",
        taxonomy_code="phys.statistical-mechanics",
        title="Foundational Susceptibility Lemma",
        source_id="doi:10.1000/retracted.paper",
        canonical_doi="10.1000/retracted.paper",
    )
    n1_path, ore_n1 = _write_valid_node(
        tmp_path,
        ral_id="RAL-MATH-0000000011",
        taxonomy_code="math.analysis",
        title="Derived Linear Response Corollary",
        source_id="doi:10.1000/healthy.paper",
        canonical_doi="10.1000/healthy.paper",
    )
    edges_path = tmp_path / "graph" / "edges" / "2026-11.ndjson"
    edges_path.parent.mkdir(parents=True, exist_ok=True)
    edges_path.write_text(
        json.dumps(
            {
                "edge_id": "E-01DEP00001",
                "src": "RAL-MATH-0000000011",
                "dst": "RAL-PHYS-0000000022",
                "relation": "DEPENDS_ON",
                "confidence": 0.99,
                "evidence_sha256": "b" * 64,
                "created_at": "2026-11-24T20:15:00Z",
                "created_by_model": "gemini-2.5-pro",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    assert validate_vault(tmp_path).ok

    # Seed Fake Drive with:
    # 1. Referenced Ore for N1 & N2 (must be kept)
    # 2. 3-day-old unreferenced orphan (must stay in 7-day grace window)
    # 3. 10-day-old unreferenced orphan (must move to _quarantine_orphans/)
    # 4. 35-day-old already-quarantined orphan (must be purged!)
    drive_svc = FakeDriveGCService()
    drive_store = DriveOreStore(drive_svc, "root-warehouse-id")

    up1 = drive_store.upload_ore_bytes(f"ore:RAL-MATH-0000000011:doi:10.1000/healthy.paper".encode(), "tar.gz")
    up2 = drive_store.upload_ore_bytes(f"ore:RAL-PHYS-0000000022:doi:10.1000/retracted.paper".encode(), "tar.gz")
    assert up1.ore_sha256 == ore_n1
    assert up2.ore_sha256 == ore_n2

    recent_orphan = drive_store.upload_ore_bytes(b"recent-unreferenced-ore", "tar.gz")
    stale_orphan = drive_store.upload_ore_bytes(b"stale-unreferenced-ore-gt7d", "tar.gz")
    drive_svc.items[recent_orphan.file_id]["createdTime"] = (now - timedelta(days=3)).strftime("%Y-%m-%dT%H:%M:%SZ")
    drive_svc.items[stale_orphan.file_id]["createdTime"] = (now - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")

    q_folder_id = drive_store._ensure_folder("_quarantine_orphans", "root-warehouse-id")
    drive_svc.items["old-q-file"] = {
        "id": "old-q-file",
        "name": f"{'c'*64}.tar.gz",
        "parents": [q_folder_id],
        "mimeType": "application/octet-stream",
        "data": b"expired-quarantine",
        "md5Checksum": hashlib.md5(b"expired-quarantine").hexdigest(),
        "createdTime": (now - timedelta(days=40)).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "modifiedTime": (now - timedelta(days=35)).strftime("%Y-%m-%dT%H:%M:%SZ"),
    }

    # Mock air-gapped Docker sandbox runner verifying --network=none and zero host secrets.
    sandbox_cmds: list[list[str]] = []

    def _mock_sandbox_runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        sandbox_cmds.append(cmd)
        assert "--network=none" in cmd
        assert "--read-only" in cmd
        return subprocess.CompletedProcess(cmd, 0, stdout="OK\n", stderr="")

    sandbox = OracleSandbox(runner=_mock_sandbox_runner)

    # Mock Crossref HTTP confirming 10.1000/retracted.paper is retracted, healthy.paper is clean.
    def _mock_crossref(url: str, **kwargs: Any) -> _FakeResp:
        if "10.1000/retracted.paper" in url:
            return _FakeResp(
                200,
                json_data={
                    "message": {
                        "type": "journal-article",
                        "update-to": [{"type": "retraction", "DOI": "10.1000/retraction.notice"}],
                    }
                },
            )
        return _FakeResp(200, json_data={"message": {"type": "journal-article", "title": ["Healthy Paper"]}})

    class _RateLimit429Error(RuntimeError):
        status_code = 429

    router = ModelRouter(
        {
            "status": "OK",
            "model_capability_epoch": 1,
            "roles": {
                "formalizer": ["gemini-2.5-pro", "gemini-2.5-flash"],
                "skeptic": ["gemma-4-27b-it"],
                "scout": ["gemini-2.5-flash"],
            },
            "exhausted_models": [],
        }
    )

    llm_calls = {"count": 0}

    def _failing_epoch_fn(r: ModelRouter) -> None:
        def _always_429(mid: str) -> str:
            llm_calls["count"] += 1
            raise _RateLimit429Error(
                f"429 RESOURCE_EXHAUSTED: daily RPD quota exceeded for {mid}"
            )

        r.call_with_fallback("formalizer", _always_429)

    router_state_file = tmp_path / "handoff" / "router_state.json"
    pivot_res = run_epoch_or_pivot(
        router,
        _failing_epoch_fn,
        repo_root=tmp_path,
        drive_store=drive_store,
        sandbox=sandbox,
        http_get=_mock_crossref,
        router_state_path=router_state_file,
        now=now,
    )

    assert pivot_res.status == "PIVOTED_TO_MAINTENANCE"
    assert pivot_res.pivoted_to_maintenance is True
    assert pivot_res.pivot_reason == "HTTP_429_RPD_EXHAUSTED"
    assert set(pivot_res.exhausted_models) == {"gemini-2.5-pro", "gemini-2.5-flash"}
    assert llm_calls["count"] == 2  # Exactly the two formalizer attempts, zero LLM calls in maintenance!
    assert json.loads(router_state_file.read_text(encoding="utf-8"))["exhausted_models"] == [
        "gemini-2.5-pro",
        "gemini-2.5-flash",
    ]

    rep = pivot_res.maintenance_report
    assert rep is not None
    assert rep.ok is True
    assert rep.llm_calls_made == 0
    assert rep.vault_validation_ok is True

    # Verify Retraction Sweep downgraded N2 -> T4_FALSIFIED and cascaded N1 -> T3_SPECULATIVE.
    assert rep.retraction_sweep is not None
    assert rep.retraction_sweep.retracted_nodes == ("RAL-PHYS-0000000022",)
    assert rep.retraction_sweep.cascaded_t1_demotions == ("RAL-MATH-0000000011",)

    n2_fm, _ = split_frontmatter(n2_path.read_text(encoding="utf-8"))
    assert n2_fm["epistemic_tier"] == "T4_FALSIFIED"
    assert n2_fm["consensus_version"] == 2
    assert n2_fm["provenance"][0]["retracted"] is True
    assert "<details>" in n2_path.read_text(encoding="utf-8")

    n1_fm, _ = split_frontmatter(n1_path.read_text(encoding="utf-8"))
    assert n1_fm["epistemic_tier"] == "T3_SPECULATIVE"
    assert n1_fm["consensus_version"] == 2

    # Verify kernel regression executed inside air-gapped Docker sandbox.
    assert rep.kernel_regression is not None
    assert rep.kernel_regression.kernels_checked == 2
    assert rep.kernel_regression.passed == 2
    assert len(sandbox_cmds) == 2

    # Verify Drive Orphan GC kept referenced + <7d grace orphan, quarantined >7d orphan, purged >=30d.
    assert rep.drive_gc is not None
    assert rep.drive_gc.kept_referenced == 2
    assert rep.drive_gc.kept_grace_window == 1
    assert rep.drive_gc.quarantined == (stale_orphan.file_id,)
    assert rep.drive_gc.purged == ("old-q-file",)
    assert "old-q-file" not in drive_svc.items
    assert q_folder_id in drive_svc.items[stale_orphan.file_id]["parents"]

    # Verify telemetry written & full vault CI passes post-maintenance.
    assert rep.telemetry_path is not None and rep.telemetry_path.exists()
    assert validate_vault(tmp_path).ok

    # Circuit breaker DRIVE_AUTH_FAIL skips LLM & Drive Ore ops completely.
    cb_res = run_epoch_or_pivot(
        router,
        lambda _r: (_ for _ in ()).throw(AssertionError("LLM must not be called under DRIVE_AUTH_FAIL")),
        repo_root=tmp_path,
        drive_store=drive_store,
        sandbox=sandbox,
        circuit_breaker="DRIVE_AUTH_FAIL",
        now=now,
    )
    assert cb_res.status == "CIRCUIT_BREAKER_MAINTENANCE"
    assert cb_res.maintenance_report is not None
    assert cb_res.maintenance_report.drive_gc is None
    assert cb_res.maintenance_report.vault_validation_ok is True

    # Smoke test director & maintenance CLIs.
    assert (
        director_main(
            [
                "--repo-root",
                str(tmp_path),
                "--date",
                "2026-11-24",
                "--output",
                str(tmp_path / "plan.json"),
            ]
        )
        == 0
    )
    assert (tmp_path / "plan.json").exists()
    assert (
        maintenance_main(
            [
                "--repo-root",
                str(tmp_path),
                "--skip-retraction-sweep",
                "--skip-kernel-regression",
                "--no-telemetry",
            ]
        )
        == 0
    )

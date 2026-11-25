"""Batch B static audit regression tests."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import gzip
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
from types import SimpleNamespace
from typing import Any

import pytest

from ci.validate_node import BANNED_FILLER_PHRASES, TAXONOMY_PATTERN, split_frontmatter
from ci.validate_vault import validate_vault
from ral.canonicalize import compute_assumptions_hash, compute_oracle_script_sha256, compute_quote_sha256, sha256_bytes
from ral.model_router import ModelRouter
from ral.orchestrator import (
    _default_fetch_sources_for_epoch,
    execute_epoch_pipeline,
    run_consolidate_job,
)
from ral.pipeline.ast_linter import lint_verification_script
from ral.pipeline.formalizer import (
    VALID_NODE_TYPES,
    FormalizedNodeDraft,
    FormalizerStage,
    normalize_taxonomy_code,
)
from ral.pipeline.oracle_sandbox import OracleSandbox, SandboxConfig
from ral.pipeline.oracles.base import get_default_registry
from ral.pipeline.scout import MAX_SCOUT_TEXT_CHARS, ScoutCandidate, ScoutStage, prepare_scout_excerpt
from ral.pipeline.skeptic import SkepticStage
from ral.sources import LicenseRejectedError, SourceRecord, normalize_license, require_allowed_license
from ral.sources.arxiv_tex import extract_tex_from_tarball, fetch_arxiv_source
from ral.sources.openalex import _extract_arxiv_id_from_work, search_oa_works
from ral.sources.pmc_jats import fetch_pmc_jats, parse_jats_xml
from ral.storage.drive_ore import DRIVE_FOLDER_MIME, DriveOreStore, build_ore_uri

REPO_ROOT = Path(__file__).resolve().parent.parent

VALID_SYMPY_KERNEL = """# RAL_ORACLE: sympy_numeric | TIMEOUT: 15s
import numpy as np

for val in (-2.5, 0.0, 3.5):
    assert np.isclose((val + 1.0) ** 2, val ** 2 + 2.0 * val + 1.0)
assert np.isclose((3.0 - 1.0) * (3.0 + 1.0), 3.0 ** 2 - 1.0)
assert np.isclose((4.0 + 1.0) ** 2, 25.0)
"""


class _FakeResp:
    def __init__(self, *, json_data: Any = None, content: bytes = b"", text: str = "") -> None:
        self._json = json_data
        self.content = content if content else text.encode("utf-8")
        self.text = text or content.decode("utf-8", errors="replace")

    def raise_for_status(self) -> None:
        return None

    def json(self) -> Any:
        return self._json


def _make_tar_gz(files: dict[str, str], *, symlinks: dict[str, str] | None = None) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tf:
        for name, text in files.items():
            raw = text.encode("utf-8")
            info = tarfile.TarInfo(name=name)
            info.size = len(raw)
            tf.addfile(info, io.BytesIO(raw))
        for name, target in (symlinks or {}).items():
            info = tarfile.TarInfo(name=name)
            info.type = tarfile.SYMTYPE
            info.linkname = target
            tf.addfile(info)
    return buf.getvalue()


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
        (cfg_dir / fname).write_text((REPO_ROOT / "config" / fname).read_text(encoding="utf-8"), encoding="utf-8")


def _local_sandbox() -> OracleSandbox:
    def _runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        assert "--network=none" in cmd
        assert "--read-only" in cmd
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


# ---------------------------------------------------------------------------
# 1. Sources: License Normalization, OpenAlex, PMC JATS, arXiv & Fallbacks
# ---------------------------------------------------------------------------


def test_batch_b_license_normalization_and_openalex_arxiv_extraction() -> None:
    assert normalize_license("cc-by") == "CC-BY-4.0"
    assert normalize_license("CC-BY") == "CC-BY-4.0"
    assert normalize_license("cc-by-sa") == "CC-BY-SA-4.0"
    assert normalize_license("CC BY-SA") == "CC-BY-SA-4.0"

    allow = {"CC-BY-4.0", "CC-BY-SA-4.0"}
    assert require_allowed_license("cc-by", allow) == "CC-BY-4.0"
    assert require_allowed_license("cc-by-sa", allow) == "CC-BY-SA-4.0"
    for bad in ("cc-by-nc", "cc-by-nd", "cc-by-nc-sa", "cc-by-nc-4.0"):
        with pytest.raises(LicenseRejectedError):
            require_allowed_license(bad, allow)

    assert _extract_arxiv_id_from_work({"doi": "https://doi.org/10.48550/arXiv.2305.12345"}) == "2305.12345"
    assert _extract_arxiv_id_from_work({"ids": {"doi": "10.48550/arxiv.hep-th/9901001v2"}}) == "hep-th/9901001v2"
    assert (
        _extract_arxiv_id_from_work(
            {"locations": [{"landing_page_url": None, "pdf_url": "https://arxiv.org/pdf/2401.09876v1.pdf"}]}
        )
        == "2401.09876v1"
    )
    assert (
        _extract_arxiv_id_from_work(
            {"locations": [{"landing_page_url": "https://arxiv.org/abs/math.GT/0601001"}]}
        )
        == "math.GT/0601001"
    )

    seen_params: list[dict[str, Any]] = []
    oa_payload = {
        "results": [
            {
                "id": "https://openalex.org/W-CCBY",
                "title": "Short CC-BY Paper",
                "doi": "https://doi.org/10.48550/arXiv.2305.12345",
                "publication_year": 2023,
                "cited_by_count": 40,
                "is_retracted": False,
                "type": "article",
                "best_oa_location": {"license": "cc-by", "pdf_url": "https://arxiv.org/pdf/2305.12345.pdf"},
                "primary_location": {"source": {"type": "repository"}},
            },
            {
                "id": "https://openalex.org/W-NC",
                "title": "Rejected NC Paper",
                "doi": "10.1000/nc",
                "publication_year": 2023,
                "cited_by_count": 99,
                "is_retracted": False,
                "best_oa_location": {"license": "cc-by-nc"},
            },
        ]
    }

    def _mock_get(url: str, **kwargs: Any) -> _FakeResp:
        seen_params.append(dict(kwargs.get("params") or {}))
        return _FakeResp(json_data=oa_payload)

    works = search_oa_works("spectral norm", allowed_licenses=allow, http_get=_mock_get)
    assert seen_params[0]["per-page"] == 25
    assert len(works) == 1
    assert works[0].source_license == "CC-BY-4.0"
    assert works[0].arxiv_id == "2305.12345"


def test_batch_b_default_source_fetcher_fallbacks_and_error_logging(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    _init_mini_repo(tmp_path)
    seen_urls: list[tuple[str, dict[str, Any]]] = []
    tar_bytes = _make_tar_gz(
        {
            "main.tex": (
                "\\documentclass{article}\n\\begin{document}\n"
                "\\section{Theorem}\nEnergy is bounded.\n"
                "\\end{document}\n"
            )
        }
    )
    atom_xml = (
        b'<?xml version="1.0" encoding="UTF-8"?>'
        b'<feed xmlns="http://www.w3.org/2005/Atom">'
        b"<entry>"
        b"<id>http://arxiv.org/abs/2301.00001v1</id>"
        b"<published>2023-01-02T00:00:00Z</published>"
        b"<title>Fallback ArXiv Theorem</title>"
        b"<summary>Spectral identity summary.</summary>"
        b"</entry></feed>"
    )

    # 1. OpenAlex errors -> direct arXiv API fallback -> NIST reference_dbs fallback; errors logged to stderr & summary.
    def _mock_http(url: str, **kwargs: Any) -> _FakeResp:
        seen_urls.append((url, dict(kwargs.get("params") or {})))
        if "api.openalex.org" in url:
            raise RuntimeError("OpenAlex 503 temporary failure")
        if "export.arxiv.org/api/query" in url:
            return _FakeResp(content=atom_xml)
        if "arxiv.org/e-print/2301.00001v1" in url:
            return _FakeResp(content=tar_bytes)
        raise RuntimeError(f"Unexpected HTTP call: {url}")

    errors: list[str] = []
    recs = _default_fetch_sources_for_epoch(
        "explore",
        "phys.statistical-mechanics",
        tmp_path,
        http_get=_mock_http,
        max_sources=2,
        errors_out=errors,
    )
    assert any("api.openalex.org" in u and p.get("per-page") == 25 for u, p in seen_urls)
    assert len(recs) == 2
    assert recs[0].source_id == "arxiv:2301.00001v1"
    assert recs[1].source_format == "reference_db"
    assert recs[1].source_id.startswith("nist:codata2022:")
    assert any("OpenAlex search failed" in e for e in errors)
    captured = capsys.readouterr()
    assert "OpenAlex search failed" in captured.err


def test_batch_b_pmc_jats_ext_links_and_license_p() -> None:
    xml_multi_ext = b"""<article article-type="research-article" xmlns:xlink="http://www.w3.org/1999/xlink">
      <front><article-meta>
        <article-id pub-id-type="pmc">PMC111222</article-id>
        <title-group><article-title>Multi ExtLink Article</article-title></title-group>
        <pub-date><year>2023</year></pub-date>
        <permissions>
          <license license-type="open-access">
            <license-p>Published by <ext-link ext-link-type="uri" xlink:href="https://example.org/journal-policy">Journal</ext-link>
            under <ext-link ext-link-type="uri" xlink:href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</ext-link>.</license-p>
          </license>
        </permissions>
      </article-meta></front>
      <body><p>Theorem statement.</p></body>
    </article>"""
    parsed = parse_jats_xml(xml_multi_ext)
    assert parsed["raw_license"] == "https://creativecommons.org/licenses/by/4.0/"
    rec1 = fetch_pmc_jats("PMC111222", http_get=lambda *a, **kw: _FakeResp(content=xml_multi_ext))
    assert rec1.source_license == "CC-BY-4.0"

    xml_prose_only = b"""<article article-type="research-article">
      <front><article-meta>
        <article-id pub-id-type="pmc">PMC333444</article-id>
        <title-group><article-title>Prose CC BY 4.0 Article</article-title></title-group>
        <pub-date><year>2024</year></pub-date>
        <permissions>
          <license license-type="open-access">
            <license-p>This is an open access article distributed under the Creative Commons Attribution 4.0 International License.</license-p>
          </license>
        </permissions>
      </article-meta></front>
      <body><p>Body text.</p></body>
    </article>"""
    rec2 = fetch_pmc_jats("PMC333444", http_get=lambda *a, **kw: _FakeResp(content=xml_prose_only))
    assert rec2.source_license == "CC-BY-4.0"

    xml_nc = b"""<article article-type="research-article">
      <front><article-meta>
        <article-id pub-id-type="pmc">PMC555666</article-id>
        <title-group><article-title>Restricted NC Article</article-title></title-group>
        <pub-date><year>2024</year></pub-date>
        <permissions>
          <license license-type="open-access">
            <license-p>Distributed under Creative Commons Attribution 4.0 (CC BY-NC 4.0).</license-p>
          </license>
        </permissions>
      </article-meta></front>
      <body><p>Body text.</p></body>
    </article>"""
    with pytest.raises(LicenseRejectedError):
        fetch_pmc_jats("PMC555666", http_get=lambda *a, **kw: _FakeResp(content=xml_nc))


def test_batch_b_arxiv_pdf_skip_sandbox_extraction_and_local_guards() -> None:
    with pytest.raises(ValueError, match="PDF-only"):
        fetch_arxiv_source(
            "2301.00001v1",
            http_get=lambda *a, **kw: _FakeResp(content=b"%PDF-1.7\n%binary"),
            allowed_licenses={"arXiv-non-exclusive-1.0"},
        )
    with pytest.raises(ValueError, match="PDF-only"):
        extract_tex_from_tarball(b"%PDF-1.7\n%binary", use_sandbox=False)

    # Path traversal and symlink checks in local bounded extractor.
    traversal_tar = _make_tar_gz({"../evil.tex": "\\documentclass{article}"})
    with pytest.raises(ValueError, match="Unsafe tar member path"):
        extract_tex_from_tarball(traversal_tar, use_sandbox=False)

    symlink_tar = _make_tar_gz({"ok.tex": "\\documentclass{article}"}, symlinks={"link.tex": "/etc/passwd"})
    with pytest.raises(ValueError, match="Unsafe link tar member"):
        extract_tex_from_tarball(symlink_tar, use_sandbox=False)

    # Bounded uncompressed size check (for both tar and gzipped .tex).
    big_tex = "\\documentclass{article}\n\\begin{document}\n" + ("A" * 4096) + "\n\\end{document}\n"
    big_tar = _make_tar_gz({"main.tex": big_tex})
    with pytest.raises(ValueError, match="maximum allowed size"):
        extract_tex_from_tarball(big_tar, max_total_bytes=1024, use_sandbox=False)
    big_gz = gzip.compress(big_tex.encode("utf-8"))
    with pytest.raises(ValueError, match="maximum allowed size"):
        extract_tex_from_tarball(big_gz, max_total_bytes=1024, use_sandbox=False)

    # Sandboxed tar extraction executes _SANDBOX_EXTRACT_SCRIPT inside OracleSandbox when Docker is available.
    sandbox_cmds: list[list[str]] = []

    def _docker_like_runner(cmd: list[str], **kwargs: Any) -> subprocess.CompletedProcess[str]:
        if cmd[1:3] == ["info"] or cmd[1:4] == ["image", "inspect", "ral-sandbox:latest"]:
            return subprocess.CompletedProcess(cmd, 0, stdout="ok", stderr="")
        sandbox_cmds.append(list(cmd))
        assert "--network=none" in cmd
        assert "--read-only" in cmd
        vol_idx = cmd.index("-v") + 1
        host_dir = cmd[vol_idx].rsplit(":/work:ro", 1)[0]
        script_path = Path(host_dir) / "extract_tex.py"
        code = script_path.read_text(encoding="utf-8")
        # Rewrite /work/ mounts to tmpdir so local Python executes the exact sandbox extractor script.
        code = code.replace("/work/", host_dir.replace("\\", "/") + "/")
        script_path.write_text(code, encoding="utf-8")
        return subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            check=False,
        )

    sbx = OracleSandbox(runner=_docker_like_runner)
    valid_tar = _make_tar_gz(
        {
            "main.tex": "\\documentclass{article}\n\\begin{document}\n\\begin{theorem}A=A\\end{theorem}\n\\end{document}\n"
        }
    )
    text, secs = extract_tex_from_tarball(valid_tar, sandbox=sbx, use_sandbox=True)
    assert len(sandbox_cmds) == 1
    assert "A=A" in text
    assert "tex:thm1" in secs


# ---------------------------------------------------------------------------
# 2. Drive Canary supportsAllDrives, Paginated GC & Weekly Sunday Job 7 GC
# ---------------------------------------------------------------------------


class _PaginatedFakeFiles:
    def __init__(self, svc: _PaginatedFakeDriveService) -> None:
        self._svc = svc

    def list(self, **kwargs: Any) -> Any:
        self._svc.list_calls.append(dict(kwargs))
        q = kwargs.get("q", "")
        name_m = __import__("re").search(r"name = '([^']*)'", q)
        parent_m = __import__("re").search(r"'([^']*)' in parents", q)
        mime_m = __import__("re").search(r"mimeType = '([^']*)'", q)
        mime_ne_m = __import__("re").search(r"mimeType != '([^']*)'", q)
        t_name = name_m.group(1) if name_m else None
        t_parent = parent_m.group(1) if parent_m else None
        t_mime = mime_m.group(1) if mime_m else None
        t_not_mime = mime_ne_m.group(1) if mime_ne_m else None
        page_size = int(kwargs.get("pageSize") or 1000)
        offset = int(kwargs.get("pageToken") or 0)

        def _run() -> dict[str, Any]:
            matches = []
            for item in self._svc.items.values():
                if item.get("trashed"):
                    continue
                if t_name is not None and item["name"] != t_name:
                    continue
                if t_parent is not None and t_parent not in item["parents"]:
                    continue
                if t_mime is not None and item.get("mimeType") != t_mime:
                    continue
                if t_not_mime is not None and item.get("mimeType") == t_not_mime:
                    continue
                matches.append(
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
            page = matches[offset : offset + page_size]
            resp: dict[str, Any] = {"files": page}
            if offset + page_size < len(matches):
                resp["nextPageToken"] = str(offset + page_size)
            return resp

        return SimpleNamespace(execute=_run)

    def create(self, **kwargs: Any) -> Any:
        body = kwargs["body"]
        media = kwargs.get("media_body")

        def _run() -> dict[str, Any]:
            self._svc.counter += 1
            fid = f"drive-{self._svc.counter}"
            data = media.getbytes(0, media.size()) if media is not None else b""
            md5 = hashlib.md5(data).hexdigest() if body.get("mimeType") != DRIVE_FOLDER_MIME else None
            ts = "2026-11-29T00:00:00Z"
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

        return SimpleNamespace(execute=_run)

    def update(self, **kwargs: Any) -> Any:
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

        return SimpleNamespace(execute=_run)

    def get(self, **kwargs: Any) -> Any:
        fid = kwargs["fileId"]
        return SimpleNamespace(execute=lambda: {"id": fid, "md5Checksum": self._svc.items[fid].get("md5Checksum")})

    def get_media(self, **kwargs: Any) -> Any:
        self._svc.get_media_calls.append(dict(kwargs))
        fid = kwargs["fileId"]
        return SimpleNamespace(execute=lambda: self._svc.items[fid]["data"])

    def delete(self, **kwargs: Any) -> Any:
        fid = kwargs["fileId"]
        return SimpleNamespace(execute=lambda: self._svc.items.pop(fid, None))


class _PaginatedFakeDriveService:
    def __init__(self, root_id: str = "root-warehouse-id") -> None:
        self.counter = 0
        self.list_calls: list[dict[str, Any]] = []
        self.get_media_calls: list[dict[str, Any]] = []
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

    def files(self) -> _PaginatedFakeFiles:
        return _PaginatedFakeFiles(self)


def test_batch_b_drive_canary_paginated_listing_and_sunday_job7_gc(tmp_path: Path) -> None:
    _init_mini_repo(tmp_path)
    svc = _PaginatedFakeDriveService()
    store = DriveOreStore(svc, "root-warehouse-id")

    canary = store.run_preflight_canary()
    assert canary.ok is True
    assert svc.get_media_calls[-1]["supportsAllDrives"] is True

    # Upload 3 Ore objects across distinct hex prefixes and verify paginated non-folder query.
    u1 = store.upload_ore_bytes(b"ore-1", "tar.gz")
    u2 = store.upload_ore_bytes(b"ore-2", "tar.gz")
    u3 = store.upload_ore_bytes(b"ore-3", "tar.gz")

    svc.list_calls.clear()
    listed = store.list_ore_objects(page_size=2)
    assert {obj["ore_sha256"] for obj in listed} == {u1.ore_sha256, u2.ore_sha256, u3.ore_sha256}
    # 1 lookup for _quarantine_orphans + 2 paginated pages (page_size=2 for 3 items) = 3 calls total.
    assert len(svc.list_calls) == 3
    assert any("mimeType !=" in c.get("q", "") for c in svc.list_calls)

    # Mark u2 as a >7 day orphan and run Job 7 on Tuesday (skip) vs Sunday (run GC).
    sunday_now = datetime(2026, 11, 29, 18, 30, tzinfo=timezone.utc)
    svc.items[u2.file_id]["createdTime"] = (sunday_now - timedelta(days=10)).strftime("%Y-%m-%dT%H:%M:%SZ")

    # Mark u1 and u3 referenced in ledger/sources so only u2 is orphaned.
    src_ledger = tmp_path / "ledger" / "sources" / "2026-11.ndjson"
    src_ledger.parent.mkdir(parents=True, exist_ok=True)
    src_ledger.write_text(
        "\n".join(
            [
                json.dumps({"ore_sha256": u1.ore_sha256, "source_license": "CC-BY-4.0"}),
                json.dumps({"ore_sha256": u3.ore_sha256, "source_license": "CC-BY-4.0"}),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    handoff = tmp_path / "ral_handoff"
    handoff.mkdir(parents=True, exist_ok=True)
    (handoff / "circuit_breaker.json").write_text(json.dumps({"circuit_breaker": None}), encoding="utf-8")

    # Tuesday (2026-11-24) -> skips Drive GC.
    con_tue = run_consolidate_job(
        tmp_path,
        handoff_dir=handoff,
        cache_dir=tmp_path / ".ral_cache_tue",
        run_date_iso="2026-11-24",
        run_id="tue1",
        dry_run=False,
        drive_store=store,
        now=datetime(2026, 11, 24, 18, 30, tzinfo=timezone.utc),
    )
    assert con_tue.drive_gc is None

    # Sunday (2026-11-29) -> runs weekly Drive Orphan GC and quarantines u2.
    con_sun = run_consolidate_job(
        tmp_path,
        handoff_dir=handoff,
        cache_dir=tmp_path / ".ral_cache_sun",
        run_date_iso="2026-11-29",
        run_id="sun1",
        dry_run=False,
        drive_store=store,
        now=sunday_now,
    )
    assert con_sun.drive_gc is not None
    assert con_sun.drive_gc.quarantined == (u2.file_id,)


# ---------------------------------------------------------------------------
# 3. Sandbox /tmp Env, AST Linter, Skeptic min_assertions=1 & Scout Chunking
# ---------------------------------------------------------------------------


def test_batch_b_sandbox_tmp_env_skeptic_min_assertions_and_scout_chunking(tmp_path: Path) -> None:
    _init_mini_repo(tmp_path)
    dockerfile = (REPO_ROOT / "ci" / "sandbox.Dockerfile").read_text(encoding="utf-8")
    for env_kv in (
        "HOME=/tmp",
        "HYPOTHESIS_STORAGE_DIRECTORY=/tmp/.hypothesis",
        "MPLCONFIGDIR=/tmp/matplotlib",
    ):
        assert env_kv in dockerfile

    cmd = OracleSandbox().build_docker_command(tmp_path, "verify.py", timeout_s=15)
    env_flags = [cmd[i + 1] for i, tok in enumerate(cmd) if tok == "-e"]
    assert set(env_flags) == {
        "HOME=/tmp",
        "HYPOTHESIS_STORAGE_DIRECTORY=/tmp/.hypothesis",
        "MPLCONFIGDIR=/tmp/matplotlib",
    }

    # AST linter blocks getattr / setattr / delattr.
    for fn_call in (
        "getattr(1, '__class__')\nx = 1\nassert x > 0\nassert x < 2\nassert x != 3",
        "setattr(object, 'x', 1)\nx = 1\nassert x > 0\nassert x < 2\nassert x != 3",
        "delattr(object, 'x')\nx = 1\nassert x > 0\nassert x < 2\nassert x != 3",
    ):
        assert any(v.startswith("FORBIDDEN_BUILTIN") for v in lint_verification_script(fn_call))

    # Skeptic enforces min_assertions=1 on falsify.py BEFORE invoking Docker.
    router = ModelRouter(
        {
            "status": "OK",
            "model_capability_epoch": 1,
            "roles": {
                "scout": ["gemini-2.5-flash"],
                "formalizer": ["gemini-2.5-pro"],
                "architect": ["gemini-2.5-pro"],
                "skeptic": ["gemma-4-27b-it"],
            },
            "exhausted_models": [],
        }
    )
    draft = FormalizedNodeDraft(
        ral_id="RAL-MATH-01SKEPTIC1",
        node_type="THEOREM",
        taxonomy_code="math.analysis",
        title="Quadratic Positivity",
        formal_signature=("operator:square",),
        assumptions=("x is real",),
        formal_statement_latex="x^2 \\ge 0",
        symbol_rows=({"symbol": "x", "meaning": "scalar", "domain": "\\mathbb{R}", "si_units": "1"},),
        failure_boundaries=("complex numbers",),
        provenance_summary="Square of real is non-negative.",
        oracle_plugin="sympy_numeric",
        verify_py=VALID_SYMPY_KERNEL,
        formalizer_model="gemini-2.5-pro",
        model_capability_epoch=1,
    )
    docker_calls: list[list[str]] = []

    def _recording_runner(c: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        docker_calls.append(c)
        return subprocess.CompletedProcess(c, 0, stdout="", stderr="")

    sbx_record = OracleSandbox(runner=_recording_runner)
    client_zero_assert = SimpleNamespace(
        models=SimpleNamespace(
            generate_content=lambda **kw: SimpleNamespace(
                text=json.dumps(
                    {
                        "attack_found": True,
                        "falsify_py": "print('fake counterexample without assert')\n",
                        "missing_assumptions": [],
                        "summary": "No assertion",
                    }
                )
            )
        )
    )
    verdict = SkepticStage(client_zero_assert, router, sandbox=sbx_record).challenge(draft, "x^2 >= 0")
    assert verdict.survived is True
    assert verdict.counterexample_verified is False
    assert verdict.sandbox_result is not None
    assert "INSUFFICIENT_ASSERTIONS" in verdict.sandbox_result.stderr
    assert docker_calls == []

    # Scout caps/chunks source text <= 24,000 chars and prioritizes theorem/equation/results sections.
    filler_intro = "Introduction background prose without equations. " * 700  # ~34,300 chars
    thm_quote = "Theorem 7: For all real matrices A, tr(A^T A) equals squared Frobenius norm."
    refs_tail = "References: [1] Smith et al. 1999. " * 300
    full_text = f"{filler_intro}\n\n{thm_quote}\n\n{refs_tail}"
    long_source = SourceRecord(
        source_id="arxiv:2401.00001v1",
        source_license="CC-BY-4.0",
        source_format="arxiv_tex",
        title="Long Paper",
        pub_year=2024,
        peer_reviewed=True,
        retracted=False,
        canonical_doi=None,
        arxiv_base_id="2401.00001",
        arxiv_version=1,
        citation_count=10,
        raw_bytes=b"long-source",
        raw_ext="tar.gz",
        extracted_text=full_text,
        sections={
            "tex:sec1": filler_intro,
            "tex:thm1": thm_quote,
            "tex:sec9": f"References\n{refs_tail}",
        },
    )
    excerpt = prepare_scout_excerpt(long_source, max_chars=MAX_SCOUT_TEXT_CHARS)
    assert len(excerpt) <= MAX_SCOUT_TEXT_CHARS
    assert thm_quote in excerpt

    prompts_seen: list[str] = []

    def _scout_gen(**kw: Any) -> Any:
        prompts_seen.append(kw["contents"])
        return SimpleNamespace(
            text=json.dumps(
                {
                    "candidates": [
                        {
                            "title": "Frobenius Trace Identity",
                            "claim_summary": "Frobenius norm squared equals trace of A^T A.",
                            "quote_text": thm_quote,
                            "source_locator": "tex:thm1",
                            "node_type": "THEOREM",
                            "taxonomy_code": "math.analysis",
                            "formal_signature": ["operator:trace", "operator:norm"],
                            "assumptions": ["A is finite-dimensional real matrix"],
                            "recommended_oracle": "sympy_numeric",
                        }
                    ]
                }
            )
        )

    scout = ScoutStage(
        SimpleNamespace(models=SimpleNamespace(generate_content=_scout_gen)),
        router,
        allowlist_path=REPO_ROOT / "config" / "license_allowlist.yaml",
    )
    dec, cands = scout.extract_candidates(long_source, repo_root=tmp_path)
    assert dec.action == "PROCEED"
    assert len(cands) == 1
    assert len(prompts_seen[0]) < len(full_text)
    assert len(prompts_seen[0]) <= MAX_SCOUT_TEXT_CHARS + 1200


# ---------------------------------------------------------------------------
# 4. Formalizer Prompt Constraints, validate_node_content Repair Loop & Correspondence
# ---------------------------------------------------------------------------


def test_batch_b_formalizer_constraints_repair_loop_and_theory_upgrade_correspondence(
    tmp_path: Path,
) -> None:
    _init_mini_repo(tmp_path)
    assert normalize_taxonomy_code("phys.statistical_mechanics") == "phys.statistical-mechanics"

    router = ModelRouter(
        {
            "status": "OK",
            "model_capability_epoch": 2,
            "roles": {
                "scout": ["gemini-2.5-flash"],
                "formalizer": ["gemini-2.5-pro"],
                "architect": ["gemini-2.5-pro"],
                "skeptic": ["gemma-4-27b-it"],
            },
            "exhausted_models": [],
        }
    )
    sbx = _local_sandbox()
    reg = get_default_registry(sandbox=sbx)

    prompts_seen: list[str] = []

    def _repair_node_gen(**kw: Any) -> Any:
        prompt = kw["contents"]
        prompts_seen.append(prompt)
        if "Formalize this verified scientific quote" in prompt:
            for nt in VALID_NODE_TYPES:
                assert nt in prompt
            assert TAXONOMY_PATTERN in prompt
            assert "350" in prompt
            assert BANNED_FILLER_PHRASES[0] in prompt
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "node_type": "INVALID_TYPE",
                        "taxonomy_code": "math.geometry_topology",
                        "title": "Polynomial Square Identity",
                        "formal_signature": ["operator:binomial_expansion"],
                        "assumptions": ["x is a real scalar"],
                        "formal_statement_latex": "(x+1)^2 = x^2 + 2x + 1",
                        "symbol_rows": [{"symbol": "x", "meaning": "real", "domain": "\\mathbb{R}", "si_units": "1"}],
                        "failure_boundaries": ["non-commutative rings"],
                        "provenance_summary": "In conclusion, it is worth noting this holds.",
                        "oracle_plugin": "sympy_numeric",
                        "verify_py": VALID_SYMPY_KERNEL,
                    }
                )
            )
        if "Repair the executable verification script" in prompt:
            assert "NODE_VALIDATION_FAILED" in prompt
            if len(prompts_seen) == 2:
                assert "SCHEMA_ERROR(node_type)" in prompt
                return SimpleNamespace(
                    text=json.dumps(
                        {
                            "node_type": "THEOREM",
                            "taxonomy_code": "math.geometry_topology",
                            "assumptions": ["x is a real scalar"],
                            "verify_py": VALID_SYMPY_KERNEL,
                        }
                    )
                )
            assert "BANNED_FILLER_PHRASE" in prompt
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "node_type": "THEOREM",
                        "taxonomy_code": "math.geometry_topology",
                        "provenance_summary": "Exact binomial expansion over commutative rings.",
                        "assumptions": ["x is a real scalar"],
                        "verify_py": VALID_SYMPY_KERNEL,
                    }
                )
            )
        raise AssertionError(f"Unexpected prompt: {prompt[:80]}")

    src = SourceRecord(
        source_id="arxiv:2402.00001v1",
        source_license="CC-BY-4.0",
        source_format="arxiv_tex",
        title="Polynomial Identity",
        pub_year=2024,
        peer_reviewed=True,
        retracted=False,
        canonical_doi="10.1000/poly.1",
        arxiv_base_id="2402.00001",
        arxiv_version=1,
        citation_count=80,
        raw_bytes=b"poly-v1",
        raw_ext="tar.gz",
        extracted_text="For any real scalar x, (x+1)^2 = x^2 + 2x + 1.",
    )
    cand = ScoutCandidate(
        source_id=src.source_id,
        title="Polynomial Square Identity",
        claim_summary="Binomial square identity",
        quote_text="For any real scalar x, (x+1)^2 = x^2 + 2x + 1.",
        quote_sha256=compute_quote_sha256("For any real scalar x, (x+1)^2 = x^2 + 2x + 1."),
        quote_char_len=45,
        quote_match_ratio=1.0,
        source_locator="tex:sec1",
        node_type="THEOREM",
        taxonomy_code="math.geometry_topology",
        formal_signature=("operator:binomial_expansion",),
        assumptions=("x is a real scalar",),
        recommended_oracle="sympy_numeric",
        dedup_action="PROCEED",
    )
    formalizer = FormalizerStage(
        SimpleNamespace(models=SimpleNamespace(generate_content=_repair_node_gen)),
        router,
        reg,
        oracle_config_path=REPO_ROOT / "config" / "oracle_plugins.yaml",
    )
    draft, res = formalizer.formalize_and_verify(cand, src, max_repairs=2, repo_root=tmp_path)
    assert draft is not None and res.passed is True
    assert draft.node_type == "THEOREM"
    assert draft.taxonomy_code == "math.geometry-topology"
    assert "In conclusion" not in draft.provenance_summary
    assert len(prompts_seen) == 3

    # End-to-end THEORY_UPGRADE inside execute_epoch_pipeline generates correspondence_script and supersedes v1 -> v2.
    upgrade_repo = tmp_path / "upgrade_repo"
    _init_mini_repo(upgrade_repo)
    old_ore = b"arxiv-preprint-v1-bytes"
    old_sha = sha256_bytes(old_ore)
    old_id = "RAL-MATH-01UPGRADE1"
    old_assump = ["x is a real scalar"]
    old_fm = {
        "ral_schema": "2.1",
        "ral_id": old_id,
        "node_type": "THEOREM",
        "epistemic_tier": "T2_ESTABLISHED",
        "consensus_version": 1,
        "taxonomy_code": "math.analysis",
        "title": "Binomial Square v1",
        "formal_signature": ["operator:binomial_expansion"],
        "assumptions": old_assump,
        "assumptions_hash": compute_assumptions_hash(old_assump),
        "verification": {
            "oracle_plugin": "sympy_numeric",
            "oracle_status": "PASS",
            "oracle_script_sha256": compute_oracle_script_sha256(VALID_SYMPY_KERNEL),
            "verified_at": "2026-06-01T00:00:00Z",
            "formalizer_model": "gemini-2.5-pro",
            "skeptic_model": "gemma-4-27b-it",
            "model_capability_epoch": 1,
        },
        "provenance": [
            {
                "source_id": "arxiv:2402.00001v1",
                "source_license": "arXiv-non-exclusive-1.0",
                "source_format": "arxiv_tex",
                "source_locator": "tex:sec1",
                "ore_uri": build_ore_uri(old_sha, "tar.gz"),
                "ore_sha256": old_sha,
                "quote_sha256": compute_quote_sha256("Binomial square v1"),
                "quote_char_len": 18,
                "pub_year": 2024,
                "peer_reviewed": True,
                "retracted": False,
            }
        ],
        "re_evaluate_after": "2027-06-01",
    }
    import yaml

    old_md = (
        f"---\n{yaml.safe_dump(old_fm, sort_keys=False).strip()}\n---\n\n"
        "## Active Consensus (v1 — Effective 2026-06-01)\n\n"
        "### Formal Statement\n$$(x+1)^2 = x^2 + 2x + 1$$\n\n"
        "| Symbol | Meaning | Domain | SI Units |\n| :--- | :--- | :--- | :--- |\n"
        "| $x$ | scalar | $\\mathbb{R}$ | $1$ |\n\n"
        "### Failure Boundaries & Known Pathologies\n- **Breaks when:** non-commutative ring.\n\n"
        "### Executable Verification Kernel\n```python\n"
        f"{VALID_SYMPY_KERNEL.strip()}\n```\n\n"
        "### Canonical Provenance Anchors\n- `arxiv:2402.00001v1` (`tex:sec1`): v1.\n\n"
        "## Active Challenges & Open Anomalies\n- *(None)*\n\n"
        "## Superseded & Archived\n*(None)*\n"
    )
    node_path = upgrade_repo / "vault" / "math" / "analysis" / f"{old_id}.md"
    node_path.parent.mkdir(parents=True, exist_ok=True)
    node_path.write_text(old_md, encoding="utf-8")

    src_ledger = upgrade_repo / "ledger" / "sources" / "2026-06.ndjson"
    src_ledger.parent.mkdir(parents=True, exist_ok=True)
    src_ledger.write_text(
        json.dumps(
            {
                "ore_sha256": old_sha,
                "canonical_doi": None,
                "arxiv_base_id": "2402.00001",
                "arxiv_version": 1,
                "source_license": "arXiv-non-exclusive-1.0",
                "ore_uri": build_ore_uri(old_sha, "tar.gz"),
                "linked_ral_ids": [old_id],
                "ingested_at": "2026-06-01T00:00:00Z",
            }
        )
        + "\n",
        encoding="utf-8",
    )

    v2_source = SourceRecord(
        source_id="arxiv:2402.00001v2",
        source_license="CC-BY-4.0",
        source_format="arxiv_tex",
        title="Binomial Identity (Peer Reviewed)",
        pub_year=2023,
        peer_reviewed=True,
        retracted=False,
        canonical_doi="10.1000/binomial.peer",
        arxiv_base_id="2402.00001",
        arxiv_version=2,
        citation_count=90,
        raw_bytes=b"arxiv-peer-reviewed-v2-bytes",
        raw_ext="tar.gz",
        extracted_text="For any real scalar x, (x+1)^2 = x^2 + 2x + 1 and (x-1)*(x+1) = x^2 - 1.",
    )

    corr_generated = {"called": False}

    def _upgrade_gen(**kw: Any) -> Any:
        prompt = kw["contents"]
        if "zero-tool primary-source scientific claim extractor" in prompt:
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "candidates": [
                            {
                                "title": "Binomial Identity (Peer Reviewed)",
                                "claim_summary": "Generalized binomial identities.",
                                "quote_text": "For any real scalar x, (x+1)^2 = x^2 + 2x + 1 and (x-1)*(x+1) = x^2 - 1.",
                                "source_locator": "tex:sec2",
                                "node_type": "THEOREM",
                                "taxonomy_code": "math.analysis",
                                "formal_signature": ["operator:binomial_expansion"],
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
                        "taxonomy_code": "math.analysis",
                        "title": "Binomial Identity (Peer Reviewed)",
                        "formal_signature": ["operator:binomial_expansion"],
                        "assumptions": ["x is a real scalar"],
                        "formal_statement_latex": "(x+1)^2 = x^2 + 2x + 1",
                        "symbol_rows": [{"symbol": "x", "meaning": "scalar", "domain": "\\mathbb{R}", "si_units": "1"}],
                        "failure_boundaries": ["non-commutative rings"],
                        "provenance_summary": "Peer-reviewed binomial identities.",
                        "oracle_plugin": "sympy_numeric",
                        "verify_py": VALID_SYMPY_KERNEL,
                    }
                )
            )
        if "Assume this formalization has a hidden defect" in prompt:
            return SimpleNamespace(
                text=json.dumps({"attack_found": False, "falsify_py": "", "missing_assumptions": [], "summary": "OK"})
            )
        if "correspondence_script" in prompt:
            corr_generated["called"] = True
            return SimpleNamespace(
                text=json.dumps(
                    {
                        "correspondence_script": (
                            "import numpy as np\n"
                            "xs = np.array([-1.0, 0.5, 2.0])\n"
                            "assert np.allclose((xs + 1.0) ** 2, xs ** 2 + 2.0 * xs + 1.0)\n"
                        )
                    }
                )
            )
        raise AssertionError(f"Unexpected prompt: {prompt[:80]}")

    drive_svc = _PaginatedFakeDriveService()
    drive_store = DriveOreStore(drive_svc, "root-warehouse-id")
    # Upload old Ore so vault stays consistent on Drive if checked, and run epoch pipeline.
    drive_store.upload_ore_bytes(old_ore, "tar.gz")
    out = execute_epoch_pipeline(
        router,
        repo_root=upgrade_repo,
        mode="explore",
        target="math.analysis",
        current_night=1,
        ore_cache_dir=tmp_path / "ore_cache_up",
        gemini_client=SimpleNamespace(models=SimpleNamespace(generate_content=_upgrade_gen)),
        drive_store=drive_store,
        sandbox=sbx,
        source_fetcher=lambda _m, _t, _r: [v2_source],
        dry_run=False,
        now=datetime(2026, 11, 25, 4, 0, 0, tzinfo=timezone.utc),
    )
    assert corr_generated["called"] is True
    assert out["nodes_upgraded"] == [old_id]
    fm_after, _ = split_frontmatter(node_path.read_text(encoding="utf-8"))
    assert fm_after["consensus_version"] == 2
    assert fm_after["epistemic_tier"] == "T1_CANONICAL"
    assert validate_vault(upgrade_repo).ok is True

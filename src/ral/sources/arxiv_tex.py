"""Native arXiv `.tex` e-print tarball fetcher and LaTeX section parser (Section 6.3)."""

from __future__ import annotations

import gzip
import io
import json
import re
import shutil
import tarfile
from typing import TYPE_CHECKING, Any, Callable, Mapping
import xml.etree.ElementTree as ET

from ral.sources import (
    RetractedSourceError,
    SourceRecord,
    default_http_get,
    require_allowed_license,
)

if TYPE_CHECKING:
    from ral.pipeline.oracle_sandbox import OracleSandbox

ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"
ARXIV_API_URL = "https://export.arxiv.org/api/query"
MAX_TEX_UNCOMPRESSED_BYTES = 16 * 1024 * 1024
_DEFAULT_DOCKER_SANDBOX_READY: bool | None = None

_ARXIV_ID_RE = re.compile(
    r"^(?:arxiv:)?(?:https?://arxiv\.org/(?:abs|pdf|e-print)/)?([a-z\-]+(?:\.[A-Z]{2})?/\d{7}|\d{4}\.\d{4,5})(?:v(\d+))?(?:\.pdf)?$",
    re.IGNORECASE,
)
_TEX_SECTION_RE = re.compile(r"\\(?:section|subsection)\*?\{([^}]+)\}")
_TEX_ENV_RE = re.compile(
    r"\\begin\{(theorem|lemma|proposition|corollary|equation\*?|align\*?)\}(.*?)\\end\{\1\}",
    re.DOTALL,
)


def parse_arxiv_id(raw_id: str) -> tuple[str, int | None]:
    """Parse arXiv identifier into `(arxiv_base_id, optional_version)`."""
    cleaned = raw_id.strip()
    match = _ARXIV_ID_RE.match(cleaned)
    if not match:
        raise ValueError(f"Unrecognized arXiv identifier: {raw_id!r}")
    base_id = match.group(1)
    ver = int(match.group(2)) if match.group(2) is not None else None
    return base_id, ver


def _parse_tex_sections(full_tex: str) -> dict[str, str]:
    sections: dict[str, str] = {}
    sec_matches = list(_TEX_SECTION_RE.finditer(full_tex))
    for idx, match in enumerate(sec_matches, start=1):
        start = match.end()
        end = sec_matches[idx].start() if idx < len(sec_matches) else len(full_tex)
        sections[f"tex:sec{idx}"] = f"{match.group(1).strip()}\n{full_tex[start:end].strip()}".strip()

    eq_idx = 0
    thm_idx = 0
    for env_match in _TEX_ENV_RE.finditer(full_tex):
        env_name = env_match.group(1)
        content = env_match.group(2).strip()
        if env_name.startswith(("equation", "align")):
            eq_idx += 1
            sections[f"tex:eq{eq_idx}"] = content
        else:
            thm_idx += 1
            sections[f"tex:thm{thm_idx}"] = content
    return sections


_SANDBOX_EXTRACT_SCRIPT = """import gzip
import io
import json
from pathlib import Path
import re
import sys
import tarfile

def validate_member(name: str) -> None:
    if not name or "\\x00" in name or name.startswith(("/", "\\\\")):
        raise ValueError(f"Unsafe tar member path: {name!r}")
    first_part = name.split("/", 1)[0].split("\\\\", 1)[0]
    if ":" in first_part:
        raise ValueError(f"Unsafe tar member path: {name!r}")
    if ".." in re.split(r"[/\\\\]", name):
        raise ValueError(f"Unsafe tar member path: {name!r}")

def main() -> int:
    limits = json.loads(Path("/work/limits.json").read_text(encoding="utf-8"))
    max_total = int(limits["max_total_bytes"])
    tar_bytes = Path("/work/payload.bin").read_bytes()
    tex_files = []
    total_read = 0

    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tf:
            for member in tf.getmembers():
                validate_member(member.name)
                if member.issym() or member.islnk():
                    raise ValueError(f"Unsafe link tar member: {member.name!r}")
                if not member.isreg() or not member.name.lower().endswith(".tex"):
                    continue
                if member.size > max_total - total_read:
                    raise ValueError("Extracted TeX source exceeds maximum allowed size")
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                remaining = max_total - total_read
                chunk = extracted.read(remaining + 1)
                total_read += len(chunk)
                if total_read > max_total:
                    raise ValueError("Extracted TeX source exceeds maximum allowed size")
                tex_files.append([member.name, chunk.decode("utf-8", errors="replace")])
    except tarfile.TarError:
        if tar_bytes.startswith(b"\\x1f\\x8b"):
            with gzip.GzipFile(fileobj=io.BytesIO(tar_bytes)) as gz:
                raw_chunk = gz.read(max_total + 1)
        else:
            raw_chunk = tar_bytes[: max_total + 1]
        if len(raw_chunk) > max_total:
            raise ValueError("Extracted TeX source exceeds maximum allowed size")
        raw_text = raw_chunk.decode("utf-8", errors="replace")
        if "\\\\begin{document}" in raw_text or "\\\\section" in raw_text or "\\\\documentclass" in raw_text:
            tex_files.append(["main.tex", raw_text])
        else:
            raise ValueError("Payload is neither a valid tar archive nor a LaTeX source document")

    if not tex_files:
        raise ValueError("No .tex files found inside arXiv source archive")
    print(json.dumps({"tex_files": tex_files}))
    return 0

if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"{type(exc).__name__}: {exc}", file=sys.stderr)
        raise SystemExit(1)
"""


def _validate_tar_member_path(name: str) -> None:
    if not name or "\x00" in name or name.startswith(("/", "\\")):
        raise ValueError(f"Unsafe tar member path: {name!r}")
    first_part = name.split("/", 1)[0].split("\\", 1)[0]
    if ":" in first_part:
        raise ValueError(f"Unsafe tar member path: {name!r}")
    if ".." in re.split(r"[/\\]", name):
        raise ValueError(f"Unsafe tar member path: {name!r}")


def _extract_tex_local(
    tar_bytes: bytes,
    *,
    max_total_bytes: int,
) -> list[tuple[str, str]]:
    tex_files: list[tuple[str, str]] = []
    total_read = 0

    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tf:
            for member in tf.getmembers():
                _validate_tar_member_path(member.name)
                if member.issym() or member.islnk():
                    raise ValueError(f"Unsafe link tar member: {member.name!r}")
                if not member.isreg() or not member.name.lower().endswith(".tex"):
                    continue
                if member.size > max_total_bytes - total_read:
                    raise ValueError("Extracted TeX source exceeds maximum allowed size")
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                remaining = max_total_bytes - total_read
                chunk = extracted.read(remaining + 1)
                total_read += len(chunk)
                if total_read > max_total_bytes:
                    raise ValueError("Extracted TeX source exceeds maximum allowed size")
                tex_files.append((member.name, chunk.decode("utf-8", errors="replace")))
    except tarfile.TarError:
        if tar_bytes.startswith(b"\x1f\x8b"):
            with gzip.GzipFile(fileobj=io.BytesIO(tar_bytes)) as gz:
                raw_chunk = gz.read(max_total_bytes + 1)
        else:
            raw_chunk = tar_bytes[: max_total_bytes + 1]
        if len(raw_chunk) > max_total_bytes:
            raise ValueError("Extracted TeX source exceeds maximum allowed size")
        raw_text = raw_chunk.decode("utf-8", errors="replace")
        if "\\begin{document}" in raw_text or "\\section" in raw_text or "\\documentclass" in raw_text:
            tex_files.append(("main.tex", raw_text))
        else:
            raise ValueError("Payload is neither a valid tar archive nor a LaTeX source document")

    if not tex_files:
        raise ValueError("No .tex files found inside arXiv source archive")
    return tex_files


def _can_use_sandbox(sandbox: OracleSandbox | None) -> OracleSandbox | None:
    global _DEFAULT_DOCKER_SANDBOX_READY
    if sandbox is not None:
        if sandbox.is_docker_available() and sandbox.is_image_available():
            return sandbox
        return None
    if _DEFAULT_DOCKER_SANDBOX_READY is None:
        if shutil.which("docker") is None:
            _DEFAULT_DOCKER_SANDBOX_READY = False
        else:
            from ral.pipeline.oracle_sandbox import OracleSandbox

            default_sbx = OracleSandbox()
            _DEFAULT_DOCKER_SANDBOX_READY = bool(
                default_sbx.is_docker_available() and default_sbx.is_image_available()
            )
    if _DEFAULT_DOCKER_SANDBOX_READY:
        from ral.pipeline.oracle_sandbox import OracleSandbox

        return OracleSandbox()
    return None


def _extract_tex_in_sandbox(
    tar_bytes: bytes,
    *,
    max_total_bytes: int,
    sandbox: OracleSandbox,
) -> list[tuple[str, str]]:
    res = sandbox.run_script(
        _SANDBOX_EXTRACT_SCRIPT,
        plugin_name="arxiv_tex_extract",
        script_name="extract_tex.py",
        timeout_s=15,
        enforce_ast_lint=False,
        extra_files={
            "payload.bin": tar_bytes,
            "limits.json": json.dumps({"max_total_bytes": int(max_total_bytes)}).encode("utf-8"),
        },
    )
    if not res.passed:
        detail = res.stderr.strip() or res.stdout.strip() or "unknown sandbox error"
        raise ValueError(f"Sandboxed TeX extraction failed: {detail}")
    try:
        parsed = json.loads(res.stdout)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Sandboxed TeX extractor emitted invalid JSON: {exc}") from exc
    raw_files = parsed.get("tex_files") or []
    tex_files = [(str(item[0]), str(item[1])) for item in raw_files if isinstance(item, list) and len(item) == 2]
    if not tex_files:
        raise ValueError("No .tex files found inside arXiv source archive")
    return tex_files


def extract_tex_from_tarball(
    tar_bytes: bytes,
    *,
    max_total_bytes: int = MAX_TEX_UNCOMPRESSED_BYTES,
    sandbox: OracleSandbox | None = None,
    use_sandbox: bool = True,
) -> tuple[str, dict[str, str]]:
    """
    Safely extract `.tex` source files from an arXiv `.tar.gz`, `.tar`, or `.tex.gz` payload.

    Rejects PDF-only payloads and path traversal members, enforcing a <=16 MiB uncompressed cap
    inside `OracleSandbox` when Docker is available (with bounded local fallback).
    """
    if tar_bytes.startswith(b"%PDF-"):
        raise ValueError("PDF-only payload contains no TeX source")
    if int(max_total_bytes) <= 0:
        raise ValueError("max_total_bytes must be positive")
    effective_cap = min(int(max_total_bytes), MAX_TEX_UNCOMPRESSED_BYTES)

    active_sandbox = _can_use_sandbox(sandbox) if use_sandbox else None
    if active_sandbox is not None:
        tex_files = _extract_tex_in_sandbox(
            tar_bytes,
            max_total_bytes=effective_cap,
            sandbox=active_sandbox,
        )
    else:
        tex_files = _extract_tex_local(
            tar_bytes,
            max_total_bytes=effective_cap,
        )

    # Put root file containing \documentclass first for deterministic reading order.
    tex_files.sort(key=lambda pair: (0 if "\\documentclass" in pair[1] else 1, pair[0]))
    combined = "\n\n".join(content for _, content in tex_files)
    return combined, _parse_tex_sections(combined)


def search_arxiv_api(
    query: str,
    *,
    max_results: int = 10,
    http_get: Callable[..., Any] | None = None,
) -> list[dict[str, Any]]:
    """Query the public arXiv Atom API directly as a primary-source fallback."""
    getter = http_get or default_http_get
    resp = getter(
        ARXIV_API_URL,
        params={
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max(1, int(max_results)),
        },
    )
    if hasattr(resp, "raise_for_status"):
        resp.raise_for_status()
    raw_bytes = resp.content if hasattr(resp, "content") else str(resp).encode("utf-8")
    root = ET.fromstring(raw_bytes)

    def _local_child_text(parent: ET.Element, tag_name: str) -> str:
        for child in parent:
            local = child.tag.rsplit("}", 1)[-1]
            if local == tag_name:
                return re.sub(r"\s+", " ", "".join(child.itertext())).strip()
        return ""

    entries = [elem for elem in root.iter() if elem.tag.rsplit("}", 1)[-1] == "entry"]
    results: list[dict[str, Any]] = []
    for entry in entries:
        raw_id = _local_child_text(entry, "id")
        if not raw_id:
            continue
        try:
            base_id, ver = parse_arxiv_id(raw_id)
        except ValueError:
            continue
        canonical_id = f"{base_id}v{ver}" if ver is not None else base_id
        published = _local_child_text(entry, "published")
        pub_year: int | None = None
        if len(published) >= 4 and published[:4].isdigit():
            pub_year = int(published[:4])
        elif len(base_id) >= 4 and base_id[:2].isdigit():
            yy = int(base_id[:2])
            pub_year = 1900 + yy if yy >= 91 else 2000 + yy

        results.append(
            {
                "arxiv_id": canonical_id,
                "arxiv_base_id": base_id,
                "arxiv_version": ver,
                "title": _local_child_text(entry, "title") or f"arXiv:{canonical_id}",
                "pub_year": int(pub_year or 2020),
                "summary": _local_child_text(entry, "summary"),
            }
        )
    return results


def fetch_arxiv_source(
    arxiv_id: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    http_get: Callable[..., Any] | None = None,
    allowed_licenses: set[str] | None = None,
    sandbox: OracleSandbox | None = None,
) -> SourceRecord:
    """
    Fetch native arXiv `.tex` source tarball first (or `oa_pdf` fallback when fallback text is provided).

    Skips PDF-only payloads with empty extracted text so Scout does not receive blank payloads.
    Enforces `config/license_allowlist.yaml` and rejects withdrawn/retracted records.
    """
    meta = dict(metadata or {})
    if meta.get("retracted") or meta.get("withdrawn"):
        raise RetractedSourceError(f"arXiv source {arxiv_id!r} is marked retracted/withdrawn")

    raw_license = meta.get("source_license") or meta.get("license") or "arXiv-non-exclusive-1.0"
    clean_license = require_allowed_license(raw_license, allowed_licenses)

    base_id, version = parse_arxiv_id(arxiv_id)
    canonical_id = f"{base_id}v{version}" if version is not None else base_id
    getter = http_get or default_http_get
    resp = getter(ARXIV_EPRINT_URL.format(arxiv_id=canonical_id))
    if hasattr(resp, "raise_for_status"):
        resp.raise_for_status()

    raw_bytes = resp.content if hasattr(resp, "content") else bytes(resp)
    if raw_bytes.startswith(b"%PDF-"):
        extracted_text = str(meta.get("fallback_pdf_text") or "").strip()
        if not extracted_text:
            raise ValueError(
                f"arXiv source {canonical_id!r} returned PDF-only payload with empty text; skipping"
            )
        source_format = "oa_pdf"
        raw_ext = "pdf"
        sections: dict[str, str] = {}
    else:
        extracted_text, sections = extract_tex_from_tarball(raw_bytes, sandbox=sandbox)
        if "this paper has been withdrawn" in extracted_text.lower():
            raise RetractedSourceError(f"arXiv paper {canonical_id!r} was withdrawn by authors")
        source_format = "arxiv_tex"
        raw_ext = "tar.gz"

    pub_year = meta.get("pub_year")
    if pub_year is None and len(base_id) >= 4 and base_id[:2].isdigit():
        yy = int(base_id[:2])
        pub_year = 1900 + yy if yy >= 91 else 2000 + yy

    return SourceRecord(
        source_id=f"arxiv:{canonical_id}",
        source_license=clean_license,
        source_format=source_format,
        title=str(meta.get("title") or f"arXiv:{canonical_id}"),
        pub_year=int(pub_year or 2020),
        peer_reviewed=bool(meta.get("peer_reviewed", False)),
        retracted=False,
        canonical_doi=meta.get("canonical_doi"),
        arxiv_base_id=base_id,
        arxiv_version=version,
        citation_count=int(meta.get("citation_count", 0)),
        raw_bytes=raw_bytes,
        raw_ext=raw_ext,
        extracted_text=extracted_text,
        sections=sections,
        metadata={"source": "arxiv"},
    )

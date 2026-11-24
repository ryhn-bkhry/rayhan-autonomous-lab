"""Native arXiv `.tex` e-print tarball fetcher and LaTeX section parser (Section 6.3)."""

from __future__ import annotations

import gzip
import io
import re
import tarfile
from typing import Any, Callable, Mapping

from ral.sources import (
    RetractedSourceError,
    SourceRecord,
    default_http_get,
    require_allowed_license,
)

ARXIV_EPRINT_URL = "https://arxiv.org/e-print/{arxiv_id}"

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


def extract_tex_from_tarball(
    tar_bytes: bytes,
    *,
    max_total_bytes: int = 16 * 1024 * 1024,
) -> tuple[str, dict[str, str]]:
    """
    Safely extract `.tex` source files from an arXiv `.tar.gz`, `.tar`, or `.tex.gz` payload.

    Rejects path traversal members and returns `(combined_tex, sections_by_locator)`.
    """
    tex_files: list[tuple[str, str]] = []
    total_read = 0

    try:
        with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:*") as tf:
            for member in tf.getmembers():
                if not member.isfile():
                    continue
                if ".." in member.name or member.name.startswith(("/", "\\")):
                    raise ValueError(f"Unsafe tar member path: {member.name!r}")
                if not member.name.lower().endswith(".tex"):
                    continue
                total_read += member.size
                if total_read > max_total_bytes:
                    raise ValueError("Extracted TeX source exceeds maximum allowed size")
                extracted = tf.extractfile(member)
                if extracted is None:
                    continue
                content = extracted.read().decode("utf-8", errors="replace")
                tex_files.append((member.name, content))
    except tarfile.TarError:
        try:
            raw_text = gzip.decompress(tar_bytes).decode("utf-8", errors="replace")
        except OSError:
            raw_text = tar_bytes.decode("utf-8", errors="replace")
        if "\\begin{document}" in raw_text or "\\section" in raw_text or "\\documentclass" in raw_text:
            tex_files.append(("main.tex", raw_text))
        else:
            raise ValueError("Payload is neither a valid tar archive nor a LaTeX source document")

    if not tex_files:
        raise ValueError("No .tex files found inside arXiv source archive")

    # Put root file containing \documentclass first for deterministic reading order.
    tex_files.sort(key=lambda pair: (0 if "\\documentclass" in pair[1] else 1, pair[0]))
    combined = "\n\n".join(content for _, content in tex_files)
    return combined, _parse_tex_sections(combined)


def fetch_arxiv_source(
    arxiv_id: str,
    *,
    metadata: Mapping[str, Any] | None = None,
    http_get: Callable[..., Any] | None = None,
    allowed_licenses: set[str] | None = None,
) -> SourceRecord:
    """
    Fetch native arXiv `.tex` source tarball first (or `oa_pdf` fallback when arXiv serves PDF).

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
        source_format = "oa_pdf"
        raw_ext = "pdf"
        extracted_text = str(meta.get("fallback_pdf_text") or "")
        sections: dict[str, str] = {}
    else:
        extracted_text, sections = extract_tex_from_tarball(raw_bytes)
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

"""Open-Access primary source models, license normalization, and HTTP helpers (Section 6.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
import yaml

DEFAULT_USER_AGENT = "RayhansAutonomousLab/2.1 (Open-Access Scientific Vault)"


class LicenseRejectedError(ValueError):
    """Raised when a candidate source carries a non-OA, missing, or disallowed license."""


class RetractedSourceError(ValueError):
    """Raised when a candidate source is flagged as retracted or withdrawn."""


@dataclass(frozen=True)
class SourceRecord:
    """Normalized primary-source artifact fetched by Stage 1 adapters."""

    source_id: str
    source_license: str
    source_format: str  # arxiv_tex | pmc_jats | oa_pdf | reference_db
    title: str
    pub_year: int
    peer_reviewed: bool
    retracted: bool
    canonical_doi: str | None
    arxiv_base_id: str | None
    arxiv_version: int | None
    citation_count: int
    raw_bytes: bytes
    raw_ext: str
    extracted_text: str
    sections: dict[str, str] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


_EXACT_LICENSE_ALIASES: dict[str, str] = {
    "cc-by-4.0": "CC-BY-4.0",
    "cc by 4.0": "CC-BY-4.0",
    "cc-by-3.0": "CC-BY-3.0",
    "cc by 3.0": "CC-BY-3.0",
    "cc-by-sa-4.0": "CC-BY-SA-4.0",
    "cc by-sa 4.0": "CC-BY-SA-4.0",
    "cc0-1.0": "CC0-1.0",
    "cc0": "CC0-1.0",
    "arxiv-non-exclusive-1.0": "arXiv-non-exclusive-1.0",
    "mit": "MIT",
    "bsd-3-clause": "BSD-3-Clause",
    "apache-2.0": "Apache-2.0",
    "public-domain": "Public-Domain",
    "public domain": "Public-Domain",
    "us-gov-nist": "US-Gov-NIST",
    "pmc-oa": "PMC-OA",
}


def load_license_allowlist(path: Path = Path("config/license_allowlist.yaml")) -> set[str]:
    """Load mandatory Open-Access / Public-Domain license allowlist from YAML."""
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    licenses = data.get("allowed_licenses", [])
    if not isinstance(licenses, list) or not licenses:
        raise ValueError(f"Invalid or empty allowed_licenses in {path}")
    return {str(item).strip() for item in licenses if str(item).strip()}


def normalize_license(raw: str | None) -> str:
    """
    Normalize known SPDX strings and Creative Commons / arXiv / PMC URIs to canonical IDs.

    Returns the original stripped value if unrecognized so `require_allowed_license` rejects it.
    """
    if not raw or not str(raw).strip():
        return ""
    cleaned = str(raw).strip()
    lowered = cleaned.lower()

    if lowered in _EXACT_LICENSE_ALIASES:
        return _EXACT_LICENSE_ALIASES[lowered]

    # Reject non-commercial or no-derivatives CC variants explicitly.
    if "by-nc" in lowered or "by-nd" in lowered:
        return cleaned

    if "creativecommons.org/licenses/by/4.0" in lowered:
        return "CC-BY-4.0"
    if "creativecommons.org/licenses/by/3.0" in lowered:
        return "CC-BY-3.0"
    if "creativecommons.org/licenses/by-sa/4.0" in lowered:
        return "CC-BY-SA-4.0"
    if "creativecommons.org/publicdomain/zero/1.0" in lowered or "creativecommons.org/licenses/zero/1.0" in lowered:
        return "CC0-1.0"
    if "arxiv.org/licenses/nonexclusive-distrib/1.0" in lowered:
        return "arXiv-non-exclusive-1.0"

    return cleaned


def require_allowed_license(
    raw_license: str | None,
    allowed_licenses: set[str] | None = None,
) -> str:
    """Normalize `raw_license` and enforce membership in `config/license_allowlist.yaml`."""
    allowlist = allowed_licenses if allowed_licenses is not None else load_license_allowlist()
    normalized = normalize_license(raw_license)
    if not normalized or normalized not in allowlist:
        raise LicenseRejectedError(
            f"Source license {raw_license!r} (normalized={normalized!r}) is not in OA allowlist"
        )
    return normalized


def default_http_get(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
    timeout: int = 20,
) -> Any:
    """Execute a synchronous HTTP GET via `requests` with RAL User-Agent."""
    import requests

    merged_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        merged_headers.update(headers)
    return requests.get(url, params=params, headers=merged_headers, timeout=timeout)

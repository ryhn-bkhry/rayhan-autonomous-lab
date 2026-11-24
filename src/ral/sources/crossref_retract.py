"""Crossref / Retraction Watch firewall and deterministic DOI normalizer (Section 6.3)."""

from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Callable, Mapping

from ral.sources import default_http_get, normalize_license

CROSSREF_WORKS_URL = "https://api.crossref.org/works/{doi}"
_DOI_PREFIX_RE = re.compile(r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)", re.IGNORECASE)


def normalize_doi(doi: str | None) -> str | None:
    """Normalize DOI strings to lowercase canonical form without URL prefixes."""
    if not doi or not str(doi).strip():
        return None
    cleaned = _DOI_PREFIX_RE.sub("", str(doi).strip()).strip().lower()
    return cleaned or None


@dataclass(frozen=True)
class RetractionStatus:
    """Crossref & Retraction Watch verification status for a scholarly DOI."""

    doi: str
    retracted: bool
    reason: str | None = None
    peer_reviewed: bool = False
    source_license: str | None = None
    pub_year: int | None = None
    updated_by: list[str] | None = None


def _extract_pub_year(msg: Mapping[str, Any]) -> int | None:
    for key in ("published-print", "published-online", "issued", "created"):
        part = msg.get(key) or {}
        date_parts = part.get("date-parts") or []
        if date_parts and isinstance(date_parts[0], list) and date_parts[0]:
            year = date_parts[0][0]
            if isinstance(year, int):
                return year
    return None


def check_crossref_status(
    doi: str,
    *,
    retraction_watch_dois: set[str] | None = None,
    http_get: Callable[..., Any] | None = None,
) -> RetractionStatus:
    """
    Check a DOI against an optional local Retraction Watch set and live Crossref metadata.
    """
    norm_doi = normalize_doi(doi)
    if not norm_doi:
        raise ValueError(f"Invalid DOI: {doi!r}")

    if retraction_watch_dois:
        rw_normalized = {d for item in retraction_watch_dois if (d := normalize_doi(item))}
        if norm_doi in rw_normalized:
            return RetractionStatus(
                doi=norm_doi,
                retracted=True,
                reason="Matched Retraction Watch firewall",
            )

    getter = http_get or default_http_get
    resp = getter(CROSSREF_WORKS_URL.format(doi=norm_doi))
    if hasattr(resp, "raise_for_status"):
        resp.raise_for_status()
    payload = resp.json() if hasattr(resp, "json") else resp
    msg = payload.get("message", payload) if isinstance(payload, Mapping) else {}

    retracted = False
    reasons: list[str] = []
    updated_by: list[str] = []

    for upd in msg.get("update-to") or []:
        upd_type = str((upd or {}).get("type") or "").lower()
        if upd_type in ("retraction", "withdrawal", "retracted"):
            retracted = True
            reasons.append(f"Crossref update-to:{upd_type}")
            if upd.get("DOI"):
                updated_by.append(str(upd["DOI"]))

    relations = msg.get("relation") or {}
    if isinstance(relations, Mapping) and relations.get("is-retracted-by"):
        retracted = True
        reasons.append("Crossref relation:is-retracted-by")
        for item in relations.get("is-retracted-by") or []:
            if isinstance(item, Mapping) and item.get("id"):
                updated_by.append(str(item["id"]))

    titles = msg.get("title") or []
    title_str = " ".join(str(t) for t in titles).strip()
    if title_str.lower().startswith(("retracted:", "retraction:", "withdrawn:")):
        retracted = True
        reasons.append("Title marked retracted/withdrawn")

    work_type = str(msg.get("type") or "").lower()
    peer_reviewed = work_type in ("journal-article", "proceedings-article", "book-chapter")

    raw_license: str | None = None
    for lic in msg.get("license") or []:
        if isinstance(lic, Mapping) and lic.get("URL"):
            raw_license = normalize_license(str(lic["URL"]))
            break

    return RetractionStatus(
        doi=norm_doi,
        retracted=retracted,
        reason="; ".join(reasons) if reasons else None,
        peer_reviewed=peer_reviewed,
        source_license=raw_license,
        pub_year=_extract_pub_year(msg),
        updated_by=updated_by or None,
    )

"""OpenAlex Open-Access work discovery and citation half-life estimator (Sections 4.3 & 6.3)."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, Mapping, Sequence

from ral.sources import (
    LicenseRejectedError,
    default_http_get,
    load_license_allowlist,
    require_allowed_license,
)
from ral.sources.crossref_retract import normalize_doi

OPENALEX_WORKS_URL = "https://api.openalex.org/works"


@dataclass(frozen=True)
class OpenAlexWork:
    """Filtered Open-Access scholarly work from OpenAlex."""

    work_id: str
    title: str
    canonical_doi: str | None
    pub_year: int
    cited_by_count: int
    source_license: str
    oa_url: str | None
    arxiv_id: str | None
    pmcid: str | None
    peer_reviewed: bool
    retracted: bool
    concepts: list[str] = field(default_factory=list)


def _extract_arxiv_id_from_work(work: Mapping[str, Any]) -> str | None:
    ids = work.get("ids") or {}
    if isinstance(ids, Mapping) and ids.get("arxiv"):
        return str(ids["arxiv"]).rsplit("/", 1)[-1]
    for loc in work.get("locations") or []:
        landing = str((loc or {}).get("landing_page_url") or "")
        if "arxiv.org/abs/" in landing:
            return landing.split("arxiv.org/abs/", 1)[1].split("?", 1)[0].strip("/")
    return None


def _extract_pmcid_from_work(work: Mapping[str, Any]) -> str | None:
    ids = work.get("ids") or {}
    if isinstance(ids, Mapping) and ids.get("pmcid"):
        raw = str(ids["pmcid"]).strip()
        return raw.rsplit("/", 1)[-1]
    return None


def search_oa_works(
    query: str,
    *,
    concept_id: str | None = None,
    per_page: int = 25,
    allowed_licenses: set[str] | None = None,
    http_get: Callable[..., Any] | None = None,
) -> list[OpenAlexWork]:
    """
    Search OpenAlex works strictly filtered to Open-Access, non-retracted, allowlisted licenses.
    """
    allowlist = allowed_licenses if allowed_licenses is not None else load_license_allowlist()
    getter = http_get or default_http_get
    filters = ["is_oa:true", "is_retracted:false"]
    if concept_id:
        filters.append(f"concepts.id:{concept_id}")
    params = {
        "search": query,
        "filter": ",".join(filters),
        "per-page": int(per_page),
    }
    resp = getter(OPENALEX_WORKS_URL, params=params)
    if hasattr(resp, "raise_for_status"):
        resp.raise_for_status()
    payload = resp.json() if hasattr(resp, "json") else resp

    works: list[OpenAlexWork] = []
    for raw in payload.get("results", []):
        if raw.get("is_retracted"):
            continue
        best_oa = raw.get("best_oa_location") or {}
        primary = raw.get("primary_location") or {}
        raw_license = best_oa.get("license") or primary.get("license")
        try:
            clean_license = require_allowed_license(raw_license, allowlist)
        except LicenseRejectedError:
            continue

        pub_year = raw.get("publication_year")
        if not isinstance(pub_year, int):
            continue

        work_type = str(raw.get("type") or "").lower()
        arxiv_id = _extract_arxiv_id_from_work(raw)
        source_type = str(((primary.get("source") or {}).get("type")) or "").lower()
        peer_reviewed = work_type == "article" and source_type != "repository" and not (
            arxiv_id and source_type == ""
        )

        works.append(
            OpenAlexWork(
                work_id=str(raw.get("id") or "").rsplit("/", 1)[-1],
                title=str(raw.get("title") or "").strip(),
                canonical_doi=normalize_doi(raw.get("doi")),
                pub_year=pub_year,
                cited_by_count=int(raw.get("cited_by_count") or 0),
                source_license=clean_license,
                oa_url=best_oa.get("pdf_url") or best_oa.get("landing_page_url"),
                arxiv_id=arxiv_id,
                pmcid=_extract_pmcid_from_work(raw),
                peer_reviewed=bool(peer_reviewed),
                retracted=False,
                concepts=[
                    str(c.get("display_name"))
                    for c in (raw.get("concepts") or [])
                    if isinstance(c, Mapping) and c.get("display_name")
                ],
            )
        )
    return works


def compute_citation_half_life(
    works: Sequence[Mapping[str, Any] | OpenAlexWork],
    *,
    reference_year: int | None = None,
    default_years: float = 10.0,
) -> float:
    """
    Compute median citation half-life `h_d` in years for a set of OpenAlex works (Section 4.3).

    Uses citation-weighted median age when positive citations exist, falling back to unweighted
    median age and `default_years` when no valid publication years are available.
    """
    ref_year = reference_year if reference_year is not None else datetime.now(timezone.utc).year
    samples: list[tuple[float, float]] = []
    for item in works:
        if isinstance(item, OpenAlexWork):
            year = item.pub_year
            cites = item.cited_by_count
        else:
            year = item.get("publication_year") or item.get("pub_year")
            cites = item.get("cited_by_count") or 0
        if isinstance(year, int) and 1600 <= year <= ref_year:
            age = float(max(0, ref_year - year))
            samples.append((age, float(max(0, int(cites)))))

    if not samples:
        return float(default_years)

    samples.sort(key=lambda pair: pair[0])
    total_cites = sum(w for _, w in samples)
    if total_cites <= 0.0:
        ages = [age for age, _ in samples]
        mid = len(ages) // 2
        return float(ages[mid]) if len(ages) % 2 == 1 else float(0.5 * (ages[mid - 1] + ages[mid]))

    half = total_cites / 2.0
    cumulative = 0.0
    for age, weight in samples:
        cumulative += weight
        if cumulative >= half:
            return float(age)
    return float(samples[-1][0])

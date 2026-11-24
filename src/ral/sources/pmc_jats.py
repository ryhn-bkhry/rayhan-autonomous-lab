"""PubMed Central JATS XML fetcher and structured parser (Section 6.3)."""

from __future__ import annotations

import re
from typing import Any, Callable
import xml.etree.ElementTree as ET

from ral.sources import (
    RetractedSourceError,
    SourceRecord,
    default_http_get,
    require_allowed_license,
)
from ral.sources.crossref_retract import normalize_doi

PMC_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
_XLINK_HREF = "{http://www.w3.org/1999/xlink}href"


def _norm_ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _elem_text(elem: ET.Element | None) -> str:
    if elem is None:
        return ""
    return _norm_ws("".join(elem.itertext()))


def parse_jats_xml(xml_bytes: bytes) -> dict[str, Any]:
    """Parse a PubMed Central JATS XML payload into structured metadata and section text."""
    root = ET.fromstring(xml_bytes)
    article = root if root.tag.endswith("article") else root.find(".//article")
    if article is None:
        raise ValueError("JATS XML does not contain an <article> element")

    article_type = str(article.attrib.get("article-type") or "").lower()
    retracted = "retract" in article_type

    title = _elem_text(article.find(".//title-group/article-title"))
    doi: str | None = None
    pmcid: str | None = None
    for aid in article.findall(".//article-meta/article-id"):
        id_type = (aid.attrib.get("pub-id-type") or "").lower()
        val = _elem_text(aid)
        if id_type == "doi" and val:
            doi = normalize_doi(val)
        elif id_type in ("pmc", "pmcid") and val:
            pmcid = val if val.upper().startswith("PMC") else f"PMC{val}"

    pub_year: int | None = None
    for pub_date in article.findall(".//article-meta/pub-date"):
        year_txt = _elem_text(pub_date.find("year"))
        if year_txt.isdigit():
            pub_year = int(year_txt)
            break

    raw_license: str | None = None
    lic_elem = article.find(".//permissions/license")
    if lic_elem is not None:
        raw_license = (
            lic_elem.attrib.get(_XLINK_HREF)
            or _elem_text(lic_elem.find(".//ext-link"))
            or lic_elem.attrib.get("license-type")
            or _elem_text(lic_elem)
        )
        # Check nested ext-link xlink:href too
        ext_link = lic_elem.find(".//ext-link")
        if ext_link is not None and ext_link.attrib.get(_XLINK_HREF):
            raw_license = ext_link.attrib[_XLINK_HREF]

    sections: dict[str, str] = {}
    body_chunks: list[str] = []
    abstract_text = _elem_text(article.find(".//abstract"))
    if abstract_text:
        sections["jats:abstract"] = abstract_text
        body_chunks.append(abstract_text)

    body = article.find(".//body")
    if body is not None:
        sec_list = body.findall("./sec")
        if sec_list:
            for idx, sec in enumerate(sec_list, start=1):
                sec_title = _elem_text(sec.find("title"))
                sec_text = _elem_text(sec)
                if sec_text:
                    sections[f"jats:sec{idx}"] = sec_text
                    body_chunks.append(f"{sec_title}: {sec_text}" if sec_title else sec_text)
        else:
            full_body = _elem_text(body)
            if full_body:
                sections["jats:body"] = full_body
                body_chunks.append(full_body)

    return {
        "title": title,
        "doi": doi,
        "pmcid": pmcid,
        "pub_year": pub_year or 2020,
        "raw_license": raw_license,
        "retracted": retracted,
        "sections": sections,
        "text": "\n\n".join(body_chunks).strip(),
    }


def fetch_pmc_jats(
    pmcid: str,
    *,
    http_get: Callable[..., Any] | None = None,
    allowed_licenses: set[str] | None = None,
) -> SourceRecord:
    """Fetch and validate an Open-Access PubMed Central article in native JATS XML."""
    clean_num = pmcid.strip().upper().removeprefix("PMC")
    canonical_pmc = f"PMC{clean_num}"
    getter = http_get or default_http_get
    resp = getter(
        PMC_EFETCH_URL,
        params={"db": "pmc", "id": clean_num, "retmode": "xml"},
    )
    if hasattr(resp, "raise_for_status"):
        resp.raise_for_status()
    raw_bytes = resp.content if hasattr(resp, "content") else str(resp).encode("utf-8")

    parsed = parse_jats_xml(raw_bytes)
    if parsed["retracted"]:
        raise RetractedSourceError(f"PMC article {canonical_pmc} is flagged as a retraction notice")
    clean_license = require_allowed_license(parsed["raw_license"], allowed_licenses)

    return SourceRecord(
        source_id=f"pmc:{parsed['pmcid'] or canonical_pmc}",
        source_license=clean_license,
        source_format="pmc_jats",
        title=parsed["title"] or canonical_pmc,
        pub_year=int(parsed["pub_year"]),
        peer_reviewed=True,
        retracted=False,
        canonical_doi=parsed["doi"],
        arxiv_base_id=None,
        arxiv_version=None,
        citation_count=0,
        raw_bytes=raw_bytes,
        raw_ext="xml",
        extracted_text=parsed["text"],
        sections=parsed["sections"],
        metadata={"source": "pmc"},
    )

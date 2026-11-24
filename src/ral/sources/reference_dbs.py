"""Deterministic canonical reference database fetchers: NIST CODATA, OEIS, DLMF (Section 6.3)."""

from __future__ import annotations

import re
from typing import Any, Callable

from ral.sources import (
    SourceRecord,
    default_http_get,
    require_allowed_license,
)

# Exact 2022 CODATA / 2019 SI redefinition defining physical constants (US Government public domain).
NIST_CODATA_2022_EXACT: dict[str, dict[str, Any]] = {
    "c": {
        "name": "Speed of light in vacuum",
        "value": "299792458",
        "uncertainty": "0 (exact)",
        "unit": "m s^-1",
        "pub_year": 2022,
    },
    "h": {
        "name": "Planck constant",
        "value": "6.62607015e-34",
        "uncertainty": "0 (exact)",
        "unit": "J Hz^-1",
        "pub_year": 2022,
    },
    "e": {
        "name": "Elementary charge",
        "value": "1.602176634e-19",
        "uncertainty": "0 (exact)",
        "unit": "C",
        "pub_year": 2022,
    },
    "k_B": {
        "name": "Boltzmann constant",
        "value": "1.380649e-23",
        "uncertainty": "0 (exact)",
        "unit": "J K^-1",
        "pub_year": 2022,
    },
    "N_A": {
        "name": "Avogadro constant",
        "value": "6.02214076e23",
        "uncertainty": "0 (exact)",
        "unit": "mol^-1",
        "pub_year": 2022,
    },
    "delta_nu_Cs": {
        "name": "Hyperfine transition frequency of Cs-133",
        "value": "9192631770",
        "uncertainty": "0 (exact)",
        "unit": "Hz",
        "pub_year": 2022,
    },
    "K_cd": {
        "name": "Luminous efficacy of monochromatic 540 THz radiation",
        "value": "683",
        "uncertainty": "0 (exact)",
        "unit": "lm W^-1",
        "pub_year": 2022,
    },
}

OEIS_SEARCH_URL = "https://oeis.org/search"
DLMF_ENTRY_URL = "https://dlmf.nist.gov/{section_id}"


def fetch_nist_constant(
    symbol: str,
    *,
    http_get: Callable[..., Any] | None = None,
) -> SourceRecord:
    """Fetch a NIST CODATA physical constant from built-in defining constants or NIST web."""
    key = symbol.strip()
    if key in NIST_CODATA_2022_EXACT and http_get is None:
        entry = NIST_CODATA_2022_EXACT[key]
        text = (
            f"NIST CODATA 2022 Constant {key} ({entry['name']}): "
            f"value = {entry['value']} {entry['unit']}, uncertainty = {entry['uncertainty']}."
        )
        return SourceRecord(
            source_id=f"nist:codata2022:{key}",
            source_license="US-Gov-NIST",
            source_format="reference_db",
            title=f"CODATA 2022: {entry['name']} ({key})",
            pub_year=int(entry["pub_year"]),
            peer_reviewed=True,
            retracted=False,
            canonical_doi=None,
            arxiv_base_id=None,
            arxiv_version=None,
            citation_count=1000,
            raw_bytes=text.encode("utf-8"),
            raw_ext="txt",
            extracted_text=text,
            sections={f"nist:codata:{key}": text},
            metadata={"symbol": key, **entry},
        )

    getter = http_get or default_http_get
    resp = getter(f"https://physics.nist.gov/cgi-bin/cuu/Value?{key}")
    if hasattr(resp, "raise_for_status"):
        resp.raise_for_status()
    raw_bytes = resp.content if hasattr(resp, "content") else str(resp).encode("utf-8")
    text = raw_bytes.decode("utf-8", errors="replace").strip()
    return SourceRecord(
        source_id=f"nist:codata2022:{key}",
        source_license="US-Gov-NIST",
        source_format="reference_db",
        title=f"NIST CODATA Constant: {key}",
        pub_year=2022,
        peer_reviewed=True,
        retracted=False,
        canonical_doi=None,
        arxiv_base_id=None,
        arxiv_version=None,
        citation_count=1000,
        raw_bytes=raw_bytes,
        raw_ext="txt",
        extracted_text=text,
        sections={f"nist:codata:{key}": text},
        metadata={"symbol": key},
    )


def fetch_oeis_sequence(
    a_number: str,
    *,
    http_get: Callable[..., Any] | None = None,
    allowed_licenses: set[str] | None = None,
) -> SourceRecord:
    """Fetch an OEIS sequence JSON record (`Axxxxxx`) with OA license validation."""
    clean_id = a_number.strip().upper()
    if not re.match(r"^A\d{6}$", clean_id):
        raise ValueError(f"Expected OEIS A-number (e.g. A000045), got {a_number!r}")

    getter = http_get or default_http_get
    resp = getter(OEIS_SEARCH_URL, params={"q": f"id:{clean_id}", "fmt": "json"})
    if hasattr(resp, "raise_for_status"):
        resp.raise_for_status()
    payload = resp.json() if hasattr(resp, "json") else resp
    results = payload.get("results") if isinstance(payload, dict) else payload
    if not results:
        raise ValueError(f"OEIS sequence {clean_id} not found")
    entry = results[0]

    clean_license = require_allowed_license(entry.get("license") or "CC-BY-4.0", allowed_licenses)
    name = str(entry.get("name") or clean_id).strip()
    data_terms = str(entry.get("data") or "").strip()
    formulas = "\n".join(str(f) for f in (entry.get("formula") or []))
    text = f"OEIS {clean_id}: {name}\nTerms: {data_terms}\nFormulas:\n{formulas}".strip()
    raw_bytes = text.encode("utf-8")

    return SourceRecord(
        source_id=f"oeis:{clean_id}",
        source_license=clean_license,
        source_format="reference_db",
        title=f"OEIS {clean_id}: {name}",
        pub_year=2020,
        peer_reviewed=True,
        retracted=False,
        canonical_doi=None,
        arxiv_base_id=None,
        arxiv_version=None,
        citation_count=100,
        raw_bytes=raw_bytes,
        raw_ext="json",
        extracted_text=text,
        sections={f"oeis:{clean_id}": text},
        metadata={"oeis_id": clean_id, "terms": data_terms},
    )


def fetch_dlmf_entry(
    section_id: str,
    *,
    http_get: Callable[..., Any] | None = None,
) -> SourceRecord:
    """Fetch a NIST Digital Library of Mathematical Functions (DLMF) entry."""
    clean_sec = section_id.strip().lstrip("/")
    if not clean_sec:
        raise ValueError("section_id must be non-empty")

    getter = http_get or default_http_get
    resp = getter(DLMF_ENTRY_URL.format(section_id=clean_sec))
    if hasattr(resp, "raise_for_status"):
        resp.raise_for_status()
    raw_bytes = resp.content if hasattr(resp, "content") else str(resp).encode("utf-8")
    raw_html = raw_bytes.decode("utf-8", errors="replace")
    stripped = re.sub(r"<[^>]+>", " ", raw_html)
    text = re.sub(r"\s+", " ", stripped).strip()

    return SourceRecord(
        source_id=f"dlmf:{clean_sec}",
        source_license="US-Gov-NIST",
        source_format="reference_db",
        title=f"NIST DLMF {clean_sec}",
        pub_year=2024,
        peer_reviewed=True,
        retracted=False,
        canonical_doi=None,
        arxiv_base_id=None,
        arxiv_version=None,
        citation_count=500,
        raw_bytes=raw_bytes,
        raw_ext="html",
        extracted_text=text,
        sections={f"dlmf:{clean_sec}": text},
        metadata={"section_id": clean_sec},
    )

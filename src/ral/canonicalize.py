"""Deterministic text canonicalization and hashing utilities (Blueprint Section 7.1)."""

from __future__ import annotations

from difflib import SequenceMatcher
import hashlib
import re
import unicodedata
from typing import Sequence

_LATEX_COMMENT_RE = re.compile(r"(?<!\\)%.*")
_WHITESPACE_RE = re.compile(r"\s+")


def sha256_bytes(data: bytes) -> str:
    """Return lowercase hexadecimal SHA-256 digest of raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_hex(text: str) -> str:
    """Return lowercase hexadecimal SHA-256 digest of UTF-8 encoded text."""
    return sha256_bytes(text.encode("utf-8"))


def normalize_newlines(text: str) -> str:
    """Normalize CRLF/CR line endings to LF for cross-platform hash parity."""
    return text.replace("\r\n", "\n").replace("\r", "\n")


def canonical_text(text: str) -> str:
    """
    Canonicalize text/LaTeX deterministically per Section 7.1:
    NFKC -> normalize newlines -> strip unescaped LaTeX '%' comments -> lowercase -> collapse whitespace.
    """
    normalized = normalize_newlines(unicodedata.normalize("NFKC", text))
    stripped_comments = _LATEX_COMMENT_RE.sub("", normalized)
    lowered = stripped_comments.lower()
    return _WHITESPACE_RE.sub(" ", lowered).strip()


def compute_assumptions_hash(assumptions: Sequence[str]) -> str:
    """Compute sorted canonical assumptions SHA-256 hash prefixed with 'sha256:'."""
    canonical_items = sorted(canonical_text(item) for item in assumptions)
    joined = "\n".join(canonical_items)
    return f"sha256:{sha256_hex(joined)}"


def compute_quote_sha256(quote_text: str) -> str:
    """Compute SHA-256 hash over canonicalized verbatim quote prefixed with 'sha256:'."""
    return f"sha256:{sha256_hex(canonical_text(quote_text))}"


def compute_oracle_script_sha256(verify_py_code: str) -> str:
    """Compute SHA-256 hash over stripped verification kernel code prefixed with 'sha256:'."""
    normalized = normalize_newlines(verify_py_code).strip()
    return f"sha256:{sha256_hex(normalized)}"


def quote_match_ratio(source_text: str, quote_text: str) -> float:
    """
    Verify verbatim quote presence after canonicalization.

    Returns 1.0 if canonical quote is an exact substring of canonical source,
    otherwise computes the best sliding-window SequenceMatcher similarity in [0.0, 1.0].
    """
    canon_quote = canonical_text(quote_text)
    if not canon_quote:
        return 0.0
    canon_source = canonical_text(source_text)
    if not canon_source:
        return 0.0
    if canon_quote in canon_source:
        return 1.0

    q_len = len(canon_quote)
    s_len = len(canon_source)
    if s_len <= q_len:
        return SequenceMatcher(None, canon_source, canon_quote).ratio()

    best = 0.0
    step = max(1, q_len // 10)
    window = int(q_len * 1.15) + 1
    for start in range(0, s_len - q_len + 1, step):
        snippet = canon_source[start : start + window]
        ratio = SequenceMatcher(None, snippet, canon_quote).ratio()
        if ratio > best:
            best = ratio
            if best >= 0.999:
                return best
    tail_snippet = canon_source[max(0, s_len - window) :]
    tail_ratio = SequenceMatcher(None, tail_snippet, canon_quote).ratio()
    return max(best, tail_ratio)

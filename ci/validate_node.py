"""Blocking CI validator for canonical Markdown vault nodes (Blueprint Sections 7.2 & 8)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
import re
import sys
from typing import Annotated, Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator
import yaml

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ral.canonicalize import (  # noqa: E402
    compute_assumptions_hash,
    compute_oracle_script_sha256,
    normalize_newlines,
)
from ral.pipeline.ast_linter import lint_verification_script  # noqa: E402

RAL_ID_PATTERN = r"^RAL-[A-Z]{2,6}-[0-9A-Z]{10}$"
SHA256_PREFIXED_PATTERN = r"^sha256:[0-9a-f]{64}$"
SHA256_HEX_PATTERN = r"^[0-9a-f]{64}$"
TAXONOMY_PATTERN = r"^(math|phys|cs|eng|econ|chem|bio)(\.[a-z0-9-]+)+$"
ORE_URI_RE = re.compile(
    r"^gdrive://[^/]+/by-sha256/([0-9a-f]{2})/([0-9a-f]{2})/([0-9a-f]{64})\.[A-Za-z0-9.]+$"
)
ISO_DATE_RE = re.compile(r"^\d{4}-\d{2}-\d{2}$")

BANNED_FILLER_PHRASES: tuple[str, ...] = (
    "as an ai",
    "as a language model",
    "in this paper",
    "in conclusion",
    "in summary",
    "certainly,",
    "sure, here",
    "here is a",
    "let's explore",
    "delve into",
    "it is worth noting",
    "it is important to note",
)

_DETAILS_BLOCK_RE = re.compile(r"<details\b[^>]*>.*?</details>", re.IGNORECASE | re.DOTALL)
_KERNEL_HEADING_RE = re.compile(
    r"^###\s+Executable Verification Kernel[ \t]*(?:\n|$)",
    re.MULTILINE,
)
_PY_FENCE_RE = re.compile(r"^\s*```(?:python|py)\s*\n(.*?)\n```(?=\s*(?:\n|$))", re.DOTALL)
_FENCE_ANY_RE = re.compile(r"```[\s\S]*?```")
_DISPLAY_MATH_DOLLAR_RE = re.compile(r"(?<!\\)\$\$[\s\S]*?(?<!\\)\$\$")
_DISPLAY_MATH_BRACKET_RE = re.compile(r"\\\[[\s\S]*?\\\]")
_INLINE_MATH_DOLLAR_RE = re.compile(r"(?<!\\)\$(?!\s)([^\n$]+?)(?<!\\)\$")
_INLINE_MATH_PAREN_RE = re.compile(r"\\\([\s\S]*?\\\)")
_INLINE_CODE_RE = re.compile(r"`[^`\n]+`")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_URL_RE = re.compile(r"\b(?:https?|gdrive)://\S+")


def _reject_bool_int(value: Any, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, not {type(value).__name__}")
    return value


class VerificationMeta(BaseModel):
    """Pydantic schema for `verification` YAML frontmatter block."""

    model_config = ConfigDict(extra="forbid")

    oracle_plugin: str = Field(min_length=1)
    oracle_status: Literal["PASS", "FAIL"]
    oracle_script_sha256: Annotated[str, Field(pattern=SHA256_PREFIXED_PATTERN)]
    verified_at: str
    formalizer_model: str = Field(min_length=1)
    skeptic_model: str = Field(min_length=1)
    model_capability_epoch: int = Field(ge=1)

    @field_validator("model_capability_epoch", mode="before")
    @classmethod
    def _validate_epoch(cls, value: Any) -> int:
        return _reject_bool_int(value, "model_capability_epoch")

    @field_validator("verified_at", mode="before")
    @classmethod
    def _validate_verified_at(cls, value: Any) -> str:
        if isinstance(value, datetime):
            if value.tzinfo is None:
                raise ValueError("verified_at timestamp must include UTC timezone")
            return value.isoformat()
        if not isinstance(value, str):
            raise ValueError("verified_at must be an ISO-8601 timestamp string")
        cleaned = value.strip()
        if not cleaned or ("T" not in cleaned and " " not in cleaned):
            raise ValueError("verified_at must be a full ISO-8601 datetime")
        parsed = datetime.fromisoformat(cleaned.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            raise ValueError("verified_at must include timezone offset or 'Z'")
        return cleaned


class ProvenanceEntry(BaseModel):
    """Pydantic schema for each anchor in `provenance[]`."""

    model_config = ConfigDict(extra="forbid")

    source_id: str = Field(min_length=1)
    source_license: str = Field(min_length=1)
    source_format: Literal["arxiv_tex", "pmc_jats", "oa_pdf", "reference_db"]
    source_locator: str = Field(min_length=1)
    ore_uri: str = Field(min_length=1)
    ore_sha256: Annotated[str, Field(pattern=SHA256_HEX_PATTERN)]
    quote_sha256: Annotated[str, Field(pattern=SHA256_PREFIXED_PATTERN)]
    quote_char_len: int = Field(ge=1)
    pub_year: int = Field(ge=1600, le=2100)
    peer_reviewed: bool
    retracted: bool

    @field_validator("quote_char_len", "pub_year", mode="before")
    @classmethod
    def _validate_ints(cls, value: Any, info: Any) -> int:
        return _reject_bool_int(value, info.field_name)

    @model_validator(mode="after")
    def _validate_ore_uri_matches_hash(self) -> ProvenanceEntry:
        match = ORE_URI_RE.match(self.ore_uri)
        if not match:
            raise ValueError(
                f"ore_uri must match gdrive://<warehouse>/by-sha256/{{h[0:2]}}/{{h[2:4]}}/{{sha256}}.{{ext}}, got {self.ore_uri!r}"
            )
        shard1, shard2, uri_hash = match.groups()
        if uri_hash != self.ore_sha256 or shard1 != self.ore_sha256[:2] or shard2 != self.ore_sha256[2:4]:
            raise ValueError("ore_uri shard prefix or filename hash does not match ore_sha256")
        return self


class NodeFrontmatter(BaseModel):
    """Strict v2.1 YAML frontmatter contract (Blueprint Section 7.2)."""

    model_config = ConfigDict(extra="forbid")

    ral_schema: Literal["2.1"]
    ral_id: Annotated[str, Field(pattern=RAL_ID_PATTERN)]
    node_type: Literal[
        "AXIOM",
        "THEOREM",
        "EMPIRICAL_LAW",
        "ALGORITHM",
        "BRIDGE_ISOMORPHISM",
        "NEGATIVE_RESULT",
    ]
    epistemic_tier: Literal["T1_CANONICAL", "T2_ESTABLISHED", "T3_SPECULATIVE", "T4_FALSIFIED"]
    consensus_version: int = Field(ge=1)
    taxonomy_code: Annotated[str, Field(pattern=TAXONOMY_PATTERN)]
    title: str = Field(min_length=3)
    formal_signature: list[str] = Field(min_length=1)
    assumptions: list[str]
    assumptions_hash: Annotated[str, Field(pattern=SHA256_PREFIXED_PATTERN)]
    verification: VerificationMeta
    provenance: list[ProvenanceEntry] = Field(min_length=1)
    re_evaluate_after: str

    @field_validator("consensus_version", mode="before")
    @classmethod
    def _validate_version(cls, value: Any) -> int:
        return _reject_bool_int(value, "consensus_version")

    @field_validator("formal_signature", "assumptions")
    @classmethod
    def _validate_non_empty_items(cls, items: list[str], info: Any) -> list[str]:
        for idx, item in enumerate(items):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(f"{info.field_name}[{idx}] must be a non-empty string")
        return items

    @field_validator("re_evaluate_after", mode="before")
    @classmethod
    def _validate_re_evaluate_after(cls, value: Any) -> str:
        if isinstance(value, date) and not isinstance(value, datetime):
            return value.isoformat()
        if not isinstance(value, str) or not ISO_DATE_RE.match(value.strip()):
            raise ValueError("re_evaluate_after must be an ISO date YYYY-MM-DD")
        date.fromisoformat(value.strip())
        return value.strip()

    @model_validator(mode="after")
    def _validate_epistemic_invariants(self) -> NodeFrontmatter:
        if self.epistemic_tier in ("T1_CANONICAL", "T2_ESTABLISHED"):
            if self.verification.oracle_status != "PASS":
                raise ValueError(f"{self.epistemic_tier} requires verification.oracle_status == 'PASS'")
            if any(p.retracted for p in self.provenance):
                raise ValueError(f"{self.epistemic_tier} cannot rely on retracted provenance")
        return self


@dataclass(frozen=True)
class NodeValidationResult:
    """Outcome of validating a single Markdown vault node."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    frontmatter: NodeFrontmatter | None = None
    prose_word_count: int = 0
    kernel_code: str | None = None


def load_allowed_licenses(allowlist_path: Path | None = None) -> set[str]:
    """Load the Open-Access license allowlist from YAML."""
    path = allowlist_path or (_REPO_ROOT / "config" / "license_allowlist.yaml")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    licenses = data.get("allowed_licenses", [])
    if not isinstance(licenses, list) or not licenses:
        raise ValueError(f"Invalid or empty allowed_licenses in {path}")
    return {str(item).strip() for item in licenses if str(item).strip()}


def split_frontmatter(raw_text: str) -> tuple[dict[str, Any], str]:
    """Split YAML frontmatter and Markdown body from a node document."""
    text = normalize_newlines(raw_text).lstrip("\ufeff")
    if not text.startswith("---\n"):
        raise ValueError("Missing opening YAML frontmatter delimiter '---'")

    rest = text[4:]
    end_idx = rest.find("\n---\n")
    if end_idx == -1:
        if rest.endswith("\n---"):
            end_idx = len(rest) - 4
            body = ""
        else:
            raise ValueError("Missing closing YAML frontmatter delimiter '---'")
    else:
        body = rest[end_idx + 5 :]

    yaml_text = rest[:end_idx]
    parsed = yaml.safe_load(yaml_text)
    if not isinstance(parsed, dict):
        raise ValueError("YAML frontmatter must deserialize to a mapping/dict")
    return parsed, body


def extract_verification_kernel(body: str) -> str | None:
    """Extract the active Executable Verification Kernel Python code block."""
    active_body = _DETAILS_BLOCK_RE.sub("", normalize_newlines(body))
    heading_match = _KERNEL_HEADING_RE.search(active_body)
    if not heading_match:
        return None
    fence_match = _PY_FENCE_RE.match(active_body[heading_match.end() :])
    if not fence_match:
        return None
    return fence_match.group(1)


def extract_natural_prose(body: str) -> str:
    """Strip code blocks, LaTeX math, Markdown tables, and archived `<details>` blocks."""
    text = normalize_newlines(body)
    text = _DETAILS_BLOCK_RE.sub(" ", text)
    text = _FENCE_ANY_RE.sub(" ", text)
    text = _DISPLAY_MATH_DOLLAR_RE.sub(" ", text)
    text = _DISPLAY_MATH_BRACKET_RE.sub(" ", text)
    text = _INLINE_MATH_DOLLAR_RE.sub(" ", text)
    text = _INLINE_MATH_PAREN_RE.sub(" ", text)

    kept_lines: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped.startswith("|") or re.match(r"^[-:| ]+$", stripped):
            continue
        cleaned = re.sub(r"^\s{0,3}#{1,6}\s*", "", line)
        cleaned = re.sub(r"^\s*[-*+]\s+", "", cleaned)
        kept_lines.append(cleaned)

    prose = "\n".join(kept_lines)
    prose = _INLINE_CODE_RE.sub(" ", prose)
    prose = _MD_LINK_RE.sub(r"\1", prose)
    prose = _URL_RE.sub(" ", prose)
    prose = prose.replace("*", " ").replace("_", " ")
    return re.sub(r"\s+", " ", prose).strip()


def count_prose_words(body: str) -> int:
    """Count natural-language prose words outside LaTeX, tables, code, and `<details>`."""
    prose = extract_natural_prose(body)
    if not prose:
        return 0
    return len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", prose))


def find_banned_filler(prose: str) -> list[str]:
    """Return any conversational LLM filler phrases detected in prose."""
    lowered = prose.lower()
    return [phrase for phrase in BANNED_FILLER_PHRASES if phrase in lowered]


def validate_node_content(
    raw_text: str,
    *,
    allowed_licenses: set[str] | None = None,
    expected_filename_stem: str | None = None,
    max_prose_words: int = 350,
) -> NodeValidationResult:
    """Run all Section 8.1 node integrity checks against raw Markdown node content."""
    errors: list[str] = []
    try:
        frontmatter_dict, body = split_frontmatter(raw_text)
    except (ValueError, yaml.YAMLError) as exc:
        return NodeValidationResult(ok=False, errors=[f"FRONTMATTER_PARSE_ERROR: {exc}"])

    try:
        fm = NodeFrontmatter.model_validate(frontmatter_dict)
    except ValidationError as exc:
        for err in exc.errors():
            loc = ".".join(str(p) for p in err["loc"])
            errors.append(f"SCHEMA_ERROR({loc}): {err['msg']}")
        return NodeValidationResult(ok=False, errors=errors)

    if expected_filename_stem is not None and expected_filename_stem != fm.ral_id:
        errors.append(
            f"FILENAME_MISMATCH: file stem {expected_filename_stem!r} != ral_id {fm.ral_id!r}"
        )

    if allowed_licenses is not None:
        for idx, prov in enumerate(fm.provenance):
            if prov.source_license not in allowed_licenses:
                errors.append(
                    f"DISALLOWED_LICENSE(provenance[{idx}]): {prov.source_license!r} not in allowlist"
                )

    for heading in ("## Active Consensus", "### Formal Statement", "### Executable Verification Kernel"):
        if heading not in body:
            errors.append(f"MISSING_SECTION: required heading {heading!r} not found")

    recomputed_assumptions_hash = compute_assumptions_hash(fm.assumptions)
    if recomputed_assumptions_hash != fm.assumptions_hash:
        errors.append(
            f"ASSUMPTIONS_HASH_MISMATCH: expected {recomputed_assumptions_hash}, got {fm.assumptions_hash}"
        )

    kernel_code = extract_verification_kernel(body)
    if kernel_code is None:
        errors.append("MISSING_KERNEL: no fenced ```python block under ### Executable Verification Kernel")
    else:
        recomputed_script_hash = compute_oracle_script_sha256(kernel_code)
        if recomputed_script_hash != fm.verification.oracle_script_sha256:
            errors.append(
                f"ORACLE_SCRIPT_HASH_MISMATCH: expected {recomputed_script_hash}, got {fm.verification.oracle_script_sha256}"
            )
        ast_violations = lint_verification_script(kernel_code, min_assertions=3, check_anti_cheat=True)
        for violation in ast_violations:
            errors.append(f"AST_LINT_ERROR: {violation}")

    prose = extract_natural_prose(body)
    prose_words = len(re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", prose)) if prose else 0
    if prose_words > max_prose_words:
        errors.append(f"PROSE_CAP_EXCEEDED: {prose_words} words > limit {max_prose_words}")

    filler_hits = find_banned_filler(prose)
    for hit in filler_hits:
        errors.append(f"BANNED_FILLER_PHRASE: {hit!r}")

    return NodeValidationResult(
        ok=len(errors) == 0,
        errors=errors,
        frontmatter=fm,
        prose_word_count=prose_words,
        kernel_code=kernel_code,
    )


def validate_node_file(
    path: Path,
    *,
    allowlist_path: Path | None = None,
    check_filename: bool = False,
    allowed_licenses: set[str] | None = None,
) -> NodeValidationResult:
    """Validate a single Markdown node file on disk."""
    licenses = allowed_licenses
    if licenses is None:
        resolved_allowlist = allowlist_path or (_REPO_ROOT / "config" / "license_allowlist.yaml")
        if resolved_allowlist.exists():
            licenses = load_allowed_licenses(resolved_allowlist)

    raw_text = path.read_text(encoding="utf-8")
    return validate_node_content(
        raw_text,
        allowed_licenses=licenses,
        expected_filename_stem=path.stem if check_filename else None,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate RAL canonical Markdown node files.")
    parser.add_argument("nodes", nargs="+", type=Path, help="Path(s) to vault node .md files")
    parser.add_argument(
        "--allowlist",
        type=Path,
        default=_REPO_ROOT / "config" / "license_allowlist.yaml",
        help="Path to license_allowlist.yaml",
    )
    parser.add_argument(
        "--check-filename",
        action="store_true",
        help="Require filename stem to match frontmatter ral_id",
    )
    args = parser.parse_args(argv)

    licenses = load_allowed_licenses(args.allowlist) if args.allowlist.exists() else None
    all_ok = True
    for node_path in args.nodes:
        result = validate_node_file(
            node_path,
            allowed_licenses=licenses,
            check_filename=args.check_filename,
        )
        if result.ok:
            print(f"PASS {node_path} ({result.prose_word_count} prose words)")
        else:
            all_ok = False
            print(f"FAIL {node_path}", file=sys.stderr)
            for err in result.errors:
                print(f"  - {err}", file=sys.stderr)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

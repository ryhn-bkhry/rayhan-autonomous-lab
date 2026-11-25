"""Stage 5: Taxonomy Architect, Append-Only Theory-Upgrade Protocol, and 2PC Drive Commit (Sections 6.3 & 6.4)."""

from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime, timezone
import difflib
import json
import math
from pathlib import Path
import re
from typing import Any, Literal, Mapping, Sequence
import uuid

import yaml

from ci.validate_node import (
    NodeFrontmatter,
    load_allowed_licenses,
    split_frontmatter,
    validate_node_content,
    validate_node_file,
)
from ci.validate_vault import scan_text_for_leaks, validate_vault
from ral.canonicalize import normalize_newlines, sha256_bytes
from ral.pipeline.formalizer import (
    FormalizedNodeDraft,
    _strip_outer_math_delimiters,
    generate_ral_id,
    render_node_markdown,
)
from ral.pipeline.oracle_sandbox import OracleSandbox
from ral.sources import SourceRecord
from ral.storage.drive_ore import DriveOreStore, OreUploadResult, build_ore_uri

TIER_RANK: dict[str, int] = {
    "T1_CANONICAL": 1,
    "T2_ESTABLISHED": 2,
    "T3_SPECULATIVE": 3,
    "T4_FALSIFIED": 4,
}

_ACTIVE_CONSENSUS_BLOCK_RE = re.compile(
    r"(##\s+Active Consensus[^\n]*\n.*?)(?=\n##\s+Active Challenges & Open Anomalies)",
    re.DOTALL,
)
_EFFECTIVE_DATE_RE = re.compile(r"Effective\s+(\d{4}-\d{2}-\d{2})")
_CHALLENGES_SECTION_RE = re.compile(
    r"(##\s+Active Challenges & Open Anomalies\s*\n)(.*?)(?=\n##\s+Superseded & Archived|\Z)",
    re.DOTALL,
)
_ARCHIVED_SECTION_RE = re.compile(
    r"(##\s+Superseded & Archived\s*\n)(.*)\Z",
    re.DOTALL,
)
_ANCHORS_SECTION_RE = re.compile(
    r"(###\s+Canonical Provenance Anchors\s*\n)(.*?)(?=\n##\s+Active Challenges & Open Anomalies)",
    re.DOTALL,
)
_DETAILS_BLOCK_RE = re.compile(r"<details\b[^>]*>.*?</details>", re.IGNORECASE | re.DOTALL)
_FORMAL_STMT_RE = re.compile(
    r"^###\s+Formal Statement\s*\n(.*?)(?=\n#{2,3}\s|\Z)",
    re.MULTILINE | re.DOTALL,
)
_MD_TABLE_SPLIT_RE = re.compile(r"\n\s*\|[^\n]*\|\s*\n\s*\|[\s:|-]+\|")


class VaultCommitValidationError(RuntimeError):
    """Raised when blocking `validate_node.py` or `validate_vault.py` fails during 2PC commit."""


@dataclass(frozen=True)
class ArchitectResult:
    """Outcome of Stage 5 archival, merge, challenge, or supersession."""

    action: Literal[
        "CREATED",
        "MERGED_PROVENANCE",
        "CHALLENGED_ONLY",
        "SUPERSEDED",
        "REJECTED_UPGRADE_GATE",
    ]
    ral_id: str
    node_path: Path | None
    challenge_node_path: Path | None
    consensus_version: int
    ore_upload: OreUploadResult | None = None
    lineage_record: dict[str, Any] | None = None
    edge_record: dict[str, Any] | None = None
    reason: str = ""


def jaccard_similarity(sig_a: Sequence[str], sig_b: Sequence[str]) -> float:
    """Compute Jaccard similarity over normalized `formal_signature` tags."""
    set_a = {s.strip().lower() for s in sig_a if s.strip()}
    set_b = {s.strip().lower() for s in sig_b if s.strip()}
    if not set_a or not set_b:
        return 0.0
    return len(set_a & set_b) / len(set_a | set_b)


def cosine_similarity(vec_a: Sequence[float], vec_b: Sequence[float]) -> float:
    """Compute cosine similarity between two equal-length vectors."""
    if len(vec_a) != len(vec_b) or not vec_a:
        return 0.0
    dot = sum(float(a) * float(b) for a, b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(float(a) * float(a) for a in vec_a))
    norm_b = math.sqrt(sum(float(b) * float(b) for b in vec_b))
    if norm_a <= 1e-12 or norm_b <= 1e-12:
        return 0.0
    return dot / (norm_a * norm_b)


def _extract_formal_statement_from_body(body: str) -> str:
    """Extract the active Formal Statement block (excluding symbol table and `<details>`) from node body."""
    active_body = _DETAILS_BLOCK_RE.sub("", normalize_newlines(body))
    match = _FORMAL_STMT_RE.search(active_body)
    if not match:
        return ""
    raw_stmt = _MD_TABLE_SPLIT_RE.split(match.group(1), maxsplit=1)[0].strip()
    return _strip_outer_math_delimiters(raw_stmt)


def _normalize_compare_text(text: str | None) -> str:
    cleaned = _strip_outer_math_delimiters(text)
    return re.sub(r"\s+", " ", cleaned).strip().lower()


def _text_similarity(text_a: str | None, text_b: str | None) -> float:
    na = _normalize_compare_text(text_a)
    nb = _normalize_compare_text(text_b)
    if not na or not nb:
        return 0.0
    return difflib.SequenceMatcher(None, na, nb).ratio()


def find_matching_node(
    repo_root: Path,
    formal_signature: Sequence[str],
    *,
    candidate_title: str | None = None,
    candidate_statement: str | None = None,
    candidate_embedding: Sequence[float] | None = None,
    existing_embeddings: Mapping[str, Sequence[float]] | None = None,
    embed_space_consistent: bool = True,
    cosine_threshold: float = 0.88,
    jaccard_threshold: float = 1.0,
    text_similarity_threshold: float = 0.85,
) -> tuple[Path, NodeFrontmatter] | None:
    """
    Locate an existing canonical node matching via cosine >= 0.88 (when consistent embeddings
    exist) OR exact `formal_signature` match (`jac == 1.0`, >=3 non-fallback tags) with
    normalized statement/title similarity >= 0.85. Never merges on `("operator:identity",)`.
    """
    vault_dir = Path(repo_root) / "vault"
    if not vault_dir.exists():
        return None

    cand_tags = {str(s).strip().lower() for s in formal_signature if str(s).strip()}
    cand_non_fallback = cand_tags - {"operator:identity"}
    allowlist_path = Path(repo_root) / "config" / "license_allowlist.yaml"

    best_match: tuple[float, Path, NodeFrontmatter] | None = None
    for md_path in sorted(vault_dir.rglob("*.md")):
        res = validate_node_file(md_path, allowlist_path=allowlist_path)
        if res.frontmatter is None:
            continue
        fm = res.frontmatter

        cos = 0.0
        has_embedding_match = False
        if (
            embed_space_consistent
            and candidate_embedding is not None
            and existing_embeddings is not None
            and fm.ral_id in existing_embeddings
        ):
            cos = cosine_similarity(candidate_embedding, existing_embeddings[fm.ral_id])
            has_embedding_match = cos >= cosine_threshold

        exist_tags = {str(s).strip().lower() for s in fm.formal_signature if str(s).strip()}
        exist_non_fallback = exist_tags - {"operator:identity"}
        jac = jaccard_similarity(formal_signature, fm.formal_signature)

        has_sig_match = False
        text_sim = 0.0
        if (
            cand_tags != {"operator:identity"}
            and exist_tags != {"operator:identity"}
            and len(cand_non_fallback) >= 3
            and len(exist_non_fallback) >= 3
            and jac >= jaccard_threshold
        ):
            try:
                raw_md = normalize_newlines(md_path.read_text(encoding="utf-8"))
                _, body = split_frontmatter(raw_md)
                existing_stmt = _extract_formal_statement_from_body(body)
            except Exception:
                existing_stmt = ""
            stmt_sim = _text_similarity(candidate_statement, existing_stmt)
            title_sim = _text_similarity(candidate_title, fm.title)
            text_sim = max(stmt_sim, title_sim)
            has_sig_match = text_sim >= text_similarity_threshold

        if has_embedding_match or has_sig_match:
            score = max(cos if has_embedding_match else 0.0, text_sim if has_sig_match else 0.0)
            if best_match is None or score > best_match[0]:
                best_match = (score, md_path, fm)

    return (best_match[1], best_match[2]) if best_match is not None else None


def _node_relative_path(repo_root: Path, taxonomy_code: str, ral_id: str) -> Path:
    parts = [p.strip() for p in taxonomy_code.split(".") if p.strip()]
    return Path(repo_root) / "vault" / Path(*parts) / f"{ral_id}.md"


def _append_ndjson(path: Path, row: Mapping[str, Any]) -> tuple[bool, int]:
    """Append one JSON row and return `(existed_before, previous_byte_size)` for rollback."""
    existed = path.exists()
    prev_size = path.stat().st_size if existed else 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(row), sort_keys=False) + "\n")
    return existed, prev_size


def _rollback_mutations(
    created_files: Sequence[Path],
    modified_files: Mapping[Path, bytes],
    appended_ndjsons: Sequence[tuple[Path, bool, int]],
) -> None:
    for path in created_files:
        if path.exists():
            path.unlink()
    for path, original_bytes in modified_files.items():
        path.write_bytes(original_bytes)
    for path, existed, prev_size in appended_ndjsons:
        if not path.exists():
            continue
        if not existed:
            path.unlink()
        else:
            with path.open("r+b") as fh:
                fh.truncate(prev_size)


def _new_id(prefix: str) -> str:
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    raw = uuid.uuid4().bytes[:10]
    return f"{prefix}-" + "".join(alphabet[b % len(alphabet)] for b in raw)


def _build_merged_provenance_markdown(
    raw_text: str,
    new_provenance: Mapping[str, Any],
    anchor_summary: str,
) -> str:
    """Build merged node Markdown with appended provenance entry without mutating disk."""
    raw = normalize_newlines(raw_text)
    fm_dict, body = split_frontmatter(raw)
    prov_list: list[dict[str, Any]] = list(fm_dict.get("provenance") or [])
    already_present = any(
        p.get("source_id") == new_provenance.get("source_id")
        and p.get("quote_sha256") == new_provenance.get("quote_sha256")
        for p in prov_list
    )
    if not already_present:
        prov_list.append(dict(new_provenance))
        fm_dict["provenance"] = prov_list

    anchor_line = (
        f"- `{new_provenance['source_id']}` (`{new_provenance['source_locator']}`): {anchor_summary}"
    )
    if anchor_line not in body:
        match = _ANCHORS_SECTION_RE.search(body)
        if match:
            sec_body = match.group(2).rstrip()
            updated_sec = f"{sec_body}\n{anchor_line}\n" if sec_body else f"{anchor_line}\n"
            body = body[: match.start(2)] + updated_sec + body[match.end(2) :]

    return f"---\n{yaml.safe_dump(fm_dict, sort_keys=False).strip()}\n---\n{body.lstrip()}"


def merge_provenance_into_node(
    node_path: Path,
    new_provenance: Mapping[str, Any],
    anchor_summary: str,
    *,
    allowed_licenses: set[str] | None = None,
) -> str:
    """
    Zero-redundancy provenance merge (Section 6.3): append a new provenance entry without
    creating a duplicate node.
    """
    updated_md = _build_merged_provenance_markdown(
        node_path.read_text(encoding="utf-8"),
        new_provenance,
        anchor_summary,
    )
    res = validate_node_content(
        updated_md,
        allowed_licenses=allowed_licenses,
        expected_filename_stem=node_path.stem,
    )
    if not res.ok:
        raise VaultCommitValidationError(f"Merged node failed validation: {res.errors}")
    leaks = scan_text_for_leaks(updated_md, node_path.name)
    if leaks:
        raise VaultCommitValidationError(f"Merged node failed leak scan: {leaks}")
    node_path.write_text(updated_md, encoding="utf-8")
    return updated_md


class ArchitectStage:
    """Stage 5 archival, zero-redundancy merge, Theory-Upgrade Protocol, and Drive 2PC."""

    def __init__(
        self,
        repo_root: Path,
        drive_store: DriveOreStore,
        sandbox: OracleSandbox | None = None,
    ) -> None:
        self.repo_root = Path(repo_root)
        self._drive = drive_store
        self._sandbox = sandbox or OracleSandbox()

    def _allowed_licenses(self) -> set[str] | None:
        allow_path = self.repo_root / "config" / "license_allowlist.yaml"
        return load_allowed_licenses(allow_path) if allow_path.exists() else None

    def _pre_upload_validate_markdown(
        self,
        markdown: str,
        *,
        expected_stem: str,
        source_label: str,
    ) -> None:
        """Validate schema, kernel AST, prose cap, and leak checks BEFORE Drive Ore upload."""
        pre_res = validate_node_content(
            markdown,
            allowed_licenses=self._allowed_licenses(),
            expected_filename_stem=expected_stem,
        )
        if not pre_res.ok:
            raise VaultCommitValidationError(f"Pre-upload node validation failed: {pre_res.errors}")
        leaks = scan_text_for_leaks(markdown, source_label)
        if leaks:
            raise VaultCommitValidationError(f"Pre-upload leak scan failed: {leaks}")

    def _run_blocking_validation(self, node_path: Path) -> None:
        allow_path = self.repo_root / "config" / "license_allowlist.yaml"
        n_res = validate_node_file(
            node_path,
            allowlist_path=allow_path if allow_path.exists() else None,
            check_filename=True,
        )
        if not n_res.ok:
            raise VaultCommitValidationError(
                f"Blocking node validation failed for {node_path}: {n_res.errors}"
            )
        v_res = validate_vault(
            self.repo_root,
            allowlist_path=allow_path if allow_path.exists() else None,
        )
        if not v_res.ok:
            raise VaultCommitValidationError(
                f"Blocking vault validation failed: {v_res.errors}"
            )

    def _source_ledger_row(
        self,
        *,
        ore_upload: OreUploadResult,
        source: SourceRecord,
        linked_ral_ids: Sequence[str],
        ts_iso: str,
    ) -> dict[str, Any]:
        return {
            "ore_sha256": ore_upload.ore_sha256,
            "canonical_doi": source.canonical_doi,
            "arxiv_base_id": source.arxiv_base_id,
            "arxiv_version": source.arxiv_version,
            "source_license": source.source_license,
            "ore_uri": ore_upload.ore_uri,
            "linked_ral_ids": list(linked_ral_ids),
            "ingested_at": ts_iso,
        }

    def _ral_id_exists_in_vault(self, ral_id: str) -> bool:
        vault_dir = self.repo_root / "vault"
        if not vault_dir.exists():
            return False
        return any(p.stem == ral_id for p in vault_dir.rglob("*.md"))

    def commit_new_node(
        self,
        draft: FormalizedNodeDraft,
        *,
        epistemic_tier: str,
        re_evaluate_after: str,
        skeptic_model: str,
        source: SourceRecord,
        quote_sha256: str,
        quote_char_len: int,
        source_locator: str,
        now: datetime | None = None,
    ) -> ArchitectResult:
        """
        Execute Verify-Before-Archive 2PC for a newly verified claim node:
        resolve collisions -> pre-validate -> upload Ore to Drive -> write Markdown + source NDJSON -> block on CI gates.
        """
        ts = now or datetime.now(timezone.utc)
        ts_iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ")

        active_draft = draft
        node_path = _node_relative_path(self.repo_root, active_draft.taxonomy_code, active_draft.ral_id)
        if node_path.exists():
            stmt_sim = 0.0
            title_sim = 0.0
            existing_stmt = ""
            try:
                existing_raw = normalize_newlines(node_path.read_text(encoding="utf-8"))
                existing_fm, existing_body = split_frontmatter(existing_raw)
                existing_stmt = _extract_formal_statement_from_body(existing_body)
                stmt_sim = _text_similarity(active_draft.formal_statement_latex, existing_stmt)
                title_sim = _text_similarity(active_draft.title, str(existing_fm.get("title") or ""))
            except Exception:
                pass
            if stmt_sim >= 0.85 or (not existing_stmt and title_sim >= 0.85):
                return self.merge_into_existing_node(
                    node_path,
                    source=source,
                    quote_sha256=quote_sha256,
                    quote_char_len=quote_char_len,
                    source_locator=source_locator,
                    anchor_summary=active_draft.provenance_summary,
                    now=ts,
                )

        nonce = 1
        while node_path.exists() or self._ral_id_exists_in_vault(active_draft.ral_id):
            new_id = generate_ral_id(
                active_draft.taxonomy_code,
                entropy=f"{draft.ral_id}:{source.source_id}:{quote_sha256}:collision:{nonce}",
            )
            active_draft = replace(active_draft, ral_id=new_id)
            node_path = _node_relative_path(
                self.repo_root, active_draft.taxonomy_code, active_draft.ral_id
            )
            nonce += 1

        local_ore_sha = sha256_bytes(source.raw_bytes)
        expected_uri = build_ore_uri(local_ore_sha, source.raw_ext, warehouse_name=self._drive.warehouse_name)

        preview_md = render_node_markdown(
            active_draft,
            epistemic_tier=epistemic_tier,
            oracle_status="PASS",
            skeptic_model=skeptic_model,
            source=source,
            quote_sha256=quote_sha256,
            quote_char_len=quote_char_len,
            source_locator=source_locator,
            re_evaluate_after=re_evaluate_after,
            consensus_version=1,
            ore_uri=expected_uri,
            ore_sha256=local_ore_sha,
            verified_at=ts_iso,
        )
        self._pre_upload_validate_markdown(
            preview_md,
            expected_stem=active_draft.ral_id,
            source_label=f"{active_draft.ral_id}.md",
        )

        ore_upload = self._drive.upload_ore_bytes(
            source.raw_bytes,
            source.raw_ext,
            expected_sha256=local_ore_sha,
        )
        if ore_upload.ore_uri != expected_uri or ore_upload.ore_sha256 != local_ore_sha:
            preview_md = render_node_markdown(
                active_draft,
                epistemic_tier=epistemic_tier,
                oracle_status="PASS",
                skeptic_model=skeptic_model,
                source=source,
                quote_sha256=quote_sha256,
                quote_char_len=quote_char_len,
                source_locator=source_locator,
                re_evaluate_after=re_evaluate_after,
                consensus_version=1,
                ore_uri=ore_upload.ore_uri,
                ore_sha256=ore_upload.ore_sha256,
                verified_at=ts_iso,
            )

        created_files: list[Path] = []
        appended: list[tuple[Path, bool, int]] = []
        try:
            node_path.parent.mkdir(parents=True, exist_ok=True)
            node_path.write_text(preview_md, encoding="utf-8")
            created_files.append(node_path)

            src_ledger = self.repo_root / "ledger" / "sources" / f"{ts:%Y-%m}.ndjson"
            appended.append(
                _append_ndjson(
                    src_ledger,
                    self._source_ledger_row(
                        ore_upload=ore_upload,
                        source=source,
                        linked_ral_ids=[active_draft.ral_id],
                        ts_iso=ts_iso,
                    ),
                )
            )
            self._run_blocking_validation(node_path)
        except Exception:
            _rollback_mutations(created_files, {}, appended)
            raise

        return ArchitectResult(
            action="CREATED",
            ral_id=active_draft.ral_id,
            node_path=node_path,
            challenge_node_path=None,
            consensus_version=1,
            ore_upload=ore_upload,
            reason="Verified and committed new node via 2PC",
        )

    def merge_into_existing_node(
        self,
        existing_node_path: Path,
        *,
        source: SourceRecord,
        quote_sha256: str,
        quote_char_len: int,
        source_locator: str,
        anchor_summary: str,
        now: datetime | None = None,
    ) -> ArchitectResult:
        """Merge corroborating provenance into an existing node with zero duplicate node creation."""
        ts = now or datetime.now(timezone.utc)
        ts_iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        local_ore_sha = sha256_bytes(source.raw_bytes)
        expected_uri = build_ore_uri(
            local_ore_sha,
            source.raw_ext,
            warehouse_name=self._drive.warehouse_name,
        )

        original_bytes = existing_node_path.read_bytes()
        original_text = original_bytes.decode("utf-8")
        fm_dict, _ = split_frontmatter(original_text)
        ral_id = str(fm_dict["ral_id"])
        version = int(fm_dict["consensus_version"])
        new_prov = {
            "source_id": source.source_id,
            "source_license": source.source_license,
            "source_format": source.source_format,
            "source_locator": source_locator,
            "ore_uri": expected_uri,
            "ore_sha256": local_ore_sha,
            "quote_sha256": quote_sha256,
            "quote_char_len": int(quote_char_len),
            "pub_year": int(source.pub_year),
            "peer_reviewed": bool(source.peer_reviewed),
            "retracted": bool(source.retracted),
        }
        preview_merged_md = _build_merged_provenance_markdown(
            original_text,
            new_prov,
            anchor_summary,
        )
        self._pre_upload_validate_markdown(
            preview_merged_md,
            expected_stem=existing_node_path.stem,
            source_label=f"{ral_id}.md",
        )

        ore_upload = self._drive.upload_ore_bytes(
            source.raw_bytes,
            source.raw_ext,
            expected_sha256=local_ore_sha,
        )
        new_prov["ore_uri"] = ore_upload.ore_uri
        new_prov["ore_sha256"] = ore_upload.ore_sha256

        appended: list[tuple[Path, bool, int]] = []
        try:
            merge_provenance_into_node(
                existing_node_path,
                new_prov,
                anchor_summary,
                allowed_licenses=self._allowed_licenses(),
            )
            src_ledger = self.repo_root / "ledger" / "sources" / f"{ts:%Y-%m}.ndjson"
            appended.append(
                _append_ndjson(
                    src_ledger,
                    self._source_ledger_row(
                        ore_upload=ore_upload,
                        source=source,
                        linked_ral_ids=[ral_id],
                        ts_iso=ts_iso,
                    ),
                )
            )
            self._run_blocking_validation(existing_node_path)
        except Exception:
            _rollback_mutations([], {existing_node_path: original_bytes}, appended)
            raise

        return ArchitectResult(
            action="MERGED_PROVENANCE",
            ral_id=ral_id,
            node_path=existing_node_path,
            challenge_node_path=None,
            consensus_version=version,
            ore_upload=ore_upload,
            reason="Zero-redundancy provenance merge",
        )

    def apply_theory_upgrade(
        self,
        existing_node_path: Path,
        candidate_draft: FormalizedNodeDraft,
        candidate_tier: str,
        *,
        re_evaluate_after: str,
        skeptic_model: str,
        source: SourceRecord,
        quote_sha256: str,
        quote_char_len: int,
        source_locator: str,
        reason: str = "DOMAIN_GENERALIZATION",
        correspondence_script: str | None = None,
        falsification_script: str | None = None,
        retraction_confirmed: bool = False,
        now: datetime | None = None,
    ) -> ArchitectResult:
        """
        Execute Section 6.4 Append-Only Theory-Upgrade Protocol:
        1. Preprint Firewall: if tier(new) > tier(old) (e.g. T3 challenging T1), leave Active
           Consensus untouched, emit CHALLENGES edge, and append pointer under Active Challenges.
        2. Full Supersession Gate: requires tier(new) <= tier(old) AND passing sandbox
           Correspondence Test, sandbox Falsification script, or confirmed retraction.
        3. Append-Only Mutation: immutable ral_id, increment consensus_version, move prior
           Active Consensus verbatim into `<details>`, and record `SUPERSEDES` in `graph/lineage/`.
        """
        ts = now or datetime.now(timezone.utc)
        ts_iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
        today_str = ts_iso[:10]

        existing_raw = normalize_newlines(existing_node_path.read_text(encoding="utf-8"))
        existing_fm, existing_body = split_frontmatter(existing_raw)
        existing_ral_id = str(existing_fm["ral_id"])
        existing_tier = str(existing_fm["epistemic_tier"])
        old_version = int(existing_fm["consensus_version"])

        # 1. Preprint Firewall: weaker tier (higher rank number) NEVER modifies Active Consensus.
        if TIER_RANK[candidate_tier] > TIER_RANK[existing_tier]:
            challenge_ral_id = (
                candidate_draft.ral_id
                if candidate_draft.ral_id != existing_ral_id
                else generate_ral_id(
                    candidate_draft.taxonomy_code,
                    entropy=f"challenge:{existing_ral_id}:{source.source_id}:{quote_sha256}",
                )
            )
            challenge_draft = FormalizedNodeDraft(
                ral_id=challenge_ral_id,
                node_type=candidate_draft.node_type,
                taxonomy_code=candidate_draft.taxonomy_code,
                title=candidate_draft.title,
                formal_signature=candidate_draft.formal_signature,
                assumptions=candidate_draft.assumptions,
                formal_statement_latex=candidate_draft.formal_statement_latex,
                symbol_rows=candidate_draft.symbol_rows,
                failure_boundaries=candidate_draft.failure_boundaries,
                provenance_summary=candidate_draft.provenance_summary,
                oracle_plugin=candidate_draft.oracle_plugin,
                verify_py=candidate_draft.verify_py,
                formalizer_model=candidate_draft.formalizer_model,
                model_capability_epoch=candidate_draft.model_capability_epoch,
                oracle_timeout_s=candidate_draft.oracle_timeout_s,
            )

            ch_match = _CHALLENGES_SECTION_RE.search(existing_body)
            if not ch_match:
                raise VaultCommitValidationError(
                    f"Existing node {existing_ral_id} missing '## Active Challenges & Open Anomalies'"
                )
            pointer_line = (
                f"{today_str} [{candidate_tier}] `{challenge_ral_id}` (`{source.source_id}`): {reason}"
            )
            ch_content = ch_match.group(2).strip()
            if not ch_content or ch_content == "- *(None)*":
                new_ch_content = f"- {pointer_line}\n"
            else:
                new_ch_content = f"{ch_content}\n- {pointer_line}\n"
            updated_existing_body = (
                existing_body[: ch_match.start(2)]
                + new_ch_content
                + existing_body[ch_match.end(2) :]
            )
            updated_existing_md = (
                f"---\n{yaml.safe_dump(existing_fm, sort_keys=False).strip()}\n---\n"
                f"{updated_existing_body.lstrip()}"
            )

            local_ore_sha = sha256_bytes(source.raw_bytes)
            expected_uri = build_ore_uri(
                local_ore_sha,
                source.raw_ext,
                warehouse_name=self._drive.warehouse_name,
            )

            challenge_md = render_node_markdown(
                challenge_draft,
                epistemic_tier=candidate_tier,
                oracle_status="PASS",
                skeptic_model=skeptic_model,
                source=source,
                quote_sha256=quote_sha256,
                quote_char_len=quote_char_len,
                source_locator=source_locator,
                re_evaluate_after=re_evaluate_after,
                consensus_version=1,
                ore_uri=expected_uri,
                ore_sha256=local_ore_sha,
                verified_at=ts_iso,
            )
            self._pre_upload_validate_markdown(
                challenge_md,
                expected_stem=challenge_ral_id,
                source_label=f"{challenge_ral_id}.md",
            )
            self._pre_upload_validate_markdown(
                updated_existing_md,
                expected_stem=existing_node_path.stem,
                source_label=f"{existing_ral_id}.md",
            )

            ore_upload = self._drive.upload_ore_bytes(
                source.raw_bytes,
                source.raw_ext,
                expected_sha256=local_ore_sha,
            )
            if ore_upload.ore_uri != expected_uri or ore_upload.ore_sha256 != local_ore_sha:
                challenge_md = render_node_markdown(
                    challenge_draft,
                    epistemic_tier=candidate_tier,
                    oracle_status="PASS",
                    skeptic_model=skeptic_model,
                    source=source,
                    quote_sha256=quote_sha256,
                    quote_char_len=quote_char_len,
                    source_locator=source_locator,
                    re_evaluate_after=re_evaluate_after,
                    consensus_version=1,
                    ore_uri=ore_upload.ore_uri,
                    ore_sha256=ore_upload.ore_sha256,
                    verified_at=ts_iso,
                )
            challenge_path = _node_relative_path(
                self.repo_root, challenge_draft.taxonomy_code, challenge_ral_id
            )

            edge_row = {
                "edge_id": _new_id("E"),
                "src": challenge_ral_id,
                "dst": existing_ral_id,
                "relation": "CHALLENGES",
                "confidence": 0.85,
                "evidence_sha256": quote_sha256,
                "created_at": ts_iso,
                "created_by_model": candidate_draft.formalizer_model,
            }

            original_existing_bytes = existing_node_path.read_bytes()
            created_files: list[Path] = []
            appended: list[tuple[Path, bool, int]] = []
            try:
                challenge_path.parent.mkdir(parents=True, exist_ok=True)
                challenge_path.write_text(challenge_md, encoding="utf-8")
                created_files.append(challenge_path)

                existing_node_path.write_text(updated_existing_md, encoding="utf-8")

                src_ledger = self.repo_root / "ledger" / "sources" / f"{ts:%Y-%m}.ndjson"
                appended.append(
                    _append_ndjson(
                        src_ledger,
                        self._source_ledger_row(
                            ore_upload=ore_upload,
                            source=source,
                            linked_ral_ids=[challenge_ral_id],
                            ts_iso=ts_iso,
                        ),
                    )
                )
                edges_ledger = self.repo_root / "graph" / "edges" / f"{ts:%Y-%m}.ndjson"
                appended.append(_append_ndjson(edges_ledger, edge_row))
                self._run_blocking_validation(existing_node_path)
            except Exception:
                _rollback_mutations(
                    created_files,
                    {existing_node_path: original_existing_bytes},
                    appended,
                )
                raise

            return ArchitectResult(
                action="CHALLENGED_ONLY",
                ral_id=existing_ral_id,
                node_path=existing_node_path,
                challenge_node_path=challenge_path,
                consensus_version=old_version,
                ore_upload=ore_upload,
                edge_record=edge_row,
                reason=f"Preprint firewall blocked {candidate_tier} overwrite of {existing_tier}",
            )

        # 2. Full Supersession Gate: requires sandbox Correspondence or Falsification or Retraction.
        gate_passed = bool(retraction_confirmed)
        if not gate_passed and correspondence_script:
            corr_res = self._sandbox.run_script(
                correspondence_script,
                plugin_name="correspondence_test",
                script_name="correspondence.py",
                timeout_s=candidate_draft.oracle_timeout_s,
                enforce_ast_lint=True,
                min_assertions=1,
            )
            gate_passed = corr_res.passed
        if not gate_passed and falsification_script:
            fals_res = self._sandbox.run_script(
                falsification_script,
                plugin_name="falsification_test",
                script_name="falsify_old.py",
                timeout_s=candidate_draft.oracle_timeout_s,
                enforce_ast_lint=True,
                min_assertions=0,
            )
            gate_passed = fals_res.passed

        if not gate_passed:
            return ArchitectResult(
                action="REJECTED_UPGRADE_GATE",
                ral_id=existing_ral_id,
                node_path=existing_node_path,
                challenge_node_path=None,
                consensus_version=old_version,
                reason="Supersession rejected: failed correspondence/falsification/retraction gate",
            )

        # 3. Append-Only Mutation: keep ral_id immutable, archive prior Active Consensus verbatim.
        active_match = _ACTIVE_CONSENSUS_BLOCK_RE.search(existing_body)
        if not active_match:
            raise VaultCommitValidationError(
                f"Existing node {existing_ral_id} missing Active Consensus block"
            )
        prior_active_block = active_match.group(1).strip()
        date_match = _EFFECTIVE_DATE_RE.search(prior_active_block)
        old_date = (
            date_match.group(1)
            if date_match
            else str(existing_fm["verification"]["verified_at"])[:10]
        )

        archived_entry = (
            f"<details>\n"
            f"<summary>v{old_version} ({old_date} to {today_str}) | Reason: {reason}</summary>\n\n"
            f"{prior_active_block}\n"
            f"</details>"
        )
        arch_match = _ARCHIVED_SECTION_RE.search(existing_body)
        prior_archive_content = arch_match.group(2).strip() if arch_match else ""
        if prior_archive_content and prior_archive_content != "*(None)*":
            combined_archive = f"{archived_entry}\n\n{prior_archive_content}"
        else:
            combined_archive = archived_entry

        ch_match = _CHALLENGES_SECTION_RE.search(existing_body)
        existing_challenges: list[str] = []
        if ch_match:
            for line in ch_match.group(2).splitlines():
                s = line.strip()
                if s.startswith("- ") and s != "- *(None)*":
                    existing_challenges.append(s[2:].strip())

        local_ore_sha = sha256_bytes(source.raw_bytes)
        expected_uri = build_ore_uri(
            local_ore_sha,
            source.raw_ext,
            warehouse_name=self._drive.warehouse_name,
        )

        new_version = old_version + 1
        upgraded_draft = FormalizedNodeDraft(
            ral_id=existing_ral_id,
            node_type=candidate_draft.node_type,
            taxonomy_code=str(existing_fm["taxonomy_code"]),
            title=candidate_draft.title,
            formal_signature=candidate_draft.formal_signature,
            assumptions=candidate_draft.assumptions,
            formal_statement_latex=candidate_draft.formal_statement_latex,
            symbol_rows=candidate_draft.symbol_rows,
            failure_boundaries=candidate_draft.failure_boundaries,
            provenance_summary=candidate_draft.provenance_summary,
            oracle_plugin=candidate_draft.oracle_plugin,
            verify_py=candidate_draft.verify_py,
            formalizer_model=candidate_draft.formalizer_model,
            model_capability_epoch=candidate_draft.model_capability_epoch,
            oracle_timeout_s=candidate_draft.oracle_timeout_s,
        )

        def _render_upgraded(uri: str, sha: str) -> str:
            upgraded_md = render_node_markdown(
                upgraded_draft,
                epistemic_tier=candidate_tier,
                oracle_status="PASS",
                skeptic_model=skeptic_model,
                source=source,
                quote_sha256=quote_sha256,
                quote_char_len=quote_char_len,
                source_locator=source_locator,
                re_evaluate_after=re_evaluate_after,
                consensus_version=new_version,
                ore_uri=uri,
                ore_sha256=sha,
                verified_at=ts_iso,
                effective_date=today_str,
                active_challenges=existing_challenges,
                archived_details=combined_archive,
            )
            # Preserve prior provenance entries alongside the new superseding source.
            new_fm, new_body = split_frontmatter(upgraded_md)
            old_provs = list(existing_fm.get("provenance") or [])
            merged_provs = old_provs + [
                p
                for p in new_fm["provenance"]
                if not any(
                    op.get("source_id") == p.get("source_id")
                    and op.get("quote_sha256") == p.get("quote_sha256")
                    for op in old_provs
                )
            ]
            new_fm["provenance"] = merged_provs
            return f"---\n{yaml.safe_dump(new_fm, sort_keys=False).strip()}\n---\n{new_body.lstrip()}"

        final_md = _render_upgraded(expected_uri, local_ore_sha)
        self._pre_upload_validate_markdown(
            final_md,
            expected_stem=existing_node_path.stem,
            source_label=f"{existing_ral_id}.md",
        )

        ore_upload = self._drive.upload_ore_bytes(
            source.raw_bytes,
            source.raw_ext,
            expected_sha256=local_ore_sha,
        )
        if ore_upload.ore_uri != expected_uri or ore_upload.ore_sha256 != local_ore_sha:
            final_md = _render_upgraded(ore_upload.ore_uri, ore_upload.ore_sha256)

        lineage_row = {
            "lineage_id": _new_id("L"),
            "ral_id": existing_ral_id,
            "relation": "SUPERSEDES",
            "from_version": old_version,
            "to_version": new_version,
            "from_tier": existing_tier,
            "to_tier": candidate_tier,
            "reason": reason,
            "source_id": source.source_id,
            "evidence_sha256": quote_sha256,
            "created_at": ts_iso,
            "created_by_model": candidate_draft.formalizer_model,
        }

        original_bytes = existing_node_path.read_bytes()
        appended: list[tuple[Path, bool, int]] = []
        try:
            existing_node_path.write_text(final_md, encoding="utf-8")
            src_ledger = self.repo_root / "ledger" / "sources" / f"{ts:%Y-%m}.ndjson"
            appended.append(
                _append_ndjson(
                    src_ledger,
                    self._source_ledger_row(
                        ore_upload=ore_upload,
                        source=source,
                        linked_ral_ids=[existing_ral_id],
                        ts_iso=ts_iso,
                    ),
                )
            )
            lineage_ledger = self.repo_root / "graph" / "lineage" / f"{ts:%Y-%m}.ndjson"
            appended.append(_append_ndjson(lineage_ledger, lineage_row))
            self._run_blocking_validation(existing_node_path)
        except Exception:
            _rollback_mutations([], {existing_node_path: original_bytes}, appended)
            raise

        return ArchitectResult(
            action="SUPERSEDED",
            ral_id=existing_ral_id,
            node_path=existing_node_path,
            challenge_node_path=None,
            consensus_version=new_version,
            ore_upload=ore_upload,
            lineage_record=lineage_row,
            reason=f"Archived v{old_version} into <details> and upgraded to v{new_version} ({reason})",
        )

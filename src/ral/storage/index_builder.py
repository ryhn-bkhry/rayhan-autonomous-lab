"""Ephemeral SQLite index compiler (FTS5 + Graph + int8 vectors) & Vector-Space Guard (Sections 1.1, 3.3, 9)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import shutil
import sqlite3
import subprocess
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from ci.validate_node import NodeFrontmatter, split_frontmatter
from ral.canonicalize import canonical_text, normalize_newlines, sha256_bytes, sha256_hex

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_MODEL_POLICY_PATH = _REPO_ROOT / "config" / "model_policy.yaml"

TIER_RANK: dict[str, int] = {
    "T1_CANONICAL": 1,
    "T2_ESTABLISHED": 2,
    "T3_SPECULATIVE": 3,
    "T4_FALSIFIED": 4,
}

_DETAILS_RE = re.compile(r"<details\b[^>]*>.*?</details>", re.IGNORECASE | re.DOTALL)
_FORMAL_STMT_RE = re.compile(
    r"^###\s+Formal Statement\s*\n(.*?)(?=\n#{2,3}\s|\Z)",
    re.MULTILINE | re.DOTALL,
)
_FTS_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


class EmbeddingDimensionError(ValueError):
    """Raised when an embedding model returns a vector whose dimension violates `embedding_output_dim`."""


class ZstdBackendError(RuntimeError):
    """Raised when no Zstandard compression/decompression backend is available."""


@dataclass(frozen=True)
class IndexBuildResult:
    """Summary returned by `build_or_update_index()`."""

    db_path: Path
    node_count: int
    edge_count: int
    embedding_model_id: str
    embedding_dim: int
    embed_space_consistent: bool
    vector_action: str
    mismatch_nights: int
    vectors_written: int
    staged_count: int
    total_nodes_for_vectors: int


# ---------------------------------------------------------------------------
# Zstandard Compression Helpers (3-tier backend: stdlib -> zstandard -> CLI)
# ---------------------------------------------------------------------------


def _compress_zstd_bytes(data: bytes, level: int = 6) -> bytes:
    try:
        from compression import zstd as stdlib_zstd  # type: ignore[import-not-found]

        return bytes(stdlib_zstd.compress(data, level=level))
    except ImportError:
        pass

    try:
        import zstandard as zstd_pkg  # type: ignore[import-not-found]

        return bytes(zstd_pkg.ZstdCompressor(level=level).compress(data))
    except ImportError:
        pass

    zstd_bin = shutil.which("zstd")
    if zstd_bin:
        proc = subprocess.run(
            [zstd_bin, f"-{int(level)}", "-q", "-c"],
            input=data,
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout
        raise ZstdBackendError(f"zstd CLI compression failed: {proc.stderr.decode('utf-8', 'replace')}")

    raise ZstdBackendError("No zstd backend available (tried compression.zstd, zstandard, zstd CLI)")


def _decompress_zstd_bytes(data: bytes) -> bytes:
    try:
        from compression import zstd as stdlib_zstd  # type: ignore[import-not-found]

        return bytes(stdlib_zstd.decompress(data))
    except ImportError:
        pass

    try:
        import zstandard as zstd_pkg  # type: ignore[import-not-found]

        return bytes(zstd_pkg.ZstdDecompressor().decompress(data))
    except ImportError:
        pass

    zstd_bin = shutil.which("zstd")
    if zstd_bin:
        proc = subprocess.run(
            [zstd_bin, "-d", "-q", "-c"],
            input=data,
            capture_output=True,
            check=False,
        )
        if proc.returncode == 0:
            return proc.stdout
        raise ZstdBackendError(f"zstd CLI decompression failed: {proc.stderr.decode('utf-8', 'replace')}")

    raise ZstdBackendError("No zstd backend available (tried compression.zstd, zstandard, zstd CLI)")


def compress_sqlite_zst(
    sqlite_path: Path,
    output_path: Path,
    *,
    level: int = 6,
) -> tuple[Path, str, int]:
    """Compress a closed SQLite database file to `.sqlite.zst` and return `(path, sha256, bytes)`."""
    src = Path(sqlite_path)
    dst = Path(output_path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    compressed = _compress_zstd_bytes(src.read_bytes(), level=level)
    digest = sha256_bytes(compressed)

    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_zst_", dir=str(dst.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(compressed)
        os.replace(tmp_path, dst)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return dst, digest, len(compressed)


def decompress_sqlite_zst(
    zst_path: Path,
    output_path: Path,
    *,
    expected_sha256: str | None = None,
) -> Path:
    """
    Verify compressed archive SHA-256 (if supplied) BEFORE decompressing atomically to `output_path`.
    """
    src = Path(zst_path)
    dst = Path(output_path)
    compressed = src.read_bytes()
    if expected_sha256 is not None:
        clean_expected = expected_sha256.removeprefix("sha256:").strip().lower()
        actual = sha256_bytes(compressed)
        if actual != clean_expected:
            raise ValueError(
                f"Compressed index SHA-256 mismatch: expected {clean_expected}, got {actual}"
            )

    raw_sqlite = _decompress_zstd_bytes(compressed)
    dst.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_sqlite_", dir=str(dst.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(raw_sqlite)
        os.replace(tmp_path, dst)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise
    return dst


# ---------------------------------------------------------------------------
# Int8 Vector Quantization & Single-String Structural Embeddings
# ---------------------------------------------------------------------------


def quantize_int8(vector: Sequence[float]) -> tuple[bytes, float]:
    """L2-normalize a float vector and quantize symmetrically to int8 + float scale."""
    arr = np.asarray(vector, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm > 1e-12:
        arr = arr / norm
    max_abs = float(np.max(np.abs(arr))) if arr.size else 0.0
    scale = (max_abs / 127.0) if max_abs > 1e-12 else 1e-8
    q = np.rint(np.clip(arr / scale, -127.0, 127.0)).astype(np.int8)
    return q.tobytes(), float(scale)


def dequantize_int8(blob: bytes, scale: float) -> np.ndarray:
    """Dequantize an int8 SQLite BLOB back into a float32 NumPy array."""
    return np.frombuffer(blob, dtype=np.int8).astype(np.float32) * float(scale)


def build_structural_embed_text(
    formal_signature: Sequence[str],
    formal_statement: str,
) -> str:
    """
    Build domain-stripped structural embedding text from `formal_signature` tags +
    `Formal Statement` LaTeX (Section 4.4) — intentionally omits prose and domain names.
    """
    tags = ", ".join(str(t).strip() for t in formal_signature if str(t).strip())
    stmt = normalize_newlines(formal_statement).strip()
    return f"SIGNATURES: {tags}\nFORMAL_STATEMENT:\n{stmt}".strip()


def embed_single_document(
    client: Any,
    model_id: str,
    text: str,
    output_dim: int = 768,
) -> list[float]:
    """
    Invoke `client.models.embed_content` with a single-string `contents` payload
    (required for `gemini-embedding-2` so list inputs are never aggregated).
    """
    res = client.models.embed_content(
        model=model_id,
        contents=str(text),
        config={"output_dimensionality": int(output_dim)},
    )
    vec = [float(x) for x in res.embeddings[0].values]
    if len(vec) != int(output_dim):
        raise EmbeddingDimensionError(
            f"Expected {output_dim}-dim embedding from {model_id!r}, got {len(vec)}"
        )
    return vec


# ---------------------------------------------------------------------------
# SQLite Schema, Compilation & Vector-Space Integrity Guard (Section 3.3)
# ---------------------------------------------------------------------------


_VECTOR_TABLE_COLS = """(
    ral_id TEXT PRIMARY KEY,
    embedding_model_id TEXT NOT NULL,
    dim INTEGER NOT NULL,
    scale REAL NOT NULL,
    vec_int8 BLOB NOT NULL,
    content_sha256 TEXT NOT NULL
)"""


def ensure_schema(conn: sqlite3.Connection) -> None:
    """Create meta, nodes, FTS5, edges, and vectors tables if absent."""
    conn.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS nodes (
            ral_id TEXT PRIMARY KEY,
            rel_path TEXT NOT NULL UNIQUE,
            node_type TEXT NOT NULL,
            epistemic_tier TEXT NOT NULL,
            consensus_version INTEGER NOT NULL,
            taxonomy_code TEXT NOT NULL,
            macro_discipline TEXT NOT NULL,
            title TEXT NOT NULL,
            formal_signature_json TEXT NOT NULL,
            assumptions_json TEXT NOT NULL,
            assumptions_hash TEXT NOT NULL,
            oracle_plugin TEXT NOT NULL,
            oracle_status TEXT NOT NULL,
            verified_at TEXT NOT NULL,
            re_evaluate_after TEXT NOT NULL,
            formal_statement TEXT NOT NULL,
            body_text TEXT NOT NULL,
            content_sha256 TEXT NOT NULL
        );

        CREATE VIRTUAL TABLE IF NOT EXISTS nodes_fts USING fts5(
            ral_id UNINDEXED,
            title,
            taxonomy_code,
            formal_signature,
            assumptions,
            formal_statement,
            body_text,
            tokenize='unicode61'
        );

        CREATE TABLE IF NOT EXISTS edges (
            edge_id TEXT PRIMARY KEY,
            src TEXT NOT NULL,
            dst TEXT NOT NULL,
            relation TEXT NOT NULL,
            formal_mapping_json TEXT,
            confidence REAL,
            evidence_sha256 TEXT,
            created_at TEXT,
            created_by_model TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
        CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
        CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(relation);

        CREATE TABLE IF NOT EXISTS vectors {_VECTOR_TABLE_COLS};
        """
    )


def _table_exists(conn: sqlite3.Connection, table_name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name = ?",
        (table_name,),
    ).fetchone()
    return row is not None


def _get_meta(conn: sqlite3.Connection) -> dict[str, str]:
    rows = conn.execute("SELECT key, value FROM meta").fetchall()
    return {str(k): str(v) for k, v in rows}


def _set_meta_many(conn: sqlite3.Connection, mapping: Mapping[str, str]) -> None:
    conn.executemany(
        "INSERT OR REPLACE INTO meta (key, value) VALUES (?, ?)",
        [(str(k), str(v)) for k, v in mapping.items()],
    )


def _extract_formal_statement(active_body: str) -> str:
    match = _FORMAL_STMT_RE.search(active_body)
    if not match:
        return ""
    return match.group(1).strip()


def _parse_vault_nodes(repo_root: Path) -> list[dict[str, Any]]:
    vault_dir = Path(repo_root) / "vault"
    if not vault_dir.exists():
        return []

    records: list[dict[str, Any]] = []
    for md_path in sorted(vault_dir.rglob("*.md")):
        raw = normalize_newlines(md_path.read_text(encoding="utf-8"))
        fm_dict, body = split_frontmatter(raw)
        fm = NodeFrontmatter.model_validate(fm_dict)
        active_body = _DETAILS_RE.sub("", body).strip()
        formal_stmt = _extract_formal_statement(active_body)
        struct_text = build_structural_embed_text(fm.formal_signature, formal_stmt)
        content_sha = sha256_hex(canonical_text(struct_text))
        rel_path = md_path.relative_to(repo_root).as_posix()

        records.append(
            {
                "ral_id": fm.ral_id,
                "rel_path": rel_path,
                "node_type": fm.node_type,
                "epistemic_tier": fm.epistemic_tier,
                "consensus_version": int(fm.consensus_version),
                "taxonomy_code": fm.taxonomy_code,
                "macro_discipline": fm.taxonomy_code.split(".", 1)[0],
                "title": fm.title,
                "formal_signature": list(fm.formal_signature),
                "assumptions": list(fm.assumptions),
                "assumptions_hash": fm.assumptions_hash,
                "oracle_plugin": fm.verification.oracle_plugin,
                "oracle_status": fm.verification.oracle_status,
                "verified_at": fm.verification.verified_at,
                "re_evaluate_after": fm.re_evaluate_after,
                "formal_statement": formal_stmt,
                "body_text": active_body,
                "struct_text": struct_text,
                "content_sha256": content_sha,
            }
        )
    return records


def _parse_graph_edges(repo_root: Path) -> list[dict[str, Any]]:
    edges_dir = Path(repo_root) / "graph" / "edges"
    if not edges_dir.exists():
        return []
    rows: list[dict[str, Any]] = []
    for ndjson_path in sorted(edges_dir.glob("*.ndjson")):
        for idx, line in enumerate(ndjson_path.read_text(encoding="utf-8").splitlines(), start=1):
            stripped = line.strip()
            if not stripped:
                continue
            row = json.loads(stripped)
            if not isinstance(row, dict):
                continue
            eid = str(row.get("edge_id") or f"{ndjson_path.stem}:{idx}")
            rows.append(
                {
                    "edge_id": eid,
                    "src": str(row.get("src") or ""),
                    "dst": str(row.get("dst") or ""),
                    "relation": str(row.get("relation") or ""),
                    "formal_mapping_json": (
                        json.dumps(row["formal_mapping"], sort_keys=True)
                        if row.get("formal_mapping") is not None
                        else None
                    ),
                    "confidence": float(row["confidence"]) if row.get("confidence") is not None else None,
                    "evidence_sha256": row.get("evidence_sha256"),
                    "created_at": row.get("created_at"),
                    "created_by_model": row.get("created_by_model"),
                }
            )
    return rows


def _load_model_policy(repo_root: Path, policy: Mapping[str, Any] | None) -> Mapping[str, Any]:
    if policy is not None:
        return policy
    local_policy = Path(repo_root) / "config" / "model_policy.yaml"
    resolved = local_policy if local_policy.exists() else DEFAULT_MODEL_POLICY_PATH
    if resolved.exists():
        data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
        if isinstance(data, Mapping):
            return data
    return {
        "embedding_output_dim": 768,
        "embedding_migration_grace_nights": 3,
    }


def _populate_vector_table(
    conn: sqlite3.Connection,
    table_name: str,
    nodes_to_embed: Sequence[Mapping[str, Any]],
    *,
    embed_client: Any,
    model_id: str,
    target_dim: int,
) -> int:
    if table_name not in ("vectors", "vectors_next"):
        raise ValueError(f"Unexpected vector table name: {table_name!r}")
    written = 0
    for node in nodes_to_embed:
        vec = embed_single_document(embed_client, model_id, node["struct_text"], target_dim)
        q_blob, scale = quantize_int8(vec)
        conn.execute(
            f"INSERT OR REPLACE INTO {table_name} "
            "(ral_id, embedding_model_id, dim, scale, vec_int8, content_sha256) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (
                node["ral_id"],
                model_id,
                int(target_dim),
                float(scale),
                sqlite3.Binary(q_blob),
                node["content_sha256"],
            ),
        )
        written += 1
    return written


def build_or_update_index(
    repo_root: Path,
    db_path: Path,
    *,
    embed_client: Any | None = None,
    discovered_embed_model: str | None = None,
    policy: Mapping[str, Any] | None = None,
    migration_batch_size: int = 100,
    now: datetime | None = None,
) -> IndexBuildResult:
    """
    Compile or incrementally update the ephemeral SQLite index (FTS5 + Graph + int8 vectors)
    while strictly enforcing Section 3.3 Vector-Space Integrity Guard.
    """
    root = Path(repo_root)
    db_file = Path(db_path)
    db_file.parent.mkdir(parents=True, exist_ok=True)
    pol = _load_model_policy(root, policy)
    target_dim = int(pol.get("embedding_output_dim", 768))
    grace_nights = int(pol.get("embedding_migration_grace_nights", 3))
    ts_iso = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")

    node_records = _parse_vault_nodes(root)
    edge_records = _parse_graph_edges(root)
    node_by_id = {rec["ral_id"]: rec for rec in node_records}

    conn = sqlite3.connect(str(db_file))
    try:
        ensure_schema(conn)
        with conn:
            conn.execute("DELETE FROM nodes")
            conn.execute("DELETE FROM nodes_fts")
            conn.executemany(
                """
                INSERT INTO nodes (
                    ral_id, rel_path, node_type, epistemic_tier, consensus_version,
                    taxonomy_code, macro_discipline, title, formal_signature_json,
                    assumptions_json, assumptions_hash, oracle_plugin, oracle_status,
                    verified_at, re_evaluate_after, formal_statement, body_text, content_sha256
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r["ral_id"],
                        r["rel_path"],
                        r["node_type"],
                        r["epistemic_tier"],
                        r["consensus_version"],
                        r["taxonomy_code"],
                        r["macro_discipline"],
                        r["title"],
                        json.dumps(r["formal_signature"]),
                        json.dumps(r["assumptions"]),
                        r["assumptions_hash"],
                        r["oracle_plugin"],
                        r["oracle_status"],
                        r["verified_at"],
                        r["re_evaluate_after"],
                        r["formal_statement"],
                        r["body_text"],
                        r["content_sha256"],
                    )
                    for r in node_records
                ],
            )
            conn.executemany(
                """
                INSERT INTO nodes_fts (
                    ral_id, title, taxonomy_code, formal_signature,
                    assumptions, formal_statement, body_text
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        r["ral_id"],
                        r["title"],
                        r["taxonomy_code"],
                        " ".join(r["formal_signature"]),
                        "\n".join(r["assumptions"]),
                        r["formal_statement"],
                        r["body_text"],
                    )
                    for r in node_records
                ],
            )

            conn.execute("DELETE FROM edges")
            conn.executemany(
                """
                INSERT INTO edges (
                    edge_id, src, dst, relation, formal_mapping_json,
                    confidence, evidence_sha256, created_at, created_by_model
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                [
                    (
                        e["edge_id"],
                        e["src"],
                        e["dst"],
                        e["relation"],
                        e["formal_mapping_json"],
                        e["confidence"],
                        e["evidence_sha256"],
                        e["created_at"],
                        e["created_by_model"],
                    )
                    for e in edge_records
                ],
            )

            meta = _get_meta(conn)
            old_model = meta.get("embedding_model_id", "").strip()
            old_dim = int(meta.get("embedding_dim", str(target_dim)) or target_dim)
            prev_mismatch = int(meta.get("embedding_mismatch_nights", "0") or 0)
            pending_model = meta.get("pending_embedding_model_id", "").strip()
            existing_vec_count = int(conn.execute("SELECT COUNT(*) FROM vectors").fetchone()[0])

            vectors_written = 0
            staged_count = 0
            mismatch_nights = 0
            embed_space_consistent = False
            active_model = old_model
            active_dim = old_dim
            vector_action = "SKIPPED_NO_EMBEDDER"

            # Case 1: Fresh index (no active embedding_model_id recorded and no existing vectors).
            if not old_model and existing_vec_count == 0:
                if discovered_embed_model and embed_client is not None:
                    vectors_written = _populate_vector_table(
                        conn,
                        "vectors",
                        node_records,
                        embed_client=embed_client,
                        model_id=discovered_embed_model,
                        target_dim=target_dim,
                    )
                    active_model = discovered_embed_model
                    active_dim = target_dim
                    embed_space_consistent = True
                    vector_action = "INITIALIZED"
                    _set_meta_many(
                        conn,
                        {
                            "embedding_model_id": active_model,
                            "embedding_dim": str(active_dim),
                            "embedding_mismatch_nights": "0",
                            "pending_embedding_model_id": "",
                            "embed_space_consistent": "1",
                            "compiled_at": ts_iso,
                        },
                    )
                else:
                    active_model = discovered_embed_model or ""
                    active_dim = target_dim
                    embed_space_consistent = False
                    vector_action = "SKIPPED_NO_EMBEDDER"
                    _set_meta_many(
                        conn,
                        {
                            "embedding_model_id": active_model,
                            "embedding_dim": str(active_dim),
                            "embedding_mismatch_nights": "0",
                            "pending_embedding_model_id": "",
                            "embed_space_consistent": "0",
                            "compiled_at": ts_iso,
                        },
                    )

            # Case 2: Discovered embedding model matches stamped space -> incremental update only.
            elif discovered_embed_model and discovered_embed_model == old_model and target_dim == old_dim:
                if _table_exists(conn, "vectors_next"):
                    conn.execute("DROP TABLE vectors_next")
                existing_rows = {
                    row[0]: (row[1], row[2])
                    for row in conn.execute(
                        "SELECT ral_id, embedding_model_id, content_sha256 FROM vectors"
                    ).fetchall()
                }
                stale_ids = [rid for rid in existing_rows if rid not in node_by_id]
                if stale_ids:
                    conn.executemany("DELETE FROM vectors WHERE ral_id = ?", [(rid,) for rid in stale_ids])

                missing_or_changed = [
                    rec
                    for rec in node_records
                    if rec["ral_id"] not in existing_rows
                    or existing_rows[rec["ral_id"]][0] != old_model
                    or existing_rows[rec["ral_id"]][1] != rec["content_sha256"]
                ]
                if embed_client is not None and missing_or_changed:
                    vectors_written = _populate_vector_table(
                        conn,
                        "vectors",
                        missing_or_changed,
                        embed_client=embed_client,
                        model_id=old_model,
                        target_dim=old_dim,
                    )

                covered = int(
                    conn.execute(
                        "SELECT COUNT(*) FROM vectors WHERE embedding_model_id = ?",
                        (old_model,),
                    ).fetchone()[0]
                )
                embed_space_consistent = covered == len(node_records)
                vector_action = "INCREMENTAL_UPDATED"
                _set_meta_many(
                    conn,
                    {
                        "embedding_model_id": old_model,
                        "embedding_dim": str(old_dim),
                        "embedding_mismatch_nights": "0",
                        "pending_embedding_model_id": "",
                        "embed_space_consistent": "1" if embed_space_consistent else "0",
                        "compiled_at": ts_iso,
                    },
                )

            # Case 3: Transient embedder unavailability while existing vector space exists.
            elif not discovered_embed_model:
                mismatch_nights = prev_mismatch + 1
                embed_space_consistent = False
                vector_action = "SKIPPED_NO_EMBEDDER"
                _set_meta_many(
                    conn,
                    {
                        "embedding_model_id": old_model,
                        "embedding_dim": str(old_dim),
                        "embedding_mismatch_nights": str(mismatch_nights),
                        "pending_embedding_model_id": pending_model,
                        "embed_space_consistent": "0",
                        "compiled_at": ts_iso,
                    },
                )

            # Case 4: Model or dimension mismatch! Enforce grace window and atomic vectors_next swap.
            else:
                if pending_model and pending_model != discovered_embed_model:
                    mismatch_nights = 1
                    if _table_exists(conn, "vectors_next"):
                        conn.execute("DROP TABLE vectors_next")
                else:
                    mismatch_nights = prev_mismatch + 1

                if mismatch_nights < grace_nights:
                    # Grace window (< 3 consecutive nights): HALT all vector writes immediately.
                    embed_space_consistent = False
                    vector_action = "HALTED_MISMATCH_GRACE"
                    _set_meta_many(
                        conn,
                        {
                            "embedding_model_id": old_model,
                            "embedding_dim": str(old_dim),
                            "embedding_mismatch_nights": str(mismatch_nights),
                            "pending_embedding_model_id": discovered_embed_model,
                            "embed_space_consistent": "0",
                            "compiled_at": ts_iso,
                        },
                    )
                else:
                    # >= 3 consecutive nights: stage into `vectors_next` without touching `vectors`.
                    conn.execute(f"CREATE TABLE IF NOT EXISTS vectors_next {_VECTOR_TABLE_COLS}")
                    conn.execute(
                        "DELETE FROM vectors_next WHERE embedding_model_id != ? OR dim != ?",
                        (discovered_embed_model, target_dim),
                    )
                    next_rows = {
                        row[0]: row[1]
                        for row in conn.execute(
                            "SELECT ral_id, content_sha256 FROM vectors_next"
                        ).fetchall()
                    }
                    stale_next = [rid for rid in next_rows if rid not in node_by_id]
                    if stale_next:
                        conn.executemany(
                            "DELETE FROM vectors_next WHERE ral_id = ?",
                            [(rid,) for rid in stale_next],
                        )

                    pending_nodes = [
                        rec
                        for rec in node_records
                        if rec["ral_id"] not in next_rows
                        or next_rows[rec["ral_id"]] != rec["content_sha256"]
                    ]
                    if embed_client is not None and pending_nodes:
                        batch = pending_nodes[: max(1, int(migration_batch_size))]
                        vectors_written = _populate_vector_table(
                            conn,
                            "vectors_next",
                            batch,
                            embed_client=embed_client,
                            model_id=discovered_embed_model,
                            target_dim=target_dim,
                        )

                    staged_valid = {
                        row[0]: row[1]
                        for row in conn.execute(
                            "SELECT ral_id, content_sha256 FROM vectors_next WHERE embedding_model_id = ? AND dim = ?",
                            (discovered_embed_model, target_dim),
                        ).fetchall()
                    }
                    staged_count = sum(
                        1
                        for rec in node_records
                        if staged_valid.get(rec["ral_id"]) == rec["content_sha256"]
                    )

                    if staged_count == len(node_records) and (len(node_records) == 0 or embed_client is not None or staged_count > 0):
                        conn.execute("DROP TABLE vectors")
                        conn.execute("ALTER TABLE vectors_next RENAME TO vectors")
                        active_model = discovered_embed_model
                        active_dim = target_dim
                        mismatch_nights = 0
                        embed_space_consistent = True
                        vector_action = "MIGRATION_SWAPPED"
                        _set_meta_many(
                            conn,
                            {
                                "embedding_model_id": active_model,
                                "embedding_dim": str(active_dim),
                                "embedding_mismatch_nights": "0",
                                "pending_embedding_model_id": "",
                                "embed_space_consistent": "1",
                                "compiled_at": ts_iso,
                            },
                        )
                    else:
                        embed_space_consistent = False
                        vector_action = "MIGRATING_STAGED"
                        _set_meta_many(
                            conn,
                            {
                                "embedding_model_id": old_model,
                                "embedding_dim": str(old_dim),
                                "embedding_mismatch_nights": str(mismatch_nights),
                                "pending_embedding_model_id": discovered_embed_model,
                                "embed_space_consistent": "0",
                                "compiled_at": ts_iso,
                            },
                        )

        return IndexBuildResult(
            db_path=db_file,
            node_count=len(node_records),
            edge_count=len(edge_records),
            embedding_model_id=active_model,
            embedding_dim=active_dim,
            embed_space_consistent=embed_space_consistent,
            vector_action=vector_action,
            mismatch_nights=mismatch_nights,
            vectors_written=vectors_written,
            staged_count=staged_count,
            total_nodes_for_vectors=len(node_records),
        )
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Read-Only Query & Embedding Loaders (FTS5 BM25 + Jaccard + Consistent Cosine)
# ---------------------------------------------------------------------------


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    resolved = Path(db_path).resolve().as_uri()
    return sqlite3.connect(f"{resolved}?mode=ro", uri=True)


def read_index_meta(db_path: Path) -> dict[str, str]:
    """Read SQLite `meta` table via a strict read-only URI connection."""
    conn = _connect_readonly(db_path)
    try:
        return _get_meta(conn)
    finally:
        conn.close()


def load_node_embeddings(db_path: Path) -> tuple[dict[str, np.ndarray], bool, str]:
    """
    Load dequantized unit-normalized embeddings from `vectors` via a read-only connection.

    Returns `(vectors_by_ral_id, embed_space_consistent, embedding_model_id)`.
    """
    conn = _connect_readonly(db_path)
    try:
        meta = _get_meta(conn)
        model_id = meta.get("embedding_model_id", "")
        consistent = meta.get("embed_space_consistent") == "1"
        vectors: dict[str, np.ndarray] = {}
        for ral_id, row_model, scale, blob in conn.execute(
            "SELECT ral_id, embedding_model_id, scale, vec_int8 FROM vectors"
        ).fetchall():
            if model_id and row_model != model_id:
                consistent = False
                continue
            vec = dequantize_int8(bytes(blob), float(scale))
            norm = float(np.linalg.norm(vec))
            if norm > 1e-12:
                vec = vec / norm
            vectors[str(ral_id)] = vec
        return vectors, consistent, model_id
    finally:
        conn.close()


def _jaccard(tags_a: Sequence[str], tags_b: Sequence[str]) -> float:
    sa = {t.strip().lower() for t in tags_a if str(t).strip()}
    sb = {t.strip().lower() for t in tags_b if str(t).strip()}
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


def hybrid_search(
    db_path: Path,
    query: str,
    *,
    formal_signature_tags: Sequence[str] | None = None,
    query_vector: Sequence[float] | None = None,
    query_embed_model_id: str | None = None,
    max_tier: str = "T2_ESTABLISHED",
    limit: int = 10,
) -> list[dict[str, Any]]:
    """
    Hybrid FTS5 BM25 + `formal_signature` Jaccard search, augmenting with vector cosine
    ONLY when `embed_space_consistent == '1'` and vector model matches `meta.embedding_model_id`.
    """
    max_rank = TIER_RANK.get(max_tier, 2)
    allowed_tiers = tuple(t for t, r in TIER_RANK.items() if r <= max_rank)
    placeholders = ", ".join("?" for _ in allowed_tiers)

    tokens = _FTS_TOKEN_RE.findall(query or "")
    fts_expr = " OR ".join(f'"{tok}"' for tok in tokens)

    conn = _connect_readonly(db_path)
    try:
        meta = _get_meta(conn)
        embed_model_id = meta.get("embedding_model_id", "")
        consistent = meta.get("embed_space_consistent") == "1"
        use_vector = (
            consistent
            and query_vector is not None
            and (query_embed_model_id is None or query_embed_model_id == embed_model_id)
        )

        q_unit: np.ndarray | None = None
        if use_vector and query_vector is not None:
            q_arr = np.asarray(query_vector, dtype=np.float32)
            q_norm = float(np.linalg.norm(q_arr))
            if q_norm > 1e-12:
                q_unit = q_arr / q_norm
            else:
                use_vector = False

        bm25_by_id: dict[str, float] = {}
        if fts_expr:
            sql_fts = (
                "SELECT n.ral_id, bm25(nodes_fts) AS rank_score "
                "FROM nodes_fts "
                "JOIN nodes n ON n.ral_id = nodes_fts.ral_id "
                f"WHERE nodes_fts MATCH ? AND n.epistemic_tier IN ({placeholders}) "
                "ORDER BY rank_score ASC LIMIT ?"
            )
            for rid, bm25_raw in conn.execute(sql_fts, (fts_expr, *allowed_tiers, max(limit * 5, 50))).fetchall():
                clamped_bm25 = max(-50.0, min(50.0, float(bm25_raw)))
                bm25_by_id[str(rid)] = 1.0 / (1.0 + math.exp(clamped_bm25))

        rows = conn.execute(
            "SELECT ral_id, rel_path, node_type, epistemic_tier, consensus_version, "
            "taxonomy_code, title, formal_signature_json "
            f"FROM nodes WHERE epistemic_tier IN ({placeholders})",
            allowed_tiers,
        ).fetchall()

        vec_map: dict[str, np.ndarray] = {}
        if use_vector and embed_model_id:
            for rid, scale, blob in conn.execute(
                "SELECT ral_id, scale, vec_int8 FROM vectors WHERE embedding_model_id = ?",
                (embed_model_id,),
            ).fetchall():
                v = dequantize_int8(bytes(blob), float(scale))
                nrm = float(np.linalg.norm(v))
                if nrm > 1e-12:
                    vec_map[str(rid)] = v / nrm

        scored: list[dict[str, Any]] = []
        sig_tags = list(formal_signature_tags or ())
        for rid, rel_path, node_type, tier, version, tax_code, title, sig_json in rows:
            sigs = json.loads(sig_json)
            fts_score = bm25_by_id.get(str(rid), 0.0)
            jac_score = _jaccard(sig_tags, sigs) if sig_tags else 0.0
            cos_score = 0.0
            if use_vector and q_unit is not None and str(rid) in vec_map:
                v = vec_map[str(rid)]
                if v.shape == q_unit.shape:
                    cos_score = max(0.0, float(np.dot(q_unit, v)))

            if (fts_expr or sig_tags or q_unit is not None) and (
                fts_score == 0.0 and jac_score == 0.0 and cos_score <= 0.0
            ):
                continue

            if use_vector:
                total = 0.50 * fts_score + 0.25 * jac_score + 0.25 * cos_score
            else:
                total = 0.65 * fts_score + 0.35 * jac_score

            scored.append(
                {
                    "ral_id": str(rid),
                    "rel_path": str(rel_path),
                    "node_type": str(node_type),
                    "epistemic_tier": str(tier),
                    "consensus_version": int(version),
                    "taxonomy_code": str(tax_code),
                    "title": str(title),
                    "formal_signature": sigs,
                    "score": round(total, 6),
                    "used_vector": bool(use_vector),
                }
            )

        scored.sort(key=lambda r: (-r["score"], r["ral_id"]))
        return scored[: max(1, int(limit))]
    finally:
        conn.close()


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for compiling and optionally compressing the ephemeral SQLite index."""
    parser = argparse.ArgumentParser(description="Build RAL ephemeral SQLite index.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--db", type=Path, required=True, help="Output SQLite DB path")
    parser.add_argument("--compress-out", type=Path, default=None, help="Optional .sqlite.zst output")
    args = parser.parse_args(argv)

    result = build_or_update_index(args.repo_root, args.db)
    summary: dict[str, Any] = {
        "db_path": str(result.db_path),
        "node_count": result.node_count,
        "edge_count": result.edge_count,
        "embedding_model_id": result.embedding_model_id,
        "embed_space_consistent": result.embed_space_consistent,
        "vector_action": result.vector_action,
    }
    if args.compress_out is not None:
        zst_path, digest, size_bytes = compress_sqlite_zst(result.db_path, args.compress_out)
        summary["zst_path"] = str(zst_path)
        summary["zst_sha256"] = digest
        summary["zst_size_bytes"] = size_bytes
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

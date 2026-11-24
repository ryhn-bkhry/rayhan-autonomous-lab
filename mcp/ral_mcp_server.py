"""Hardened read-only MCP server for Rayhan's Autonomous Lab (Blueprint Section 9)."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
import heapq
import json
import os
from pathlib import Path
import re
import sqlite3
import sys
import tempfile
from typing import Any, Mapping, Sequence, TextIO

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ral.canonicalize import sha256_bytes  # noqa: E402
from ral.github_client import (  # noqa: E402
    INDEX_RELEASE_TAG,
    LATEST_POINTER_NAME,
    GitHubClient,
)
from ral.storage.index_builder import (  # noqa: E402
    TIER_RANK,
    decompress_sqlite_zst,
    hybrid_search,
)

RAL_ID_RE = re.compile(r"^RAL-[A-Z]{2,6}-[0-9A-Z]{10}\Z")
DOMAIN_FILTER_RE = re.compile(r"^[a-z]{2,16}(?:\.[a-z0-9-]+)*\Z")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}\Z")
ALLOWED_TIERS: frozenset[str] = frozenset(TIER_RANK.keys())

MCP_PROTOCOL_VERSION = "2024-11-05"

MCP_TOOLS_SCHEMA: list[dict[str, Any]] = [
    {
        "name": "ral_search",
        "description": (
            "Hybrid FTS5 BM25 + formal_signature Jaccard (+ consistent-space vector cosine) "
            "search returning ral_id, epistemic_tier, taxonomy_code, and consensus_version."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "formal_signature_tags": {"type": "array", "items": {"type": "string"}},
                "max_tier": {
                    "type": "string",
                    "enum": sorted(ALLOWED_TIERS),
                    "default": "T2_ESTABLISHED",
                },
                "limit": {"type": "integer", "minimum": 1, "maximum": 100, "default": 10},
            },
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ral_get_node",
        "description": (
            "Retrieve canonical Markdown node by ral_id, automatically attaching linked "
            "BREAKS_WITHOUT_ASSUMPTION failure modes and T3/T4 epistemic warning banners."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ral_id": {"type": "string", "pattern": "^RAL-[A-Z]{2,6}-[0-9A-Z]{10}$"},
            },
            "required": ["ral_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ral_get_dependency_cone",
        "description": (
            "Traverse the acyclic DEPENDS_ON DAG from ral_id and return its minimal "
            "prerequisite cone in topological order (foundations first)."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "ral_id": {"type": "string", "pattern": "^RAL-[A-Z]{2,6}-[0-9A-Z]{10}$"},
                "max_depth": {"type": "integer", "minimum": 0, "maximum": 20, "default": 5},
            },
            "required": ["ral_id"],
            "additionalProperties": False,
        },
    },
    {
        "name": "ral_find_isomorphisms",
        "description": (
            "Query cross-discipline ISOMORPHIC_TO edges and return structured formal_mapping "
            "variable dictionaries."
        ),
        "inputSchema": {
            "type": "object",
            "properties": {
                "domain_a": {"type": ["string", "null"]},
                "domain_b": {"type": ["string", "null"]},
                "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 50},
            },
            "additionalProperties": False,
        },
    },
]


def validate_ral_id(ral_id: Any) -> str:
    """Strictly validate a RAL node identifier against `^RAL-[A-Z]{2,6}-[0-9A-Z]{10}$`."""
    if not isinstance(ral_id, str) or not RAL_ID_RE.match(ral_id):
        raise ValueError(
            f"Invalid ral_id {ral_id!r}: must match ^RAL-[A-Z]{{2,6}}-[0-9A-Z]{{10}}$"
        )
    return ral_id


def validate_domain_filter(domain: Any) -> str | None:
    """Strictly validate an optional taxonomy prefix filter (rejects SQL/path metacharacters)."""
    if domain is None:
        return None
    if not isinstance(domain, str) or not DOMAIN_FILTER_RE.match(domain):
        raise ValueError(
            f"Invalid domain filter {domain!r}: must match ^[a-z]{{2,16}}(\\.[a-z0-9-]+)*$"
        )
    return domain


def _connect_readonly(db_path: Path) -> sqlite3.Connection:
    resolved = Path(db_path).resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"SQLite index database not found: {resolved}")
    conn = sqlite3.connect(f"{resolved.as_uri()}?mode=ro", uri=True)
    conn.execute("PRAGMA query_only = ON")
    return conn


def _resolve_vault_node_path(repo_root: Path, rel_path: str) -> Path:
    """Resolve `rel_path` strictly inside `<repo_root>/vault/` via `is_relative_to()`."""
    if not isinstance(rel_path, str) or not rel_path.strip():
        raise PermissionError("Empty node rel_path is forbidden")
    rel_obj = Path(rel_path)
    if rel_obj.is_absolute():
        raise PermissionError(f"Absolute node path forbidden: {rel_path!r}")

    vault_root = (Path(repo_root) / "vault").resolve()
    candidate = (Path(repo_root) / rel_obj).resolve()
    if not candidate.is_relative_to(vault_root):
        raise PermissionError(f"Node path escapes vault/ root: {rel_path!r}")
    if candidate.suffix.lower() != ".md":
        raise PermissionError(f"Vault node must be a .md file: {rel_path!r}")
    if not candidate.is_file():
        raise FileNotFoundError(f"Vault node Markdown file missing: {candidate}")
    return candidate


def _domain_matches(taxonomy_code: str, prefix: str) -> bool:
    return taxonomy_code == prefix or taxonomy_code.startswith(f"{prefix}.")


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=".tmp_ptr_", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(payload), indent=2) + "\n")
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _tier_warning_banner(epistemic_tier: str, re_evaluate_after: str) -> str | None:
    if epistemic_tier == "T3_SPECULATIVE":
        return (
            f"WARNING [T3_SPECULATIVE]: Unrefereed or speculative claim "
            f"(re_evaluate_after={re_evaluate_after}). Do not treat as canonical truth "
            "or as a T1_CANONICAL dependency."
        )
    if epistemic_tier == "T4_FALSIFIED":
        return (
            "WARNING [T4_FALSIFIED]: Falsified or retracted claim. "
            "Retained strictly as a negative result / failure boundary."
        )
    return None


class RALMCPServer:
    """Read-only, injection-hardened MCP tool server backed by compiled SQLite + vault Markdown."""

    def __init__(
        self,
        repo_root: Path,
        *,
        db_path: Path | None = None,
        cache_dir: Path | None = None,
    ) -> None:
        self.repo_root = Path(repo_root).resolve()
        self.cache_dir = (
            Path(cache_dir).resolve()
            if cache_dir is not None
            else (self.repo_root / ".ral_cache").resolve()
        )
        self.db_path = (
            Path(db_path).resolve()
            if db_path is not None
            else (self.cache_dir / "ral-index.sqlite").resolve()
        )

    def sync_index(
        self,
        *,
        github_client: GitHubClient | None = None,
        local_pointer_path: Path | None = None,
        tag: str = INDEX_RELEASE_TAG,
    ) -> dict[str, Any]:
        """
        Synchronize local read-only SQLite DB using `ral-index-latest.json`.

        Verifies SHA-256 BEFORE decompressing `.sqlite.zst` so corrupt/tampered archives
        never overwrite the active SQLite database.
        """
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cached_pointer_path = self.cache_dir / LATEST_POINTER_NAME

        pointer: dict[str, Any] | None = None
        if local_pointer_path is not None:
            ptr_file = Path(local_pointer_path)
            if not ptr_file.is_file():
                raise FileNotFoundError(f"Pointer file not found: {ptr_file}")
            parsed = json.loads(ptr_file.read_text(encoding="utf-8"))
            if not isinstance(parsed, Mapping):
                raise ValueError(f"Invalid pointer JSON in {ptr_file}")
            pointer = dict(parsed)
        elif github_client is not None:
            pointer = github_client.fetch_latest_index_pointer(tag=tag)
            if pointer is None:
                return {"synced": False, "reason": "NO_RELEASE_POINTER"}
        elif cached_pointer_path.is_file():
            parsed = json.loads(cached_pointer_path.read_text(encoding="utf-8"))
            if isinstance(parsed, Mapping):
                pointer = dict(parsed)

        if pointer is None:
            raise FileNotFoundError("No ral-index-latest.json pointer available to sync")

        expected_sha = (
            str(pointer.get("sha256") or "").removeprefix("sha256:").strip().lower()
        )
        if not _SHA256_RE.match(expected_sha):
            raise ValueError(f"Invalid pointer sha256: {pointer.get('sha256')!r}")

        asset_name = str(pointer.get("asset_name") or "ral-index-latest.sqlite.zst").strip()
        if "/" in asset_name or "\\" in asset_name or ".." in asset_name:
            raise ValueError(f"Unsafe asset_name in pointer: {asset_name!r}")

        if cached_pointer_path.is_file() and self.db_path.is_file():
            try:
                cached_ptr = json.loads(cached_pointer_path.read_text(encoding="utf-8"))
                cached_sha = (
                    str(cached_ptr.get("sha256") or "")
                    .removeprefix("sha256:")
                    .strip()
                    .lower()
                )
                if cached_sha == expected_sha:
                    return {
                        "synced": True,
                        "cached": True,
                        "sha256": expected_sha,
                        "asset_name": asset_name,
                        "db_path": str(self.db_path),
                    }
            except Exception:
                pass

        zst_path: Path | None = None
        if local_pointer_path is not None:
            local_candidates = [
                self.cache_dir / asset_name,
                Path(local_pointer_path).resolve().parent / asset_name,
            ]
            for cand in local_candidates:
                if cand.is_file():
                    zst_path = cand
                    break
            if zst_path is None:
                raise FileNotFoundError(
                    f"Compressed index archive {asset_name!r} not found alongside pointer or in cache"
                )
        elif github_client is not None:
            dl_url = str(pointer.get("download_url") or "").strip()
            if not dl_url:
                raise ValueError("Remote pointer missing download_url")
            raw_zst = github_client.download_asset_bytes(dl_url)
            actual_sha = sha256_bytes(raw_zst)
            if actual_sha != expected_sha:
                raise ValueError(
                    f"Downloaded index SHA-256 mismatch: expected {expected_sha}, got {actual_sha}"
                )
            zst_path = self.cache_dir / asset_name
            zst_path.write_bytes(raw_zst)
        else:
            cand = self.cache_dir / asset_name
            if not cand.is_file():
                raise FileNotFoundError(f"Cached index archive missing: {cand}")
            zst_path = cand

        decompress_sqlite_zst(zst_path, self.db_path, expected_sha256=expected_sha)
        _atomic_write_json(cached_pointer_path, pointer)
        return {
            "synced": True,
            "cached": False,
            "sha256": expected_sha,
            "asset_name": asset_name,
            "db_path": str(self.db_path),
        }

    def ral_search(
        self,
        query: str,
        *,
        formal_signature_tags: Sequence[str] | None = None,
        max_tier: str = "T2_ESTABLISHED",
        limit: int = 10,
        query_vector: Sequence[float] | None = None,
        query_embed_model_id: str | None = None,
    ) -> dict[str, Any]:
        """Hybrid FTS5 BM25 + `formal_signature` Jaccard search with epistemic tier metadata."""
        if not isinstance(query, str):
            raise ValueError("query must be a string")
        if max_tier not in ALLOWED_TIERS:
            raise ValueError(f"Invalid max_tier {max_tier!r}; allowed: {sorted(ALLOWED_TIERS)}")
        safe_limit = max(1, min(100, int(limit)))

        if not self.db_path.is_file():
            raise FileNotFoundError(f"SQLite index database not found: {self.db_path}")

        results = hybrid_search(
            self.db_path,
            query,
            formal_signature_tags=formal_signature_tags,
            query_vector=query_vector,
            query_embed_model_id=query_embed_model_id,
            max_tier=max_tier,
            limit=safe_limit,
        )
        for item in results:
            banner = _tier_warning_banner(str(item.get("epistemic_tier") or ""), "")
            if banner is not None:
                item["warning"] = banner

        return {
            "query": query,
            "max_tier": max_tier,
            "count": len(results),
            "results": results,
        }

    def ral_get_node(self, ral_id: str) -> dict[str, Any]:
        """
        Return canonical Markdown node + linked `BREAKS_WITHOUT_ASSUMPTION` (`NEGATIVE_RESULT`)
        failure modes + explicit `WARNING` banner for `T3_SPECULATIVE` / `T4_FALSIFIED`.
        """
        clean_id = validate_ral_id(ral_id)
        conn = _connect_readonly(self.db_path)
        try:
            row = conn.execute(
                """
                SELECT
                    ral_id, rel_path, node_type, epistemic_tier, consensus_version,
                    taxonomy_code, macro_discipline, title, formal_signature_json,
                    assumptions_json, assumptions_hash, oracle_plugin, oracle_status,
                    verified_at, re_evaluate_after, formal_statement
                FROM nodes
                WHERE ral_id = ?
                """,
                (clean_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"Node not found in index: {clean_id}")

            (
                rid,
                rel_path,
                node_type,
                tier,
                version,
                tax_code,
                macro,
                title,
                sig_json,
                assump_json,
                assump_hash,
                oracle_plugin,
                oracle_status,
                verified_at,
                re_eval,
                formal_stmt,
            ) = row

            node_file = _resolve_vault_node_path(self.repo_root, str(rel_path))
            markdown = node_file.read_text(encoding="utf-8")

            edge_rows = conn.execute(
                """
                SELECT edge_id, src, dst, confidence, evidence_sha256, created_at
                FROM edges
                WHERE relation = 'BREAKS_WITHOUT_ASSUMPTION' AND (dst = ? OR src = ?)
                ORDER BY edge_id ASC
                """,
                (clean_id, clean_id),
            ).fetchall()

            failure_modes: list[dict[str, Any]] = []
            for edge_id, src, dst, conf, ev_sha, created_at in edge_rows:
                peer_id = str(src) if str(dst) == clean_id else str(dst)
                peer_row = conn.execute(
                    """
                    SELECT ral_id, rel_path, node_type, epistemic_tier, title, formal_statement
                    FROM nodes
                    WHERE ral_id = ?
                    """,
                    (peer_id,),
                ).fetchone()
                peer_meta: dict[str, Any] = {
                    "edge_id": str(edge_id),
                    "relation": "BREAKS_WITHOUT_ASSUMPTION",
                    "src": str(src),
                    "dst": str(dst),
                    "negative_result_ral_id": peer_id,
                    "confidence": float(conf) if conf is not None else None,
                    "evidence_sha256": ev_sha,
                    "created_at": created_at,
                }
                if peer_row is not None:
                    p_id, p_rel, p_type, p_tier, p_title, p_stmt = peer_row
                    peer_meta.update(
                        {
                            "ral_id": str(p_id),
                            "rel_path": str(p_rel),
                            "node_type": str(p_type),
                            "epistemic_tier": str(p_tier),
                            "title": str(p_title),
                            "formal_statement": str(p_stmt),
                        }
                    )
                    try:
                        p_file = _resolve_vault_node_path(self.repo_root, str(p_rel))
                        peer_meta["markdown"] = p_file.read_text(encoding="utf-8")
                    except (PermissionError, FileNotFoundError):
                        pass
                failure_modes.append(peer_meta)
        finally:
            conn.close()

        warning_banner = _tier_warning_banner(str(tier), str(re_eval))
        annotated_parts: list[str] = []
        if warning_banner is not None:
            annotated_parts.append(f"> **{warning_banner}**\n")
        annotated_parts.append(markdown.rstrip())
        if failure_modes:
            annotated_parts.append("\n## Linked Failure Modes (BREAKS_WITHOUT_ASSUMPTION)")
            for fm in failure_modes:
                fm_id = fm.get("ral_id") or fm["negative_result_ral_id"]
                fm_title = fm.get("title") or fm_id
                fm_tier = fm.get("epistemic_tier") or "UNKNOWN"
                annotated_parts.append(f"- `{fm_id}` [{fm_tier}]: {fm_title}")
        annotated_markdown = "\n".join(annotated_parts).rstrip() + "\n"

        return {
            "ral_id": str(rid),
            "rel_path": str(rel_path),
            "node_type": str(node_type),
            "epistemic_tier": str(tier),
            "consensus_version": int(version),
            "taxonomy_code": str(tax_code),
            "macro_discipline": str(macro),
            "title": str(title),
            "formal_signature": json.loads(sig_json),
            "assumptions": json.loads(assump_json),
            "assumptions_hash": str(assump_hash),
            "oracle_plugin": str(oracle_plugin),
            "oracle_status": str(oracle_status),
            "verified_at": str(verified_at),
            "re_evaluate_after": str(re_eval),
            "formal_statement": str(formal_stmt),
            "warning_banner": warning_banner,
            "failure_modes": failure_modes,
            "markdown": markdown,
            "annotated_markdown": annotated_markdown,
        }

    def ral_get_dependency_cone(self, ral_id: str, *, max_depth: int = 5) -> dict[str, Any]:
        """
        Traverse the acyclic `DEPENDS_ON` DAG from `ral_id` up to `max_depth` hops and return
        the minimal prerequisite cone in topological order (foundations first).
        """
        clean_id = validate_ral_id(ral_id)
        depth_limit = max(0, min(20, int(max_depth)))

        conn = _connect_readonly(self.db_path)
        try:
            root_row = conn.execute(
                """
                SELECT ral_id, rel_path, node_type, epistemic_tier, consensus_version, taxonomy_code, title
                FROM nodes
                WHERE ral_id = ?
                """,
                (clean_id,),
            ).fetchone()
            if root_row is None:
                raise KeyError(f"Node not found in index: {clean_id}")

            def _row_to_node_meta(r: tuple[Any, ...], depth: int) -> dict[str, Any]:
                return {
                    "ral_id": str(r[0]),
                    "rel_path": str(r[1]),
                    "node_type": str(r[2]),
                    "epistemic_tier": str(r[3]),
                    "consensus_version": int(r[4]),
                    "taxonomy_code": str(r[5]),
                    "title": str(r[6]),
                    "depth": int(depth),
                }

            visited_nodes: dict[str, dict[str, Any]] = {
                clean_id: _row_to_node_meta(root_row, 0)
            }
            depth_by_id: dict[str, int] = {clean_id: 0}
            collected_edges: list[dict[str, Any]] = []
            unresolved_stubs: set[str] = set()
            queue: deque[str] = deque([clean_id])

            while queue:
                curr = queue.popleft()
                curr_depth = depth_by_id[curr]
                if curr_depth >= depth_limit:
                    continue
                edge_rows = conn.execute(
                    """
                    SELECT edge_id, src, dst, confidence, evidence_sha256
                    FROM edges
                    WHERE relation = 'DEPENDS_ON' AND src = ?
                    ORDER BY dst ASC, edge_id ASC
                    """,
                    (curr,),
                ).fetchall()
                for edge_id, src, dst, conf, ev_sha in edge_rows:
                    src_id = str(src)
                    dst_id = str(dst)
                    collected_edges.append(
                        {
                            "edge_id": str(edge_id),
                            "src": src_id,
                            "dst": dst_id,
                            "relation": "DEPENDS_ON",
                            "confidence": float(conf) if conf is not None else None,
                            "evidence_sha256": ev_sha,
                        }
                    )
                    if dst_id.startswith("STUB-"):
                        unresolved_stubs.add(dst_id)
                        continue
                    dst_row = conn.execute(
                        """
                        SELECT ral_id, rel_path, node_type, epistemic_tier, consensus_version, taxonomy_code, title
                        FROM nodes
                        WHERE ral_id = ?
                        """,
                        (dst_id,),
                    ).fetchone()
                    if dst_row is None:
                        unresolved_stubs.add(dst_id)
                        continue
                    if dst_id not in depth_by_id:
                        depth_by_id[dst_id] = curr_depth + 1
                        visited_nodes[dst_id] = _row_to_node_meta(dst_row, curr_depth + 1)
                        queue.append(dst_id)
        finally:
            conn.close()

        # Kahn's topological sort on prerequisite -> dependent orientation (dst -> src)
        # so foundational axioms appear FIRST and `ral_id` appears LAST.
        prereq_to_dependents: dict[str, list[str]] = defaultdict(list)
        in_degree: dict[str, int] = {nid: 0 for nid in visited_nodes}
        seen_pairs: set[tuple[str, str]] = set()
        for edge in collected_edges:
            src_id = edge["src"]
            dst_id = edge["dst"]
            if src_id in visited_nodes and dst_id in visited_nodes:
                pair = (dst_id, src_id)
                if pair not in seen_pairs:
                    seen_pairs.add(pair)
                    prereq_to_dependents[dst_id].append(src_id)
                    in_degree[src_id] += 1

        heap: list[tuple[int, str]] = [
            (-depth_by_id[nid], nid) for nid, deg in in_degree.items() if deg == 0
        ]
        heapq.heapify(heap)

        topo_order: list[dict[str, Any]] = []
        while heap:
            _, current = heapq.heappop(heap)
            topo_order.append(visited_nodes[current])
            for dependent in sorted(prereq_to_dependents.get(current, ())):
                in_degree[dependent] -= 1
                if in_degree[dependent] == 0:
                    heapq.heappush(heap, (-depth_by_id[dependent], dependent))

        if len(topo_order) != len(visited_nodes):
            raise ValueError(f"Cycle detected in DEPENDS_ON subgraph rooted at {clean_id}")

        return {
            "root_ral_id": clean_id,
            "max_depth": depth_limit,
            "node_count": len(topo_order),
            "topological_order": topo_order,
            "edges": collected_edges,
            "unresolved_stubs": sorted(unresolved_stubs),
        }

    def ral_find_isomorphisms(
        self,
        *,
        domain_a: str | None = None,
        domain_b: str | None = None,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Query `ISOMORPHIC_TO` edges and return structured `formal_mapping` dictionaries."""
        dom_a = validate_domain_filter(domain_a)
        dom_b = validate_domain_filter(domain_b)
        safe_limit = max(1, min(200, int(limit)))

        conn = _connect_readonly(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT
                    e.edge_id, e.src, e.dst, e.formal_mapping_json, e.confidence,
                    e.evidence_sha256, e.created_at, e.created_by_model,
                    ns.title AS src_title, ns.taxonomy_code AS src_taxonomy, ns.epistemic_tier AS src_tier,
                    nd.title AS dst_title, nd.taxonomy_code AS dst_taxonomy, nd.epistemic_tier AS dst_tier
                FROM edges e
                JOIN nodes ns ON ns.ral_id = e.src
                JOIN nodes nd ON nd.ral_id = e.dst
                WHERE e.relation = 'ISOMORPHIC_TO'
                ORDER BY COALESCE(e.confidence, 0.0) DESC, e.edge_id ASC
                """
            ).fetchall()
        finally:
            conn.close()

        matches: list[dict[str, Any]] = []
        for (
            edge_id,
            src,
            dst,
            mapping_json,
            conf,
            ev_sha,
            created_at,
            created_by,
            src_title,
            src_tax,
            src_tier,
            dst_title,
            dst_tax,
            dst_tier,
        ) in rows:
            s_tax = str(src_tax)
            d_tax = str(dst_tax)
            if dom_a and dom_b:
                ok = (_domain_matches(s_tax, dom_a) and _domain_matches(d_tax, dom_b)) or (
                    _domain_matches(s_tax, dom_b) and _domain_matches(d_tax, dom_a)
                )
            elif dom_a:
                ok = _domain_matches(s_tax, dom_a) or _domain_matches(d_tax, dom_a)
            elif dom_b:
                ok = _domain_matches(s_tax, dom_b) or _domain_matches(d_tax, dom_b)
            else:
                ok = True
            if not ok:
                continue

            formal_mapping: dict[str, Any] = {}
            if mapping_json:
                try:
                    parsed_map = json.loads(mapping_json)
                    if isinstance(parsed_map, Mapping):
                        formal_mapping = dict(parsed_map)
                except json.JSONDecodeError:
                    formal_mapping = {}

            matches.append(
                {
                    "edge_id": str(edge_id),
                    "relation": "ISOMORPHIC_TO",
                    "src": str(src),
                    "src_title": str(src_title),
                    "src_taxonomy": s_tax,
                    "src_tier": str(src_tier),
                    "dst": str(dst),
                    "dst_title": str(dst_title),
                    "dst_taxonomy": d_tax,
                    "dst_tier": str(dst_tier),
                    "formal_mapping": formal_mapping,
                    "confidence": float(conf) if conf is not None else None,
                    "evidence_sha256": ev_sha,
                    "created_at": created_at,
                    "created_by_model": created_by,
                }
            )
            if len(matches) >= safe_limit:
                break

        return {
            "domain_a": dom_a,
            "domain_b": dom_b,
            "count": len(matches),
            "isomorphisms": matches,
        }

    def dispatch_tool(self, name: str, arguments: Mapping[str, Any] | None = None) -> dict[str, Any]:
        """Dispatch one MCP tool call by name with explicit keyword validation."""
        args = dict(arguments or {})
        if name == "ral_search":
            return self.ral_search(
                query=args.get("query"),
                formal_signature_tags=args.get("formal_signature_tags"),
                max_tier=args.get("max_tier", "T2_ESTABLISHED"),
                limit=args.get("limit", 10),
            )
        if name == "ral_get_node":
            return self.ral_get_node(args.get("ral_id"))
        if name == "ral_get_dependency_cone":
            return self.ral_get_dependency_cone(
                args.get("ral_id"),
                max_depth=args.get("max_depth", 5),
            )
        if name == "ral_find_isomorphisms":
            return self.ral_find_isomorphisms(
                domain_a=args.get("domain_a"),
                domain_b=args.get("domain_b"),
                limit=args.get("limit", 50),
            )
        raise KeyError(f"Unknown MCP tool: {name!r}")

    def handle_jsonrpc_request(self, request: Mapping[str, Any]) -> dict[str, Any] | None:
        """Handle a single MCP JSON-RPC 2.0 request/notification dict."""
        method = str(request.get("method") or "")
        req_id = request.get("id")
        params = request.get("params") or {}

        if method == "notifications/initialized":
            return None

        if method == "initialize":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "protocolVersion": MCP_PROTOCOL_VERSION,
                    "capabilities": {"tools": {}},
                    "serverInfo": {
                        "name": "ral-mcp-server",
                        "version": "2.1.0",
                    },
                },
            }

        if method == "ping":
            return {"jsonrpc": "2.0", "id": req_id, "result": {}}

        if method == "tools/list":
            return {
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {"tools": MCP_TOOLS_SCHEMA},
            }

        if method == "tools/call":
            if not isinstance(params, Mapping):
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": "params must be an object"},
                }
            tool_name = str(params.get("name") or "")
            tool_args = params.get("arguments") or {}
            if not isinstance(tool_args, Mapping):
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "error": {"code": -32602, "message": "arguments must be an object"},
                }
            try:
                result = self.dispatch_tool(tool_name, tool_args)
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [{"type": "text", "text": json.dumps(result, indent=2)}],
                        "isError": False,
                    },
                }
            except Exception as exc:
                return {
                    "jsonrpc": "2.0",
                    "id": req_id,
                    "result": {
                        "content": [
                            {
                                "type": "text",
                                "text": json.dumps(
                                    {"error": type(exc).__name__, "detail": str(exc)}
                                ),
                            }
                        ],
                        "isError": True,
                    },
                }

        if req_id is None:
            return None
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }


def serve_stdio(
    server: RALMCPServer,
    *,
    stdin: TextIO | None = None,
    stdout: TextIO | None = None,
) -> int:
    """Serve newline-delimited MCP JSON-RPC 2.0 requests over stdio."""
    in_stream = stdin or sys.stdin
    out_stream = stdout or sys.stdout

    for raw_line in in_stream:
        line = raw_line.strip()
        if not line:
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32700, "message": f"Parse error: {exc}"},
            }
            out_stream.write(json.dumps(err_resp) + "\n")
            out_stream.flush()
            continue

        if not isinstance(payload, Mapping):
            err_resp = {
                "jsonrpc": "2.0",
                "id": None,
                "error": {"code": -32600, "message": "Invalid Request"},
            }
            out_stream.write(json.dumps(err_resp) + "\n")
            out_stream.flush()
            continue

        response = server.handle_jsonrpc_request(payload)
        if response is not None:
            out_stream.write(json.dumps(response) + "\n")
            out_stream.flush()

    return 0


def _split_repo_slug(repo_slug: str) -> tuple[str, str]:
    cleaned = repo_slug.strip().strip("/")
    if "/" not in cleaned:
        raise ValueError(f"Expected owner/repo slug, got {repo_slug!r}")
    owner, repo = cleaned.split("/", 1)
    return owner, repo


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint to sync the RAL SQLite index and run the MCP stdio server."""
    parser = argparse.ArgumentParser(description="Hardened read-only RAL MCP server (Section 9).")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--db", type=Path, default=None, help="Path to local compiled SQLite DB")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Cache dir for pointer & archive")
    parser.add_argument(
        "--sync-release",
        action="store_true",
        help="Sync latest compiled SQLite index from GitHub Release before serving",
    )
    parser.add_argument(
        "--github-repo",
        default=os.environ.get("GITHUB_REPOSITORY", ""),
        help="GitHub owner/repo slug (defaults to $GITHUB_REPOSITORY)",
    )
    parser.add_argument("--release-tag", default=INDEX_RELEASE_TAG)
    args = parser.parse_args(argv)

    server = RALMCPServer(
        args.repo_root,
        db_path=args.db,
        cache_dir=args.cache_dir,
    )

    if args.sync_release:
        if not args.github_repo:
            raise SystemExit("--sync-release requires --github-repo or GITHUB_REPOSITORY")
        owner, repo = _split_repo_slug(args.github_repo)
        gh = GitHubClient(owner, repo)
        server.sync_index(github_client=gh, tag=args.release_tag)

    return serve_stdio(server)


if __name__ == "__main__":
    raise SystemExit(main())

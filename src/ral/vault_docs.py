"""AI-native Vault index (`INDEX.md`) and frontier entry-point (`llms.txt`) generator (Sections 2, 5, 9)."""

from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any

from ci.validate_node import split_frontmatter
from ral.canonicalize import normalize_newlines
from ral.director import (
    MACRO_DISCIPLINES,
    extract_seed_leaves,
    load_taxonomy_seeds,
    macro_of,
)

TIER_ORDER: tuple[str, ...] = (
    "T1_CANONICAL",
    "T2_ESTABLISHED",
    "T3_SPECULATIVE",
    "T4_FALSIFIED",
)


def estimate_tokens(text: str) -> int:
    """Conservative deterministic token count estimator for Markdown/prose sizing."""
    if not text:
        return 0
    return max(len(text.split()), math.ceil(len(text) / 3.5))


def _atomic_write_text(target: Path, content: str) -> None:
    path = Path(target)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".tmp_{path.name}_", dir=str(path.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _count_ndjson_rows(dir_path: Path) -> int:
    if not dir_path.exists():
        return 0
    total = 0
    for ndjson_path in sorted(dir_path.glob("*.ndjson")):
        for line in ndjson_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                total += 1
    return total


def collect_vault_summary(repo_root: Path) -> dict[str, Any]:
    """Scan `vault/`, `graph/edges/`, `ledger/sources/`, `config/`, and `metrics/` for summary stats."""
    root = Path(repo_root)
    seeds_path = root / "config" / "taxonomy_seeds.yaml"
    seeds_cfg = load_taxonomy_seeds(seeds_path if seeds_path.exists() else None)
    seed_leaves = extract_seed_leaves(seeds_cfg)
    macro_cfg = seeds_cfg.get("macro_disciplines") or {}

    bench_path = root / "metrics" / "model_benchmarks.json"
    capability_epoch = 1
    if bench_path.exists():
        try:
            bench_data = json.loads(bench_path.read_text(encoding="utf-8"))
            capability_epoch = int(bench_data.get("model_capability_epoch", 1))
        except (ValueError, TypeError, json.JSONDecodeError):
            capability_epoch = 1

    tier_counts: Counter[str] = Counter()
    macro_node_counts: Counter[str] = Counter()
    macro_verified_counts: Counter[str] = Counter()
    nodes_by_leaf: dict[str, list[dict[str, Any]]] = defaultdict(list)

    vault_dir = root / "vault"
    if vault_dir.exists():
        for md_path in sorted(vault_dir.rglob("*.md")):
            raw = normalize_newlines(md_path.read_text(encoding="utf-8"))
            try:
                fm, _ = split_frontmatter(raw)
            except Exception:
                continue
            ral_id = str(fm.get("ral_id") or "").strip()
            tax_code = str(fm.get("taxonomy_code") or "").strip()
            tier = str(fm.get("epistemic_tier") or "").strip()
            if not ral_id or not tax_code:
                continue
            macro = macro_of(tax_code)
            rel_path = md_path.relative_to(root).as_posix()
            tier_counts[tier] += 1
            macro_node_counts[macro] += 1
            if tier in ("T1_CANONICAL", "T2_ESTABLISHED"):
                macro_verified_counts[macro] += 1
            nodes_by_leaf[tax_code].append(
                {
                    "ral_id": ral_id,
                    "epistemic_tier": tier,
                    "consensus_version": int(fm.get("consensus_version") or 1),
                    "node_type": str(fm.get("node_type") or ""),
                    "taxonomy_code": tax_code,
                    "title": str(fm.get("title") or ral_id).strip(),
                    "rel_path": rel_path,
                }
            )

    for leaf_nodes in nodes_by_leaf.values():
        leaf_nodes.sort(key=lambda n: (TIER_ORDER.index(n["epistemic_tier"]) if n["epistemic_tier"] in TIER_ORDER else 99, n["ral_id"]))

    edge_count = _count_ndjson_rows(root / "graph" / "edges")
    source_count = _count_ndjson_rows(root / "ledger" / "sources")
    total_nodes = sum(tier_counts.values())

    active_leaves_by_macro: dict[str, list[str]] = {m: [] for m in MACRO_DISCIPLINES}
    for code in sorted(set(seed_leaves.keys()) | set(nodes_by_leaf.keys())):
        m = macro_of(code)
        active_leaves_by_macro.setdefault(m, []).append(code)

    macros_summary: list[dict[str, Any]] = []
    for m in MACRO_DISCIPLINES:
        info = macro_cfg.get(m) or {}
        leaf_codes = active_leaves_by_macro.get(m, [])
        macros_summary.append(
            {
                "code": m,
                "name": str(info.get("name") or m.upper()),
                "authority": str(info.get("authority") or "Custom"),
                "leaf_count": len(leaf_codes),
                "leaves": leaf_codes,
                "node_count": macro_node_counts.get(m, 0),
                "verified_count": macro_verified_counts.get(m, 0),
            }
        )

    return {
        "schema_version": "2.1",
        "capability_epoch": capability_epoch,
        "total_nodes": total_nodes,
        "verified_nodes": tier_counts.get("T1_CANONICAL", 0) + tier_counts.get("T2_ESTABLISHED", 0),
        "tier_counts": {t: tier_counts.get(t, 0) for t in TIER_ORDER},
        "edge_count": edge_count,
        "source_count": source_count,
        "seed_leaves": seed_leaves,
        "macros": macros_summary,
        "nodes_by_leaf": dict(nodes_by_leaf),
    }


def _build_index_header_and_table(summary: dict[str, Any], ts_iso: str) -> list[str]:
    tc = summary["tier_counts"]
    lines = [
        "# Rayhan's Autonomous Lab (RAL) — Canonical Vault Index",
        "",
        f"- **Schema Version:** `{summary['schema_version']}`",
        f"- **Generated At (UTC):** `{ts_iso}`",
        f"- **Model Capability Epoch:** `{summary['capability_epoch']}`",
        f"- **Total Nodes:** `{summary['total_nodes']}` "
        f"(`T1_CANONICAL`: {tc['T1_CANONICAL']}, `T2_ESTABLISHED`: {tc['T2_ESTABLISHED']}, "
        f"`T3_SPECULATIVE`: {tc['T3_SPECULATIVE']}, `T4_FALSIFIED`: {tc['T4_FALSIFIED']})",
        f"- **Graph Edges:** `{summary['edge_count']}` | **Ingested Sources:** `{summary['source_count']}`",
        "",
        "## Macro-Discipline Overview",
        "",
        "| Macro | Discipline | Authority | Active Leaves | Verified (`T1`+`T2`) | Total Nodes |",
        "| :--- | :--- | :--- | ---: | ---: | ---: |",
    ]
    for m in summary["macros"]:
        lines.append(
            f"| `{m['code']}` | {m['name']} | `{m['authority']}` | "
            f"{m['leaf_count']} | {m['verified_count']} | {m['node_count']} |"
        )
    lines.append("")
    return lines


def _render_leaf_sections(
    summary: dict[str, Any],
    *,
    per_leaf_node_cap: int | None,
    include_node_lines: bool,
) -> list[str]:
    lines = ["## Hierarchical Taxonomy Map", ""]
    seed_leaves = summary["seed_leaves"]
    nodes_by_leaf = summary["nodes_by_leaf"]

    for m in summary["macros"]:
        lines.append(f"### `{m['code']}` — {m['name']} ({m['authority']})")
        lines.append("")
        for code in m["leaves"]:
            meta = seed_leaves.get(code)
            leaf_title = meta["name"] if meta else code
            nodes = nodes_by_leaf.get(code, [])
            verified_in_leaf = sum(
                1 for n in nodes if n["epistemic_tier"] in ("T1_CANONICAL", "T2_ESTABLISHED")
            )
            lines.append(
                f"- **`{code}`** — {leaf_title} "
                f"(`{len(nodes)}` nodes, `{verified_in_leaf}` verified)"
            )
            if include_node_lines and nodes:
                shown = nodes if per_leaf_node_cap is None else nodes[:per_leaf_node_cap]
                for node in shown:
                    lines.append(
                        f"  - `{node['ral_id']}` [{node['epistemic_tier']} v{node['consensus_version']}]: "
                        f"[{node['title']}]({node['rel_path']})"
                    )
                omitted = len(nodes) - len(shown)
                if omitted > 0:
                    lines.append(
                        f"  - *(+{omitted} additional node(s) omitted for `<15,000` token cap; "
                        "query via `mcp/ral_mcp_server.py` or `ral-index-latest.json`)*"
                    )
        lines.append("")
    return lines


def _render_pointer_footer() -> list[str]:
    return [
        "## Compiled Index & Machine Query Entry Point",
        "",
        "- For full-text BM25, `formal_signature` Jaccard, topological dependency cones, and "
        "cross-domain isomorphisms, query `mcp/ral_mcp_server.py` backed by the Blue-Green "
        "`nightly-index` GitHub Release (`ral-index-latest.json`).",
        "- See `llms.txt` for machine-readable agent bootstrap instructions.",
        "",
    ]


def render_index_md(
    repo_root: Path,
    *,
    now: datetime | None = None,
    max_tokens: int = 15000,
) -> str:
    """
    Render `/INDEX.md` as a hierarchical vault map strictly bounded below `max_tokens` (15,000).
    """
    ts = now or datetime.now(timezone.utc)
    ts_iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = collect_vault_summary(repo_root)

    header_lines = _build_index_header_and_table(summary, ts_iso)
    footer_lines = _render_pointer_footer()

    candidates: list[list[str]] = [
        header_lines
        + _render_leaf_sections(summary, per_leaf_node_cap=None, include_node_lines=True)
        + footer_lines,
        header_lines
        + _render_leaf_sections(summary, per_leaf_node_cap=10, include_node_lines=True)
        + footer_lines,
        header_lines
        + _render_leaf_sections(summary, per_leaf_node_cap=5, include_node_lines=True)
        + footer_lines,
        header_lines
        + _render_leaf_sections(summary, per_leaf_node_cap=3, include_node_lines=True)
        + footer_lines,
        header_lines
        + _render_leaf_sections(summary, per_leaf_node_cap=1, include_node_lines=True)
        + footer_lines,
        header_lines
        + _render_leaf_sections(summary, per_leaf_node_cap=0, include_node_lines=False)
        + footer_lines,
        header_lines + footer_lines,
    ]

    for block in candidates:
        rendered = "\n".join(block).rstrip() + "\n"
        if estimate_tokens(rendered) < max_tokens:
            return rendered

    # Ultimate compact fallback if caller supplies a tiny custom `max_tokens`.
    minimal = (
        "# Rayhan's Autonomous Lab (RAL) — Canonical Vault Index\n\n"
        f"Nodes: {summary['total_nodes']} | Edges: {summary['edge_count']} | "
        f"Epoch: {summary['capability_epoch']}\n"
        "Query `mcp/ral_mcp_server.py` or `ral-index-latest.json`.\n"
    )
    return minimal


def render_llms_txt(
    repo_root: Path,
    *,
    repo_slug: str = "rayhans-autonomous-lab",
    now: datetime | None = None,
) -> str:
    """Render machine-readable `/llms.txt` entry point for future frontier AI models."""
    ts = now or datetime.now(timezone.utc)
    ts_iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    summary = collect_vault_summary(repo_root)
    tc = summary["tier_counts"]

    lines = [
        f"# {repo_slug}",
        "",
        "> Rayhan's Autonomous Lab (RAL) — Self-verifying autonomous scientific & mathematical knowledge vault.",
        "> Enforces Generator-Verifier Asymmetry: primary-source verbatim quote hashes + air-gapped Docker deterministic oracles.",
        "",
        "## Live Vault Snapshot",
        f"- schema_version: {summary['schema_version']}",
        f"- updated_at_utc: {ts_iso}",
        f"- model_capability_epoch: {summary['capability_epoch']}",
        f"- total_nodes: {summary['total_nodes']}",
        f"- T1_CANONICAL: {tc['T1_CANONICAL']}",
        f"- T2_ESTABLISHED: {tc['T2_ESTABLISHED']}",
        f"- T3_SPECULATIVE: {tc['T3_SPECULATIVE']}",
        f"- T4_FALSIFIED: {tc['T4_FALSIFIED']}",
        f"- total_edges: {summary['edge_count']}",
        f"- total_sources: {summary['source_count']}",
        "",
        "## Dual-Storage Topology",
        "- Distilled Gold (this Git repository): canonical Markdown + LaTeX + verified Python kernels + sharded NDJSON ledgers (`vault/`, `graph/`, `ledger/`, `metrics/`). Open-Access licenses only.",
        "- Compiled Ephemeral Index (GitHub Release tag `nightly-index`): Blue-Green swapped `ral-index-YYYYMMDD-{run_id}.sqlite.zst` + pointer `ral-index-latest.json`. Always verify SHA-256 before Zstandard decompression.",
        "- Cold Ore Archive (`RAL_Ore_Warehouse`): content-addressed primary tarballs/XML/PDFs stored after passing Stages 1-5 (Verify-Before-Archive 2PC).",
        "",
        "## Epistemic Tier Contract",
        "- T1_CANONICAL: Passes `sympy_numeric` or `z3_smt` oracle + peer-reviewed/canonical reference backing + survives blinded cross-family Skeptic.",
        "- T2_ESTABLISHED: Peer-reviewed empirical result passing `empirical_stats` with uncertainty bounds + survives blinded Skeptic.",
        "- T3_SPECULATIVE: Unrefereed preprint or open anomaly (`re_evaluate_after = +180d`); forbidden as `DEPENDS_ON` prerequisite for `T1_CANONICAL`.",
        "- T4_FALSIFIED: Retracted result or verified counterexample (`NEGATIVE_RESULT`), attached via `BREAKS_WITHOUT_ASSUMPTION`.",
        "",
        "## Read-Only MCP Server (One-Command Launch)",
        "- Launch (stdio JSON-RPC 2.0): `python mcp/ral_mcp_server.py --repo-root .`",
        "- Sync compiled index first: `python mcp/ral_mcp_server.py --repo-root . --sync-release`",
        "- Security invariants: SQLite `?mode=ro`, parameterized SQL only, strict `ral_id` regex `^RAL-[A-Z]{2,6}-[0-9A-Z]{10}$`, `vault/` path confinement.",
        "",
        "## Deterministic MCP Tools",
        "1. `ral_search(query, formal_signature_tags=None, max_tier='T2_ESTABLISHED', limit=10)`: Hybrid FTS5 BM25 + `formal_signature` Jaccard (+ consistent-space vector cosine).",
        "2. `ral_get_node(ral_id)`: Canonical Markdown + linked `BREAKS_WITHOUT_ASSUMPTION` (`NEGATIVE_RESULT`) failure modes + `T3`/`T4` warning banners.",
        "3. `ral_get_dependency_cone(ral_id, max_depth=5)`: Acyclic `DEPENDS_ON` prerequisite subgraph in topological order (foundations first).",
        "4. `ral_find_isomorphisms(domain_a=None, domain_b=None)`: Structured `ISOMORPHIC_TO` cross-discipline `formal_mapping` variable dictionaries.",
        "",
        "## Key Repository Files",
        "- `INDEX.md`: Hierarchical vault map (<15,000 tokens).",
        "- `docs/RAL_Blueprint_v2.1.md`: Authoritative architecture & epistemology specification.",
        "- `config/taxonomy_seeds.yaml`: 7 macro-disciplines and 35 Day-0 leaves.",
        "- `config/license_allowlist.yaml`: Open-Access / Public-Domain license allowlist.",
        "- `config/model_policy.yaml`: Frontier exclusion regexes & embedding model discovery policy.",
        "- `ci/validate_node.py` & `ci/validate_vault.py`: Blocking schema, AST anti-cheat, DAG, and leak gates.",
        "",
        "## Macro-Disciplines & Seed Leaves",
    ]

    for m in summary["macros"]:
        leaf_list = ", ".join(m["leaves"])
        lines.append(f"- `{m['code']}` ({m['name']}, {m['authority']}): {leaf_list}")

    return "\n".join(lines).rstrip() + "\n"


def regenerate_vault_docs(
    repo_root: Path,
    *,
    now: datetime | None = None,
    repo_slug: str = "rayhans-autonomous-lab",
    max_index_tokens: int = 15000,
) -> dict[str, Any]:
    """Generate and atomically write `<repo_root>/INDEX.md` and `<repo_root>/llms.txt`."""
    root = Path(repo_root)
    ts = now or datetime.now(timezone.utc)
    index_md = render_index_md(root, now=ts, max_tokens=max_index_tokens)
    llms_txt = render_llms_txt(root, repo_slug=repo_slug, now=ts)

    index_path = root / "INDEX.md"
    llms_path = root / "llms.txt"
    _atomic_write_text(index_path, index_md)
    _atomic_write_text(llms_path, llms_txt)

    summary = collect_vault_summary(root)
    return {
        "index_path": index_path,
        "llms_path": llms_path,
        "index_tokens": estimate_tokens(index_md),
        "llms_tokens": estimate_tokens(llms_txt),
        "node_count": summary["total_nodes"],
        "edge_count": summary["edge_count"],
    }

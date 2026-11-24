"""Blocking repository-wide CI validator for RAL vault, graph, ledgers, and leaks (Section 8)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
import json
import math
from pathlib import Path
import re
import sys
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ci.validate_node import (  # noqa: E402
    NodeFrontmatter,
    load_allowed_licenses,
    validate_node_file,
)

ALLOWED_EDGE_RELATIONS: frozenset[str] = frozenset(
    {
        "DEPENDS_ON",
        "GENERALIZES",
        "INSTANTIATES",
        "ISOMORPHIC_TO",
        "CONTRADICTS",
        "CHALLENGES",
        "SUPERSEDES",
        "BREAKS_WITHOUT_ASSUMPTION",
    }
)

_LOCAL_PATH_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("MAC_USERS_PATH", re.compile(r"(?<![A-Za-z0-9])/(?:Users|home)/(?:[A-Za-z0-9._-]+)?")),
    ("WIN_USERS_PATH", re.compile(r"[A-Za-z]:[/\\]Users[/\\](?:[A-Za-z0-9._-]+)?")),
)

_SECRET_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("GOOGLE_API_KEY", re.compile(r"\bAIza[0-9A-Za-z\-_]{10,}\b")),
    ("GITHUB_TOKEN", re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{16,}\b")),
    ("GITHUB_FINE_PAT", re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}\b")),
    ("GOOGLE_OAUTH_ACCESS", re.compile(r"\bya29\.[0-9A-Za-z\-_]{12,}\b")),
    ("GOOGLE_OAUTH_REFRESH", re.compile(r"\b1//[0-9A-Za-z\-_]{16,}\b")),
    ("AWS_ACCESS_KEY", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("SK_SECRET_KEY", re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b")),
    ("PRIVATE_KEY_HEADER", re.compile(r"-----BEGIN (?:RSA |EC |DSA |OPENSSH |)PRIVATE KEY-----")),
    (
        "OAUTH_JSON_SECRET",
        re.compile(
            r'"(?:client_secret|refresh_token|private_key|private_key_id|access_token)"\s*:\s*"[^"]{8,}"'
        ),
    ),
    ("EMAIL_ADDRESS", re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b")),
)

_HEX_RE = re.compile(r"^[0-9a-fA-F]{16,}$")
_TOKEN_SPLIT_RE = re.compile(r"""[\s"'`,;(){}\[\]<>]+""")


def shannon_entropy(text: str) -> float:
    """Compute Shannon entropy in bits per character."""
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum((c / length) * math.log2(c / length) for c in counts.values())


def scan_text_for_leaks(text: str, source_label: str) -> list[str]:
    """Scan text for local machine paths, secret prefixes, OAuth JSON, emails, and high-entropy tokens."""
    findings: list[str] = []
    for line_no, line in enumerate(text.splitlines(), start=1):
        for tag, pat in _LOCAL_PATH_PATTERNS:
            match = pat.search(line)
            if match:
                findings.append(
                    f"PRIVATE_PATH_LEAK({tag} in {source_label}:{line_no}): {match.group(0)!r}"
                )
        for tag, pat in _SECRET_PATTERNS:
            match = pat.search(line)
            if match:
                findings.append(
                    f"SECRET_LEAK({tag} in {source_label}:{line_no})"
                )

        for token in _TOKEN_SPLIT_RE.split(line):
            if len(token) < 32:
                continue
            if token.startswith(("sha256:", "http://", "https://", "gdrive://")):
                continue
            if "/" in token or "\\" in token:
                continue
            if _HEX_RE.match(token):
                continue
            has_upper = any("A" <= ch <= "Z" for ch in token)
            has_lower = any("a" <= ch <= "z" for ch in token)
            has_digit = any("0" <= ch <= "9" for ch in token)
            if has_upper and has_lower and has_digit and shannon_entropy(token) >= 4.5:
                findings.append(
                    f"HIGH_ENTROPY_SECRET({source_label}:{line_no}): token length {len(token)}"
                )
    return findings


def detect_depends_on_cycles(edges: Sequence[dict[str, Any]]) -> list[str]:
    """
    Detect cycles in the DEPENDS_ON subgraph using Kahn's topological sort algorithm.

    Returns a sorted list of node IDs participating in or blocked by a cycle; empty if acyclic.
    """
    adj: dict[str, list[str]] = defaultdict(list)
    in_degree: dict[str, int] = {}

    for edge in edges:
        if edge.get("relation") != "DEPENDS_ON":
            continue
        src = edge.get("src")
        dst = edge.get("dst")
        if not src or not dst:
            continue
        in_degree.setdefault(src, 0)
        in_degree.setdefault(dst, 0)
        adj[src].append(dst)
        in_degree[dst] += 1

    queue: deque[str] = deque(sorted(node for node, deg in in_degree.items() if deg == 0))
    visited = 0
    while queue:
        current = queue.popleft()
        visited += 1
        for nxt in adj.get(current, ()):
            in_degree[nxt] -= 1
            if in_degree[nxt] == 0:
                queue.append(nxt)

    if visited == len(in_degree):
        return []
    return sorted(node for node, deg in in_degree.items() if deg > 0)


def check_tier_firewall(edges: Sequence[dict[str, Any]], node_tiers: dict[str, str]) -> list[str]:
    """Enforce that no T1_CANONICAL node has a DEPENDS_ON edge to T3_SPECULATIVE or T4_FALSIFIED."""
    violations: list[str] = []
    for edge in edges:
        if edge.get("relation") != "DEPENDS_ON":
            continue
        src = edge.get("src", "")
        dst = edge.get("dst", "")
        src_tier = node_tiers.get(src)
        dst_tier = node_tiers.get(dst)
        if src_tier == "T1_CANONICAL" and dst_tier in ("T3_SPECULATIVE", "T4_FALSIFIED"):
            violations.append(
                f"TIER_FIREWALL_VIOLATION: {src} ({src_tier}) DEPENDS_ON {dst} ({dst_tier})"
            )
    return violations


def compute_dir_size_mb(path: Path) -> float:
    """Compute recursive directory size in MiB (0.0 if directory does not exist)."""
    if not path.exists():
        return 0.0
    total_bytes = sum(p.stat().st_size for p in path.rglob("*") if p.is_file())
    return total_bytes / (1024.0 * 1024.0)


def _iter_ndjson_rows(path: Path) -> list[tuple[int, dict[str, Any]]]:
    rows: list[tuple[int, dict[str, Any]]] = []
    for line_no, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        stripped = raw_line.strip()
        if not stripped:
            continue
        parsed = json.loads(stripped)
        if not isinstance(parsed, dict):
            raise ValueError(f"{path}:{line_no} must be a JSON object")
        rows.append((line_no, parsed))
    return rows


@dataclass(frozen=True)
class VaultValidationReport:
    """Summary returned by `validate_vault()`."""

    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    node_count: int = 0
    edge_count: int = 0
    git_size_mb: float = 0.0
    git_size_alert: bool = False


def validate_vault(
    repo_root: Path,
    *,
    max_git_mb: float = 700.0,
    allowlist_path: Path | None = None,
) -> VaultValidationReport:
    """Run all Section 8.2 repository-wide blocking CI validation gates."""
    root = Path(repo_root)
    errors: list[str] = []
    warnings: list[str] = []

    resolved_allowlist = allowlist_path or (root / "config" / "license_allowlist.yaml")
    if not resolved_allowlist.exists():
        errors.append(f"MISSING_ALLOWLIST: {resolved_allowlist} does not exist")
        allowed_licenses: set[str] = set()
    else:
        allowed_licenses = load_allowed_licenses(resolved_allowlist)

    vault_nodes: dict[str, NodeFrontmatter] = {}
    node_tiers: dict[str, str] = {}
    referenced_ore_hashes: set[str] = set()

    vault_dir = root / "vault"
    if vault_dir.exists():
        for md_path in sorted(vault_dir.rglob("*.md")):
            rel_label = md_path.relative_to(root).as_posix()
            res = validate_node_file(
                md_path,
                allowed_licenses=allowed_licenses if allowed_licenses else None,
                check_filename=True,
            )
            if not res.ok:
                for err in res.errors:
                    errors.append(f"NODE_INVALID({rel_label}): {err}")
            if res.frontmatter is not None:
                rid = res.frontmatter.ral_id
                if rid in vault_nodes:
                    errors.append(f"DUPLICATE_RAL_ID: {rid} duplicated at {rel_label}")
                else:
                    vault_nodes[rid] = res.frontmatter
                    node_tiers[rid] = res.frontmatter.epistemic_tier
                for prov in res.frontmatter.provenance:
                    referenced_ore_hashes.add(prov.ore_sha256)

    known_ore_hashes: set[str] = set()
    sources_dir = root / "ledger" / "sources"
    if sources_dir.exists():
        for ndjson_path in sorted(sources_dir.glob("*.ndjson")):
            rel_label = ndjson_path.relative_to(root).as_posix()
            try:
                rows = _iter_ndjson_rows(ndjson_path)
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"NDJSON_PARSE_ERROR({rel_label}): {exc}")
                continue
            for line_no, row in rows:
                ore_sha = row.get("ore_sha256")
                if not isinstance(ore_sha, str) or not re.match(r"^[0-9a-f]{64}$", ore_sha):
                    errors.append(f"INVALID_SOURCE_HASH({rel_label}:{line_no}): {ore_sha!r}")
                else:
                    known_ore_hashes.add(ore_sha)
                lic = row.get("source_license")
                if allowed_licenses and lic not in allowed_licenses:
                    errors.append(
                        f"DISALLOWED_SOURCE_LICENSE({rel_label}:{line_no}): {lic!r}"
                    )

    for ore_sha in sorted(referenced_ore_hashes - known_ore_hashes):
        errors.append(
            f"MISSING_SOURCE_LEDGER_ENTRY: ore_sha256 {ore_sha} referenced in vault/ not found in ledger/sources/*.ndjson"
        )

    registered_stubs: set[str] = set()
    prereq_debt_path = root / "ledger" / "prereq_debt.ndjson"
    if prereq_debt_path.exists():
        rel_label = prereq_debt_path.relative_to(root).as_posix()
        try:
            for line_no, row in _iter_ndjson_rows(prereq_debt_path):
                stub_id = row.get("stub_id")
                if isinstance(stub_id, str) and stub_id.startswith("STUB-"):
                    registered_stubs.add(stub_id)
                else:
                    errors.append(f"INVALID_STUB_ID({rel_label}:{line_no}): {stub_id!r}")
        except (ValueError, json.JSONDecodeError) as exc:
            errors.append(f"NDJSON_PARSE_ERROR({rel_label}): {exc}")

    edges: list[dict[str, Any]] = []
    edges_dir = root / "graph" / "edges"
    if edges_dir.exists():
        for ndjson_path in sorted(edges_dir.glob("*.ndjson")):
            rel_label = ndjson_path.relative_to(root).as_posix()
            try:
                rows = _iter_ndjson_rows(ndjson_path)
            except (ValueError, json.JSONDecodeError) as exc:
                errors.append(f"NDJSON_PARSE_ERROR({rel_label}): {exc}")
                continue
            for line_no, row in rows:
                edges.append(row)
                rel = row.get("relation")
                if rel not in ALLOWED_EDGE_RELATIONS:
                    errors.append(f"INVALID_EDGE_RELATION({rel_label}:{line_no}): {rel!r}")
                src = row.get("src")
                dst = row.get("dst")
                if src not in vault_nodes:
                    errors.append(f"DANGLING_EDGE_SRC({rel_label}:{line_no}): unknown node {src!r}")
                if dst not in vault_nodes and dst not in registered_stubs:
                    errors.append(
                        f"DANGLING_EDGE_DST({rel_label}:{line_no}): unknown target {dst!r}"
                    )

    cycle_nodes = detect_depends_on_cycles(edges)
    if cycle_nodes:
        errors.append(
            f"DEPENDS_ON_CYCLE: topological sort failed; cycle involves nodes {', '.join(cycle_nodes)}"
        )

    errors.extend(check_tier_firewall(edges, node_tiers))

    scan_targets: list[Path] = []
    for folder in ("vault", "ledger", "graph", "metrics", "config"):
        subdir = root / folder
        if subdir.exists():
            scan_targets.extend(sorted(p for p in subdir.rglob("*") if p.is_file()))
    for top_file in ("INDEX.md", "llms.txt"):
        candidate = root / top_file
        if candidate.is_file():
            scan_targets.append(candidate)

    for target in scan_targets:
        rel_label = target.relative_to(root).as_posix()
        try:
            content = target.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        errors.extend(scan_text_for_leaks(content, rel_label))

    git_size_mb = compute_dir_size_mb(root / ".git")
    git_size_alert = git_size_mb > max_git_mb
    if git_size_alert:
        warnings.append(
            f"GIT_SIZE_CRITICAL_ALERT: .git size {git_size_mb:.2f} MB exceeds {max_git_mb:.2f} MB"
        )

    return VaultValidationReport(
        ok=len(errors) == 0,
        errors=errors,
        warnings=warnings,
        node_count=len(vault_nodes),
        edge_count=len(edges),
        git_size_mb=round(git_size_mb, 3),
        git_size_alert=git_size_alert,
    )


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate full RAL repository vault & CI rails.")
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root directory")
    parser.add_argument("--max-git-mb", type=float, default=700.0, help=".git alert threshold in MiB")
    parser.add_argument("--allowlist", type=Path, default=None, help="Optional license_allowlist.yaml path")
    args = parser.parse_args(argv)

    report = validate_vault(args.root, max_git_mb=args.max_git_mb, allowlist_path=args.allowlist)
    for warn in report.warnings:
        print(f"WARN: {warn}", file=sys.stderr)
    if report.ok:
        print(
            f"VAULT_OK: nodes={report.node_count} edges={report.edge_count} "
            f"git_size_mb={report.git_size_mb:.3f}"
        )
        return 0
    print(
        f"VAULT_FAIL: {len(report.errors)} error(s), nodes={report.node_count}, edges={report.edge_count}",
        file=sys.stderr,
    )
    for err in report.errors:
        print(f"  - {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

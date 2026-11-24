"""Zero-LLM maintenance engine: retraction sweep, kernel regression, Drive GC, and 429 RPD pivot (Sections 5 & 11)."""

from __future__ import annotations

import argparse
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import re
from typing import Any, Callable, Literal, Mapping, Sequence
import uuid

import yaml

from ci.validate_node import (
    extract_verification_kernel,
    load_allowed_licenses,
    split_frontmatter,
    validate_node_content,
)
from ci.validate_vault import validate_vault
from ral.canonicalize import normalize_newlines
from ral.model_router import AllModelsExhaustedError, ModelRouter, is_rate_limit_error
from ral.pipeline.oracle_sandbox import OracleSandbox
from ral.pipeline.oracles.base import OraclePluginRegistry, get_default_registry
from ral.sources.crossref_retract import RetractionStatus, check_crossref_status, normalize_doi
from ral.storage.drive_ore import DriveOreStore, OrphanGCReport

_ACTIVE_CONSENSUS_RE = re.compile(
    r"(##\s+Active Consensus[^\n]*\n)(.*?)(?=\n##\s+Active Challenges & Open Anomalies)",
    re.DOTALL,
)
_EFFECTIVE_DATE_RE = re.compile(r"Effective\s+(\d{4}-\d{2}-\d{2})")
_FAILURES_RE = re.compile(
    r"(###\s+Failure Boundaries & Known Pathologies\s*\n)(.*?)(?=\n###\s+Executable Verification Kernel)",
    re.DOTALL,
)
_ARCHIVED_RE = re.compile(
    r"(##\s+Superseded & Archived[ \t]*(?:\n|\Z))(.*)\Z",
    re.DOTALL,
)


@dataclass(frozen=True)
class RetractionSweepReport:
    """Outcome of zero-LLM Crossref / Retraction Watch sweep across `vault/`."""

    nodes_scanned: int
    dois_checked: int
    retracted_nodes: tuple[str, ...]
    cascaded_t1_demotions: tuple[str, ...]
    lineage_records: tuple[dict[str, Any], ...]
    errors: tuple[str, ...]


@dataclass(frozen=True)
class KernelRegressionReport:
    """Outcome of re-executing existing verification kernels inside the air-gapped Docker sandbox."""

    kernels_checked: int
    passed: int
    failed: int
    failures: tuple[dict[str, str], ...]


@dataclass(frozen=True)
class MaintenanceReport:
    """Combined zero-LLM maintenance execution summary."""

    ok: bool
    mode: str
    circuit_breaker: str | None
    retraction_sweep: RetractionSweepReport | None
    kernel_regression: KernelRegressionReport | None
    drive_gc: OrphanGCReport | None
    vault_validation_ok: bool
    vault_validation_errors: tuple[str, ...]
    telemetry_path: Path | None
    llm_calls_made: int = 0


@dataclass(frozen=True)
class EpochPivotResult:
    """Result returned by `run_epoch_or_pivot()` covering normal completion or 429/CB pivot."""

    status: Literal[
        "LLM_EPOCH_COMPLETE",
        "PIVOTED_TO_MAINTENANCE",
        "CIRCUIT_BREAKER_MAINTENANCE",
    ]
    pivoted_to_maintenance: bool
    pivot_reason: str | None
    epoch_output: Any
    maintenance_report: MaintenanceReport | None
    exhausted_models: tuple[str, ...]


def _new_lineage_id() -> str:
    alphabet = "0123456789ABCDEFGHJKMNPQRSTVWXYZ"
    raw = uuid.uuid4().bytes[:10]
    return "L-" + "".join(alphabet[b % len(alphabet)] for b in raw)


def _iter_ndjson_dir(dir_path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not dir_path.exists():
        return rows
    for path in sorted(dir_path.glob("*.ndjson")):
        for line in path.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            parsed = json.loads(stripped)
            if isinstance(parsed, dict):
                rows.append(parsed)
    return rows


def _extract_doi_from_provenance(
    prov: Mapping[str, Any],
    ore_to_doi: Mapping[str, str],
) -> str | None:
    ore_sha = str(prov.get("ore_sha256") or "").strip().lower()
    if ore_sha in ore_to_doi:
        return ore_to_doi[ore_sha]
    source_id = str(prov.get("source_id") or "").strip()
    lowered = source_id.lower()
    if lowered.startswith("doi:") or source_id.startswith("10.") or "doi.org/" in lowered:
        return normalize_doi(source_id)
    return None


def _append_only_tier_downgrade(
    node_path: Path,
    *,
    new_tier: str,
    re_evaluate_after: str,
    reason: str,
    notice_bullet: str,
    retracted_ore_hashes: set[str] | None,
    allowed_licenses: set[str] | None,
    ts: datetime,
) -> dict[str, Any] | None:
    """
    Perform an append-only Theory-Upgrade/Downgrade mutation preserving `ral_id`,
    verification kernel hashes, and archiving the previous Active Consensus verbatim in `<details>`.
    """
    raw = normalize_newlines(node_path.read_text(encoding="utf-8"))
    fm, body = split_frontmatter(raw)
    ral_id = str(fm["ral_id"])
    old_tier = str(fm["epistemic_tier"])
    old_version = int(fm["consensus_version"])

    changed_prov = False
    if retracted_ore_hashes:
        for entry in fm.get("provenance") or []:
            if str(entry.get("ore_sha256") or "").lower() in retracted_ore_hashes:
                if not entry.get("retracted"):
                    entry["retracted"] = True
                    changed_prov = True

    if old_tier == new_tier and not changed_prov:
        return None

    ts_iso = ts.strftime("%Y-%m-%dT%H:%M:%SZ")
    today_str = ts_iso[:10]

    active_match = _ACTIVE_CONSENSUS_RE.search(body)
    if not active_match:
        raise ValueError(f"Node {ral_id} at {node_path} missing Active Consensus block")

    prior_header = active_match.group(1)
    prior_inner = active_match.group(2)
    prior_full_block = (prior_header + prior_inner).strip()
    date_match = _EFFECTIVE_DATE_RE.search(prior_header)
    old_date = date_match.group(1) if date_match else str(fm["verification"]["verified_at"])[:10]

    new_version = old_version + 1
    new_header = f"## Active Consensus (v{new_version} — Effective {today_str})\n"

    fail_match = _FAILURES_RE.search(prior_inner)
    if fail_match:
        existing_fail_body = fail_match.group(2).strip()
        if notice_bullet not in existing_fail_body:
            updated_fail_body = (
                f"{notice_bullet}\n{existing_fail_body}\n" if existing_fail_body else f"{notice_bullet}\n"
            )
            new_inner = (
                prior_inner[: fail_match.start(2)]
                + updated_fail_body
                + prior_inner[fail_match.end(2) :]
            )
        else:
            new_inner = prior_inner
    else:
        new_inner = prior_inner

    archived_entry = (
        f"<details>\n"
        f"<summary>v{old_version} ({old_date} to {today_str}) | Reason: {reason}</summary>\n\n"
        f"{prior_full_block}\n"
        f"</details>"
    )

    updated_body = body[: active_match.start(1)] + new_header + new_inner + body[active_match.end(2) :]
    arch_match = _ARCHIVED_RE.search(updated_body)
    if arch_match:
        existing_arch = arch_match.group(2).strip()
        if existing_arch and existing_arch != "*(None)*":
            combined_arch = f"{archived_entry}\n\n{existing_arch}"
        else:
            combined_arch = archived_entry
        arch_header = arch_match.group(1)
        if not arch_header.endswith("\n"):
            arch_header += "\n"
        updated_body = updated_body[: arch_match.start(1)] + arch_header + combined_arch + "\n"
    else:
        updated_body = updated_body.rstrip() + f"\n\n## Superseded & Archived\n{archived_entry}\n"

    fm["epistemic_tier"] = new_tier
    fm["consensus_version"] = new_version
    fm["re_evaluate_after"] = re_evaluate_after

    updated_md = f"---\n{yaml.safe_dump(fm, sort_keys=False).strip()}\n---\n{updated_body.lstrip()}"
    res = validate_node_content(
        updated_md,
        allowed_licenses=allowed_licenses,
        expected_filename_stem=node_path.stem,
    )
    if not res.ok:
        raise ValueError(f"Downgraded node {ral_id} failed validation: {res.errors}")

    node_path.write_text(updated_md, encoding="utf-8")

    return {
        "lineage_id": _new_lineage_id(),
        "ral_id": ral_id,
        "relation": "SUPERSEDES",
        "from_version": old_version,
        "to_version": new_version,
        "from_tier": old_tier,
        "to_tier": new_tier,
        "reason": reason,
        "source_id": str((fm.get("provenance") or [{}])[0].get("source_id") or ""),
        "evidence_sha256": str((fm.get("provenance") or [{}])[0].get("quote_sha256") or ""),
        "created_at": ts_iso,
        "created_by_model": "ral-maintenance-zero-llm",
    }


def run_retraction_sweep(
    repo_root: Path,
    *,
    retraction_watch_dois: set[str] | None = None,
    http_get: Callable[..., Any] | None = None,
    now: datetime | None = None,
) -> RetractionSweepReport:
    """
    Zero-LLM sweep checking vault node DOIs against Crossref / Retraction Watch.

    Confirmed retractions trigger append-only downgrade to `T4_FALSIFIED` and cascade
    any dependent `T1_CANONICAL` nodes (`DEPENDS_ON`) to `T3_SPECULATIVE` to preserve
    the Section 8.2 Tier Firewall invariant.
    """
    root = Path(repo_root)
    ts = now or datetime.now(timezone.utc)
    today_date = ts.date()
    today_str = ts.strftime("%Y-%m-%d")

    allow_path = root / "config" / "license_allowlist.yaml"
    allowed_licenses = load_allowed_licenses(allow_path) if allow_path.exists() else None

    ore_to_doi: dict[str, str] = {}
    for src_row in _iter_ndjson_dir(root / "ledger" / "sources"):
        ore_sha = str(src_row.get("ore_sha256") or "").strip().lower()
        norm = normalize_doi(src_row.get("canonical_doi"))
        if ore_sha and norm:
            ore_to_doi[ore_sha] = norm

    vault_dir = root / "vault"
    if not vault_dir.exists():
        return RetractionSweepReport(0, 0, (), (), (), ())

    node_paths: dict[str, Path] = {}
    node_tiers: dict[str, str] = {}
    doi_cache: dict[str, RetractionStatus] = {}
    retracted_nodes: list[str] = []
    cascaded_demotions: list[str] = []
    lineage_records: list[dict[str, Any]] = []
    errors: list[str] = []
    nodes_scanned = 0

    for md_path in sorted(vault_dir.rglob("*.md")):
        nodes_scanned += 1
        raw = normalize_newlines(md_path.read_text(encoding="utf-8"))
        try:
            fm, _ = split_frontmatter(raw)
        except Exception as exc:
            errors.append(f"{md_path}: {exc}")
            continue
        ral_id = str(fm["ral_id"])
        node_paths[ral_id] = md_path
        node_tiers[ral_id] = str(fm["epistemic_tier"])

        retracted_hashes: set[str] = set()
        reasons: list[str] = []
        for prov in fm.get("provenance") or []:
            if prov.get("retracted"):
                continue
            doi = _extract_doi_from_provenance(prov, ore_to_doi)
            if not doi:
                continue
            if doi not in doi_cache:
                try:
                    doi_cache[doi] = check_crossref_status(
                        doi,
                        retraction_watch_dois=retraction_watch_dois,
                        http_get=http_get,
                    )
                except Exception as exc:
                    errors.append(f"DOI check failed for {doi}: {type(exc).__name__}: {exc}")
                    continue
            status = doi_cache[doi]
            if status.retracted:
                retracted_hashes.add(str(prov.get("ore_sha256") or "").lower())
                reasons.append(status.reason or f"DOI {doi} retracted")

        if retracted_hashes:
            re_eval_t4 = (today_date + timedelta(days=3650)).isoformat()
            reason_summary = "; ".join(sorted(set(reasons)))
            lineage = _append_only_tier_downgrade(
                md_path,
                new_tier="T4_FALSIFIED",
                re_evaluate_after=re_eval_t4,
                reason="RETRACTION_FALSIFIED",
                notice_bullet=f"- **RETRACTED ({today_str}):** Primary source retracted ({reason_summary}).",
                retracted_ore_hashes=retracted_hashes,
                allowed_licenses=allowed_licenses,
                ts=ts,
            )
            node_tiers[ral_id] = "T4_FALSIFIED"
            retracted_nodes.append(ral_id)
            if lineage is not None:
                lineage_records.append(lineage)

    # Cascade Tier Firewall protection across `DEPENDS_ON` reverse edges.
    if retracted_nodes:
        rev_deps: dict[str, list[str]] = defaultdict(list)
        for edge in _iter_ndjson_dir(root / "graph" / "edges"):
            if edge.get("relation") == "DEPENDS_ON":
                src = str(edge.get("src") or "")
                dst = str(edge.get("dst") or "")
                if src and dst:
                    rev_deps[dst].append(src)

        queue: deque[str] = deque(retracted_nodes)
        visited: set[str] = set(retracted_nodes)
        re_eval_t3 = (today_date + timedelta(days=180)).isoformat()
        while queue:
            target_id = queue.popleft()
            for dependent_id in sorted(rev_deps.get(target_id, ())):
                if dependent_id in visited:
                    continue
                if node_tiers.get(dependent_id) == "T1_CANONICAL" and dependent_id in node_paths:
                    lineage = _append_only_tier_downgrade(
                        node_paths[dependent_id],
                        new_tier="T3_SPECULATIVE",
                        re_evaluate_after=re_eval_t3,
                        reason="UPSTREAM_RETRACTION",
                        notice_bullet=(
                            f"- **UPSTREAM_RETRACTION ({today_str}):** Prerequisite `{target_id}` "
                            "was retracted or demoted; downgraded to `T3_SPECULATIVE`."
                        ),
                        retracted_ore_hashes=None,
                        allowed_licenses=allowed_licenses,
                        ts=ts,
                    )
                    node_tiers[dependent_id] = "T3_SPECULATIVE"
                    cascaded_demotions.append(dependent_id)
                    if lineage is not None:
                        lineage_records.append(lineage)
                    visited.add(dependent_id)
                    queue.append(dependent_id)

    if lineage_records:
        lineage_path = root / "graph" / "lineage" / f"{ts:%Y-%m}.ndjson"
        lineage_path.parent.mkdir(parents=True, exist_ok=True)
        with lineage_path.open("a", encoding="utf-8") as fh:
            for rec in lineage_records:
                fh.write(json.dumps(rec, sort_keys=False) + "\n")

    return RetractionSweepReport(
        nodes_scanned=nodes_scanned,
        dois_checked=len(doi_cache),
        retracted_nodes=tuple(retracted_nodes),
        cascaded_t1_demotions=tuple(cascaded_demotions),
        lineage_records=tuple(lineage_records),
        errors=tuple(errors),
    )


def run_kernel_regression(
    repo_root: Path,
    *,
    sandbox: OracleSandbox | None = None,
    registry: OraclePluginRegistry | None = None,
    max_kernels: int | None = None,
) -> KernelRegressionReport:
    """
    Re-run stored Executable Verification Kernels inside the air-gapped Docker sandbox
    with full AST anti-cheating lint enforcement (Zero-LLM).
    """
    root = Path(repo_root)
    vault_dir = root / "vault"
    if not vault_dir.exists():
        return KernelRegressionReport(0, 0, 0, ())

    sbx = sandbox or OracleSandbox()
    reg = registry or get_default_registry(sandbox=sbx)

    checked = 0
    passed = 0
    failed = 0
    failures: list[dict[str, str]] = []

    for md_path in sorted(vault_dir.rglob("*.md")):
        if max_kernels is not None and checked >= int(max_kernels):
            break
        raw = normalize_newlines(md_path.read_text(encoding="utf-8"))
        try:
            fm, body = split_frontmatter(raw)
        except Exception as exc:
            checked += 1
            failed += 1
            failures.append({"path": str(md_path), "error": f"FRONTMATTER_ERROR: {exc}"})
            continue

        ral_id = str(fm.get("ral_id") or md_path.stem)
        plugin_name = str((fm.get("verification") or {}).get("oracle_plugin") or "sympy_numeric")
        kernel_code = extract_verification_kernel(body)
        if kernel_code is None:
            checked += 1
            failed += 1
            failures.append({"ral_id": ral_id, "plugin": plugin_name, "error": "MISSING_KERNEL"})
            continue

        checked += 1
        try:
            plugin = reg.get(plugin_name)
            res = plugin.execute(kernel_code)
        except KeyError:
            res = sbx.run_script(
                kernel_code,
                plugin_name=plugin_name,
                enforce_ast_lint=True,
                min_assertions=3,
            )

        if res.passed:
            passed += 1
        else:
            failed += 1
            failures.append(
                {
                    "ral_id": ral_id,
                    "plugin": plugin_name,
                    "error": (res.stderr or res.stdout or "SANDBOX_FAIL")[:400],
                }
            )

    return KernelRegressionReport(
        kernels_checked=checked,
        passed=passed,
        failed=failed,
        failures=tuple(failures),
    )


def collect_referenced_ore_hashes(repo_root: Path) -> set[str]:
    """Collect all `ore_sha256` digests referenced by `vault/` nodes or `ledger/sources/`."""
    root = Path(repo_root)
    refs: set[str] = set()

    vault_dir = root / "vault"
    if vault_dir.exists():
        for md_path in sorted(vault_dir.rglob("*.md")):
            try:
                fm, _ = split_frontmatter(md_path.read_text(encoding="utf-8"))
            except Exception:
                continue
            for prov in fm.get("provenance") or []:
                h = str(prov.get("ore_sha256") or "").strip().lower()
                if h:
                    refs.add(h)

    for row in _iter_ndjson_dir(root / "ledger" / "sources"):
        h = str(row.get("ore_sha256") or "").strip().lower()
        if h:
            refs.add(h)

    return refs


def run_drive_orphan_gc(
    repo_root: Path,
    drive_store: DriveOreStore | None,
    *,
    circuit_breaker: str | None = None,
    now: datetime | None = None,
    quarantine_after_days: int = 7,
    purge_after_days: int = 30,
    dry_run: bool = False,
) -> OrphanGCReport | None:
    """
    Execute Drive Orphan GC unless Drive is unavailable or `circuit_breaker == 'DRIVE_AUTH_FAIL'`.
    """
    if drive_store is None or circuit_breaker == "DRIVE_AUTH_FAIL":
        return None
    refs = collect_referenced_ore_hashes(repo_root)
    return drive_store.run_orphan_gc(
        refs,
        now=now,
        quarantine_after_days=quarantine_after_days,
        purge_after_days=purge_after_days,
        dry_run=dry_run,
    )


def append_telemetry_row(
    repo_root: Path,
    row: Mapping[str, Any],
    *,
    now: datetime | None = None,
) -> Path:
    """Append a structured telemetry row to `metrics/telemetry/YYYY-MM.ndjson`."""
    ts = now or datetime.now(timezone.utc)
    tel_path = Path(repo_root) / "metrics" / "telemetry" / f"{ts:%Y-%m}.ndjson"
    tel_path.parent.mkdir(parents=True, exist_ok=True)
    with tel_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(dict(row), sort_keys=False) + "\n")
    return tel_path


def run_zero_llm_maintenance(
    repo_root: Path,
    *,
    drive_store: DriveOreStore | None = None,
    sandbox: OracleSandbox | None = None,
    registry: OraclePluginRegistry | None = None,
    http_get: Callable[..., Any] | None = None,
    retraction_watch_dois: set[str] | None = None,
    circuit_breaker: str | None = None,
    mode: str = "ZERO_LLM_MAINTENANCE",
    do_retraction_sweep: bool = True,
    do_kernel_regression: bool = True,
    do_drive_gc: bool = True,
    max_kernels: int | None = None,
    record_telemetry: bool = True,
    now: datetime | None = None,
) -> MaintenanceReport:
    """
    Execute deterministic Zero-LLM maintenance:
    retraction sweep, sandbox kernel regression, Drive Orphan GC, and `validate_vault()`.
    """
    root = Path(repo_root)
    ts = now or datetime.now(timezone.utc)

    retraction_rep: RetractionSweepReport | None = None
    if do_retraction_sweep and circuit_breaker != "DRIVE_AUTH_FAIL":
        retraction_rep = run_retraction_sweep(
            root,
            retraction_watch_dois=retraction_watch_dois,
            http_get=http_get,
            now=ts,
        )

    kernel_rep: KernelRegressionReport | None = None
    if do_kernel_regression:
        kernel_rep = run_kernel_regression(
            root,
            sandbox=sandbox,
            registry=registry,
            max_kernels=max_kernels,
        )

    gc_rep: OrphanGCReport | None = None
    if do_drive_gc and circuit_breaker != "DRIVE_AUTH_FAIL":
        gc_rep = run_drive_orphan_gc(
            root,
            drive_store,
            circuit_breaker=circuit_breaker,
            now=ts,
        )

    allow_path = root / "config" / "license_allowlist.yaml"
    vault_res = validate_vault(
        root,
        allowlist_path=allow_path if allow_path.exists() else None,
    )

    telemetry_path: Path | None = None
    if record_telemetry:
        telemetry_path = append_telemetry_row(
            root,
            {
                "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                "run_date": ts.strftime("%Y-%m-%d"),
                "mode": mode,
                "circuit_breaker": circuit_breaker,
                "llm_calls_made": 0,
                "vault_ok": vault_res.ok,
                "node_count": vault_res.node_count,
                "edge_count": vault_res.edge_count,
                "retracted_count": len(retraction_rep.retracted_nodes) if retraction_rep else 0,
                "kernels_checked": kernel_rep.kernels_checked if kernel_rep else 0,
                "kernels_failed": kernel_rep.failed if kernel_rep else 0,
                "drive_quarantined": len(gc_rep.quarantined) if gc_rep else 0,
                "drive_purged": len(gc_rep.purged) if gc_rep else 0,
            },
            now=ts,
        )

    overall_ok = bool(vault_res.ok) and (kernel_rep is None or kernel_rep.failed == 0)
    return MaintenanceReport(
        ok=overall_ok,
        mode=mode,
        circuit_breaker=circuit_breaker,
        retraction_sweep=retraction_rep,
        kernel_regression=kernel_rep,
        drive_gc=gc_rep,
        vault_validation_ok=vault_res.ok,
        vault_validation_errors=tuple(vault_res.errors),
        telemetry_path=telemetry_path,
        llm_calls_made=0,
    )


def _persist_router_state(path: Path | None, router: ModelRouter) -> None:
    if path is None:
        return
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(router.routing_table, indent=2) + "\n", encoding="utf-8")


def run_epoch_or_pivot(
    router: ModelRouter,
    llm_epoch_fn: Callable[[ModelRouter], Any],
    *,
    repo_root: Path,
    drive_store: DriveOreStore | None = None,
    sandbox: OracleSandbox | None = None,
    registry: OraclePluginRegistry | None = None,
    http_get: Callable[..., Any] | None = None,
    retraction_watch_dois: set[str] | None = None,
    circuit_breaker: str | None = None,
    router_state_path: Path | None = None,
    now: datetime | None = None,
    record_telemetry: bool = True,
) -> EpochPivotResult:
    """
    Execute an LLM epoch or gracefully pivot to Zero-LLM maintenance (Section 5):
    - If `circuit_breaker == 'DRIVE_AUTH_FAIL'`: skip LLM & Drive Ore ops, run local validation only.
    - If all LLM models are already exhausted or hit HTTP 429 RPD exhaustion mid-epoch:
      persist `router_state.json` and pivot remaining epoch time to `run_zero_llm_maintenance()`.
    """
    if circuit_breaker == "DRIVE_AUTH_FAIL":
        report = run_zero_llm_maintenance(
            repo_root,
            drive_store=None,
            sandbox=sandbox,
            registry=registry,
            circuit_breaker=circuit_breaker,
            mode="CIRCUIT_BREAKER_MAINTENANCE",
            do_retraction_sweep=False,
            do_kernel_regression=True,
            do_drive_gc=False,
            record_telemetry=record_telemetry,
            now=now,
        )
        _persist_router_state(router_state_path, router)
        return EpochPivotResult(
            status="CIRCUIT_BREAKER_MAINTENANCE",
            pivoted_to_maintenance=True,
            pivot_reason="DRIVE_AUTH_FAIL",
            epoch_output=None,
            maintenance_report=report,
            exhausted_models=tuple(router.exhausted_models),
        )

    if router.all_models_exhausted():
        _persist_router_state(router_state_path, router)
        report = run_zero_llm_maintenance(
            repo_root,
            drive_store=drive_store,
            sandbox=sandbox,
            registry=registry,
            http_get=http_get,
            retraction_watch_dois=retraction_watch_dois,
            circuit_breaker=circuit_breaker,
            mode="ZERO_LLM_RPD_PIVOT",
            record_telemetry=record_telemetry,
            now=now,
        )
        return EpochPivotResult(
            status="PIVOTED_TO_MAINTENANCE",
            pivoted_to_maintenance=True,
            pivot_reason="ALL_MODELS_EXHAUSTED_BEFORE_EPOCH",
            epoch_output=None,
            maintenance_report=report,
            exhausted_models=tuple(router.exhausted_models),
        )

    try:
        output = llm_epoch_fn(router)
        _persist_router_state(router_state_path, router)
        return EpochPivotResult(
            status="LLM_EPOCH_COMPLETE",
            pivoted_to_maintenance=False,
            pivot_reason=None,
            epoch_output=output,
            maintenance_report=None,
            exhausted_models=tuple(router.exhausted_models),
        )
    except Exception as exc:
        if not (isinstance(exc, AllModelsExhaustedError) or is_rate_limit_error(exc)):
            raise
        if not isinstance(exc, AllModelsExhaustedError) and is_rate_limit_error(exc):
            for chain in router.routing_table.get("roles", {}).values():
                for mid in chain:
                    router.mark_exhausted(mid)
        _persist_router_state(router_state_path, router)
        report = run_zero_llm_maintenance(
            repo_root,
            drive_store=drive_store,
            sandbox=sandbox,
            registry=registry,
            http_get=http_get,
            retraction_watch_dois=retraction_watch_dois,
            circuit_breaker=circuit_breaker,
            mode="ZERO_LLM_RPD_PIVOT",
            record_telemetry=record_telemetry,
            now=now,
        )
        return EpochPivotResult(
            status="PIVOTED_TO_MAINTENANCE",
            pivoted_to_maintenance=True,
            pivot_reason="HTTP_429_RPD_EXHAUSTED",
            epoch_output=None,
            maintenance_report=report,
            exhausted_models=tuple(router.exhausted_models),
        )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for Zero-LLM maintenance tasks."""
    parser = argparse.ArgumentParser(description="Run RAL Zero-LLM maintenance tasks.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument("--circuit-breaker", default=None)
    parser.add_argument("--skip-retraction-sweep", action="store_true")
    parser.add_argument("--skip-kernel-regression", action="store_true")
    parser.add_argument("--max-kernels", type=int, default=None)
    parser.add_argument("--no-telemetry", action="store_true")
    args = parser.parse_args(argv)

    report = run_zero_llm_maintenance(
        args.repo_root,
        circuit_breaker=args.circuit_breaker,
        do_retraction_sweep=not args.skip_retraction_sweep,
        do_kernel_regression=not args.skip_kernel_regression,
        do_drive_gc=False,
        max_kernels=args.max_kernels,
        record_telemetry=not args.no_telemetry,
    )
    print(
        json.dumps(
            {
                "ok": report.ok,
                "mode": report.mode,
                "vault_validation_ok": report.vault_validation_ok,
                "llm_calls_made": report.llm_calls_made,
                "retraction_sweep": asdict(report.retraction_sweep) if report.retraction_sweep else None,
                "kernel_regression": asdict(report.kernel_regression) if report.kernel_regression else None,
            },
            indent=2,
        )
    )
    return 0 if report.ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

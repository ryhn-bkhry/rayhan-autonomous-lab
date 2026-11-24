"""End-to-end 8-job nightly orchestrator for Job 0, Epochs 1..6, and Job 7 (Sections 1, 5, 6, 11)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping, Sequence

from ci.validate_node import split_frontmatter, validate_node_file
from ci.validate_vault import validate_vault
from ral.canonicalize import normalize_newlines, sha256_bytes
from ral.director import (
    build_nightly_plan,
    load_prereq_debt,
    record_prereq_attempt_failure,
    resolve_prereq_stub,
)
from ral.github_client import (
    CRITICAL_ALERT_LABEL,
    INDEX_RELEASE_TAG,
    BlueGreenSwapResult,
    GitHubClient,
)
from ral.maintenance import (
    append_telemetry_row,
    run_epoch_or_pivot,
    run_kernel_regression,
    run_retraction_sweep,
    run_zero_llm_maintenance,
)
from ral.model_router import (
    BENCHMARK_PATH,
    SUITE_PATH,
    ModelRouter,
    build_routing_table,
    ensure_benchmarked,
)
from ral.pipeline.architect import ArchitectResult, ArchitectStage, find_matching_node
from ral.pipeline.formalizer import FormalizerStage, append_rejected_record
from ral.pipeline.oracle_sandbox import OracleSandbox
from ral.pipeline.oracles.base import OraclePluginRegistry, get_default_registry
from ral.pipeline.scout import ScoutStage
from ral.pipeline.skeptic import SkepticStage, assign_epistemic_tier
from ral.preflight import (
    discover_embedding_model,
    discover_live_generate_models,
    load_policy,
)
from ral.sources import (
    LicenseRejectedError,
    RetractedSourceError,
    SourceRecord,
    load_license_allowlist,
)
from ral.sources.arxiv_tex import fetch_arxiv_source
from ral.sources.crossref_retract import check_crossref_status
from ral.sources.openalex import search_oa_works
from ral.sources.pmc_jats import fetch_pmc_jats
from ral.storage.drive_ore import (
    DRIVE_FOLDER_MIME,
    CanaryResult,
    DriveAuthError,
    DriveOreStore,
    build_drive_service_from_env,
)
from ral.storage.index_builder import build_or_update_index, compress_sqlite_zst
from ral.vault_docs import regenerate_vault_docs


@dataclass(frozen=True)
class PreflightJobResult:
    """Outcome of Job 0 (`preflight_and_plan`)."""

    ok: bool
    run_date: str
    run_id: str
    circuit_breaker: str | None
    drive_canary: dict[str, Any] | None
    embedding_model: dict[str, Any] | None
    epoch_upgraded: bool
    available_epochs: int
    plan_path: Path
    router_state_path: Path
    circuit_breaker_path: Path
    alert_issue_opened: bool
    github_api_calls_used: int


@dataclass(frozen=True)
class EpochJobResult:
    """Outcome of one sequential execution epoch (Jobs 1..6)."""

    epoch_index: int
    mode: str
    target: Any
    status: str
    pivoted_to_maintenance: bool
    pivot_reason: str | None
    sources_processed: int
    nodes_created: tuple[str, ...]
    nodes_merged: tuple[str, ...]
    nodes_upgraded: tuple[str, ...]
    nodes_challenged: tuple[str, ...]
    nodes_verified_tonight: int
    rejected_count: int
    ore_uploaded_count: int
    exhausted_models: tuple[str, ...]
    summary_path: Path


@dataclass(frozen=True)
class DeadMansSwitchResult:
    """Evaluation outcome for Job 7 Dead-Man's Switch triggers."""

    triggered: bool
    reasons: tuple[str, ...]
    consecutive_zero_verified_nights: int
    git_size_mb: float
    alert_issue: dict[str, Any] | None = None


@dataclass(frozen=True)
class ConsolidateJobResult:
    """Outcome of Job 7 (`consolidate_and_watchdog`)."""

    ok: bool
    vault_ok: bool
    vault_errors: tuple[str, ...]
    node_count: int
    edge_count: int
    nodes_verified_tonight: int
    index_db_path: Path
    index_zst_path: Path
    index_zst_sha256: str
    embed_space_consistent: bool
    vector_action: str
    blue_green_swap: BlueGreenSwapResult | None
    docs_summary: dict[str, Any]
    telemetry_path: Path
    dead_mans_switch: DeadMansSwitchResult
    github_api_calls_used: int


@dataclass(frozen=True)
class NightlyDryRunSummary:
    """Aggregated summary of a complete 8-job nightly pipeline run."""

    preflight: PreflightJobResult
    epochs: tuple[EpochJobResult, ...]
    consolidate: ConsolidateJobResult
    artifact_ops_count: int
    git_pushes_count: int
    github_rest_calls_count: int
    estimated_total_github_api_calls: int


class _DryRunMemoryFilesResource:
    """In-memory Drive files() shim used strictly for secret-free local dry runs."""

    def __init__(self) -> None:
        self._items: dict[str, dict[str, Any]] = {
            "dryrun-root": {
                "id": "dryrun-root",
                "name": "RAL_Ore_Warehouse",
                "mimeType": DRIVE_FOLDER_MIME,
                "parents": [],
                "trashed": False,
                "payload": b"",
            }
        }
        self._seq = 0

    def _next_id(self) -> str:
        self._seq += 1
        return f"dryrun-item-{self._seq}"

    def list(self, **kwargs: Any) -> Any:
        q = str(kwargs.get("q") or "")
        items = list(self._items.values())
        matched: list[dict[str, Any]] = []
        for item in items:
            if item.get("trashed"):
                continue
            if "mimeType =" in q and item.get("mimeType") not in q:
                continue
            name = item.get("name", "")
            if "name =" in q and f"name = '{name}'" not in q:
                continue
            parents = item.get("parents") or []
            if "in parents" in q and not any(f"'{p}' in parents" in q for p in parents):
                continue
            matched.append({k: v for k, v in item.items() if k != "payload"})
        return type("_Req", (), {"execute": lambda _self: {"files": matched}})()

    def create(self, **kwargs: Any) -> Any:
        body = dict(kwargs.get("body") or {})
        media = kwargs.get("media_body")
        payload = b""
        if media is not None:
            if hasattr(media, "payload"):
                payload = bytes(media.payload)
            elif hasattr(media, "getbytes"):
                payload = bytes(media.getbytes(0, None))
        fid = self._next_id()
        md5_hex = hashlib.md5(payload).hexdigest() if payload else None
        record = {
            "id": fid,
            "name": body.get("name", fid),
            "mimeType": body.get("mimeType", "application/octet-stream"),
            "parents": list(body.get("parents") or []),
            "md5Checksum": md5_hex,
            "size": str(len(payload)),
            "trashed": False,
            "payload": payload,
        }
        self._items[fid] = record
        return type("_Req", (), {"execute": lambda _self: {k: v for k, v in record.items() if k != "payload"}})()

    def get(self, **kwargs: Any) -> Any:
        fid = str(kwargs.get("fileId"))
        rec = self._items[fid]
        return type("_Req", (), {"execute": lambda _self: {k: v for k, v in rec.items() if k != "payload"}})()

    def get_media(self, **kwargs: Any) -> Any:
        fid = str(kwargs.get("fileId"))
        payload = bytes(self._items[fid]["payload"])
        return type("_Req", (), {"execute": lambda _self: payload})()

    def update(self, **kwargs: Any) -> Any:
        fid = str(kwargs.get("fileId"))
        rec = self._items[fid]
        parents = list(rec.get("parents") or [])
        rem = str(kwargs.get("removeParents") or "")
        add = str(kwargs.get("addParents") or "")
        if rem in parents:
            parents.remove(rem)
        if add and add not in parents:
            parents.append(add)
        rec["parents"] = parents
        return type("_Req", (), {"execute": lambda _self: {"id": fid, "parents": parents}})()

    def delete(self, **kwargs: Any) -> Any:
        fid = str(kwargs.get("fileId"))
        self._items.pop(fid, None)
        return type("_Req", (), {"execute": lambda _self: None})()


class _DryRunDriveService:
    def __init__(self) -> None:
        self._files = _DryRunMemoryFilesResource()

    def files(self) -> _DryRunMemoryFilesResource:
        return self._files


def parse_github_repo(repo_slug: str) -> tuple[str, str]:
    """Split `owner/repo` string into `(owner, repo)`."""
    cleaned = repo_slug.strip().strip("/")
    if "/" not in cleaned:
        raise ValueError(f"Expected owner/repo slug, got {repo_slug!r}")
    owner, repo = cleaned.split("/", 1)
    if not owner or not repo:
        raise ValueError(f"Invalid owner/repo slug: {repo_slug!r}")
    return owner, repo


def build_gemini_client_from_env(environ: Mapping[str, str] | None = None) -> Any:
    """Construct a `google.genai.Client` from `RAL_GEMINI_FREE_KEY`."""
    env = environ if environ is not None else os.environ
    api_key = (env.get("RAL_GEMINI_FREE_KEY") or "").strip()
    if not api_key:
        raise RuntimeError("Missing RAL_GEMINI_FREE_KEY environment variable")
    from google import genai

    return genai.Client(api_key=api_key)


def _atomic_write_json(path: Path, data: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".tmp_{target.name}_", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(dict(data), indent=2, sort_keys=True) + "\n")
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink()
        raise


def _has_drive_env(environ: Mapping[str, str] | None = None) -> bool:
    env = environ if environ is not None else os.environ
    root_id = (env.get("RAL_GDRIVE_ROOT_FOLDER_ID") or "").strip()
    if not root_id:
        return False
    if (env.get("RAL_GDRIVE_SA_JSON") or "").strip():
        return True
    return bool(
        (env.get("RAL_GDRIVE_CLIENT_ID") or "").strip()
        and (env.get("RAL_GDRIVE_CLIENT_SECRET") or "").strip()
        and (env.get("RAL_GDRIVE_REFRESH_TOKEN") or "").strip()
    )


def _resolve_drive_store(
    drive_store: DriveOreStore | None,
    *,
    dry_run: bool,
) -> DriveOreStore:
    if drive_store is not None:
        return drive_store
    if _has_drive_env():
        service, root_id = build_drive_service_from_env()
        return DriveOreStore(service, root_id)
    if dry_run:
        return DriveOreStore(_DryRunDriveService(), "dryrun-root")
    service, root_id = build_drive_service_from_env()
    return DriveOreStore(service, root_id)


def _find_node_path_by_id(repo_root: Path, ral_id: str) -> Path | None:
    vault_dir = Path(repo_root) / "vault"
    if not vault_dir.exists():
        return None
    for md_path in sorted(vault_dir.rglob("*.md")):
        if md_path.stem == ral_id:
            return md_path
        try:
            fm, _ = split_frontmatter(normalize_newlines(md_path.read_text(encoding="utf-8")))
            if fm.get("ral_id") == ral_id:
                return md_path
        except Exception:
            continue
    return None


def _default_fetch_sources_for_epoch(
    mode: str,
    target: Any,
    repo_root: Path,
    *,
    http_get: Callable[..., Any] | None = None,
    max_sources: int = 3,
) -> list[SourceRecord]:
    """Fetch Open-Access candidate primary sources for an epoch target."""
    allow_path = Path(repo_root) / "config" / "license_allowlist.yaml"
    allowed = load_license_allowlist(allow_path) if allow_path.exists() else load_license_allowlist()

    if isinstance(target, Mapping):
        query_terms = [
            str(target.get("domain_a") or "").replace(".", " "),
            str(target.get("domain_b") or "").replace(".", " "),
        ]
        query = " ".join(t for t in query_terms if t).strip() or "mathematical physics isomorphism"
    elif isinstance(target, str) and target.strip():
        query = target.replace(".", " ").replace("-", " ")
    else:
        query = "mathematical theorem physics statistical mechanics"

    try:
        works = search_oa_works(
            query,
            per_page=max(1, int(max_sources) * 2),
            allowed_licenses=allowed,
            http_get=http_get,
        )
    except Exception:
        return []

    records: list[SourceRecord] = []
    for work in works:
        if len(records) >= max_sources:
            break
        try:
            if work.canonical_doi:
                ret_status = check_crossref_status(work.canonical_doi, http_get=http_get)
                if ret_status.retracted:
                    continue
            if work.arxiv_id:
                rec = fetch_arxiv_source(
                    work.arxiv_id,
                    metadata={
                        "title": work.title,
                        "pub_year": work.pub_year,
                        "peer_reviewed": work.peer_reviewed,
                        "canonical_doi": work.canonical_doi,
                        "citation_count": work.cited_by_count,
                        "source_license": work.source_license,
                    },
                    http_get=http_get,
                    allowed_licenses=allowed,
                )
                records.append(rec)
            elif work.pmcid:
                rec = fetch_pmc_jats(
                    work.pmcid,
                    http_get=http_get,
                    allowed_licenses=allowed,
                )
                records.append(rec)
        except (LicenseRejectedError, RetractedSourceError, Exception):
            continue

    _ = mode  # Mode-specific ranking can refine queries in future extensions.
    return records


def run_preflight_job(
    repo_root: Path,
    *,
    handoff_dir: Path,
    cache_dir: Path,
    run_date_iso: str,
    current_night: int = 0,
    available_epochs: int = 6,
    run_id: str = "local",
    dry_run: bool = False,
    simulate_drive_fail: bool = False,
    gemini_client: Any | None = None,
    drive_store: DriveOreStore | None = None,
    github_client: GitHubClient | None = None,
    now: datetime | None = None,
) -> PreflightJobResult:
    """
    Job 0 (`preflight_and_plan`):
    1. Pull latest SQLite index via `ral-index-latest.json` (Blue-Green pointer).
    2. Run 16-byte Google Drive canary (if fail/simulated: trip `DRIVE_AUTH_FAIL`, open alert Issue, K=0).
    3. Discover live Gemini/Gemma models + dynamic 768-d embedding model.
    4. Run 20-problem calibration ONLY for newly discovered `model_id`s.
    5. Run `director.build_nightly_plan()` and emit ephemeral handoff files in `handoff_dir`.
    """
    root = Path(repo_root)
    handoff = Path(handoff_dir)
    cache = Path(cache_dir)
    handoff.mkdir(parents=True, exist_ok=True)
    cache.mkdir(parents=True, exist_ok=True)

    calls_before = github_client.api_call_count if github_client is not None else 0
    index_db_path = cache / "ral-index.sqlite"
    if github_client is not None:
        try:
            dl = github_client.download_latest_index(index_db_path, cache_dir=cache)
            if dl is None:
                index_db_path = cache / "ral-index.sqlite"
        except Exception:
            pass

    circuit_breaker: str | None = None
    drive_canary_dict: dict[str, Any] | None = None
    alert_opened = False

    if simulate_drive_fail:
        canary = CanaryResult(
            ok=False,
            canary_name="_ral_simulated_canary.bin",
            canary_size_bytes=16,
            local_md5="0" * 32,
            remote_md5=None,
            cleaned_up=False,
            error="SIMULATED_DRIVE_CANARY_FAILURE",
        )
        drive_canary_dict = asdict(canary)
        circuit_breaker = "DRIVE_AUTH_FAIL"
    else:
        resolved_store: DriveOreStore | None = drive_store
        if resolved_store is None and _has_drive_env():
            try:
                svc, root_id = build_drive_service_from_env()
                resolved_store = DriveOreStore(svc, root_id)
            except DriveAuthError as exc:
                canary = CanaryResult(
                    ok=False,
                    canary_name="_ral_auth_check.bin",
                    canary_size_bytes=16,
                    local_md5="0" * 32,
                    remote_md5=None,
                    cleaned_up=False,
                    error=str(exc),
                )
                drive_canary_dict = asdict(canary)
                circuit_breaker = "DRIVE_AUTH_FAIL"

        if resolved_store is not None:
            canary = resolved_store.run_preflight_canary()
            drive_canary_dict = asdict(canary)
            if not canary.ok:
                circuit_breaker = "DRIVE_AUTH_FAIL"
        elif circuit_breaker is None:
            if dry_run:
                drive_canary_dict = {
                    "ok": True,
                    "canary_name": "_ral_dryrun_canary.bin",
                    "canary_size_bytes": 16,
                    "local_md5": "dryrun",
                    "remote_md5": "dryrun",
                    "cleaned_up": True,
                    "error": None,
                }
            else:
                circuit_breaker = "DRIVE_AUTH_FAIL"
                drive_canary_dict = {
                    "ok": False,
                    "canary_name": "_ral_missing_creds.bin",
                    "canary_size_bytes": 16,
                    "local_md5": "0" * 32,
                    "remote_md5": None,
                    "cleaned_up": False,
                    "error": "Missing Google Drive credentials in environment",
                }

    if circuit_breaker == "DRIVE_AUTH_FAIL" and github_client is not None:
        try:
            github_client.upsert_alert_issue(
                title=f"RAL Critical Alert: Google Drive canary failed ({run_date_iso})",
                body=(
                    f"Job 0 preflight tripped `circuit_breaker=DRIVE_AUTH_FAIL` on {run_date_iso} "
                    f"(run_id={run_id}). LLM epochs halted (`K_llm=0`).\n\n"
                    f"Canary detail: `{json.dumps(drive_canary_dict)}`"
                ),
                label=CRITICAL_ALERT_LABEL,
            )
            alert_opened = True
        except Exception:
            pass

    policy_path = root / "config" / "model_policy.yaml"
    policy = load_policy(policy_path if policy_path.exists() else Path("config/model_policy.yaml"))
    bench_path = root / "metrics" / "model_benchmarks.json"
    suite_path = root / "config" / "calibration_suite.json"
    resolved_suite = suite_path if suite_path.exists() else SUITE_PATH

    client = gemini_client
    if client is None and (os.environ.get("RAL_GEMINI_FREE_KEY") or "").strip():
        client = build_gemini_client_from_env()

    live_statuses: list[dict[str, Any]] = []
    embedding_info: dict[str, Any] | None = None
    epoch_upgraded = False

    if client is not None and circuit_breaker != "DRIVE_AUTH_FAIL":
        statuses = discover_live_generate_models(client, policy)
        live_statuses = [s.to_dict() for s in statuses]
        embedding_info = discover_embedding_model(client, policy)
        db, epoch_upgraded = ensure_benchmarked(
            client,
            statuses,
            policy,
            benchmark_path=bench_path,
            suite_path=resolved_suite,
        )
        routing_table = build_routing_table(statuses, db, policy)
    else:
        routing_table = {"status": "ZERO_HEALTHY_MODELS", "roles": {}, "exhausted_models": []}

    if circuit_breaker == "DRIVE_AUTH_FAIL" or routing_table.get("status") != "OK":
        effective_epochs = 0
    else:
        effective_epochs = max(0, min(6, int(available_epochs)))

    plan = build_nightly_plan(
        root,
        run_date_iso=run_date_iso,
        current_night=current_night,
        available_epochs=effective_epochs,
        epoch_upgraded=epoch_upgraded,
        circuit_breaker=circuit_breaker,
        index_db_path=index_db_path if index_db_path.is_file() else None,
        execute_splits=(not dry_run),
    )

    ts_iso = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
    cb_payload = {
        "schema_version": "2.1",
        "run_date": run_date_iso,
        "run_id": run_id,
        "dry_run": bool(dry_run),
        "circuit_breaker": circuit_breaker,
        "drive_canary": drive_canary_dict,
        "embedding_model": embedding_info,
        "live_model_count": len(live_statuses),
        "epoch_upgraded": bool(epoch_upgraded),
        "available_epochs": effective_epochs,
        "alert_issue_opened_in_preflight": alert_opened,
        "updated_at": ts_iso,
    }

    plan_path = handoff / "plan.json"
    router_path = handoff / "router_state.json"
    cb_path = handoff / "circuit_breaker.json"
    _atomic_write_json(plan_path, plan)
    _atomic_write_json(router_path, routing_table)
    _atomic_write_json(cb_path, cb_payload)

    api_calls = (
        github_client.api_call_count - calls_before if github_client is not None else 0
    )
    return PreflightJobResult(
        ok=True,
        run_date=run_date_iso,
        run_id=run_id,
        circuit_breaker=circuit_breaker,
        drive_canary=drive_canary_dict,
        embedding_model=embedding_info,
        epoch_upgraded=epoch_upgraded,
        available_epochs=effective_epochs,
        plan_path=plan_path,
        router_state_path=router_path,
        circuit_breaker_path=cb_path,
        alert_issue_opened=alert_opened,
        github_api_calls_used=api_calls,
    )


def execute_epoch_pipeline(
    router: ModelRouter,
    *,
    repo_root: Path,
    mode: str,
    target: Any,
    current_night: int,
    ore_cache_dir: Path,
    gemini_client: Any | None,
    drive_store: DriveOreStore | None,
    sandbox: OracleSandbox | None = None,
    registry: OraclePluginRegistry | None = None,
    source_fetcher: Callable[[str, Any, Path], Sequence[SourceRecord]] | None = None,
    http_get: Callable[..., Any] | None = None,
    retraction_watch_dois: set[str] | None = None,
    max_sources: int = 3,
    soft_stop_seconds: float = 2880.0,
    dry_run: bool = False,
    now: datetime | None = None,
    monotonic_fn: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """Execute one LLM pipeline epoch across Stages 1-5 with Verify-Before-Archive 2PC."""
    root = Path(repo_root)
    ore_cache = Path(ore_cache_dir)
    ore_cache.mkdir(parents=True, exist_ok=True)
    ts = now or datetime.now(timezone.utc)
    mono = monotonic_fn or time.monotonic
    t_start = mono()

    sbx = sandbox or OracleSandbox()
    reg = registry or get_default_registry(sandbox=sbx)

    created: list[str] = []
    merged: list[str] = []
    upgraded: list[str] = []
    challenged: list[str] = []
    verified_tonight = 0
    rejected_count = 0
    ore_uploaded_count = 0
    sources_processed = 0

    if mode == "audit":
        run_retraction_sweep(
            root,
            retraction_watch_dois=retraction_watch_dois,
            http_get=http_get,
            now=ts,
        )
        run_kernel_regression(root, sandbox=sbx, registry=reg)

    if gemini_client is None:
        return {
            "sources_processed": 0,
            "nodes_created": created,
            "nodes_merged": merged,
            "nodes_upgraded": upgraded,
            "nodes_challenged": challenged,
            "nodes_verified_tonight": verified_tonight,
            "rejected_count": rejected_count,
            "ore_uploaded_count": ore_uploaded_count,
        }

    fetcher = source_fetcher or (
        lambda m, tgt, r: _default_fetch_sources_for_epoch(
            m, tgt, r, http_get=http_get, max_sources=max_sources
        )
    )
    candidate_sources = list(fetcher(mode, target, root))[: max(0, int(max_sources))]

    allow_path = root / "config" / "license_allowlist.yaml"
    oracle_cfg = root / "config" / "oracle_plugins.yaml"
    scout = ScoutStage(
        gemini_client,
        router,
        allowlist_path=allow_path if allow_path.exists() else None,
    )
    formalizer = FormalizerStage(
        gemini_client,
        router,
        reg,
        oracle_config_path=oracle_cfg if oracle_cfg.exists() else Path("config/oracle_plugins.yaml"),
    )
    skeptic = SkepticStage(gemini_client, router, sandbox=sbx)

    resolved_drive: DriveOreStore | None = drive_store
    architect: ArchitectStage | None = None

    for source in candidate_sources:
        if mono() - t_start >= soft_stop_seconds:
            break
        sources_processed += 1

        ore_sha = sha256_bytes(source.raw_bytes)
        staged_ore = ore_cache / f"{ore_sha}.{source.raw_ext}"
        staged_ore.write_bytes(source.raw_bytes)

        try:
            try:
                dedup, candidates = scout.extract_candidates(source, repo_root=root)
            except (LicenseRejectedError, RetractedSourceError):
                rejected_count += 1
                continue

            if dedup.action in ("SKIP_DUPLICATE", "SKIP_REJECTED") or not candidates:
                continue

            for cand in candidates:
                if mono() - t_start >= soft_stop_seconds:
                    break

                draft, oracle_res = formalizer.formalize_and_verify(
                    cand,
                    source,
                    max_repairs=2,
                    repo_root=root,
                    current_epoch=router.capability_epoch,
                )
                if draft is None or not oracle_res.passed:
                    rejected_count += 1
                    continue

                surviving, verdict = skeptic.run_adversarial_loop(
                    draft,
                    source,
                    formalizer,
                    quote_text=cand.quote_text,
                    max_tightening_loops=1,
                )
                if surviving is None:
                    rejected_count += 1
                    append_rejected_record(
                        root,
                        {
                            "source_sha256": ore_sha,
                            "canonical_doi": source.canonical_doi,
                            "arxiv_base_id": source.arxiv_base_id,
                            "stage_failed": "STAGE4_SKEPTIC",
                            "reason": "SKEPTIC_COUNTEREXAMPLE_UNRESOLVED",
                            "retry_on_model_upgrade": True,
                            "model_capability_epoch": router.capability_epoch,
                            "rejected_at": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
                        },
                        now=ts,
                    )
                    continue

                tier, re_eval = assign_epistemic_tier(
                    oracle_plugin=surviving.oracle_plugin,
                    oracle_passed=True,
                    skeptic_survived=True,
                    peer_reviewed=source.peer_reviewed,
                    retracted=source.retracted,
                    pub_year=source.pub_year,
                    citation_count=source.citation_count,
                    source_format=source.source_format,
                    today=ts.date(),
                )

                if architect is None:
                    resolved_drive = _resolve_drive_store(resolved_drive, dry_run=dry_run)
                    architect = ArchitectStage(root, resolved_drive, sandbox=sbx)

                # Stage 5: Check Theory-Upgrade target, Hybrid Dedup Merge, or New 2PC Commit.
                target_path: Path | None = None
                if cand.dedup_action == "THEORY_UPGRADE":
                    for existing_id in cand.existing_ral_ids:
                        found_p = _find_node_path_by_id(root, existing_id)
                        if found_p is not None:
                            target_path = found_p
                            break
                if target_path is None:
                    match = find_matching_node(root, surviving.formal_signature)
                    if match is not None:
                        target_path = match[0]

                arch_res: ArchitectResult
                if target_path is not None and cand.dedup_action == "THEORY_UPGRADE":
                    arch_res = architect.apply_theory_upgrade(
                        target_path,
                        surviving,
                        tier,
                        re_evaluate_after=re_eval,
                        skeptic_model=verdict.skeptic_model,
                        source=source,
                        quote_sha256=cand.quote_sha256,
                        quote_char_len=cand.quote_char_len,
                        source_locator=cand.source_locator,
                        reason="PEER_REVIEWED_OR_VERSION_UPGRADE",
                        retraction_confirmed=source.retracted,
                        now=ts,
                    )
                elif target_path is not None:
                    arch_res = architect.merge_into_existing_node(
                        target_path,
                        source=source,
                        quote_sha256=cand.quote_sha256,
                        quote_char_len=cand.quote_char_len,
                        source_locator=cand.source_locator,
                        anchor_summary=cand.claim_summary or surviving.provenance_summary,
                        now=ts,
                    )
                else:
                    arch_res = architect.commit_new_node(
                        surviving,
                        epistemic_tier=tier,
                        re_evaluate_after=re_eval,
                        skeptic_model=verdict.skeptic_model,
                        source=source,
                        quote_sha256=cand.quote_sha256,
                        quote_char_len=cand.quote_char_len,
                        source_locator=cand.source_locator,
                        now=ts,
                    )

                if arch_res.ore_upload is not None and not arch_res.ore_upload.deduplicated:
                    ore_uploaded_count += 1

                if arch_res.action == "CREATED":
                    created.append(arch_res.ral_id)
                    if tier in ("T1_CANONICAL", "T2_ESTABLISHED"):
                        verified_tonight += 1
                elif arch_res.action == "MERGED_PROVENANCE":
                    merged.append(arch_res.ral_id)
                elif arch_res.action == "SUPERSEDED":
                    upgraded.append(arch_res.ral_id)
                    if tier in ("T1_CANONICAL", "T2_ESTABLISHED"):
                        verified_tonight += 1
                elif arch_res.action == "CHALLENGED_ONLY":
                    challenged.append(arch_res.ral_id)
        finally:
            if staged_ore.exists():
                staged_ore.unlink()

    if mode == "exploit" and isinstance(target, str) and target:
        stubs = load_prereq_debt(root / "ledger" / "prereq_debt.ndjson").values()
        for stub in stubs:
            if stub.domain == target and stub.status == "OPEN" and current_night >= stub.cooldown_until_night:
                if created or upgraded:
                    resolve_prereq_stub(root, stub.stub_id, current_night)
                elif sources_processed > 0:
                    record_prereq_attempt_failure(root, stub.stub_id, target, current_night)

    return {
        "sources_processed": sources_processed,
        "nodes_created": created,
        "nodes_merged": merged,
        "nodes_upgraded": upgraded,
        "nodes_challenged": challenged,
        "nodes_verified_tonight": verified_tonight,
        "rejected_count": rejected_count,
        "ore_uploaded_count": ore_uploaded_count,
    }


def run_epoch_job(
    repo_root: Path,
    *,
    handoff_dir: Path,
    epoch_index: int,
    cache_dir: Path | None = None,
    ore_cache_dir: Path | None = None,
    soft_stop_seconds: float = 2880.0,
    max_sources: int = 3,
    dry_run: bool = False,
    gemini_client: Any | None = None,
    drive_store: DriveOreStore | None = None,
    sandbox: OracleSandbox | None = None,
    registry: OraclePluginRegistry | None = None,
    source_fetcher: Callable[[str, Any, Path], Sequence[SourceRecord]] | None = None,
    http_get: Callable[..., Any] | None = None,
    retraction_watch_dois: set[str] | None = None,
    now: datetime | None = None,
    monotonic_fn: Callable[[], float] | None = None,
) -> EpochJobResult:
    """
    Execute one scheduled epoch (`1..6`) with circuit-breaker checks and HTTP 429 RPD pivot.
    """
    root = Path(repo_root)
    handoff = Path(handoff_dir)
    _ = cache_dir
    ore_cache = Path(ore_cache_dir) if ore_cache_dir is not None else Path(tempfile.gettempdir()) / "ore_cache"
    ts = now or datetime.now(timezone.utc)

    plan = json.loads((handoff / "plan.json").read_text(encoding="utf-8"))
    router_path = handoff / "router_state.json"
    router_state = json.loads(router_path.read_text(encoding="utf-8"))
    cb_state = json.loads((handoff / "circuit_breaker.json").read_text(encoding="utf-8"))

    circuit_breaker = cb_state.get("circuit_breaker")
    current_night = int(plan.get("current_night", 0))
    schedule = list(plan.get("epoch_schedule") or [])
    slot = next((s for s in schedule if int(s.get("epoch_index", -1)) == int(epoch_index)), None)

    mode = str(slot["mode"]) if slot is not None else "maintenance"
    target = slot.get("target") if slot is not None else None
    router = ModelRouter(router_state)

    client = gemini_client
    if (
        client is None
        and circuit_breaker != "DRIVE_AUTH_FAIL"
        and slot is not None
        and not router.all_models_exhausted()
        and (os.environ.get("RAL_GEMINI_FREE_KEY") or "").strip()
    ):
        client = build_gemini_client_from_env()

    resolved_drive = drive_store
    if resolved_drive is None and circuit_breaker != "DRIVE_AUTH_FAIL" and _has_drive_env():
        try:
            svc, root_id = build_drive_service_from_env()
            resolved_drive = DriveOreStore(svc, root_id)
        except DriveAuthError:
            circuit_breaker = "DRIVE_AUTH_FAIL"

    if circuit_breaker == "DRIVE_AUTH_FAIL":
        pivot_res = run_epoch_or_pivot(
            router,
            lambda _r: None,
            repo_root=root,
            drive_store=None,
            sandbox=sandbox,
            registry=registry,
            http_get=http_get,
            retraction_watch_dois=retraction_watch_dois,
            circuit_breaker="DRIVE_AUTH_FAIL",
            router_state_path=router_path,
            now=ts,
            record_telemetry=False,
        )
    elif slot is None:
        maint_report = run_zero_llm_maintenance(
            root,
            drive_store=resolved_drive if (ts.weekday() == 6 and not dry_run) else None,
            sandbox=sandbox,
            registry=registry,
            http_get=http_get,
            retraction_watch_dois=retraction_watch_dois,
            circuit_breaker=circuit_breaker,
            mode="ZERO_LLM_UNALLOCATED_SLOT",
            do_drive_gc=(ts.weekday() == 6 and not dry_run and resolved_drive is not None),
            record_telemetry=False,
            now=ts,
        )
        _atomic_write_json(router_path, router.routing_table)
        pivot_res = type(
            "_SlotPivot",
            (),
            {
                "status": "PIVOTED_TO_MAINTENANCE",
                "pivoted_to_maintenance": True,
                "pivot_reason": "NO_LLM_SLOT_ALLOCATED",
                "epoch_output": None,
                "maintenance_report": maint_report,
                "exhausted_models": tuple(router.exhausted_models),
            },
        )()
    else:
        pivot_res = run_epoch_or_pivot(
            router,
            lambda active_router: execute_epoch_pipeline(
                active_router,
                repo_root=root,
                mode=mode,
                target=target,
                current_night=current_night,
                ore_cache_dir=ore_cache,
                gemini_client=client,
                drive_store=resolved_drive,
                sandbox=sandbox,
                registry=registry,
                source_fetcher=source_fetcher,
                http_get=http_get,
                retraction_watch_dois=retraction_watch_dois,
                max_sources=max_sources,
                soft_stop_seconds=soft_stop_seconds,
                dry_run=dry_run,
                now=ts,
                monotonic_fn=monotonic_fn,
            ),
            repo_root=root,
            drive_store=resolved_drive if (ts.weekday() == 6 and not dry_run) else None,
            sandbox=sandbox,
            registry=registry,
            http_get=http_get,
            retraction_watch_dois=retraction_watch_dois,
            circuit_breaker=circuit_breaker,
            router_state_path=router_path,
            now=ts,
            record_telemetry=False,
        )

    out = dict(pivot_res.epoch_output or {})
    summary_payload = {
        "epoch_index": int(epoch_index),
        "mode": mode,
        "target": target,
        "status": pivot_res.status,
        "pivoted_to_maintenance": bool(pivot_res.pivoted_to_maintenance),
        "pivot_reason": pivot_res.pivot_reason,
        "sources_processed": int(out.get("sources_processed", 0)),
        "nodes_created": list(out.get("nodes_created", [])),
        "nodes_merged": list(out.get("nodes_merged", [])),
        "nodes_upgraded": list(out.get("nodes_upgraded", [])),
        "nodes_challenged": list(out.get("nodes_challenged", [])),
        "nodes_verified_tonight": int(out.get("nodes_verified_tonight", 0)),
        "rejected_count": int(out.get("rejected_count", 0)),
        "ore_uploaded_count": int(out.get("ore_uploaded_count", 0)),
        "exhausted_models": list(pivot_res.exhausted_models),
        "completed_at": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
    }
    summary_path = handoff / f"epoch_{int(epoch_index)}_summary.json"
    _atomic_write_json(summary_path, summary_payload)

    return EpochJobResult(
        epoch_index=int(epoch_index),
        mode=mode,
        target=target,
        status=str(pivot_res.status),
        pivoted_to_maintenance=bool(pivot_res.pivoted_to_maintenance),
        pivot_reason=pivot_res.pivot_reason,
        sources_processed=summary_payload["sources_processed"],
        nodes_created=tuple(summary_payload["nodes_created"]),
        nodes_merged=tuple(summary_payload["nodes_merged"]),
        nodes_upgraded=tuple(summary_payload["nodes_upgraded"]),
        nodes_challenged=tuple(summary_payload["nodes_challenged"]),
        nodes_verified_tonight=summary_payload["nodes_verified_tonight"],
        rejected_count=summary_payload["rejected_count"],
        ore_uploaded_count=summary_payload["ore_uploaded_count"],
        exhausted_models=tuple(summary_payload["exhausted_models"]),
        summary_path=summary_path,
    )


def count_consecutive_zero_verified_nights(
    repo_root: Path,
    *,
    current_run_date: str | None = None,
    current_verified_count: int | None = None,
) -> int:
    """
    Count consecutive trailing nights with 0 verified nodes from `metrics/telemetry/*.ndjson`.
    """
    tel_dir = Path(repo_root) / "metrics" / "telemetry"
    by_date: dict[str, int] = {}
    if tel_dir.exists():
        for path in sorted(tel_dir.glob("*.ndjson")):
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    row = json.loads(stripped)
                except json.JSONDecodeError:
                    continue
                if not isinstance(row, Mapping):
                    continue
                is_summary = (
                    row.get("record_type") == "nightly_summary"
                    or "nodes_verified_tonight" in row
                    or "verified_nodes" in row
                )
                if not is_summary:
                    continue
                r_date = str(row.get("run_date") or "")[:10]
                if not r_date:
                    continue
                cnt = int(
                    row.get("nodes_verified_tonight")
                    if "nodes_verified_tonight" in row
                    else row.get("verified_nodes", 0)
                )
                by_date[r_date] = max(by_date.get(r_date, 0), cnt)

    if current_run_date is not None and current_verified_count is not None:
        by_date[str(current_run_date)[:10]] = int(current_verified_count)

    streak = 0
    for d in sorted(by_date.keys(), reverse=True):
        if by_date[d] == 0:
            streak += 1
        else:
            break
    return streak


def evaluate_dead_mans_switch(
    repo_root: Path,
    *,
    circuit_breaker: str | None,
    drive_canary: Mapping[str, Any] | None,
    git_size_mb: float,
    max_git_mb: float = 700.0,
    current_run_date: str,
    current_verified_count: int,
    github_client: GitHubClient | None = None,
) -> DeadMansSwitchResult:
    """
    Evaluate Job 7 Dead-Man's Switch conditions (Blueprint Section 5):
    - Drive canary failed (`circuit_breaker == 'DRIVE_AUTH_FAIL'` or `drive_canary.ok is False`)
    - `.git` size > 700 MB
    - 0 verified nodes for 3 consecutive nights
    """
    reasons: list[str] = []
    drive_failed = circuit_breaker == "DRIVE_AUTH_FAIL" or (
        isinstance(drive_canary, Mapping) and drive_canary.get("ok") is False
    )
    if drive_failed:
        reasons.append("DRIVE_CANARY_FAILED")

    if float(git_size_mb) > float(max_git_mb):
        reasons.append(f"GIT_SIZE_EXCEEDED({git_size_mb:.2f}MB>{max_git_mb:.2f}MB)")

    zero_streak = count_consecutive_zero_verified_nights(
        repo_root,
        current_run_date=current_run_date,
        current_verified_count=current_verified_count,
    )
    if zero_streak >= 3:
        reasons.append(f"ZERO_VERIFIED_NODES_{zero_streak}_CONSECUTIVE_NIGHTS")

    triggered = len(reasons) > 0
    alert_issue: dict[str, Any] | None = None
    if triggered and github_client is not None:
        title = f"RAL Critical Alert: {', '.join(reasons)} ({current_run_date})"
        body = (
            "## RAL Nightly Dead-Man's Switch Triggered\n\n"
            f"- **Run Date:** `{current_run_date}`\n"
            f"- **Triggers:** `{', '.join(reasons)}`\n"
            f"- **Circuit Breaker:** `{circuit_breaker}`\n"
            f"- **Git Size (MiB):** `{git_size_mb:.3f}` (threshold `{max_git_mb:.1f}`)\n"
            f"- **Consecutive Zero-Verified Nights:** `{zero_streak}`\n"
        )
        alert_issue = github_client.upsert_alert_issue(
            title=title,
            body=body,
            label=CRITICAL_ALERT_LABEL,
        )

    return DeadMansSwitchResult(
        triggered=triggered,
        reasons=tuple(reasons),
        consecutive_zero_verified_nights=zero_streak,
        git_size_mb=float(git_size_mb),
        alert_issue=alert_issue,
    )


def run_consolidate_job(
    repo_root: Path,
    *,
    handoff_dir: Path,
    cache_dir: Path,
    run_date_iso: str,
    run_id: str = "local",
    max_git_mb: float = 700.0,
    dry_run: bool = False,
    allow_release_in_dry_run: bool = False,
    github_client: GitHubClient | None = None,
    embed_client: Any | None = None,
    now: datetime | None = None,
    repo_slug: str = "rayhans-autonomous-lab",
) -> ConsolidateJobResult:
    """
    Job 7 (`consolidate_and_watchdog`):
    1. Run full `ci/validate_vault.py` checks.
    2. Compile SQLite index (`FTS5` + Graph + `int8` vectors) + compress `.sqlite.zst`.
    3. Execute Blue-Green Release Swap (`keep_snapshots=2`, ZERO `--clobber`).
    4. Regenerate `/INDEX.md` (<15k tokens) & `/llms.txt`.
    5. Append `metrics/telemetry/YYYY-MM.ndjson`.
    6. Evaluate Dead-Man's Switch (`ral-critical-alert` Issue).
    """
    root = Path(repo_root)
    handoff = Path(handoff_dir)
    cache = Path(cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    ts = now or datetime.now(timezone.utc)
    calls_before = github_client.api_call_count if github_client is not None else 0

    cb_path = handoff / "circuit_breaker.json"
    cb_state = json.loads(cb_path.read_text(encoding="utf-8")) if cb_path.is_file() else {}
    circuit_breaker = cb_state.get("circuit_breaker")
    drive_canary = cb_state.get("drive_canary")
    embed_info = cb_state.get("embedding_model")
    discovered_embed_model = (
        str(embed_info.get("model_id") or "") if isinstance(embed_info, Mapping) else None
    ) or None

    epoch_summaries: list[dict[str, Any]] = []
    for k in range(1, 7):
        s_path = handoff / f"epoch_{k}_summary.json"
        if s_path.is_file():
            epoch_summaries.append(json.loads(s_path.read_text(encoding="utf-8")))

    nodes_verified_tonight = sum(int(s.get("nodes_verified_tonight", 0)) for s in epoch_summaries)
    domain_epochs: dict[str, int] = {}
    for s in epoch_summaries:
        tgt = s.get("target")
        if isinstance(tgt, str) and tgt and not s.get("pivoted_to_maintenance"):
            domain_epochs[tgt] = domain_epochs.get(tgt, 0) + 1

    allow_path = root / "config" / "license_allowlist.yaml"
    pre_val = validate_vault(
        root,
        max_git_mb=max_git_mb,
        allowlist_path=allow_path if allow_path.exists() else None,
    )

    db_path = cache / "ral-index.sqlite"
    idx_res = build_or_update_index(
        root,
        db_path,
        embed_client=embed_client,
        discovered_embed_model=discovered_embed_model,
        now=ts,
    )
    clean_date = run_date_iso[:10].replace("-", "")
    zst_out = cache / f"ral-index-{clean_date}-{run_id}.sqlite.zst"
    zst_path, zst_sha, _ = compress_sqlite_zst(db_path, zst_out)

    swap_res: BlueGreenSwapResult | None = None
    if (
        pre_val.ok
        and github_client is not None
        and (not dry_run or allow_release_in_dry_run)
    ):
        swap_res = github_client.blue_green_swap_index(
            zst_path,
            run_date=clean_date,
            run_id=run_id,
            embedding_model_id=idx_res.embedding_model_id,
            embedding_dim=idx_res.embedding_dim,
            node_count=idx_res.node_count,
            tag=INDEX_RELEASE_TAG,
            keep_snapshots=2,
            verify_download=True,
            now=ts,
        )

    docs_summary = regenerate_vault_docs(root, now=ts, repo_slug=repo_slug)

    dms_res = evaluate_dead_mans_switch(
        root,
        circuit_breaker=circuit_breaker,
        drive_canary=drive_canary,
        git_size_mb=pre_val.git_size_mb,
        max_git_mb=max_git_mb,
        current_run_date=run_date_iso[:10],
        current_verified_count=nodes_verified_tonight,
        github_client=github_client,
    )

    api_calls_used = (
        github_client.api_call_count - calls_before if github_client is not None else 0
    )
    tel_path = append_telemetry_row(
        root,
        {
            "record_type": "nightly_summary",
            "timestamp": ts.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "run_date": run_date_iso[:10],
            "run_id": run_id,
            "dry_run": bool(dry_run),
            "circuit_breaker": circuit_breaker,
            "nodes_verified_tonight": nodes_verified_tonight,
            "node_count": idx_res.node_count,
            "edge_count": idx_res.edge_count,
            "git_size_mb": pre_val.git_size_mb,
            "domain_epochs": domain_epochs,
            "embed_space_consistent": idx_res.embed_space_consistent,
            "vector_action": idx_res.vector_action,
            "index_snapshot_name": swap_res.snapshot_name if swap_res else zst_path.name,
            "dead_mans_switch_triggered": dms_res.triggered,
            "dead_mans_switch_reasons": list(dms_res.reasons),
            "github_api_calls_used": api_calls_used,
        },
        now=ts,
    )

    post_val = validate_vault(
        root,
        max_git_mb=max_git_mb,
        allowlist_path=allow_path if allow_path.exists() else None,
    )

    return ConsolidateJobResult(
        ok=bool(pre_val.ok and post_val.ok),
        vault_ok=bool(pre_val.ok and post_val.ok),
        vault_errors=tuple(post_val.errors or pre_val.errors),
        node_count=idx_res.node_count,
        edge_count=idx_res.edge_count,
        nodes_verified_tonight=nodes_verified_tonight,
        index_db_path=db_path,
        index_zst_path=zst_path,
        index_zst_sha256=zst_sha,
        embed_space_consistent=idx_res.embed_space_consistent,
        vector_action=idx_res.vector_action,
        blue_green_swap=swap_res,
        docs_summary=docs_summary,
        telemetry_path=tel_path,
        dead_mans_switch=dms_res,
        github_api_calls_used=api_calls_used,
    )


def run_full_nightly_pipeline(
    repo_root: Path,
    *,
    handoff_dir: Path,
    cache_dir: Path,
    ore_cache_dir: Path,
    run_date_iso: str,
    current_night: int = 0,
    available_epochs: int = 6,
    run_id: str = "dryrun1",
    dry_run: bool = True,
    allow_release_in_dry_run: bool = True,
    simulate_drive_fail: bool = False,
    max_git_mb: float = 700.0,
    gemini_client: Any | None = None,
    drive_store: DriveOreStore | None = None,
    github_client: GitHubClient | None = None,
    sandbox: OracleSandbox | None = None,
    registry: OraclePluginRegistry | None = None,
    source_fetcher: Callable[[str, Any, Path], Sequence[SourceRecord]] | None = None,
    http_get: Callable[..., Any] | None = None,
    retraction_watch_dois: set[str] | None = None,
    now: datetime | None = None,
) -> NightlyDryRunSummary:
    """
    Execute the complete 8-job workflow (Job 0 -> Epochs 1..6 -> Job 7) strictly communicating
    between jobs via `handoff_dir` JSON artifacts and tracking total GitHub API budget (<40).
    """
    calls_before = github_client.api_call_count if github_client is not None else 0
    preflight_res = run_preflight_job(
        repo_root,
        handoff_dir=handoff_dir,
        cache_dir=cache_dir,
        run_date_iso=run_date_iso,
        current_night=current_night,
        available_epochs=available_epochs,
        run_id=run_id,
        dry_run=dry_run,
        simulate_drive_fail=simulate_drive_fail,
        gemini_client=gemini_client,
        drive_store=drive_store,
        github_client=github_client,
        now=now,
    )

    epoch_results: list[EpochJobResult] = []
    for k in range(1, 7):
        ep_res = run_epoch_job(
            repo_root,
            handoff_dir=handoff_dir,
            epoch_index=k,
            cache_dir=cache_dir,
            ore_cache_dir=ore_cache_dir,
            dry_run=dry_run,
            gemini_client=gemini_client,
            drive_store=drive_store,
            sandbox=sandbox,
            registry=registry,
            source_fetcher=source_fetcher,
            http_get=http_get,
            retraction_watch_dois=retraction_watch_dois,
            now=now,
        )
        epoch_results.append(ep_res)

    consolidate_res = run_consolidate_job(
        repo_root,
        handoff_dir=handoff_dir,
        cache_dir=cache_dir,
        run_date_iso=run_date_iso,
        run_id=run_id,
        max_git_mb=max_git_mb,
        dry_run=dry_run,
        allow_release_in_dry_run=allow_release_in_dry_run,
        github_client=github_client,
        embed_client=gemini_client,
        now=now,
    )

    rest_calls = (
        github_client.api_call_count - calls_before if github_client is not None else 0
    )
    artifact_ops = 14  # 1 upload in Job 0 + (1 download + 1 upload) * 6 epochs + 1 download in Job 7
    git_pushes = 1 + sum(
        1 for e in epoch_results if (e.nodes_created or e.nodes_merged or e.nodes_upgraded or e.nodes_challenged)
    )
    total_estimated = artifact_ops + git_pushes + rest_calls

    return NightlyDryRunSummary(
        preflight=preflight_res,
        epochs=tuple(epoch_results),
        consolidate=consolidate_res,
        artifact_ops_count=artifact_ops,
        git_pushes_count=git_pushes,
        github_rest_calls_count=rest_calls,
        estimated_total_github_api_calls=total_estimated,
    )

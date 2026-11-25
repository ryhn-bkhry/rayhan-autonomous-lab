"""Autonomous Curriculum & Curiosity Engine: U(d,t), S(di,dj,t), backoff, splits, and epochs (Section 4)."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import date, datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import re
import shutil
import subprocess
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from ci.validate_node import split_frontmatter, validate_node_file
from ral.canonicalize import normalize_newlines

MACRO_DISCIPLINES: tuple[str, ...] = ("math", "phys", "cs", "eng", "econ", "chem", "bio")

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_CURRICULUM_PARAMS_PATH = _REPO_ROOT / "config" / "curriculum_params.yaml"
DEFAULT_TAXONOMY_SEEDS_PATH = _REPO_ROOT / "config" / "taxonomy_seeds.yaml"

_SLUG_CLEAN_RE = re.compile(r"[^a-z0-9-]+")


@dataclass(frozen=True)
class PrereqStub:
    """Dangling prerequisite stub tracked in `ledger/prereq_debt.ndjson` (Sections 4.2 & 7.3)."""

    stub_id: str
    domain: str
    cited_by: tuple[str, ...] = ()
    attempt_count: int = 0
    last_attempt_night: int | None = None
    cooldown_until_night: int = 0
    status: str = "OPEN"

    def to_dict(self) -> dict[str, Any]:
        """Serialize to exact Section 7.3 `prereq_debt.ndjson` row format."""
        return {
            "stub_id": self.stub_id,
            "domain": self.domain,
            "cited_by": list(self.cited_by),
            "attempt_count": int(self.attempt_count),
            "last_attempt_night": self.last_attempt_night,
            "cooldown_until_night": int(self.cooldown_until_night),
            "status": self.status,
        }


@dataclass(frozen=True)
class DomainState:
    """Observed per-leaf-domain metrics used to score $U(d, t)$ at night $t$."""

    domain: str
    macro: str
    verified_count: int = 0
    open_conflicts: int = 0
    expired_t3: int = 0
    trailing_epochs_30d: float = 0.0
    nights_since_last_work: int | None = None  # None => +inf on Day 0
    ewma_yield: float | None = None
    half_life_years: float = 10.0
    seed_prior: float = 1.0
    avg_rigor: float = 1.0


@dataclass(frozen=True)
class DomainPriority:
    """Computed priority score and sorting metadata for a candidate leaf domain."""

    domain: str
    macro: str
    raw_score: float
    effective_score: float
    sigma: float
    rho: float
    pagerank_norm: float
    exploitation: float
    exploration: float
    yield_term: float
    masked: bool
    boosted: bool
    seed_prior: float
    tie_break_hash: str


@dataclass(frozen=True)
class SynthesisCandidate:
    """Cross-discipline synthesis candidate scored by whitened $S(d_i, d_j, t)$."""

    domain_a: str
    domain_b: str
    score: float
    isotropic_cosine: float
    unbridged_factor: float
    rigor_factor: float
    bridge_edges: int


@dataclass(frozen=True)
class LeafSplitPlan:
    """Deterministic sub-branch split plan for a leaf exceeding 40 verified nodes."""

    parent_domain: str
    k: int
    silhouette_score: float
    assignments: dict[str, str]  # ral_id -> new taxonomy_code
    moves: tuple[tuple[Path, Path], ...]


def macro_of(domain_code: str) -> str:
    """Return the root macro-discipline prefix of a taxonomy code."""
    return domain_code.split(".", 1)[0].strip().lower()


def load_curriculum_params(path: Path | None = None) -> dict[str, Any]:
    """Load curriculum parameters from `config/curriculum_params.yaml`."""
    resolved = Path(path) if path is not None else DEFAULT_CURRICULUM_PARAMS_PATH
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {resolved}")
    return data


def load_taxonomy_seeds(path: Path | None = None) -> dict[str, Any]:
    """Load Day-0 macro-discipline seed taxonomy from `config/taxonomy_seeds.yaml`."""
    resolved = Path(path) if path is not None else DEFAULT_TAXONOMY_SEEDS_PATH
    data = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected mapping in {resolved}")
    return data


def extract_seed_leaves(taxonomy_cfg: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Flatten Day-0 `taxonomy_seeds.yaml` leaves keyed by `taxonomy_code`."""
    leaves: dict[str, dict[str, Any]] = {}
    macros = taxonomy_cfg.get("macro_disciplines") or {}
    for macro_key, macro_info in macros.items():
        if not isinstance(macro_info, Mapping):
            continue
        for leaf in macro_info.get("leaves") or []:
            if not isinstance(leaf, Mapping) or not leaf.get("code"):
                continue
            code = str(leaf["code"]).strip()
            leaves[code] = {
                "code": code,
                "macro": str(macro_key).strip().lower(),
                "name": str(leaf.get("name") or code),
                "authority_codes": list(leaf.get("authority_codes") or []),
                "seed_prior": float(leaf.get("seed_prior", 1.0)),
                "default_half_life_years": float(leaf.get("default_half_life_years", 10.0)),
            }
    return leaves


def _normalize_taxonomy_slug(raw: str | None) -> str:
    text = (
        str(raw or "")
        .strip()
        .lower()
        .replace("_", "-")
        .replace("/", "-")
        .replace(" ", "-")
    )
    segments: list[str] = []
    for part in text.split("."):
        cleaned = re.sub(r"-{2,}", "-", part).strip("-")
        if cleaned:
            segments.append(cleaned)
    return ".".join(segments)


def get_active_taxonomy_leaves(
    repo_root: Path | None = None,
    *,
    seeds_path: Path | None = None,
) -> set[str]:
    """
    Return the active leaf set `L(T_t)` from `taxonomy_seeds.yaml` and existing `vault/` nodes.

    A seed leaf stays active unless already present or superseded by split sub-branches in `vault/`.
    """
    resolved_seeds = seeds_path
    if resolved_seeds is None and repo_root is not None:
        candidate = Path(repo_root) / "config" / "taxonomy_seeds.yaml"
        if candidate.exists():
            resolved_seeds = candidate
    seeds_cfg = load_taxonomy_seeds(resolved_seeds)
    seed_leaves = extract_seed_leaves(seeds_cfg)

    observed_domains: set[str] = set()
    if repo_root is not None:
        vault_dir = Path(repo_root) / "vault"
        if vault_dir.exists():
            for md_path in sorted(vault_dir.rglob("*.md")):
                try:
                    raw = normalize_newlines(md_path.read_text(encoding="utf-8"))
                    fm, _ = split_frontmatter(raw)
                except Exception:
                    continue
                dom = _normalize_taxonomy_slug(str(fm.get("taxonomy_code") or ""))
                if dom:
                    observed_domains.add(dom)

    active: set[str] = set(observed_domains)
    for seed_code in seed_leaves:
        norm_seed = _normalize_taxonomy_slug(seed_code)
        if not norm_seed:
            continue
        if norm_seed in observed_domains or any(
            d.startswith(f"{norm_seed}.") for d in observed_domains
        ):
            continue
        active.add(norm_seed)
    return active


def clamp_taxonomy_code(
    raw_code: str | None,
    *,
    active_leaves: Sequence[str] | set[str],
    target_domain: str | None = None,
    fallback: str = "math.analysis",
) -> str:
    """
    Clamp `raw_code` to an existing active leaf in `active_leaves` (defaulting to
    `target_domain` during explore/exploit) so sub-branches are created strictly by
    `director.plan_leaf_split()` when `n_d > 40`.
    """
    leaf_set = {_normalize_taxonomy_slug(c) for c in active_leaves if str(c).strip()}
    leaf_set.discard("")
    norm_raw = _normalize_taxonomy_slug(raw_code)
    norm_target = _normalize_taxonomy_slug(target_domain)
    norm_fallback = _normalize_taxonomy_slug(fallback)

    if not leaf_set:
        return norm_raw or norm_target or norm_fallback or "math.analysis"

    if norm_raw in leaf_set:
        return norm_raw

    if norm_raw:
        parts = norm_raw.split(".")
        for end in range(len(parts) - 1, 1, -1):
            prefix = ".".join(parts[:end])
            if prefix in leaf_set:
                return prefix

    if norm_target in leaf_set:
        return norm_target
    if norm_target:
        t_parts = norm_target.split(".")
        for end in range(len(t_parts) - 1, 1, -1):
            t_prefix = ".".join(t_parts[:end])
            if t_prefix in leaf_set:
                return t_prefix

    if norm_fallback in leaf_set:
        return norm_fallback

    if norm_raw:
        macro = macro_of(norm_raw)
        same_macro = sorted(c for c in leaf_set if macro_of(c) == macro)
        if same_macro:
            return same_macro[0]

    return sorted(leaf_set)[0]


# ---------------------------------------------------------------------------
# Section 4.2: Unresolvable Prerequisite Exponential Backoff & Tension
# ---------------------------------------------------------------------------


def compute_prereq_cooldown(
    last_attempt_night: int,
    attempt_count: int,
    params: Mapping[str, Any] | None = None,
) -> int:
    """Compute cooldown night `t_cool = t_p + min(90, 2 ** k_p)` (Section 4.2)."""
    backoff_cfg = (params or {}).get("prereq_backoff", {})
    max_cool = int(backoff_cfg.get("max_cooldown_nights", 90))
    base = int(backoff_cfg.get("backoff_base", 2))
    step = min(max_cool, base ** max(0, int(attempt_count)))
    return int(last_attempt_night) + step


def record_prereq_failure(
    stub: PrereqStub,
    current_night: int,
    params: Mapping[str, Any] | None = None,
) -> PrereqStub:
    """
    Increment prerequisite resolution failure counter `k_p <- k_p + 1`, apply exponential
    cooldown, and quarantine to `UNRESOLVABLE_PAYWALLED_OR_MISSING` once `k_p >= 5`.
    """
    backoff_cfg = (params or {}).get("prereq_backoff", {})
    quarantine_k = int(backoff_cfg.get("quarantine_attempts", 5))
    new_k = int(stub.attempt_count) + 1
    cooldown_until = compute_prereq_cooldown(current_night, new_k, params=params)
    new_status = "UNRESOLVABLE_PAYWALLED_OR_MISSING" if new_k >= quarantine_k else "OPEN"
    return PrereqStub(
        stub_id=stub.stub_id,
        domain=stub.domain,
        cited_by=stub.cited_by,
        attempt_count=new_k,
        last_attempt_night=int(current_night),
        cooldown_until_night=cooldown_until,
        status=new_status,
    )


def is_eligible_for_passive_recheck(
    stub: PrereqStub,
    current_night: int,
    params: Mapping[str, Any] | None = None,
) -> bool:
    """Return True if a quarantined stub is due for its 180-night passive re-check."""
    if stub.status != "UNRESOLVABLE_PAYWALLED_OR_MISSING":
        return False
    backoff_cfg = (params or {}).get("prereq_backoff", {})
    recheck_nights = int(backoff_cfg.get("passive_recheck_nights", 180))
    last_night = 0 if stub.last_attempt_night is None else int(stub.last_attempt_night)
    return int(current_night) >= last_night + recheck_nights


def load_prereq_debt(ledger_path: Path) -> dict[str, PrereqStub]:
    """Load latest per-stub state from append-only `ledger/prereq_debt.ndjson`."""
    path = Path(ledger_path)
    if not path.exists():
        return {}

    state: dict[str, PrereqStub] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        row = json.loads(stripped)
        if not isinstance(row, Mapping):
            continue
        sid = str(row.get("stub_id") or "").strip()
        if not sid:
            continue
        prev = state.get(sid)
        prev_cited = set(prev.cited_by) if prev is not None else set()
        new_cited = {str(c).strip() for c in (row.get("cited_by") or []) if str(c).strip()}
        merged_cited = tuple(sorted(prev_cited | new_cited))
        last_night_raw = row.get("last_attempt_night")
        state[sid] = PrereqStub(
            stub_id=sid,
            domain=str(row.get("domain") or (prev.domain if prev else "")).strip(),
            cited_by=merged_cited,
            attempt_count=int(row.get("attempt_count", 0)),
            last_attempt_night=(int(last_night_raw) if last_night_raw is not None else None),
            cooldown_until_night=int(row.get("cooldown_until_night", 0)),
            status=str(row.get("status") or "OPEN").strip(),
        )
    return state


def append_prereq_stub(ledger_path: Path, stub: PrereqStub) -> None:
    """Validate and append one `PrereqStub` state snapshot to `ledger/prereq_debt.ndjson`."""
    if not stub.stub_id.startswith("STUB-"):
        raise ValueError(f"stub_id must start with 'STUB-', got {stub.stub_id!r}")
    path = Path(ledger_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(stub.to_dict(), sort_keys=False) + "\n")


def record_prereq_attempt_failure(
    repo_root: Path,
    stub_id: str,
    domain: str,
    current_night: int,
    *,
    cited_by: Sequence[str] = (),
    params: Mapping[str, Any] | None = None,
) -> PrereqStub:
    """Record a failed OA prerequisite resolution attempt to `ledger/prereq_debt.ndjson`."""
    ledger_path = Path(repo_root) / "ledger" / "prereq_debt.ndjson"
    existing = load_prereq_debt(ledger_path)
    prev = existing.get(stub_id)
    merged_cited = tuple(
        sorted(
            (set(prev.cited_by) if prev is not None else set())
            | {str(c).strip() for c in cited_by if str(c).strip()}
        )
    )
    base_stub = PrereqStub(
        stub_id=stub_id,
        domain=domain or (prev.domain if prev else ""),
        cited_by=merged_cited,
        attempt_count=prev.attempt_count if prev is not None else 0,
        last_attempt_night=prev.last_attempt_night if prev is not None else None,
        cooldown_until_night=prev.cooldown_until_night if prev is not None else 0,
        status=prev.status if prev is not None else "OPEN",
    )
    updated = record_prereq_failure(base_stub, current_night, params=params)
    append_prereq_stub(ledger_path, updated)
    return updated


def resolve_prereq_stub(
    repo_root: Path,
    stub_id: str,
    current_night: int,
) -> PrereqStub | None:
    """Opportunistically mark a dangling or quarantined stub `RESOLVED` in `prereq_debt.ndjson`."""
    ledger_path = Path(repo_root) / "ledger" / "prereq_debt.ndjson"
    existing = load_prereq_debt(ledger_path)
    prev = existing.get(stub_id)
    if prev is None:
        return None
    resolved = PrereqStub(
        stub_id=prev.stub_id,
        domain=prev.domain,
        cited_by=prev.cited_by,
        attempt_count=prev.attempt_count,
        last_attempt_night=int(current_night),
        cooldown_until_night=int(current_night),
        status="RESOLVED",
    )
    append_prereq_stub(ledger_path, resolved)
    return resolved


def compute_epistemic_tension(
    domain: str,
    *,
    stubs: Sequence[PrereqStub],
    current_night: int,
    open_conflicts: int = 0,
    expired_t3: int = 0,
    params: Mapping[str, Any] | None = None,
) -> float:
    """
    Compute effective epistemic tension `sigma_d(t)` (Section 4.2):
    `w1 * sum(I[t >= t_cool] * exp(-0.5 * k_p)) + w2 * |OpenConflicts| + w3 * |ExpiredT3|`.
    """
    tw = (params or {}).get("tension_weights", {})
    w1 = float(tw.get("w1_dangling_prereq", 1.0))
    w2 = float(tw.get("w2_open_conflicts", 1.5))
    w3 = float(tw.get("w3_expired_t3", 0.5))
    decay = float(tw.get("dangling_attempt_decay", 0.5))

    dangling_sum = 0.0
    for stub in stubs:
        if stub.domain != domain or stub.status != "OPEN":
            continue
        if int(current_night) >= int(stub.cooldown_until_night):
            dangling_sum += math.exp(-decay * max(0, int(stub.attempt_count)))

    return (
        w1 * dangling_sum
        + w2 * max(0, int(open_conflicts))
        + w3 * max(0, int(expired_t3))
    )


# ---------------------------------------------------------------------------
# Section 4.3: Domain Priority U(d, t), Balance Guard & Tie-Breaking
# ---------------------------------------------------------------------------


def compute_cross_domain_pagerank(
    domains: Sequence[str],
    edges: Sequence[Mapping[str, Any]],
    node_domains: Mapping[str, str],
    *,
    damping: float = 0.85,
    max_iter: int = 50,
    tol: float = 1e-9,
) -> dict[str, float]:
    """
    Compute normalized cross-domain reverse PageRank over `DEPENDS_ON` edges.

    If node `u` in domain `d_a` depends on node `v` in domain `d_b` (`d_a != d_b`),
    domain `d_b` receives foundational endorsement from `d_a`.
    Returns scores min-max normalized to `[0.0, 1.0]` across `domains`.
    """
    domain_list = sorted(set(domains))
    n = len(domain_list)
    if n == 0:
        return {}
    idx = {d: i for i, d in enumerate(domain_list)}

    adj = np.zeros((n, n), dtype=np.float64)
    has_cross_edge = False
    for edge in edges:
        if edge.get("relation") != "DEPENDS_ON":
            continue
        src = str(edge.get("src") or "")
        dst = str(edge.get("dst") or "")
        d_src = node_domains.get(src)
        d_dst = node_domains.get(dst)
        if not d_src or not d_dst or d_src == d_dst:
            continue
        if d_src not in idx or d_dst not in idx:
            continue
        adj[idx[d_src], idx[d_dst]] += 1.0
        has_cross_edge = True

    if not has_cross_edge:
        return {d: 0.0 for d in domain_list}

    row_sums = adj.sum(axis=1)
    pr = np.full(n, 1.0 / n, dtype=np.float64)
    teleport = (1.0 - damping) / n

    for _ in range(max_iter):
        next_pr = np.full(n, teleport, dtype=np.float64)
        dangling_mass = float(pr[row_sums == 0.0].sum())
        if dangling_mass > 0.0:
            next_pr += damping * dangling_mass / n
        nonzero_rows = row_sums > 0.0
        if np.any(nonzero_rows):
            weights = pr[nonzero_rows] / row_sums[nonzero_rows]
            next_pr += damping * (weights @ adj[nonzero_rows])
        if float(np.abs(next_pr - pr).sum()) <= tol:
            pr = next_pr
            break
        pr = next_pr

    min_v = float(pr.min())
    max_v = float(pr.max())
    span = max_v - min_v
    if span <= 1e-12:
        return {d: 0.0 for d in domain_list}
    return {d: max(0.0, min(1.0, (float(pr[idx[d]]) - min_v) / span)) for d in domain_list}


def compute_foundational_leverage(
    pagerank_norm: float,
    half_life_years: float | None = None,
    params: Mapping[str, Any] | None = None,
) -> float:
    """Compute Foundational Leverage `rho_d(t)` in `[0, 1]` (Section 4.3)."""
    fl_cfg = (params or {}).get("foundational_leverage", {})
    w_pr = float(fl_cfg.get("pagerank_weight", 0.5))
    w_hl = float(fl_cfg.get("half_life_weight", 0.5))
    h_max = float(fl_cfg.get("h_max_years", 50.0))
    default_h = float(fl_cfg.get("default_half_life_years", 10.0))

    h_d = default_h if half_life_years is None else max(0.0, float(half_life_years))
    hl_term = math.log1p(h_d) / math.log1p(h_max) if h_max > 0.0 else 0.0
    pr_clamped = max(0.0, min(1.0, float(pagerank_norm)))
    return max(0.0, min(1.0, w_pr * pr_clamped + w_hl * hl_term))


def compute_domain_priorities(
    domain_states: Sequence[DomainState],
    *,
    stubs: Sequence[PrereqStub] = (),
    current_night: int = 0,
    run_date_iso: str,
    pagerank_scores: Mapping[str, float] | None = None,
    global_mean_yield: float | None = None,
    params: Mapping[str, Any] | None = None,
) -> list[DomainPriority]:
    """
    Compute `U(d, t)` for every active leaf domain, apply the Macro-Discipline Balance Guard,
    and sort deterministically using unbiased `sha256(date:d)[:8]` tie-breaking (Section 4.3).
    """
    if not domain_states:
        return []

    ps_cfg = (params or {}).get("priority_score", {})
    alpha = float(ps_cfg.get("alpha", 1.0))
    beta = float(ps_cfg.get("beta", 0.85))
    gamma = float(ps_cfg.get("gamma", 0.5))
    lambda_sat = float(ps_cfg.get("lambda_sat", 0.35))
    tau_0 = float(ps_cfg.get("tau_0", 14.0))
    clip_min = float(ps_cfg.get("yield_clip_min", -0.2))
    clip_max = float(ps_cfg.get("yield_clip_max", 0.2))
    tie_digits = int(ps_cfg.get("tie_break_round_digits", 9))

    bal_cfg = (params or {}).get("discipline_balance", {})
    ceiling_share = float(bal_cfg.get("ceiling_share", 0.20))
    floor_share = float(bal_cfg.get("floor_share", 0.04))
    floor_mult = float(bal_cfg.get("floor_boost_multiplier", 2.0))

    total_T = max(1, sum(max(0, int(s.verified_count)) + 1 for s in domain_states))

    observed_yields = [float(s.ewma_yield) for s in domain_states if s.ewma_yield is not None]
    if global_mean_yield is not None:
        r_bar = float(global_mean_yield)
    elif observed_yields:
        r_bar = sum(observed_yields) / len(observed_yields)
    else:
        r_bar = 0.0

    macro_epochs: dict[str, float] = defaultdict(float)
    for s in domain_states:
        macro_epochs[s.macro] += max(0.0, float(s.trailing_epochs_30d))
    total_trailing_epochs = sum(macro_epochs.values())

    masked_macros: set[str] = set()
    boosted_macros: set[str] = set()
    if total_trailing_epochs > 0.0:
        macro_shares = {m: c / total_trailing_epochs for m, c in macro_epochs.items()}
        candidate_masked = {m for m, sh in macro_shares.items() if sh >= ceiling_share}
        all_macros_present = {s.macro for s in domain_states}
        if candidate_masked and candidate_masked < all_macros_present:
            masked_macros = candidate_masked
        boosted_macros = {
            m for m, sh in macro_shares.items() if m not in masked_macros and sh < floor_share
        }

    pr_map = pagerank_scores or {}
    scored: list[DomainPriority] = []
    for s in domain_states:
        sigma = compute_epistemic_tension(
            s.domain,
            stubs=stubs,
            current_night=current_night,
            open_conflicts=s.open_conflicts,
            expired_t3=s.expired_t3,
            params=params,
        )
        pr_norm = float(pr_map.get(s.domain, 0.0))
        rho = compute_foundational_leverage(
            pr_norm,
            half_life_years=s.half_life_years,
            params=params,
        )
        c_d = max(0.0, float(s.trailing_epochs_30d))
        exploitation = alpha * (rho * (1.0 + math.log1p(sigma))) / (1.0 + lambda_sat * c_d)

        # Day-0 invariant: unvisited tau_d = +inf => staleness factor = 1.0 identically.
        if s.nights_since_last_work is None:
            staleness = 1.0
        else:
            tau_d = max(0.0, float(s.nights_since_last_work))
            staleness = 1.0 - math.exp(-tau_d / tau_0) if tau_0 > 0.0 else 1.0

        exploration = beta * math.sqrt(math.log(total_T + 1.0) / (max(0, int(s.verified_count)) + 1.0)) * staleness
        r_hat = float(s.ewma_yield) if s.ewma_yield is not None else r_bar
        yield_term = gamma * max(clip_min, min(clip_max, r_hat - r_bar))
        raw_score = exploitation + exploration + yield_term

        is_masked = s.macro in masked_macros
        is_boosted = s.macro in boosted_macros
        effective_score = raw_score * floor_mult if is_boosted else raw_score
        tie_hash = hashlib.sha256(f"{run_date_iso}:{s.domain}".encode("utf-8")).hexdigest()[:8]

        scored.append(
            DomainPriority(
                domain=s.domain,
                macro=s.macro,
                raw_score=raw_score,
                effective_score=effective_score,
                sigma=sigma,
                rho=rho,
                pagerank_norm=pr_norm,
                exploitation=exploitation,
                exploration=exploration,
                yield_term=yield_term,
                masked=is_masked,
                boosted=is_boosted,
                seed_prior=float(s.seed_prior),
                tie_break_hash=tie_hash,
            )
        )

    scored.sort(
        key=lambda item: (
            1 if item.masked else 0,
            -round(item.effective_score, tie_digits),
            -item.seed_prior,
            item.tie_break_hash,
        )
    )
    return scored


# ---------------------------------------------------------------------------
# Section 4.4: Isotropic Cross-Domain Synthesis Score S(d_i, d_j, t)
# ---------------------------------------------------------------------------


def whiten_embeddings(
    embeddings: Mapping[str, Sequence[float]],
    *,
    top_pcs: int = 2,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """
    Apply All-but-the-Top isotropic whitening (Section 4.4) in pure NumPy:
    L2-normalize -> subtract global mean -> project out top-`m` principal directions -> re-normalize.
    """
    if not embeddings:
        return {}
    node_ids = sorted(embeddings.keys())
    mat = np.asarray([embeddings[nid] for nid in node_ids], dtype=np.float64)
    if mat.ndim != 2 or mat.shape[0] == 0 or mat.shape[1] == 0:
        return {}

    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    unit = mat / np.maximum(norms, eps)

    if unit.shape[0] == 1:
        return {node_ids[0]: unit[0]}

    mu = unit.mean(axis=0, keepdims=True)
    centered = unit - mu

    max_removable = max(0, min(int(top_pcs), unit.shape[0] - 1, unit.shape[1]))
    if max_removable > 0:
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        pcs = vt[:max_removable]
        projected = centered - (centered @ pcs.T) @ pcs
    else:
        projected = centered

    proj_norms = np.linalg.norm(projected, axis=1, keepdims=True)
    whitened = projected / np.maximum(proj_norms, eps)
    return {nid: whitened[i] for i, nid in enumerate(node_ids)}


def compute_domain_centroids(
    whitened_vectors: Mapping[str, np.ndarray],
    node_domains: Mapping[str, str],
    *,
    eps: float = 1e-8,
) -> dict[str, np.ndarray]:
    """Compute unit-norm isotropic domain centroids `e_struct(d)` (Section 4.4)."""
    buckets: dict[str, list[np.ndarray]] = defaultdict(list)
    for nid, vec in whitened_vectors.items():
        dom = node_domains.get(nid)
        if dom:
            buckets[dom].append(np.asarray(vec, dtype=np.float64))

    centroids: dict[str, np.ndarray] = {}
    for dom, vecs in buckets.items():
        summed = np.sum(np.stack(vecs, axis=0), axis=0)
        norm = float(np.linalg.norm(summed))
        centroids[dom] = summed / max(norm, eps)
    return centroids


def score_synthesis_pairs(
    embeddings: Mapping[str, Sequence[float]],
    node_domains: Mapping[str, str],
    domain_rigor: Mapping[str, float],
    bridge_edge_counts: Mapping[tuple[str, str], int] | None = None,
    *,
    params: Mapping[str, Any] | None = None,
) -> list[SynthesisCandidate]:
    """
    Score cross-discipline pairs `(d_i, d_j)` with `M(d_i) != M(d_j)` via whitened
    isotropic cosine, unbridged structural hole, and rigor gate (Section 4.4).
    """
    synth_cfg = (params or {}).get("synthesis", {})
    top_pcs = int(synth_cfg.get("whitening_top_pcs", 2))
    eps = float(synth_cfg.get("norm_epsilon", 1e-8))
    min_score = float(synth_cfg.get("min_score_threshold", 0.35))
    min_rigor = float(synth_cfg.get("min_domain_rigor", 0.70))

    whitened = whiten_embeddings(embeddings, top_pcs=top_pcs, eps=eps)
    if not whitened:
        return []
    centroids = compute_domain_centroids(whitened, node_domains, eps=eps)

    domain_counts: Counter[str] = Counter()
    for nid in whitened:
        dom = node_domains.get(nid)
        if dom:
            domain_counts[dom] += 1

    normalized_bridges: dict[tuple[str, str], int] = {}
    for (da, db), cnt in (bridge_edge_counts or {}).items():
        pair = (da, db) if da <= db else (db, da)
        normalized_bridges[pair] = normalized_bridges.get(pair, 0) + int(cnt)

    domains = sorted(centroids.keys())
    candidates: list[SynthesisCandidate] = []
    for i, d_i in enumerate(domains):
        q_i = float(domain_rigor.get(d_i, 0.0))
        if q_i < min_rigor:
            continue
        m_i = macro_of(d_i)
        for d_j in domains[i + 1 :]:
            if macro_of(d_j) == m_i:
                continue
            q_j = float(domain_rigor.get(d_j, 0.0))
            if q_j < min_rigor:
                continue

            iso_cos = max(0.0, float(np.dot(centroids[d_i], centroids[d_j])))
            pair = (d_i, d_j)
            bridges = normalized_bridges.get(pair, 0)
            denom = 1.0 + float(min(domain_counts[d_i], domain_counts[d_j]))
            unbridged = max(0.0, 1.0 - (bridges / denom))
            rigor = math.sqrt(max(0.0, q_i) * max(0.0, q_j))
            score = iso_cos * unbridged * rigor

            if score >= min_score:
                candidates.append(
                    SynthesisCandidate(
                        domain_a=d_i,
                        domain_b=d_j,
                        score=score,
                        isotropic_cosine=iso_cos,
                        unbridged_factor=unbridged,
                        rigor_factor=rigor,
                        bridge_edges=bridges,
                    )
                )

    candidates.sort(key=lambda c: (-round(c.score, 9), c.domain_a, c.domain_b))
    return candidates


# ---------------------------------------------------------------------------
# Section 4.1: Deterministic Ward Clustering, Silhouette & Sub-Branch Splits
# ---------------------------------------------------------------------------


def ward_cluster(vectors: np.ndarray, k: int) -> np.ndarray:
    """
    Deterministic Ward's minimum-variance agglomerative clustering in pure NumPy.

    Returns integer cluster labels in `0..k-1`, ordered deterministically by earliest sample index.
    """
    pts = np.asarray(vectors, dtype=np.float64)
    if pts.ndim != 2 or pts.shape[0] == 0:
        raise ValueError("vectors must be a non-empty 2D array")
    n = pts.shape[0]
    if not (1 <= k <= n):
        raise ValueError(f"k={k} must satisfy 1 <= k <= {n}")
    if k == n:
        return np.arange(n, dtype=np.int64)
    if k == 1:
        return np.zeros(n, dtype=np.int64)

    active: dict[int, tuple[list[int], np.ndarray]] = {
        i: ([i], pts[i].copy()) for i in range(n)
    }
    diff = pts[:, None, :] - pts[None, :, :]
    sq_dists = np.sum(diff * diff, axis=2)
    cost = 0.5 * sq_dists
    np.fill_diagonal(cost, np.inf)

    active_ids = list(range(n))
    while len(active_ids) > k:
        sub = cost[np.ix_(active_ids, active_ids)]
        flat_idx = int(np.argmin(sub))
        r_pos, c_pos = divmod(flat_idx, len(active_ids))
        id_a = active_ids[r_pos]
        id_b = active_ids[c_pos]
        if id_a > id_b:
            id_a, id_b = id_b, id_a

        mem_a, cent_a = active[id_a]
        mem_b, cent_b = active.pop(id_b)
        n_a = len(mem_a)
        n_b = len(mem_b)
        new_mem = mem_a + mem_b
        new_cent = (n_a * cent_a + n_b * cent_b) / (n_a + n_b)
        active[id_a] = (new_mem, new_cent)
        active_ids.remove(id_b)

        for oid in active_ids:
            if oid == id_a:
                cost[id_a, id_a] = np.inf
                continue
            o_mem, o_cent = active[oid]
            n_o = len(o_mem)
            n_m = len(new_mem)
            d_sq = float(np.sum((new_cent - o_cent) ** 2))
            w_cost = ((n_m * n_o) / (n_m + n_o)) * d_sq
            cost[id_a, oid] = w_cost
            cost[oid, id_a] = w_cost

    ordered_clusters = sorted(active.values(), key=lambda entry: min(entry[0]))
    labels = np.empty(n, dtype=np.int64)
    for cluster_idx, (members, _) in enumerate(ordered_clusters):
        for m in members:
            labels[m] = cluster_idx
    return labels


def compute_silhouette_score(vectors: np.ndarray, labels: np.ndarray) -> float:
    """Compute mean Euclidean silhouette score in pure NumPy."""
    pts = np.asarray(vectors, dtype=np.float64)
    lbls = np.asarray(labels, dtype=np.int64)
    n = pts.shape[0]
    unique_lbls = np.unique(lbls)
    if n < 2 or len(unique_lbls) < 2 or len(unique_lbls) >= n:
        return 0.0

    diff = pts[:, None, :] - pts[None, :, :]
    dists = np.sqrt(np.maximum(0.0, np.sum(diff * diff, axis=2)))
    cluster_indices = {int(c): np.where(lbls == c)[0] for c in unique_lbls}

    scores = np.zeros(n, dtype=np.float64)
    for i in range(n):
        c_i = int(lbls[i])
        same = cluster_indices[c_i]
        if len(same) <= 1:
            scores[i] = 0.0
            continue
        a_i = float(dists[i, same].sum() / (len(same) - 1))
        b_i = math.inf
        for c_other, other_idx in cluster_indices.items():
            if c_other == c_i:
                continue
            mean_other = float(dists[i, other_idx].mean())
            if mean_other < b_i:
                b_i = mean_other
        denom = max(a_i, b_i)
        scores[i] = (b_i - a_i) / denom if denom > 1e-12 else 0.0
    return float(scores.mean())


def _derive_cluster_slugs(
    cluster_members: Mapping[int, list[str]],
    node_signatures: Mapping[str, Sequence[str]] | None,
) -> dict[int, str]:
    """Derive deterministic taxonomy-safe lowercase slugs for split sub-branches."""
    used: set[str] = set()
    slugs: dict[int, str] = {}
    all_tag_counts: Counter[str] = Counter()
    if node_signatures:
        for members in cluster_members.values():
            for nid in members:
                for raw_tag in node_signatures.get(nid, ()):
                    token = raw_tag.split(":", 1)[-1].lower().replace("_", "-")
                    token = _SLUG_CLEAN_RE.sub("-", token).strip("-")
                    if token:
                        all_tag_counts[token] += 1

    for cid in sorted(cluster_members.keys()):
        members = cluster_members[cid]
        local_counts: Counter[str] = Counter()
        if node_signatures:
            for nid in members:
                for raw_tag in node_signatures.get(nid, ()):
                    token = raw_tag.split(":", 1)[-1].lower().replace("_", "-")
                    token = _SLUG_CLEAN_RE.sub("-", token).strip("-")
                    if token:
                        local_counts[token] += 1

        chosen: str | None = None
        if local_counts:
            ranked = sorted(
                local_counts.keys(),
                key=lambda tok: (
                    -(local_counts[tok] / max(1, all_tag_counts[tok])),
                    -local_counts[tok],
                    tok,
                ),
            )
            for candidate in ranked:
                if candidate not in used:
                    chosen = candidate
                    break
        if not chosen:
            chosen = f"subbranch-{cid + 1}"
            while chosen in used:
                chosen = f"{chosen}-x"
        used.add(chosen)
        slugs[cid] = chosen
    return slugs


def plan_leaf_split(
    parent_domain: str,
    node_ids: Sequence[str],
    embeddings: Mapping[str, Sequence[float]],
    node_paths: Mapping[str, Path],
    node_signatures: Mapping[str, Sequence[str]] | None = None,
    *,
    embed_space_consistent: bool = True,
    params: Mapping[str, Any] | None = None,
) -> LeafSplitPlan | None:
    """
    Evaluate deterministic Ward clustering over node embeddings when `n_d > 40` and
    `embed_space_consistent` is True (Section 4.1). Returns None if split gates are unmet.
    """
    if not embed_space_consistent:
        return None
    split_cfg = (params or {}).get("leaf_splitting", {})
    max_nodes = int(split_cfg.get("max_nodes_per_leaf", 40))
    min_k = int(split_cfg.get("min_clusters", 2))
    max_k = int(split_cfg.get("max_clusters", 5))
    min_sil = float(split_cfg.get("min_silhouette_score", 0.25))

    ordered_ids = sorted(node_ids)
    if len(ordered_ids) <= max_nodes:
        return None
    if not all(nid in embeddings and nid in node_paths for nid in ordered_ids):
        return None

    mat = np.asarray([embeddings[nid] for nid in ordered_ids], dtype=np.float64)
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    unit_mat = mat / np.maximum(norms, 1e-8)

    upper_k = min(max_k, len(ordered_ids) - 1)
    if upper_k < min_k:
        return None

    best_k: int | None = None
    best_sil = -1.0
    best_labels: np.ndarray | None = None
    for k in range(min_k, upper_k + 1):
        labels = ward_cluster(unit_mat, k)
        sil = compute_silhouette_score(unit_mat, labels)
        if best_k is None or sil > best_sil + 1e-12:
            best_k = k
            best_sil = sil
            best_labels = labels

    if best_k is None or best_labels is None or best_sil <= min_sil:
        return None

    cluster_members: dict[int, list[str]] = defaultdict(list)
    for idx, nid in enumerate(ordered_ids):
        cluster_members[int(best_labels[idx])].append(nid)

    slugs = _derive_cluster_slugs(cluster_members, node_signatures)
    assignments: dict[str, str] = {}
    moves: list[tuple[Path, Path]] = []
    for cid, members in sorted(cluster_members.items()):
        new_domain = f"{parent_domain}.{slugs[cid]}"
        for nid in members:
            src_path = Path(node_paths[nid])
            dst_path = src_path.parent / slugs[cid] / src_path.name
            assignments[nid] = new_domain
            moves.append((src_path, dst_path))

    return LeafSplitPlan(
        parent_domain=parent_domain,
        k=best_k,
        silhouette_score=round(best_sil, 6),
        assignments=assignments,
        moves=tuple(moves),
    )


def execute_leaf_split(
    repo_root: Path,
    split_plan: LeafSplitPlan,
    *,
    use_git_mv: bool = True,
) -> list[tuple[Path, Path]]:
    """
    Execute a `LeafSplitPlan` via `git mv` (when tracked) or `shutil.move`, update `taxonomy_code`,
    and validate every relocated node via `ci.validate_node.validate_node_file` (zero broken links).
    """
    root = Path(repo_root).resolve()
    allowlist_path = root / "config" / "license_allowlist.yaml"
    allowlist = allowlist_path if allowlist_path.exists() else None
    is_git_repo = (root / ".git").exists()

    executed: list[tuple[Path, Path]] = []
    for src, dst in split_plan.moves:
        src_abs = Path(src).resolve()
        dst_abs = Path(dst).resolve()
        dst_abs.parent.mkdir(parents=True, exist_ok=True)

        moved_with_git = False
        if use_git_repo_cmd(use_git_mv, is_git_repo, root, src_abs):
            proc = subprocess.run(
                [
                    "git",
                    "-C",
                    str(root),
                    "mv",
                    src_abs.relative_to(root).as_posix(),
                    dst_abs.relative_to(root).as_posix(),
                ],
                capture_output=True,
                text=True,
                check=False,
            )
            moved_with_git = proc.returncode == 0

        if not moved_with_git and src_abs != dst_abs:
            shutil.move(str(src_abs), str(dst_abs))

        raw_text = normalize_newlines(dst_abs.read_text(encoding="utf-8"))
        fm_dict, body = split_frontmatter(raw_text)
        ral_id = str(fm_dict["ral_id"])
        if ral_id in split_plan.assignments:
            fm_dict["taxonomy_code"] = split_plan.assignments[ral_id]
            updated_md = f"---\n{yaml.safe_dump(fm_dict, sort_keys=False).strip()}\n---\n{body.lstrip()}"
            dst_abs.write_text(updated_md, encoding="utf-8")

        res = validate_node_file(dst_abs, allowlist_path=allowlist, check_filename=True)
        if not res.ok:
            raise ValueError(f"Split relocated node {dst_abs} failed validation: {res.errors}")
        executed.append((src_abs, dst_abs))
    return executed


def use_git_repo_cmd(
    use_git_mv: bool,
    is_git_repo: bool,
    root: Path,
    src_abs: Path,
) -> bool:
    if not (use_git_mv and is_git_repo):
        return False
    try:
        rel = src_abs.relative_to(root).as_posix()
    except ValueError:
        return False
    probe = subprocess.run(
        ["git", "-C", str(root), "ls-files", "--error-unmatch", rel],
        capture_output=True,
        text=True,
        check=False,
    )
    return probe.returncode == 0


# ---------------------------------------------------------------------------
# Section 4.5: Continuous Portfolio Allocation & Hamilton Epoch Allocator
# ---------------------------------------------------------------------------


def compute_coverage_entropy(macro_counts: Mapping[str, int]) -> float:
    """
    Compute normalized Shannon entropy `H_cov(t)` in `[0, 1]` across the 7 macro-disciplines.
    Returns `0.0` when `|V_t| == 0`.
    """
    counts = [max(0, int(macro_counts.get(m, 0))) for m in MACRO_DISCIPLINES]
    total = sum(counts)
    if total <= 0:
        return 0.0
    entropy = 0.0
    for c in counts:
        if c > 0:
            p = c / total
            entropy -= p * math.log(p)
    max_entropy = math.log(len(MACRO_DISCIPLINES))
    return max(0.0, min(1.0, entropy / max_entropy))


def compute_portfolio_weights(
    *,
    total_verified_nodes: int,
    macro_counts: Mapping[str, int],
    total_sigma: float,
    global_oracle_pass_rate: float,
    expired_t3_count: int = 0,
    epoch_upgraded: bool = False,
    embed_space_consistent: bool = False,
    params: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """
    Compute continuous portfolio weights `[w_exploit, w_explore, w_audit, w_synth]` (Section 4.5).
    """
    pa = (params or {}).get("portfolio_allocation", {})
    explore_base = float(pa.get("explore_base", 0.15))
    explore_ent_w = float(pa.get("explore_entropy_weight", 0.45))
    exploit_base = float(pa.get("exploit_base", 0.25))
    exploit_debt_w = float(pa.get("exploit_debt_weight", 0.30))
    exploit_min_nodes = int(pa.get("exploit_min_nodes", 10))
    audit_base = float(pa.get("audit_base", 0.15))
    audit_trig_w = float(pa.get("audit_trigger_weight", 0.20))
    audit_t3_thresh = int(pa.get("audit_expired_t3_threshold", 20))
    audit_min_nodes = int(pa.get("audit_min_nodes", 25))
    synth_w_const = float(pa.get("synth_weight", 0.25))
    debt_node_factor = float(pa.get("debt_denom_node_factor", 0.25))
    debt_offset = float(pa.get("debt_denom_offset", 10.0))

    synth_cfg = (params or {}).get("synthesis", {})
    synth_min_nodes = int(synth_cfg.get("min_vault_nodes", 150))

    n_nodes = max(0, int(total_verified_nodes))
    h_cov = compute_coverage_entropy(macro_counts) if n_nodes > 0 else 0.0
    debt_denom = debt_node_factor * n_nodes + debt_offset
    d_debt = min(1.0, max(0.0, float(total_sigma)) / debt_denom) if debt_denom > 0.0 else 0.0
    q_bar = max(0.0, min(1.0, float(global_oracle_pass_rate)))

    w_explore = explore_base + explore_ent_w * (1.0 - h_cov)
    w_exploit = (exploit_base + exploit_debt_w * d_debt) if n_nodes >= exploit_min_nodes else 0.0
    audit_triggered = bool(epoch_upgraded) or (int(expired_t3_count) > audit_t3_thresh)
    w_audit = (audit_base + (audit_trig_w if audit_triggered else 0.0)) if n_nodes >= audit_min_nodes else 0.0
    w_synth = (
        synth_w_const * h_cov * q_bar
        if (n_nodes >= synth_min_nodes and embed_space_consistent)
        else 0.0
    )

    return {
        "exploit": w_exploit,
        "explore": w_explore,
        "audit": w_audit,
        "synth": w_synth,
    }


def allocate_epochs(weights: dict[str, float], available_epochs: int) -> dict[str, int]:
    """
    Deterministically partition `available_epochs` (0 <= K <= 6) across modes.
    Zero-division safe when K == 0 or all weights degenerate to 0.
    """
    modes = ["exploit", "explore", "audit", "synth"]
    if available_epochs <= 0:
        return {m: 0 for m in modes}

    clean = {m: max(0.0, float(weights.get(m, 0.0))) for m in modes}
    total_w = sum(clean.values())
    if total_w <= 1e-12:
        clean = {"explore": 1.0, "exploit": 0.0, "audit": 0.0, "synth": 0.0}
        total_w = 1.0

    exact = {m: (clean[m] / total_w) * available_epochs for m in modes}
    floors = {m: math.floor(exact[m]) for m in modes}
    remainder = available_epochs - sum(floors.values())
    tie_order = sorted(modes, key=lambda m: (-(exact[m] - floors[m]), modes.index(m)))
    for m in tie_order[:remainder]:
        floors[m] += 1

    assert sum(floors.values()) == available_epochs
    return floors


# ---------------------------------------------------------------------------
# Nightly Vault Scanner & Plan Orchestration
# ---------------------------------------------------------------------------


def _load_ndjson_rows(dir_path: Path) -> list[dict[str, Any]]:
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


def _resolve_seed_meta(
    domain: str,
    seed_leaves: Mapping[str, Mapping[str, Any]],
    default_half_life: float,
) -> tuple[float, float]:
    parts = domain.split(".")
    for end in range(len(parts), 1, -1):
        prefix = ".".join(parts[:end])
        if prefix in seed_leaves:
            meta = seed_leaves[prefix]
            return float(meta["seed_prior"]), float(meta["default_half_life_years"])
    return 1.0, default_half_life


def build_nightly_plan(
    repo_root: Path,
    *,
    run_date_iso: str,
    current_night: int = 0,
    available_epochs: int = 6,
    epoch_upgraded: bool = False,
    circuit_breaker: str | None = None,
    index_db_path: Path | None = None,
    discovered_embed_model_id: str | None = None,
    execute_splits: bool = False,
    params_path: Path | None = None,
    seeds_path: Path | None = None,
) -> dict[str, Any]:
    """
    Scan vault state, ledgers, graph edges, and optional SQLite embeddings to emit a
    deterministic nightly curriculum plan (`plan.json` artifact, never committed to Git).
    """
    root = Path(repo_root)
    params = load_curriculum_params(
        params_path or (root / "config" / "curriculum_params.yaml" if (root / "config" / "curriculum_params.yaml").exists() else None)
    )
    seeds_cfg = load_taxonomy_seeds(
        seeds_path or (root / "config" / "taxonomy_seeds.yaml" if (root / "config" / "taxonomy_seeds.yaml").exists() else None)
    )
    seed_leaves = extract_seed_leaves(seeds_cfg)
    default_hl = float((params.get("foundational_leverage") or {}).get("default_half_life_years", 10.0))

    run_date_obj = date.fromisoformat(run_date_iso[:10])
    vault_dir = root / "vault"

    node_domains: dict[str, str] = {}
    node_paths: dict[str, Path] = {}
    node_signatures: dict[str, tuple[str, ...]] = {}
    verified_counts: Counter[str] = Counter()
    macro_counts: Counter[str] = Counter()
    expired_t3_by_domain: Counter[str] = Counter()
    rigor_scores_by_domain: dict[str, list[float]] = defaultdict(list)
    last_verified_date_by_domain: dict[str, date] = {}
    oracle_passes = 0
    total_nodes = 0
    total_verified_nodes = 0

    if vault_dir.exists():
        for md_path in sorted(vault_dir.rglob("*.md")):
            raw = normalize_newlines(md_path.read_text(encoding="utf-8"))
            try:
                fm, _ = split_frontmatter(raw)
            except Exception:
                continue
            ral_id = str(fm.get("ral_id") or "").strip()
            dom = str(fm.get("taxonomy_code") or "").strip()
            tier = str(fm.get("epistemic_tier") or "").strip()
            if not ral_id or not dom:
                continue
            total_nodes += 1
            node_domains[ral_id] = dom
            node_paths[ral_id] = md_path
            node_signatures[ral_id] = tuple(str(x) for x in (fm.get("formal_signature") or []))

            ver = fm.get("verification") or {}
            if ver.get("oracle_status") == "PASS":
                oracle_passes += 1

            if tier in ("T1_CANONICAL", "T2_ESTABLISHED"):
                verified_counts[dom] += 1
                macro_counts[macro_of(dom)] += 1
                total_verified_nodes += 1
                rigor_scores_by_domain[dom].append(1.0 if tier == "T1_CANONICAL" else 0.85)
            elif tier == "T3_SPECULATIVE":
                rigor_scores_by_domain[dom].append(0.4)
                re_eval = str(fm.get("re_evaluate_after") or "").strip()
                if re_eval:
                    try:
                        if run_date_obj >= date.fromisoformat(re_eval[:10]):
                            expired_t3_by_domain[dom] += 1
                    except ValueError:
                        pass

            v_at = str(ver.get("verified_at") or "")[:10]
            if v_at:
                try:
                    v_date = date.fromisoformat(v_at)
                    prev_d = last_verified_date_by_domain.get(dom)
                    if prev_d is None or v_date > prev_d:
                        last_verified_date_by_domain[dom] = v_date
                except ValueError:
                    pass

    edges = _load_ndjson_rows(root / "graph" / "edges")
    open_conflicts: Counter[str] = Counter()
    bridge_counts: Counter[tuple[str, str]] = Counter()
    for edge in edges:
        rel = edge.get("relation")
        src = str(edge.get("src") or "")
        dst = str(edge.get("dst") or "")
        d_src = node_domains.get(src)
        d_dst = node_domains.get(dst)
        if rel in ("CONTRADICTS", "CHALLENGES") and d_dst:
            open_conflicts[d_dst] += 1
        elif rel == "ISOMORPHIC_TO" and d_src and d_dst and macro_of(d_src) != macro_of(d_dst):
            pair = (d_src, d_dst) if d_src <= d_dst else (d_dst, d_src)
            bridge_counts[pair] += 1

    stubs = list(load_prereq_debt(root / "ledger" / "prereq_debt.ndjson").values())

    # Trailing 30-day epoch spend from telemetry if present.
    bal_cfg = params.get("discipline_balance", {})
    window_days = int(bal_cfg.get("trailing_window_nights", 30))
    trailing_epochs: Counter[str] = Counter()
    for row in _load_ndjson_rows(root / "metrics" / "telemetry"):
        r_date_str = str(row.get("run_date") or row.get("date") or "")[:10]
        if r_date_str:
            try:
                if (run_date_obj - date.fromisoformat(r_date_str)).days > window_days:
                    continue
            except ValueError:
                pass
        for dom, cnt in (row.get("domain_epochs") or {}).items():
            trailing_epochs[str(dom)] += float(cnt)

    # Determine active leaf domains L(T_t).
    observed_domains = set(node_domains.values())
    active_domains: set[str] = set(observed_domains)
    for seed_code in seed_leaves:
        if seed_code in observed_domains or any(d.startswith(f"{seed_code}.") for d in observed_domains):
            continue
        active_domains.add(seed_code)

    # Optional index embedding load for whitened synthesis & leaf splits.
    embeddings: dict[str, np.ndarray] = {}
    embed_space_consistent = False
    embedding_model_id = ""
    if index_db_path is not None and Path(index_db_path).exists():
        from ral.storage.index_builder import load_node_embeddings

        embeddings, embed_space_consistent, embedding_model_id = load_node_embeddings(
            Path(index_db_path)
        )

    # Dynamic sub-branch splits (Section 4.1) when n_d > 40 & consistent embeddings.
    split_plans: list[dict[str, Any]] = []
    if embed_space_consistent and embeddings:
        nodes_by_dom: dict[str, list[str]] = defaultdict(list)
        for nid, dom in node_domains.items():
            nodes_by_dom[dom].append(nid)
        for dom in sorted(nodes_by_dom.keys()):
            sp_plan = plan_leaf_split(
                dom,
                nodes_by_dom[dom],
                embeddings,
                node_paths,
                node_signatures,
                embed_space_consistent=embed_space_consistent,
                params=params,
            )
            if sp_plan is not None:
                if execute_splits:
                    execute_leaf_split(root, sp_plan)
                split_plans.append(
                    {
                        "parent_domain": sp_plan.parent_domain,
                        "k": sp_plan.k,
                        "silhouette_score": sp_plan.silhouette_score,
                        "moved_count": len(sp_plan.moves),
                    }
                )

    pagerank = compute_cross_domain_pagerank(sorted(active_domains), edges, node_domains)
    domain_states: list[DomainState] = []
    domain_rigor: dict[str, float] = {}
    for dom in sorted(active_domains):
        seed_prior, hl_years = _resolve_seed_meta(dom, seed_leaves, default_hl)
        last_d = last_verified_date_by_domain.get(dom)
        tau_d = max(0, (run_date_obj - last_d).days) if last_d is not None else None
        rigors = rigor_scores_by_domain.get(dom)
        avg_rigor = (sum(rigors) / len(rigors)) if rigors else 0.0
        domain_rigor[dom] = avg_rigor
        domain_states.append(
            DomainState(
                domain=dom,
                macro=macro_of(dom),
                verified_count=verified_counts.get(dom, 0),
                open_conflicts=open_conflicts.get(dom, 0),
                expired_t3=expired_t3_by_domain.get(dom, 0),
                trailing_epochs_30d=trailing_epochs.get(dom, 0.0),
                nights_since_last_work=tau_d,
                ewma_yield=None,
                half_life_years=hl_years,
                seed_prior=seed_prior,
                avg_rigor=avg_rigor,
            )
        )

    priorities = compute_domain_priorities(
        domain_states,
        stubs=stubs,
        current_night=current_night,
        run_date_iso=run_date_iso,
        pagerank_scores=pagerank,
        params=params,
    )

    synth_candidates = (
        score_synthesis_pairs(
            embeddings,
            node_domains,
            domain_rigor,
            bridge_counts,
            params=params,
        )
        if embed_space_consistent and embeddings
        else []
    )

    total_sigma = sum(p.sigma for p in priorities)
    global_pass_rate = (oracle_passes / total_nodes) if total_nodes > 0 else 1.0
    total_expired_t3 = sum(expired_t3_by_domain.values())

    effective_epochs = 0 if circuit_breaker == "DRIVE_AUTH_FAIL" else max(0, int(available_epochs))
    weights = compute_portfolio_weights(
        total_verified_nodes=total_verified_nodes,
        macro_counts=macro_counts,
        total_sigma=total_sigma,
        global_oracle_pass_rate=global_pass_rate,
        expired_t3_count=total_expired_t3,
        epoch_upgraded=epoch_upgraded,
        embed_space_consistent=embed_space_consistent,
        params=params,
    )
    allocation = allocate_epochs(weights, effective_epochs)

    ranked_domains = [p.domain for p in priorities if not p.masked] or [p.domain for p in priorities]
    exploit_domains = [p.domain for p in sorted(priorities, key=lambda x: (-x.sigma, -x.effective_score, x.tie_break_hash)) if not p.masked] or ranked_domains

    epoch_schedule: list[dict[str, Any]] = []
    idx_counter = {"explore": 0, "exploit": 0, "audit": 0, "synth": 0}
    for mode in ("exploit", "explore", "audit", "synth"):
        for _ in range(allocation[mode]):
            slot = idx_counter[mode]
            idx_counter[mode] += 1
            if mode == "synth" and synth_candidates:
                cand = synth_candidates[slot % len(synth_candidates)]
                target: Any = {"domain_a": cand.domain_a, "domain_b": cand.domain_b, "score": round(cand.score, 6)}
            elif mode == "exploit" and exploit_domains:
                target = exploit_domains[slot % len(exploit_domains)]
            else:
                target = ranked_domains[slot % len(ranked_domains)] if ranked_domains else None
            epoch_schedule.append(
                {
                    "epoch_index": len(epoch_schedule) + 1,
                    "mode": mode,
                    "target": target,
                }
            )

    discovered_embed = (discovered_embed_model_id or "").strip()
    index_embed = (embedding_model_id or "").strip()
    plan_embed_model_id = discovered_embed or index_embed
    plan_embedding_model_id = index_embed or discovered_embed

    return {
        "schema_version": "2.1",
        "run_date": run_date_iso,
        "current_night": int(current_night),
        "circuit_breaker": circuit_breaker,
        "total_verified_nodes": total_verified_nodes,
        "embed_space_consistent": embed_space_consistent,
        "embedding_model_id": plan_embedding_model_id,
        "embed_model_id": plan_embed_model_id,
        "portfolio_weights": {k: round(v, 6) for k, v in weights.items()},
        "epoch_allocation": allocation,
        "epoch_schedule": epoch_schedule,
        "top_domains": [asdict(p) for p in priorities[:10]],
        "synthesis_candidates": [asdict(s) for s in synth_candidates[:5]],
        "leaf_splits": split_plans,
    }


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint to generate an ephemeral nightly curriculum `plan.json`."""
    parser = argparse.ArgumentParser(description="Run RAL Director curriculum & epoch planner.")
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).date().isoformat(),
        help="ISO run date YYYY-MM-DD",
    )
    parser.add_argument("--night", type=int, default=0, help="Monotonic night counter")
    parser.add_argument("--epochs", type=int, default=6, help="Available LLM epochs (0..6)")
    parser.add_argument("--epoch-upgraded", action="store_true")
    parser.add_argument("--circuit-breaker", default=None)
    parser.add_argument("--index-db", type=Path, default=None)
    parser.add_argument("--execute-splits", action="store_true")
    parser.add_argument("--output", type=Path, default=None, help="Output path for ephemeral plan.json")
    args = parser.parse_args(argv)

    plan = build_nightly_plan(
        args.repo_root,
        run_date_iso=args.date,
        current_night=args.night,
        available_epochs=args.epochs,
        epoch_upgraded=args.epoch_upgraded,
        circuit_breaker=args.circuit_breaker,
        index_db_path=args.index_db,
        execute_splits=args.execute_splits,
    )
    rendered = json.dumps(plan, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    else:
        print(rendered, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

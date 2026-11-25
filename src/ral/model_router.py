"""Persisted 20-problem capability calibration, epoch ratchet, and cross-family routing (Section 3.2)."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence, TypeVar

import jsonschema

from ral.preflight import (
    LiveModelStatus,
    is_json_mode_unsupported_error,
    parse_json_response,
    supports_json_mime_type,
)

BENCHMARK_PATH = Path("metrics/model_benchmarks.json")
SUITE_PATH = Path("config/calibration_suite.json")

T = TypeVar("T")

_SAFE_MATH_CONSTANTS: dict[str, float] = {
    "pi": math.pi,
    "e": math.e,
}

_SAFE_MATH_FUNCS: dict[str, Callable[..., float]] = {
    "exp": math.exp,
    "log": math.log,
    "ln": math.log,
    "sqrt": math.sqrt,
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sinh": math.sinh,
    "cosh": math.cosh,
    "tanh": math.tanh,
    "coth": lambda x: 1.0 / math.tanh(x),
    "abs": abs,
}


class AllModelsExhaustedError(RuntimeError):
    """Raised when every eligible model for a role has hit HTTP 429 / RPD exhaustion."""


def _safe_eval_ast(node: ast.AST, variables: Mapping[str, float]) -> float:
    if isinstance(node, ast.Expression):
        return _safe_eval_ast(node.body, variables)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) and not isinstance(node.value, bool):
        return float(node.value)
    if isinstance(node, ast.Name):
        if node.id in variables:
            return float(variables[node.id])
        if node.id in _SAFE_MATH_CONSTANTS:
            return _SAFE_MATH_CONSTANTS[node.id]
        raise ValueError(f"Unknown symbol in expression: {node.id!r}")
    if isinstance(node, ast.UnaryOp):
        operand = _safe_eval_ast(node.operand, variables)
        if isinstance(node.op, ast.UAdd):
            return +operand
        if isinstance(node.op, ast.USub):
            return -operand
        raise ValueError(f"Unsupported unary operator: {type(node.op).__name__}")
    if isinstance(node, ast.BinOp):
        left = _safe_eval_ast(node.left, variables)
        right = _safe_eval_ast(node.right, variables)
        if isinstance(node.op, ast.Add):
            return left + right
        if isinstance(node.op, ast.Sub):
            return left - right
        if isinstance(node.op, ast.Mult):
            return left * right
        if isinstance(node.op, ast.Div):
            return left / right
        if isinstance(node.op, ast.Pow):
            return left**right
        raise ValueError(f"Unsupported binary operator: {type(node.op).__name__}")
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        func_name = node.func.id
        if func_name not in _SAFE_MATH_FUNCS:
            raise ValueError(f"Disallowed math function: {func_name!r}")
        if len(node.args) != 1:
            raise ValueError(f"Function {func_name!r} expects exactly 1 argument")
        arg_val = _safe_eval_ast(node.args[0], variables)
        return float(_SAFE_MATH_FUNCS[func_name](arg_val))
    raise ValueError(f"Unsupported AST node in expression: {type(node).__name__}")


def safe_eval_math_expr(expr: str, variables: Mapping[str, float]) -> float:
    """Safely evaluate a scalar SymPy/Python mathematical expression without `eval()`."""
    tree = ast.parse(expr.strip(), mode="eval")
    return _safe_eval_ast(tree, variables)


def _strip_json_Strictly(raw_response: str) -> Any:
    return parse_json_response(raw_response)


def grade_math_problem(problem: Mapping[str, Any], raw_response: str) -> bool:
    """Grade a `sympy_math` calibration problem symbolically (when SymPy available) & numerically."""
    try:
        parsed = _strip_json_Strictly(raw_response)
    except Exception:
        return False
    if not isinstance(parsed, dict):
        return False
    candidate_expr = parsed.get("expr")
    if not isinstance(candidate_expr, str) or not candidate_expr.strip():
        return False

    try:
        import sympy as sp  # type: ignore[import-not-found]

        sym_map: dict[str, Any] = {}
        assumptions = problem.get("assumptions") or {}
        for sym_name in problem.get("symbols", []):
            sym_map[sym_name] = sp.Symbol(sym_name, **assumptions.get(sym_name, {}))
        for const_name, const_val in (("e", sp.E), ("E", sp.E), ("pi", sp.pi)):
            sym_map.setdefault(const_name, const_val)
        cand_sp = sp.sympify(candidate_expr, locals=sym_map)
        exp_sp = sp.sympify(problem["expected_expr"], locals=sym_map)
        if sp.simplify(cand_sp - exp_sp) == 0:
            return True
    except ImportError:
        pass
    except Exception:
        pass

    test_points = problem.get("numeric_test_points", [])
    if not test_points:
        return False
    for point in test_points:
        try:
            val = safe_eval_math_expr(candidate_expr, point["inputs"])
        except Exception:
            return False
        if not math.isfinite(val):
            return False
        if not math.isclose(val, float(point["expected"]), rel_tol=1e-6, abs_tol=1e-9):
            return False
    return True


def grade_trap_problem(problem: Mapping[str, Any], raw_response: str) -> bool:
    """Grade a `counterexample_trap` problem on falsification and missing hypothesis recall."""
    try:
        parsed = _strip_json_Strictly(raw_response)
    except Exception:
        return False
    if not isinstance(parsed, dict):
        return False
    if parsed.get("is_valid") is not bool(problem.get("expected_is_valid", False)):
        return False

    witness = parsed.get("counterexample_witness")
    if not isinstance(witness, dict) or len(witness) == 0:
        return False

    missing = parsed.get("missing_hypothesis")
    if not isinstance(missing, str) or not missing.strip():
        return False
    lowered_missing = missing.lower()
    keywords = problem.get("required_missing_assumption_keywords", [])
    if keywords and not any(str(kw).lower() in lowered_missing for kw in keywords):
        return False
    return True


def grade_json_problem(problem: Mapping[str, Any], raw_response: str) -> bool:
    """Grade a `strict_json` problem via strict `json.loads` and JSON Schema validation."""
    try:
        parsed = _strip_json_Strictly(raw_response)
        jsonschema.validate(instance=parsed, schema=problem["json_schema"])
    except Exception:
        return False
    return True


def is_rate_limit_error(exc: Exception) -> bool:
    """Detect HTTP 429 / Gemini RESOURCE_EXHAUSTED RPD exhaustion errors."""
    for attr in ("status_code", "code"):
        if getattr(exc, attr, None) == 429:
            return True
    msg = str(exc).upper()
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "RATE_LIMIT" in msg


def is_transient_service_error(exc: Exception) -> bool:
    """Detect transient HTTP 503 / model-overloaded errors suitable for bounded backoff."""
    for attr in ("status_code", "code"):
        if getattr(exc, attr, None) == 503:
            return True
    msg = str(exc).upper()
    return any(tok in msg for tok in ("503", "SERVICE_UNAVAILABLE", "UNAVAILABLE", "OVERLOADED"))


def is_hard_quota_exhausted_error(exc: Exception) -> bool:
    """Detect daily or zero-quota exhaustion where retrying cannot succeed."""
    if not is_rate_limit_error(exc):
        return False
    msg = str(exc).lower()
    if any(tok in msg for tok in ("limit: 0", "limit:0", "perday", "per_day", "daily", "per day")):
        return True
    return re.search(r"\brpd\b", msg) is not None


def _diagnose_grade_failure(cat: str, problem: Mapping[str, Any], raw_response: str) -> str:
    """Produce a concise diagnostic reason when a parsed calibration response fails grading."""
    try:
        parsed = parse_json_response(raw_response)
    except Exception as exc:
        return f"JSON_PARSE_ERROR: {type(exc).__name__}: {str(exc)[:120]}"
    if cat == "strict_json":
        try:
            jsonschema.validate(instance=parsed, schema=problem["json_schema"])
        except jsonschema.ValidationError as exc:
            return f"SCHEMA_MISMATCH: {exc.message[:160]}"
        except Exception as exc:
            return f"SCHEMA_ERROR: {type(exc).__name__}: {str(exc)[:120]}"
        return "SCHEMA_MISMATCH"
    if cat == "sympy_math":
        return "MATH_GRADE_MISMATCH"
    if cat == "counterexample_trap":
        return "TRAP_GRADE_MISMATCH"
    return f"GRADE_MISMATCH: {cat}"


def _record_sample_error(sample_errors: list[str], error: str | None) -> None:
    if not error or len(sample_errors) >= 3:
        return
    collapsed = " ".join(error.split())
    if len(collapsed) > 200:
        collapsed = collapsed[:197] + "..."
    if collapsed not in sample_errors:
        sample_errors.append(collapsed)


def run_local_calibration_suite(
    client: Any,
    model_id: str,
    problems: Sequence[Mapping[str, Any]],
    scoring: Mapping[str, Any] | None = None,
    *,
    max_429_retries: int = 1,
    retry_backoff_s: float = 0.0,
    sleep_fn: Callable[[float], None] | None = None,
    deadline_monotonic: float | None = None,
    monotonic_fn: Callable[[], float] | None = None,
) -> dict[str, Any]:
    """
    Execute the deterministic 20-problem calibration suite once for a newly discovered `model_id`.
    """
    graders: dict[str, Callable[[Mapping[str, Any], str], bool]] = {
        "sympy_math": grade_math_problem,
        "counterexample_trap": grade_trap_problem,
        "strict_json": grade_json_problem,
    }
    counts: dict[str, int] = {"sympy_math": 0, "counterexample_trap": 0, "strict_json": 0}
    passes: dict[str, int] = {"sympy_math": 0, "counterexample_trap": 0, "strict_json": 0}
    per_problem: list[dict[str, Any]] = []

    allow_json_mime = supports_json_mime_type(model_id)
    allow_json_schema = allow_json_mime
    sleep = sleep_fn or time.sleep
    mono = monotonic_fn or time.monotonic
    retries = max(0, int(max_429_retries))
    api_errors = 0
    rate_limit_errors = 0
    consecutive_rate_limits = 0
    aborted_reason: str | None = None
    sample_errors: list[str] = []

    for problem in problems:
        if deadline_monotonic is not None and mono() >= deadline_monotonic:
            aborted_reason = "CALIBRATION_TIME_BUDGET_EXCEEDED"
            break
        cat = str(problem["category"])
        counts[cat] = counts.get(cat, 0) + 1
        passed = False
        error: str | None = None
        prompt = problem["prompt"]
        if cat == "strict_json":
            prompt = f"{prompt}\n\nJSON Schema:\n{json.dumps(problem['json_schema'])}"

        raw_text: str | None = None
        abort_loop = False
        for attempt in range(retries + 1):
            try:
                while True:
                    cfg: dict[str, Any] = {"temperature": 0.0}
                    if allow_json_mime:
                        cfg["response_mime_type"] = "application/json"
                        if cat == "strict_json" and allow_json_schema:
                            cfg["response_json_schema"] = dict(problem["json_schema"])
                    try:
                        resp = client.models.generate_content(
                            model=model_id,
                            contents=prompt,
                            config=cfg,
                        )
                        raw_text = (getattr(resp, "text", None) or "").strip()
                        break
                    except Exception as inner_exc:
                        if not is_json_mode_unsupported_error(inner_exc):
                            raise
                        if "response_json_schema" in cfg and allow_json_schema:
                            allow_json_schema = False
                            continue
                        if "response_mime_type" in cfg and allow_json_mime:
                            allow_json_mime = False
                            continue
                        raise
                break
            except Exception as exc:
                is_rl = is_rate_limit_error(exc)
                is_transient = is_transient_service_error(exc)
                if is_rl or is_transient:
                    if is_rl and is_hard_quota_exhausted_error(exc):
                        error = f"{type(exc).__name__}: {exc}"
                        api_errors += 1
                        rate_limit_errors += 1
                        aborted_reason = "CALIBRATION_QUOTA_EXHAUSTED"
                        abort_loop = True
                        break
                    if attempt < retries:
                        if deadline_monotonic is not None and (mono() + retry_backoff_s) >= deadline_monotonic:
                            error = f"{type(exc).__name__}: {exc}"
                            api_errors += 1
                            if is_rl:
                                rate_limit_errors += 1
                            aborted_reason = "CALIBRATION_TIME_BUDGET_EXCEEDED"
                            abort_loop = True
                            break
                        if retry_backoff_s > 0:
                            sleep(retry_backoff_s)
                        continue
                    error = f"{type(exc).__name__}: {exc}"
                    api_errors += 1
                    if is_rl:
                        rate_limit_errors += 1
                    consecutive_rate_limits += 1
                    if consecutive_rate_limits >= 2:
                        aborted_reason = "CALIBRATION_RATE_LIMITED"
                        abort_loop = True
                    break
                error = f"{type(exc).__name__}: {exc}"
                api_errors += 1
                consecutive_rate_limits = 0
                break

        if raw_text is not None:
            consecutive_rate_limits = 0
            grader = graders.get(cat)
            if grader is None:
                error = f"ValueError: Unsupported calibration category: {cat!r}"
            else:
                passed = grader(problem, raw_text)
                if not passed:
                    error = _diagnose_grade_failure(cat, problem, raw_text)

        if passed:
            passes[cat] = passes.get(cat, 0) + 1
        else:
            _record_sample_error(sample_errors, error)
        per_problem.append({"id": problem["id"], "category": cat, "passed": passed, "error": error})
        if abort_loop:
            break

    math_score = passes["sympy_math"] / counts["sympy_math"] if counts["sympy_math"] else 0.0
    skeptic_score = (
        passes["counterexample_trap"] / counts["counterexample_trap"]
        if counts["counterexample_trap"]
        else 0.0
    )
    json_reliability = passes["strict_json"] / counts["strict_json"] if counts["strict_json"] else 0.0

    w_math = float((scoring or {}).get("sympy_math", {}).get("weight", 0.40))
    w_skep = float((scoring or {}).get("counterexample_trap", {}).get("weight", 0.30))
    w_json = float((scoring or {}).get("strict_json", {}).get("weight", 0.30))
    w_total = w_math + w_skep + w_json
    overall = (
        (w_math * math_score + w_skep * skeptic_score + w_json * json_reliability) / w_total
        if w_total > 0
        else 0.0
    )

    return {
        "overall_score": round(overall, 6),
        "math_score": round(math_score, 6),
        "skeptic_score": round(skeptic_score, 6),
        "json_reliability": round(json_reliability, 6),
        "total_passed": sum(passes.values()),
        "total_problems": len(problems),
        "api_errors": api_errors,
        "rate_limit_errors": rate_limit_errors,
        "aborted_reason": aborted_reason,
        "strict_json_passed": passes["strict_json"],
        "strict_json_total": counts["strict_json"],
        "sample_errors": sample_errors,
        "results": per_problem,
    }


def _normalize_live_entry(entry: Mapping[str, Any] | LiveModelStatus) -> dict[str, Any]:
    if isinstance(entry, LiveModelStatus):
        return entry.to_dict()
    return dict(entry)


def _interleave_by_family(entries: Sequence[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin candidate models across distinct families so capped cold starts stay cross-family."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for entry in entries:
        fam = str(entry.get("family") or entry["model_id"])
        buckets.setdefault(fam, []).append(entry)
    ordered: list[dict[str, Any]] = []
    family_keys = list(buckets.keys())
    idx = 0
    while True:
        progressed = False
        for fam in family_keys:
            bucket = buckets[fam]
            if idx < len(bucket):
                ordered.append(bucket[idx])
                progressed = True
        if not progressed:
            break
        idx += 1
    return ordered


def _write_benchmarks_db(bm_path: Path, db: Mapping[str, Any]) -> None:
    bm_path.parent.mkdir(parents=True, exist_ok=True)
    bm_path.write_text(json.dumps(dict(db), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def ensure_benchmarked(
    client: Any,
    live_models: Sequence[Mapping[str, Any] | LiveModelStatus],
    policy: Mapping[str, Any],
    *,
    benchmark_path: Path = BENCHMARK_PATH,
    suite_path: Path = SUITE_PATH,
    diagnostics_out: dict[str, dict[str, Any]] | None = None,
    retry_backoff_s: float = 0.0,
    max_calibration_seconds: float | None = None,
    max_new_models_per_run: int | None = 3,
    monotonic_fn: Callable[[], float] | None = None,
) -> tuple[dict[str, Any], bool]:
    """
    Load cached benchmark scores from `metrics/model_benchmarks.json`.

    Runs the deterministic 20-problem calibration suite ONLY for newly discovered live `model_id`s
    (or if `suite_version` changed), capped to `max_new_models_per_run` and `max_calibration_seconds`.
    Returns `(benchmarks_db, capability_epoch_upgraded)`.
    """
    mono = monotonic_fn or time.monotonic
    t_start = mono()
    deadline_monotonic = (
        t_start + float(max_calibration_seconds)
        if max_calibration_seconds is not None
        else None
    )
    suite = json.loads(Path(suite_path).read_text(encoding="utf-8"))
    suite_ver = suite["suite_version"]
    bm_path = Path(benchmark_path)
    db: dict[str, Any] = (
        json.loads(bm_path.read_text(encoding="utf-8"))
        if bm_path.exists()
        else {
            "model_capability_epoch": 1,
            "historical_max_score": 0.0,
            "suite_version": suite_ver,
            "models": {},
        }
    )

    changed = False
    if db.get("suite_version") != suite_ver:
        db["models"] = {}
        db["suite_version"] = suite_ver
        changed = True

    normalized_live = [_normalize_live_entry(m) for m in live_models]
    unbenchmarked_live: list[dict[str, Any]] = []
    for m in normalized_live:
        mid = m["model_id"]
        if not (m.get("live") and m.get("json_ok")):
            if diagnostics_out is not None:
                diagnostics_out[mid] = {
                    "stage": "liveness_probe",
                    "status": "REJECTED",
                    "reason": "LIVENESS_PROBE_FAILED" if not m.get("live") else "LIVENESS_JSON_INVALID",
                    "detail": m.get("error"),
                    "live": bool(m.get("live")),
                    "json_ok": bool(m.get("json_ok")),
                    "family": m.get("family"),
                }
            continue
        if mid in db["models"]:
            if diagnostics_out is not None:
                rec = db["models"][mid]
                diagnostics_out[mid] = {
                    "stage": "calibration",
                    "status": "BENCHMARKED",
                    "cached": True,
                    "overall_score": rec["overall_score"],
                    "math_score": rec["math_score"],
                    "skeptic_score": rec["skeptic_score"],
                    "json_reliability": rec["json_reliability"],
                    "family": rec.get("family", m.get("family")),
                }
            continue
        unbenchmarked_live.append(m)

    newly_calibrated = 0
    for m in _interleave_by_family(unbenchmarked_live):
        mid = m["model_id"]
        if max_new_models_per_run is not None and newly_calibrated >= int(max_new_models_per_run):
            if diagnostics_out is not None:
                diagnostics_out[mid] = {
                    "stage": "calibration",
                    "status": "REJECTED",
                    "reason": "CALIBRATION_DEFERRED_MODEL_CAP",
                    "detail": f"Deferred beyond max_new_models_per_run={int(max_new_models_per_run)} cap",
                    "family": m.get("family"),
                }
            continue
        if deadline_monotonic is not None and mono() >= deadline_monotonic:
            if diagnostics_out is not None:
                diagnostics_out[mid] = {
                    "stage": "calibration",
                    "status": "REJECTED",
                    "reason": "CALIBRATION_DEFERRED_TIME_BUDGET",
                    "detail": f"Exceeded {float(max_calibration_seconds):.1f}s calibration budget",
                    "family": m.get("family"),
                }
            continue

        res = run_local_calibration_suite(
            client,
            mid,
            suite["problems"],
            scoring=suite.get("scoring"),
            retry_backoff_s=retry_backoff_s,
            deadline_monotonic=deadline_monotonic,
            monotonic_fn=mono,
        )
        calibration_failed = (
            res.get("aborted_reason") is not None
            or int(res.get("rate_limit_errors", 0)) > 0
            or (
                int(res.get("total_problems", 0)) > 0
                and int(res.get("api_errors", 0)) == int(res.get("total_problems", 0))
            )
        )
        if calibration_failed:
            reason = res.get("aborted_reason") or (
                "CALIBRATION_RATE_LIMITED"
                if int(res.get("rate_limit_errors", 0)) > 0
                else "CALIBRATION_API_ERROR"
            )
            if diagnostics_out is not None:
                diagnostics_out[mid] = {
                    "stage": "calibration",
                    "status": "REJECTED",
                    "reason": reason,
                    "api_errors": int(res.get("api_errors", 0)),
                    "rate_limit_errors": int(res.get("rate_limit_errors", 0)),
                    "sample_errors": list(res.get("sample_errors") or []),
                    "family": m.get("family"),
                }
            continue

        db["models"][mid] = {
            "overall_score": res["overall_score"],
            "math_score": res["math_score"],
            "skeptic_score": res["skeptic_score"],
            "json_reliability": res["json_reliability"],
            "family": m["family"],
            "benchmarked_at": datetime.now(timezone.utc).isoformat(),
        }
        newly_calibrated += 1
        changed = True
        _write_benchmarks_db(bm_path, db)
        if diagnostics_out is not None:
            diagnostics_out[mid] = {
                "stage": "calibration",
                "status": "BENCHMARKED",
                "cached": False,
                "overall_score": res["overall_score"],
                "math_score": res["math_score"],
                "skeptic_score": res["skeptic_score"],
                "json_reliability": res["json_reliability"],
                "strict_json_passed": int(res.get("strict_json_passed", 0)),
                "strict_json_total": int(res.get("strict_json_total", 0)),
                "sample_errors": list(res.get("sample_errors") or []),
                "family": m.get("family"),
            }

    live_scores = [
        db["models"][m["model_id"]]["overall_score"]
        for m in normalized_live
        if m.get("live") and m.get("json_ok") and m["model_id"] in db["models"]
    ]
    epoch_upgraded = False
    if live_scores:
        best_live = max(live_scores)
        if db["historical_max_score"] == 0.0:
            db["historical_max_score"] = best_live
            changed = True
        elif round(best_live - float(db["historical_max_score"]), 9) >= round(
            float(policy["epoch_upgrade_delta"]), 9
        ):
            db["model_capability_epoch"] = int(db["model_capability_epoch"]) + 1
            db["historical_max_score"] = best_live
            epoch_upgraded = True
            changed = True

    if changed:
        _write_benchmarks_db(bm_path, db)
    return db, epoch_upgraded


def build_routing_table(
    live_models: Sequence[Mapping[str, Any] | LiveModelStatus],
    db: Mapping[str, Any],
    policy: Mapping[str, Any],
    *,
    calibration_diagnostics: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """
    Assign ranked fallback chains for each swarm role using persisted benchmark scores.

    Enforces orthogonal model-family diversity between Formalizer and Skeptic when available.
    """
    eligible: list[dict[str, Any]] = []
    rejected_models: dict[str, dict[str, Any]] = {}
    model_diagnostics: dict[str, dict[str, Any]] = {}
    min_json_rel = float(policy["min_json_reliability"])
    min_formalizer_score = float(policy.get("min_formalizer_score", 0.55))
    models_db = db.get("models", {})

    for raw_m in live_models:
        m = _normalize_live_entry(raw_m)
        mid = m["model_id"]
        cal_diag = dict((calibration_diagnostics or {}).get(mid) or {})

        if not m.get("live"):
            entry: dict[str, Any] = {
                "stage": "liveness_probe",
                "status": "REJECTED",
                "reason": "LIVENESS_PROBE_FAILED",
                "detail": m.get("error") or "probe returned live=False",
                "family": m.get("family"),
                "latency_ms": m.get("latency_ms"),
            }
            rejected_models[mid] = entry
            model_diagnostics[mid] = entry
            continue

        if not m.get("json_ok"):
            entry = {
                "stage": "liveness_probe",
                "status": "REJECTED",
                "reason": "LIVENESS_JSON_INVALID",
                "detail": m.get("error") or 'liveness response was not {"ok": true}',
                "family": m.get("family"),
                "latency_ms": m.get("latency_ms"),
            }
            rejected_models[mid] = entry
            model_diagnostics[mid] = entry
            continue

        if mid not in models_db:
            sample_errs = list(cal_diag.get("sample_errors") or [])
            detail = cal_diag.get("detail") or (
                "; ".join(sample_errs) if sample_errs else "Model absent from benchmark database"
            )
            entry = {
                "stage": "calibration",
                "status": "REJECTED",
                "reason": cal_diag.get("reason") or "NOT_BENCHMARKED",
                "detail": detail,
                "family": m.get("family"),
                "latency_ms": m.get("latency_ms"),
            }
            for key in ("api_errors", "rate_limit_errors", "sample_errors"):
                if key in cal_diag:
                    entry[key] = cal_diag[key]
            rejected_models[mid] = entry
            model_diagnostics[mid] = entry
            continue

        rec = models_db[mid]
        if float(rec["json_reliability"]) < min_json_rel:
            entry = {
                "stage": "routing",
                "status": "REJECTED",
                "reason": "BELOW_MIN_JSON_RELIABILITY",
                "detail": (
                    f"json_reliability={rec['json_reliability']} "
                    f"< min_json_reliability={policy['min_json_reliability']}"
                ),
                "family": rec.get("family", m.get("family")),
                "latency_ms": m.get("latency_ms"),
                "overall_score": rec["overall_score"],
                "math_score": rec["math_score"],
                "skeptic_score": rec["skeptic_score"],
                "json_reliability": rec["json_reliability"],
            }
            if cal_diag.get("sample_errors"):
                entry["sample_errors"] = list(cal_diag["sample_errors"])
            rejected_models[mid] = entry
            model_diagnostics[mid] = entry
            continue

        eligible.append({"model_id": mid, "latency_ms": m["latency_ms"], **rec})
        model_diagnostics[mid] = {
            "stage": "routing",
            "status": "ELIGIBLE",
            "family": rec.get("family", m.get("family")),
            "latency_ms": m.get("latency_ms"),
            "overall_score": rec["overall_score"],
            "math_score": rec["math_score"],
            "skeptic_score": rec["skeptic_score"],
            "json_reliability": rec["json_reliability"],
            "formalizer_eligible": float(rec["overall_score"]) >= min_formalizer_score,
        }

    if not eligible:
        return {
            "status": "ZERO_HEALTHY_MODELS",
            "roles": {},
            "exhausted_models": [],
            "rejected_models": rejected_models,
            "model_diagnostics": model_diagnostics,
        }

    formalizer_pool = [
        x for x in eligible if float(x["overall_score"]) >= min_formalizer_score
    ]
    if not formalizer_pool:
        for x in eligible:
            mid = x["model_id"]
            entry = {
                **model_diagnostics[mid],
                "status": "REJECTED",
                "reason": "BELOW_MIN_FORMALIZER_SCORE",
                "detail": (
                    f"overall_score={x['overall_score']} "
                    f"< min_formalizer_score={min_formalizer_score}"
                ),
            }
            rejected_models[mid] = entry
            model_diagnostics[mid] = entry
        return {
            "status": "ZERO_HEALTHY_MODELS",
            "roles": {},
            "exhausted_models": [],
            "rejected_models": rejected_models,
            "model_diagnostics": model_diagnostics,
            "routing_reason": "NO_MODEL_MEETS_MIN_FORMALIZER_SCORE",
        }

    formalizer_chain = sorted(
        formalizer_pool,
        key=lambda x: (-x["math_score"], -x["overall_score"], x["latency_ms"]),
    )
    top_family = formalizer_chain[0]["family"]

    skeptic_chain = sorted(
        eligible,
        key=lambda x: (
            0 if x["family"] != top_family else 1,
            -x["skeptic_score"],
            -x["overall_score"],
        ),
    )

    scout_pool = [x for x in eligible if x["overall_score"] >= policy["min_scout_score"]] or eligible
    scout_chain = sorted(scout_pool, key=lambda x: (x["latency_ms"], -x["overall_score"]))

    return {
        "status": "OK",
        "model_capability_epoch": db["model_capability_epoch"],
        "roles": {
            "formalizer": [x["model_id"] for x in formalizer_chain],
            "architect": [x["model_id"] for x in formalizer_chain],
            "skeptic": [x["model_id"] for x in skeptic_chain],
            "scout": [x["model_id"] for x in scout_chain],
        },
        "exhausted_models": [],
        "rejected_models": rejected_models,
        "model_diagnostics": model_diagnostics,
    }


class ModelRouter:
    """Stateful role-based router handling transient RPM/503 backoff and hard RPD exhaustion fallback."""

    def __init__(
        self,
        routing_table: dict[str, Any],
        *,
        retry_backoffs: Sequence[float] = (5.0, 15.0, 30.0),
        sleep_fn: Callable[[float], None] | None = None,
    ) -> None:
        self.routing_table = routing_table
        self.routing_table.setdefault("exhausted_models", [])
        self.routing_table.setdefault("roles", {})
        self._retry_backoffs = tuple(float(x) for x in retry_backoffs)
        self._sleep = sleep_fn or time.sleep

    @property
    def capability_epoch(self) -> int:
        return int(self.routing_table.get("model_capability_epoch", 1))

    @property
    def exhausted_models(self) -> list[str]:
        return list(self.routing_table["exhausted_models"])

    def mark_exhausted(self, model_id: str) -> None:
        """Record a model as RPD-exhausted for the remainder of the nightly run."""
        if model_id not in self.routing_table["exhausted_models"]:
            self.routing_table["exhausted_models"].append(model_id)

    def get_chain(self, role: str) -> list[str]:
        """Return ordered non-exhausted fallback models for `role`."""
        exhausted = set(self.routing_table["exhausted_models"])
        return [mid for mid in self.routing_table["roles"].get(role, []) if mid not in exhausted]

    def get_active_model(self, role: str) -> str | None:
        """Return the highest-ranked non-exhausted model for `role`, or None."""
        chain = self.get_chain(role)
        return chain[0] if chain else None

    def all_models_exhausted(self) -> bool:
        """Return True if no non-exhausted models remain for any configured role."""
        roles = self.routing_table.get("roles", {})
        if not roles:
            return True
        exhausted = set(self.routing_table["exhausted_models"])
        for chain in roles.values():
            if any(mid not in exhausted for mid in chain):
                return False
        return True

    def call_with_fallback(
        self,
        role: str,
        invoke_fn: Callable[[str], T],
    ) -> tuple[str, T]:
        """
        Execute `invoke_fn(model_id)` across `role`'s fallback chain.

        Retries transient RPM 429 / 503 errors with bounded exponential backoff (`5s -> 15s -> 30s`)
        without permanently blacklisting the model; only marks a model permanently exhausted on
        daily or zero-quota errors (`is_hard_quota_exhausted_error`).
        """
        chain = self.get_chain(role)
        if not chain:
            raise AllModelsExhaustedError(f"No non-exhausted models available for role '{role}'")

        last_exc: Exception | None = None
        for model_id in chain:
            attempt = 0
            while True:
                try:
                    return model_id, invoke_fn(model_id)
                except Exception as exc:
                    if is_hard_quota_exhausted_error(exc):
                        self.mark_exhausted(model_id)
                        last_exc = exc
                        break
                    if is_rate_limit_error(exc) or is_transient_service_error(exc):
                        if attempt < len(self._retry_backoffs):
                            delay = self._retry_backoffs[attempt]
                            if delay > 0:
                                self._sleep(delay)
                            attempt += 1
                            continue
                        last_exc = exc
                        break
                    raise

        raise AllModelsExhaustedError(
            f"All models exhausted for role '{role}' due to HTTP 429/503 rate limits"
        ) from last_exc

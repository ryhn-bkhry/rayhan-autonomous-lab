"""Persisted 20-problem capability calibration, epoch ratchet, and cross-family routing (Section 3.2)."""

from __future__ import annotations

import ast
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence, TypeVar

import jsonschema

from ral.preflight import LiveModelStatus

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
    text = (raw_response or "").strip()
    if not text or text.startswith("```"):
        raise ValueError("Response must be raw JSON without markdown fences")
    return json.loads(text)


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
        cand_sp = sp.sympify(candidate_expr, locals=sym_map)
        exp_sp = sp.sympify(problem["expected_expr"], locals=sym_map)
        if sp.simplify(cand_sp - exp_sp) == 0:
            return True
    except ImportError:
        pass
    except Exception:
        return False

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


def run_local_calibration_suite(
    client: Any,
    model_id: str,
    problems: Sequence[Mapping[str, Any]],
    scoring: Mapping[str, Any] | None = None,
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

    for problem in problems:
        cat = str(problem["category"])
        counts[cat] = counts.get(cat, 0) + 1
        passed = False
        error: str | None = None
        try:
            prompt = problem["prompt"]
            if cat == "strict_json":
                prompt = f"{prompt}\n\nJSON Schema:\n{json.dumps(problem['json_schema'])}"
            resp = client.models.generate_content(
                model=model_id,
                contents=prompt,
                config={
                    "temperature": 0.0,
                    "response_mime_type": "application/json",
                },
            )
            raw_text = (getattr(resp, "text", None) or "").strip()
            grader = graders.get(cat)
            if grader is None:
                raise ValueError(f"Unsupported calibration category: {cat!r}")
            passed = grader(problem, raw_text)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        if passed:
            passes[cat] = passes.get(cat, 0) + 1
        per_problem.append({"id": problem["id"], "category": cat, "passed": passed, "error": error})

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
        "results": per_problem,
    }


def _normalize_live_entry(entry: Mapping[str, Any] | LiveModelStatus) -> dict[str, Any]:
    if isinstance(entry, LiveModelStatus):
        return entry.to_dict()
    return dict(entry)


def ensure_benchmarked(
    client: Any,
    live_models: Sequence[Mapping[str, Any] | LiveModelStatus],
    policy: Mapping[str, Any],
    *,
    benchmark_path: Path = BENCHMARK_PATH,
    suite_path: Path = SUITE_PATH,
) -> tuple[dict[str, Any], bool]:
    """
    Load cached benchmark scores from `metrics/model_benchmarks.json`.

    Runs the deterministic 20-problem calibration suite ONLY for newly discovered live `model_id`s
    (or if `suite_version` changed). Returns `(benchmarks_db, capability_epoch_upgraded)`.
    """
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
    for m in normalized_live:
        mid = m["model_id"]
        if not (m.get("live") and m.get("json_ok")):
            continue
        if mid not in db["models"]:
            res = run_local_calibration_suite(
                client,
                mid,
                suite["problems"],
                scoring=suite.get("scoring"),
            )
            db["models"][mid] = {
                "overall_score": res["overall_score"],
                "math_score": res["math_score"],
                "skeptic_score": res["skeptic_score"],
                "json_reliability": res["json_reliability"],
                "family": m["family"],
                "benchmarked_at": datetime.now(timezone.utc).isoformat(),
            }
            changed = True

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
        bm_path.parent.mkdir(parents=True, exist_ok=True)
        bm_path.write_text(json.dumps(db, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return db, epoch_upgraded


def build_routing_table(
    live_models: Sequence[Mapping[str, Any] | LiveModelStatus],
    db: Mapping[str, Any],
    policy: Mapping[str, Any],
) -> dict[str, Any]:
    """
    Assign ranked fallback chains for each swarm role using persisted benchmark scores.

    Enforces orthogonal model-family diversity between Formalizer and Skeptic when available.
    """
    eligible: list[dict[str, Any]] = []
    for raw_m in live_models:
        m = _normalize_live_entry(raw_m)
        mid = m["model_id"]
        if not (m.get("live") and m.get("json_ok")) or mid not in db.get("models", {}):
            continue
        rec = db["models"][mid]
        if rec["json_reliability"] >= policy["min_json_reliability"]:
            eligible.append({"model_id": mid, "latency_ms": m["latency_ms"], **rec})

    if not eligible:
        return {"status": "ZERO_HEALTHY_MODELS", "roles": {}, "exhausted_models": []}

    formalizer_chain = sorted(
        eligible,
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
    }


def is_rate_limit_error(exc: Exception) -> bool:
    """Detect HTTP 429 / Gemini RESOURCE_EXHAUSTED RPD exhaustion errors."""
    for attr in ("status_code", "code"):
        if getattr(exc, attr, None) == 429:
            return True
    msg = str(exc).upper()
    return "429" in msg or "RESOURCE_EXHAUSTED" in msg or "RATE_LIMIT" in msg


class ModelRouter:
    """Stateful role-based router handling HTTP 429 RPD exhaustion fallback."""

    def __init__(self, routing_table: dict[str, Any]) -> None:
        self.routing_table = routing_table
        self.routing_table.setdefault("exhausted_models", [])
        self.routing_table.setdefault("roles", {})

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
        Execute `invoke_fn(model_id)` across `role`'s fallback chain on HTTP 429.

        Returns `(model_id_used, result)` or raises `AllModelsExhaustedError`.
        """
        chain = self.get_chain(role)
        if not chain:
            raise AllModelsExhaustedError(f"No non-exhausted models available for role '{role}'")

        last_exc: Exception | None = None
        for model_id in chain:
            try:
                return model_id, invoke_fn(model_id)
            except Exception as exc:
                if is_rate_limit_error(exc):
                    self.mark_exhausted(model_id)
                    last_exc = exc
                    continue
                raise

        raise AllModelsExhaustedError(
            f"All models exhausted for role '{role}' due to HTTP 429 rate limits"
        ) from last_exc

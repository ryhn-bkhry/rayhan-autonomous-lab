"""Z3 SMT solver and Hypothesis property-based oracle plugin (Section 6.2)."""

from __future__ import annotations

from ral.pipeline.oracles.base import SandboxedOraclePlugin


class Z3SMTOracle(SandboxedOraclePlugin):
    """Oracle plugin verifying logical satisfiability and property-based invariants."""

    name = "z3_smt"
    enabled_by_default = True
    required_any_modules = frozenset({"z3", "hypothesis"})
    min_assertions = 3

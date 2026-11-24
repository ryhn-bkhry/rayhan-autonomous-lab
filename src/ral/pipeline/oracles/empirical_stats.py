"""Empirical unit/conservation and CI/p-value arithmetic reproduction oracle (Section 6.2)."""

from __future__ import annotations

from ral.pipeline.oracles.base import SandboxedOraclePlugin


class EmpiricalStatsOracle(SandboxedOraclePlugin):
    """Oracle plugin verifying dimensional consistency and statistical test reproduction."""

    name = "empirical_stats"
    enabled_by_default = True
    required_any_modules = frozenset({"scipy", "numpy", "pint", "statistics", "math"})
    min_assertions = 3

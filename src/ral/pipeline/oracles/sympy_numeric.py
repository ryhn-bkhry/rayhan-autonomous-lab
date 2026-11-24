"""Symbolic identity, pint units, and boundary fuzzing oracle plugin (Section 6.2)."""

from __future__ import annotations

from ral.pipeline.oracles.base import SandboxedOraclePlugin


class SymPyNumericOracle(SandboxedOraclePlugin):
    """Oracle plugin verifying symbolic identities and numerical boundary limits."""

    name = "sympy_numeric"
    enabled_by_default = True
    required_any_modules = frozenset({"sympy", "numpy", "scipy", "pint", "math"})
    min_assertions = 3

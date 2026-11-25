"""Static anti-cheating and sandbox-escape AST linter (Blueprint Section 6.3 & 8)."""

from __future__ import annotations

import ast
from typing import Iterable

FORBIDDEN_MODULES: frozenset[str] = frozenset(
    {
        "os",
        "sys",
        "subprocess",
        "socket",
        "urllib",
        "requests",
        "http",
        "pathlib",
        "shutil",
        "ctypes",
        "importlib",
        "builtins",
        "ssl",
        "httpx",
        "aiohttp",
        "urllib3",
        "socketserver",
        "ftplib",
        "smtplib",
        "xmlrpc",
        "runpy",
        "webbrowser",
        "pty",
        "pickle",
        "marshal",
        "posix",
        "nt",
        "inspect",
        "gc",
        "code",
        "multiprocessing",
        "signal",
        "fcntl",
        "tempfile",
        "glob",
        "io",
    }
)

FORBIDDEN_BUILTINS: frozenset[str] = frozenset(
    {
        "open",
        "__import__",
        "eval",
        "exec",
        "compile",
        "breakpoint",
        "input",
        "globals",
        "locals",
        "vars",
        "getattr",
        "setattr",
        "delattr",
    }
)

FORBIDDEN_DUNDER_ATTRS: frozenset[str] = frozenset(
    {
        "__subclasses__",
        "__builtins__",
        "__globals__",
        "__import__",
        "__code__",
        "__closure__",
        "__loader__",
        "__spec__",
        "__mro__",
        "__bases__",
        "__dict__",
    }
)

_REFLEXIVE_CMP_OPS = (ast.Eq, ast.LtE, ast.GtE, ast.Is)
_TRIVIAL_BUILTIN_NAMES: frozenset[str] = frozenset(
    {"bool", "int", "float", "str", "len", "abs", "min", "max", "sum", "all", "any"}
)
_EQUALITY_HELPERS: frozenset[str] = frozenset(
    {"isclose", "allclose", "array_equal", "equal", "eq"}
)


def _dump(node: ast.AST) -> str:
    return ast.dump(node, include_attributes=False)


def _is_pure_constant_expr(node: ast.AST) -> bool:
    """Return True if an expression evaluates solely over literals with no dynamic symbols."""
    if isinstance(node, ast.Constant):
        return True
    if isinstance(node, ast.UnaryOp):
        return _is_pure_constant_expr(node.operand)
    if isinstance(node, ast.BinOp):
        return _is_pure_constant_expr(node.left) and _is_pure_constant_expr(node.right)
    if isinstance(node, ast.BoolOp):
        return all(_is_pure_constant_expr(v) for v in node.values)
    if isinstance(node, ast.Compare):
        return _is_pure_constant_expr(node.left) and all(
            _is_pure_constant_expr(c) for c in node.comparators
        )
    if isinstance(node, (ast.Tuple, ast.List, ast.Set)):
        return all(_is_pure_constant_expr(elt) for elt in node.elts)
    if isinstance(node, ast.Dict):
        return all(
            (k is None or _is_pure_constant_expr(k)) and _is_pure_constant_expr(v)
            for k, v in zip(node.keys, node.values)
        )
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
        if node.func.id in _TRIVIAL_BUILTIN_NAMES and not node.keywords:
            return all(_is_pure_constant_expr(a) for a in node.args)
    return False


def _references_exception_names(type_node: ast.AST, banned: Iterable[str]) -> set[str]:
    banned_set = set(banned)
    found: set[str] = set()
    for sub in ast.walk(type_node):
        if isinstance(sub, ast.Name) and sub.id in banned_set:
            found.add(sub.id)
        elif isinstance(sub, ast.Attribute) and sub.attr in banned_set:
            found.add(sub.attr)
    return found


def _is_noop_stmt(stmt: ast.stmt) -> bool:
    if isinstance(stmt, (ast.Pass, ast.Continue, ast.Break)):
        return True
    if isinstance(stmt, ast.Expr):
        if isinstance(stmt.value, ast.Constant):
            return True
        if isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Name):
            if stmt.value.func.id == "print":
                return True
    return False


def _call_name(func: ast.AST) -> str | None:
    if isinstance(func, ast.Name):
        return func.id.lower()
    if isinstance(func, ast.Attribute):
        return func.attr.lower()
    return None


def _check_assertion(node: ast.Assert) -> list[str]:
    issues: list[str] = []
    test = node.test
    line = getattr(node, "lineno", "?")

    if isinstance(test, ast.Constant):
        issues.append(f"TRIVIAL_ASSERT(line {line}): constant assertion `assert {test.value!r}`")
        return issues

    if _is_pure_constant_expr(test):
        issues.append(f"TRIVIAL_ASSERT(line {line}): pure constant expression with no dynamic computation")
        return issues

    if isinstance(test, ast.BoolOp):
        if isinstance(test.op, ast.Or) and any(
            isinstance(v, ast.Constant) and bool(v.value) for v in test.values
        ):
            issues.append(f"TRIVIAL_ASSERT(line {line}): `or <truthy constant>` tautology")
        elif isinstance(test.op, ast.And) and any(isinstance(v, ast.Constant) for v in test.values):
            issues.append(f"TRIVIAL_ASSERT(line {line}): constant operand inside `assert ... and ...`")

    for sub in ast.walk(test):
        if isinstance(sub, ast.Compare):
            operands = [sub.left, *sub.comparators]
            for left_op, op, right_op in zip(operands, sub.ops, operands[1:]):
                if isinstance(op, _REFLEXIVE_CMP_OPS) and _dump(left_op) == _dump(right_op):
                    issues.append(f"TAUTOLOGY_COMPARE(line {line}): identical left and right expressions")
        elif isinstance(sub, ast.BinOp) and isinstance(sub.op, (ast.Sub, ast.Div)):
            if _dump(sub.left) == _dump(sub.right):
                op_name = "subtraction" if isinstance(sub.op, ast.Sub) else "division"
                issues.append(f"TAUTOLOGY_{op_name.upper()}(line {line}): identical operands")
        elif isinstance(sub, ast.Call):
            cname = _call_name(sub.func)
            if cname in _EQUALITY_HELPERS and len(sub.args) >= 2:
                if _dump(sub.args[0]) == _dump(sub.args[1]):
                    issues.append(f"TAUTOLOGY_CALL(line {line}): identical arguments to `{cname}`")

    return issues


def lint_forbidden_apis(source_code: str) -> list[str]:
    """Check syntax, forbidden imports, forbidden builtins, and dunder sandbox escapes."""
    try:
        tree = ast.parse(source_code)
    except SyntaxError as exc:
        return [f"SYNTAX_ERROR(line {exc.lineno}): {exc.msg}"]

    violations: list[str] = []
    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.Import):
            for alias in node.names:
                root_mod = alias.name.split(".")[0]
                if root_mod in FORBIDDEN_MODULES:
                    violations.append(f"FORBIDDEN_IMPORT(line {line}): module `{alias.name}`")
        elif isinstance(node, ast.ImportFrom):
            if node.level > 0:
                violations.append(f"FORBIDDEN_IMPORT(line {line}): relative import not allowed")
            if node.module:
                root_mod = node.module.split(".")[0]
                if root_mod in FORBIDDEN_MODULES:
                    violations.append(f"FORBIDDEN_IMPORT(line {line}): module `{node.module}`")
        elif isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
            if node.id in FORBIDDEN_BUILTINS:
                violations.append(f"FORBIDDEN_BUILTIN(line {line}): `{node.id}`")
        elif isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_DUNDER_ATTRS or node.attr in {"open", "__import__"}:
                violations.append(f"FORBIDDEN_ATTRIBUTE(line {line}): `{node.attr}`")

    return violations


def lint_verification_script(
    source_code: str,
    *,
    min_assertions: int = 3,
    check_anti_cheat: bool = True,
) -> list[str]:
    """
    Validate a Python verification or falsification script using static AST analysis.

    Returns a list of violation descriptions; an empty list indicates the script passed.
    """
    violations = lint_forbidden_apis(source_code)
    if violations and violations[0].startswith("SYNTAX_ERROR"):
        return violations

    if not check_anti_cheat:
        return violations

    tree = ast.parse(source_code)
    distinct_assertions: set[str] = set()

    for node in ast.walk(tree):
        line = getattr(node, "lineno", "?")
        if isinstance(node, ast.ExceptHandler):
            if node.type is None:
                violations.append(f"BARE_EXCEPT(line {line}): bare `except:` handler is forbidden")
            else:
                banned_exc = _references_exception_names(node.type, ("AssertionError", "BaseException"))
                if banned_exc:
                    names = ", ".join(sorted(banned_exc))
                    violations.append(f"FORBIDDEN_EXCEPT_TARGET(line {line}): catching `{names}` is forbidden")
            if not node.body or all(_is_noop_stmt(stmt) for stmt in node.body):
                violations.append(f"EMPTY_EXCEPT(line {line}): empty or swallowing `except` block")

        elif isinstance(node, ast.Assert):
            assert_issues = _check_assertion(node)
            if assert_issues:
                violations.extend(assert_issues)
            else:
                distinct_assertions.add(_dump(node.test))

    if len(distinct_assertions) < min_assertions:
        violations.append(
            f"INSUFFICIENT_ASSERTIONS: found {len(distinct_assertions)} distinct non-trivial "
            f"assertions, minimum required is {min_assertions}"
        )

    return violations

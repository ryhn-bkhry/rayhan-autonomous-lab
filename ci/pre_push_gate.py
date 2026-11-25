"""Cross-platform blocking pre-push validation gate and guarded Git commit helper."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path, PurePosixPath
import subprocess
import sys
from typing import Callable, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ci.validate_node import load_allowed_licenses, validate_node_file  # noqa: E402
from ci.validate_vault import validate_vault  # noqa: E402


@dataclass(frozen=True)
class PrePushGateResult:
    """Outcome of running the blocking pre-push validation gate and git commit/push."""

    ok: bool
    committed: bool
    pushed: bool
    staged_paths: tuple[str, ...]
    validated_nodes: tuple[str, ...]
    vault_ok: bool
    errors: tuple[str, ...] = ()


@dataclass(frozen=True)
class NdjsonRebaseResult:
    """Outcome of attempting automatic conflict resolution for append-only NDJSON rebase conflicts."""

    ok: bool
    resolved_paths: tuple[str, ...] = ()
    aborted: bool = False
    errors: tuple[str, ...] = ()


_CONFLICT_MARKER_PREFIXES: tuple[str, ...] = ("<<<<<<<", "=======", ">>>>>>>", "|||||||")


def _run_git(
    repo_root: Path,
    args: Sequence[str],
    *,
    run_cmd: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> subprocess.CompletedProcess[str]:
    return run_cmd(
        ["git", "-C", str(repo_root), *args],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )


def is_append_only_ndjson_path(rel_path: str) -> bool:
    """Return True if `rel_path` targets an append-only `.ndjson` file under `ledger/` or `metrics/`."""
    norm = str(rel_path).strip().replace("\\", "/")
    while norm.startswith("./"):
        norm = norm[2:]
    if not norm or norm.startswith("/"):
        return False
    posix = PurePosixPath(norm)
    if posix.is_absolute() or ".." in posix.parts:
        return False
    return norm.endswith(".ndjson") and norm.startswith(("ledger/", "metrics/"))


def merge_ndjson_sides(ours_text: str, theirs_text: str) -> str:
    """
    Combine append-only NDJSON lines from `OURS` and `THEIRS`, stripping any conflict markers
    and deduplicating while preserving first-seen line order (`OURS` first, then new `THEIRS` lines).
    """
    merged: list[str] = []
    seen: set[str] = set()
    for block in (ours_text, theirs_text):
        for raw_line in block.splitlines():
            stripped = raw_line.strip()
            if not stripped or stripped.startswith(_CONFLICT_MARKER_PREFIXES):
                continue
            if stripped not in seen:
                seen.add(stripped)
                merged.append(stripped)
    return ("\n".join(merged) + "\n") if merged else ""


def _is_rebase_in_progress(
    repo_root: Path,
    *,
    run_cmd: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> bool:
    for git_subpath in ("rebase-merge", "rebase-apply"):
        proc = _run_git(repo_root, ["rev-parse", "--git-path", git_subpath], run_cmd=run_cmd)
        if proc.returncode != 0:
            continue
        raw = proc.stdout.strip()
        if not raw:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            candidate = Path(repo_root) / candidate
        if candidate.exists():
            return True
    return False


def _abort_rebase(
    repo_root: Path,
    *,
    run_cmd: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> None:
    _run_git(repo_root, ["rebase", "--abort"], run_cmd=run_cmd)


def resolve_ndjson_rebase_conflicts(
    repo_root: Path,
    *,
    max_steps: int = 50,
    run_cmd: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> NdjsonRebaseResult:
    """
    Automatically resolve rebase conflicts restricted exclusively to append-only
    `ledger/**/*.ndjson` and `metrics/**/*.ndjson` files; abort if any non-NDJSON file conflicts.
    """
    root = Path(repo_root)
    if not _is_rebase_in_progress(root, run_cmd=run_cmd):
        return NdjsonRebaseResult(
            ok=False,
            aborted=False,
            errors=("No active git rebase in progress",),
        )

    resolved_all: list[str] = []
    seen_resolved: set[str] = set()

    for _ in range(max_steps):
        if not _is_rebase_in_progress(root, run_cmd=run_cmd):
            return NdjsonRebaseResult(
                ok=True,
                resolved_paths=tuple(resolved_all),
                aborted=False,
                errors=(),
            )

        diff_proc = _run_git(
            root,
            ["-c", "core.quotepath=false", "diff", "--name-only", "--diff-filter=U"],
            run_cmd=run_cmd,
        )
        if diff_proc.returncode != 0:
            _abort_rebase(root, run_cmd=run_cmd)
            return NdjsonRebaseResult(
                ok=False,
                resolved_paths=tuple(resolved_all),
                aborted=True,
                errors=(_format_cmd_error("git diff --diff-filter=U failed", diff_proc),),
            )

        conflicts: list[str] = []
        for raw_line in diff_proc.stdout.splitlines():
            entry = raw_line.strip()
            if not entry:
                continue
            if len(entry) >= 2 and entry.startswith('"') and entry.endswith('"'):
                entry = entry[1:-1]
            conflicts.append(entry)

        if not conflicts:
            _abort_rebase(root, run_cmd=run_cmd)
            return NdjsonRebaseResult(
                ok=False,
                resolved_paths=tuple(resolved_all),
                aborted=True,
                errors=("Rebase paused with no unmerged files",),
            )

        non_ndjson = [p for p in conflicts if not is_append_only_ndjson_path(p)]
        if non_ndjson:
            _abort_rebase(root, run_cmd=run_cmd)
            return NdjsonRebaseResult(
                ok=False,
                resolved_paths=tuple(resolved_all),
                aborted=True,
                errors=(f"Non-NDJSON rebase conflict(s): {', '.join(non_ndjson)}",),
            )

        for rel_path in conflicts:
            ours_proc = _run_git(root, ["show", f":2:{rel_path}"], run_cmd=run_cmd)
            theirs_proc = _run_git(root, ["show", f":3:{rel_path}"], run_cmd=run_cmd)
            if ours_proc.returncode != 0 or theirs_proc.returncode != 0:
                _abort_rebase(root, run_cmd=run_cmd)
                errs: list[str] = []
                if ours_proc.returncode != 0:
                    errs.append(_format_cmd_error(f"git show :2:{rel_path} failed", ours_proc))
                if theirs_proc.returncode != 0:
                    errs.append(_format_cmd_error(f"git show :3:{rel_path} failed", theirs_proc))
                return NdjsonRebaseResult(
                    ok=False,
                    resolved_paths=tuple(resolved_all),
                    aborted=True,
                    errors=tuple(errs),
                )

            merged = merge_ndjson_sides(ours_proc.stdout, theirs_proc.stdout)
            target_file = root / rel_path
            target_file.parent.mkdir(parents=True, exist_ok=True)
            with target_file.open("w", encoding="utf-8", newline="\n") as fh:
                fh.write(merged)

            add_proc = _run_git(root, ["add", "--", rel_path], run_cmd=run_cmd)
            if add_proc.returncode != 0:
                _abort_rebase(root, run_cmd=run_cmd)
                return NdjsonRebaseResult(
                    ok=False,
                    resolved_paths=tuple(resolved_all),
                    aborted=True,
                    errors=(_format_cmd_error(f"git add {rel_path} failed", add_proc),),
                )

            if rel_path not in seen_resolved:
                seen_resolved.add(rel_path)
                resolved_all.append(rel_path)

        cached_proc = _run_git(root, ["diff", "--cached", "--quiet"], run_cmd=run_cmd)
        if cached_proc.returncode == 0:
            step_proc = _run_git(root, ["rebase", "--skip"], run_cmd=run_cmd)
        elif cached_proc.returncode == 1:
            commit_proc = _run_git(root, ["commit", "--no-edit"], run_cmd=run_cmd)
            if commit_proc.returncode != 0:
                _abort_rebase(root, run_cmd=run_cmd)
                return NdjsonRebaseResult(
                    ok=False,
                    resolved_paths=tuple(resolved_all),
                    aborted=True,
                    errors=(_format_cmd_error("git commit --no-edit failed", commit_proc),),
                )
            step_proc = _run_git(root, ["rebase", "--continue"], run_cmd=run_cmd)
        else:
            _abort_rebase(root, run_cmd=run_cmd)
            return NdjsonRebaseResult(
                ok=False,
                resolved_paths=tuple(resolved_all),
                aborted=True,
                errors=(_format_cmd_error("git diff --cached failed", cached_proc),),
            )

        if step_proc.returncode == 0:
            return NdjsonRebaseResult(
                ok=True,
                resolved_paths=tuple(resolved_all),
                aborted=False,
                errors=(),
            )
        if not _is_rebase_in_progress(root, run_cmd=run_cmd):
            return NdjsonRebaseResult(
                ok=False,
                resolved_paths=tuple(resolved_all),
                aborted=False,
                errors=(_format_cmd_error("git rebase --continue failed", step_proc),),
            )

    _abort_rebase(root, run_cmd=run_cmd)
    return NdjsonRebaseResult(
        ok=False,
        resolved_paths=tuple(resolved_all),
        aborted=True,
        errors=(f"Exceeded max_steps={max_steps} during NDJSON rebase resolution",),
    )


def _format_cmd_error(label: str, proc: subprocess.CompletedProcess[str]) -> str:
    detail = (proc.stderr or proc.stdout or "").strip()
    return f"{label} (exit {proc.returncode}): {detail}" if detail else f"{label} (exit {proc.returncode})"


def resolve_existing_or_tracked_paths(
    repo_root: Path,
    candidate_paths: Sequence[str],
    *,
    run_cmd: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[str]:
    """Return only candidate pathspecs that exist on disk or are tracked in Git."""
    root = Path(repo_root)
    resolved: list[str] = []
    seen: set[str] = set()
    for raw in candidate_paths:
        pathspec = str(raw).strip()
        if not pathspec or pathspec in seen:
            continue
        seen.add(pathspec)
        if (root / pathspec).exists():
            resolved.append(pathspec)
            continue
        probe = _run_git(root, ["ls-files", "--", pathspec], run_cmd=run_cmd)
        if probe.returncode == 0 and probe.stdout.strip():
            resolved.append(pathspec)
    return resolved


def find_changed_vault_nodes(
    repo_root: Path,
    *,
    run_cmd: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> list[Path]:
    """List modified or untracked Markdown node files inside `vault/` (skipping missing/deleted)."""
    root = Path(repo_root)
    if not (root / "vault").is_dir():
        return []

    proc = _run_git(
        root,
        ["-c", "core.quotepath=false", "status", "--porcelain", "-uall", "--", "vault/"],
        run_cmd=run_cmd,
    )
    if proc.returncode != 0:
        raise RuntimeError(_format_cmd_error("git status vault/ failed", proc))

    changed: list[Path] = []
    seen: set[Path] = set()
    for raw_line in proc.stdout.splitlines():
        if len(raw_line) < 4:
            continue
        entry = raw_line[3:].strip()
        if " -> " in entry:
            entry = entry.split(" -> ", 1)[1].strip()
        if len(entry) >= 2 and entry.startswith('"') and entry.endswith('"'):
            entry = entry[1:-1]
        if not entry.endswith(".md"):
            continue
        candidate = root / entry
        if candidate.is_file() and candidate not in seen:
            seen.add(candidate)
            changed.append(candidate)
    return sorted(changed)


def run_pre_push_gate(
    repo_root: Path,
    *,
    paths: Sequence[str],
    commit_message: str,
    always_validate_vault: bool = False,
    validate_vault_nodes: bool = True,
    push: bool = True,
    remote: str = "origin",
    branch: str = "main",
    bot_name: str = "ral-bot[bot]",
    bot_email: str = "41898282+github-actions[bot]@users.noreply.github.com",
    max_git_mb: float = 700.0,
    run_cmd: Callable[..., subprocess.CompletedProcess[str]] = subprocess.run,
) -> PrePushGateResult:
    """
    Guard optional pathspecs, validate changed nodes & full vault, and commit/push changes.
    """
    root = Path(repo_root)
    stage_paths = resolve_existing_or_tracked_paths(root, paths, run_cmd=run_cmd)

    dirty = False
    if stage_paths:
        status_proc = _run_git(
            root,
            ["status", "--porcelain", "-uall", "--", *stage_paths],
            run_cmd=run_cmd,
        )
        if status_proc.returncode != 0:
            return PrePushGateResult(
                ok=False,
                committed=False,
                pushed=False,
                staged_paths=tuple(stage_paths),
                validated_nodes=(),
                vault_ok=False,
                errors=(_format_cmd_error("git status failed", status_proc),),
            )
        dirty = bool(status_proc.stdout.strip())

    if not dirty:
        if always_validate_vault:
            report = validate_vault(root, max_git_mb=max_git_mb)
            return PrePushGateResult(
                ok=report.ok,
                committed=False,
                pushed=False,
                staged_paths=tuple(stage_paths),
                validated_nodes=(),
                vault_ok=report.ok,
                errors=tuple(report.errors),
            )
        return PrePushGateResult(
            ok=True,
            committed=False,
            pushed=False,
            staged_paths=tuple(stage_paths),
            validated_nodes=(),
            vault_ok=True,
            errors=(),
        )

    validated_rel_paths: list[str] = []
    node_errors: list[str] = []
    if validate_vault_nodes and (root / "vault").is_dir():
        allowlist_file = root / "config" / "license_allowlist.yaml"
        allowed_licenses = load_allowed_licenses(allowlist_file) if allowlist_file.exists() else None
        try:
            changed_nodes = find_changed_vault_nodes(root, run_cmd=run_cmd)
        except RuntimeError as exc:
            return PrePushGateResult(
                ok=False,
                committed=False,
                pushed=False,
                staged_paths=tuple(stage_paths),
                validated_nodes=(),
                vault_ok=False,
                errors=(str(exc),),
            )
        for node_path in changed_nodes:
            rel = node_path.relative_to(root).as_posix()
            validated_rel_paths.append(rel)
            node_res = validate_node_file(
                node_path,
                allowed_licenses=allowed_licenses,
                check_filename=True,
            )
            if not node_res.ok:
                for err in node_res.errors:
                    node_errors.append(f"NODE_INVALID({rel}): {err}")
        if node_errors:
            return PrePushGateResult(
                ok=False,
                committed=False,
                pushed=False,
                staged_paths=tuple(stage_paths),
                validated_nodes=tuple(validated_rel_paths),
                vault_ok=False,
                errors=tuple(node_errors),
            )

    report = validate_vault(root, max_git_mb=max_git_mb)
    if not report.ok:
        return PrePushGateResult(
            ok=False,
            committed=False,
            pushed=False,
            staged_paths=tuple(stage_paths),
            validated_nodes=tuple(validated_rel_paths),
            vault_ok=False,
            errors=tuple(report.errors),
        )

    for cfg_key, cfg_val in (("user.name", bot_name), ("user.email", bot_email)):
        cfg_proc = _run_git(root, ["config", cfg_key, cfg_val], run_cmd=run_cmd)
        if cfg_proc.returncode != 0:
            return PrePushGateResult(
                ok=False,
                committed=False,
                pushed=False,
                staged_paths=tuple(stage_paths),
                validated_nodes=tuple(validated_rel_paths),
                vault_ok=True,
                errors=(_format_cmd_error(f"git config {cfg_key} failed", cfg_proc),),
            )

    add_proc = _run_git(root, ["add", "--", *stage_paths], run_cmd=run_cmd)
    if add_proc.returncode != 0:
        return PrePushGateResult(
            ok=False,
            committed=False,
            pushed=False,
            staged_paths=tuple(stage_paths),
            validated_nodes=tuple(validated_rel_paths),
            vault_ok=True,
            errors=(_format_cmd_error("git add failed", add_proc),),
        )

    cached_proc = _run_git(root, ["diff", "--cached", "--quiet"], run_cmd=run_cmd)
    if cached_proc.returncode == 0:
        return PrePushGateResult(
            ok=True,
            committed=False,
            pushed=False,
            staged_paths=tuple(stage_paths),
            validated_nodes=tuple(validated_rel_paths),
            vault_ok=True,
            errors=(),
        )
    if cached_proc.returncode != 1:
        return PrePushGateResult(
            ok=False,
            committed=False,
            pushed=False,
            staged_paths=tuple(stage_paths),
            validated_nodes=tuple(validated_rel_paths),
            vault_ok=True,
            errors=(_format_cmd_error("git diff --cached failed", cached_proc),),
        )

    commit_proc = _run_git(root, ["commit", "-m", commit_message], run_cmd=run_cmd)
    if commit_proc.returncode != 0:
        return PrePushGateResult(
            ok=False,
            committed=False,
            pushed=False,
            staged_paths=tuple(stage_paths),
            validated_nodes=tuple(validated_rel_paths),
            vault_ok=True,
            errors=(_format_cmd_error("git commit failed", commit_proc),),
        )

    if push:
        pull_proc = _run_git(root, ["pull", "--rebase", remote, branch], run_cmd=run_cmd)
        if pull_proc.returncode != 0:
            rebase_res = resolve_ndjson_rebase_conflicts(root, run_cmd=run_cmd)
            if not rebase_res.ok:
                return PrePushGateResult(
                    ok=False,
                    committed=True,
                    pushed=False,
                    staged_paths=tuple(stage_paths),
                    validated_nodes=tuple(validated_rel_paths),
                    vault_ok=True,
                    errors=(
                        _format_cmd_error("git pull --rebase failed", pull_proc),
                        *rebase_res.errors,
                    ),
                )
        push_proc = _run_git(root, ["push", remote, branch], run_cmd=run_cmd)
        if push_proc.returncode != 0:
            return PrePushGateResult(
                ok=False,
                committed=True,
                pushed=False,
                staged_paths=tuple(stage_paths),
                validated_nodes=tuple(validated_rel_paths),
                vault_ok=True,
                errors=(_format_cmd_error("git push failed", push_proc),),
            )

    return PrePushGateResult(
        ok=True,
        committed=True,
        pushed=push,
        staged_paths=tuple(stage_paths),
        validated_nodes=tuple(validated_rel_paths),
        vault_ok=True,
        errors=(),
    )


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entrypoint for blocking pre-push gate and guarded git commits."""
    parser = argparse.ArgumentParser(
        description="Run RAL pre-push gate with guarded Git pathspecs and atomic commit/push."
    )
    parser.add_argument("--root", type=Path, default=Path("."), help="Repository root directory")
    parser.add_argument("--message", default=None, help="Commit message when changes are staged")
    parser.add_argument("--paths", nargs="+", default=None, help="Candidate Git pathspecs to guard")
    parser.add_argument(
        "--resolve-ndjson-rebase",
        action="store_true",
        help="Resolve active rebase conflicts restricted to append-only ledger/metrics .ndjson files",
    )
    parser.add_argument(
        "--always-validate-vault",
        action="store_true",
        help="Run validate_vault() even when no candidate pathspecs have changes",
    )
    parser.add_argument(
        "--no-push",
        action="store_true",
        help="Commit locally after validation without running git pull --rebase / git push",
    )
    parser.add_argument("--remote", default="origin", help="Git remote name")
    parser.add_argument("--branch", default="main", help="Git target branch name")
    parser.add_argument("--max-git-mb", type=float, default=700.0, help=".git alert threshold in MiB")
    args = parser.parse_args(argv)

    if args.resolve_ndjson_rebase:
        rebase_res = resolve_ndjson_rebase_conflicts(args.root)
        print(json.dumps(asdict(rebase_res), indent=2))
        if not rebase_res.ok:
            for err in rebase_res.errors:
                print(f"REBASE_FAIL: {err}", file=sys.stderr)
            return 1
        return 0

    if not args.message or not args.paths:
        parser.error("--message and --paths are required unless --resolve-ndjson-rebase is set")

    res = run_pre_push_gate(
        args.root,
        paths=args.paths,
        commit_message=args.message,
        always_validate_vault=args.always_validate_vault,
        push=(not args.no_push),
        remote=args.remote,
        branch=args.branch,
        max_git_mb=args.max_git_mb,
    )
    print(json.dumps(asdict(res), indent=2))
    if not res.ok:
        for err in res.errors:
            print(f"GATE_FAIL: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

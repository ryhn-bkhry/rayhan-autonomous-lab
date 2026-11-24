"""Unified CLI entrypoint for RAL nightly workflow jobs, doc rendering, validation, and MCP."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from typing import Sequence

from ci.validate_vault import validate_vault
from ral.github_client import GitHubClient
from ral.orchestrator import (
    parse_github_repo,
    run_consolidate_job,
    run_epoch_job,
    run_preflight_job,
)
from ral.vault_docs import regenerate_vault_docs


def _maybe_github_client(repo_slug: str | None) -> GitHubClient | None:
    slug = (repo_slug or os.environ.get("GITHUB_REPOSITORY") or "").strip()
    token = (os.environ.get("GITHUB_TOKEN") or "").strip()
    if not slug or not token:
        return None
    owner, repo = parse_github_repo(slug)
    return GitHubClient(owner, repo, token=token)


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch RAL CLI subcommands (`preflight`, `run-epoch`, `consolidate`, `render-docs`, `validate`, `mcp-server`)."""
    parser = argparse.ArgumentParser(prog="ral", description="Rayhan's Autonomous Lab CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    p_pre = sub.add_parser("preflight", help="Run Job 0 preflight & curriculum planning.")
    p_pre.add_argument("--repo-root", type=Path, default=Path("."))
    p_pre.add_argument("--handoff-dir", type=Path, default=Path(".ral_handoff"))
    p_pre.add_argument("--cache-dir", type=Path, default=Path(".ral_cache"))
    p_pre.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    p_pre.add_argument("--night", type=int, default=0)
    p_pre.add_argument("--epochs", type=int, default=6)
    p_pre.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    p_pre.add_argument("--dry-run", action="store_true")
    p_pre.add_argument("--simulate-drive-fail", action="store_true")
    p_pre.add_argument("--github-repo", default=os.environ.get("GITHUB_REPOSITORY", ""))

    p_ep = sub.add_parser("run-epoch", help="Run sequential execution epoch 1..6.")
    p_ep.add_argument("--repo-root", type=Path, default=Path("."))
    p_ep.add_argument("--handoff-dir", type=Path, default=Path(".ral_handoff"))
    p_ep.add_argument("--cache-dir", type=Path, default=Path(".ral_cache"))
    p_ep.add_argument("--ore-cache-dir", type=Path, default=Path("/tmp/ore_cache"))
    p_ep.add_argument("--epoch-index", type=int, required=True)
    p_ep.add_argument("--soft-stop-seconds", type=float, default=2880.0)
    p_ep.add_argument("--max-sources", type=int, default=3)
    p_ep.add_argument("--dry-run", action="store_true")

    p_con = sub.add_parser("consolidate", help="Run Job 7 consolidation, index swap & watchdog.")
    p_con.add_argument("--repo-root", type=Path, default=Path("."))
    p_con.add_argument("--handoff-dir", type=Path, default=Path(".ral_handoff"))
    p_con.add_argument("--cache-dir", type=Path, default=Path(".ral_cache"))
    p_con.add_argument("--date", default=datetime.now(timezone.utc).date().isoformat())
    p_con.add_argument("--run-id", default=os.environ.get("GITHUB_RUN_ID", "local"))
    p_con.add_argument("--max-git-mb", type=float, default=700.0)
    p_con.add_argument("--dry-run", action="store_true")
    p_con.add_argument("--allow-release-in-dry-run", action="store_true")
    p_con.add_argument("--github-repo", default=os.environ.get("GITHUB_REPOSITORY", ""))

    p_docs = sub.add_parser("render-docs", help="Regenerate /INDEX.md and /llms.txt.")
    p_docs.add_argument("--repo-root", type=Path, default=Path("."))
    p_docs.add_argument("--repo-slug", default="rayhans-autonomous-lab")
    p_docs.add_argument("--max-index-tokens", type=int, default=15000)

    p_val = sub.add_parser("validate", help="Run blocking vault validation checks.")
    p_val.add_argument("--repo-root", type=Path, default=Path("."))
    p_val.add_argument("--max-git-mb", type=float, default=700.0)

    p_mcp = sub.add_parser("mcp-server", help="Launch hardened read-only MCP stdio server.")
    p_mcp.add_argument("--repo-root", type=Path, default=Path("."))
    p_mcp.add_argument("--db", type=Path, default=None)
    p_mcp.add_argument("--cache-dir", type=Path, default=None)
    p_mcp.add_argument("--sync-release", action="store_true")
    p_mcp.add_argument("--github-repo", default=os.environ.get("GITHUB_REPOSITORY", ""))

    args = parser.parse_args(argv)

    if args.command == "preflight":
        gh = _maybe_github_client(args.github_repo)
        res = run_preflight_job(
            args.repo_root,
            handoff_dir=args.handoff_dir,
            cache_dir=args.cache_dir,
            run_date_iso=args.date,
            current_night=args.night,
            available_epochs=args.epochs,
            run_id=args.run_id,
            dry_run=args.dry_run,
            simulate_drive_fail=args.simulate_drive_fail,
            github_client=gh,
        )
        print(
            json.dumps(
                {
                    "ok": res.ok,
                    "run_date": res.run_date,
                    "circuit_breaker": res.circuit_breaker,
                    "available_epochs": res.available_epochs,
                    "epoch_upgraded": res.epoch_upgraded,
                    "github_api_calls_used": res.github_api_calls_used,
                },
                indent=2,
            )
        )
        return 0 if res.ok else 1

    if args.command == "run-epoch":
        res = run_epoch_job(
            args.repo_root,
            handoff_dir=args.handoff_dir,
            epoch_index=args.epoch_index,
            cache_dir=args.cache_dir,
            ore_cache_dir=args.ore_cache_dir,
            soft_stop_seconds=args.soft_stop_seconds,
            max_sources=args.max_sources,
            dry_run=args.dry_run,
        )
        print(
            json.dumps(
                {
                    "epoch_index": res.epoch_index,
                    "mode": res.mode,
                    "status": res.status,
                    "pivoted_to_maintenance": res.pivoted_to_maintenance,
                    "nodes_verified_tonight": res.nodes_verified_tonight,
                },
                indent=2,
            )
        )
        return 0

    if args.command == "consolidate":
        gh = _maybe_github_client(args.github_repo)
        res = run_consolidate_job(
            args.repo_root,
            handoff_dir=args.handoff_dir,
            cache_dir=args.cache_dir,
            run_date_iso=args.date,
            run_id=args.run_id,
            max_git_mb=args.max_git_mb,
            dry_run=args.dry_run,
            allow_release_in_dry_run=args.allow_release_in_dry_run,
            github_client=gh,
        )
        print(
            json.dumps(
                {
                    "ok": res.ok,
                    "vault_ok": res.vault_ok,
                    "node_count": res.node_count,
                    "edge_count": res.edge_count,
                    "nodes_verified_tonight": res.nodes_verified_tonight,
                    "dead_mans_switch_triggered": res.dead_mans_switch.triggered,
                    "dead_mans_switch_reasons": list(res.dead_mans_switch.reasons),
                },
                indent=2,
            )
        )
        return 0 if res.ok else 1

    if args.command == "render-docs":
        info = regenerate_vault_docs(
            args.repo_root,
            repo_slug=args.repo_slug,
            max_index_tokens=args.max_index_tokens,
        )
        print(
            json.dumps(
                {
                    "index_path": str(info["index_path"]),
                    "llms_path": str(info["llms_path"]),
                    "index_tokens": info["index_tokens"],
                    "llms_tokens": info["llms_tokens"],
                    "node_count": info["node_count"],
                    "edge_count": info["edge_count"],
                },
                indent=2,
            )
        )
        return 0

    if args.command == "validate":
        report = validate_vault(args.repo_root, max_git_mb=args.max_git_mb)
        print(
            json.dumps(
                {
                    "ok": report.ok,
                    "node_count": report.node_count,
                    "edge_count": report.edge_count,
                    "git_size_mb": report.git_size_mb,
                    "errors": report.errors,
                    "warnings": report.warnings,
                },
                indent=2,
            )
        )
        return 0 if report.ok else 1

    if args.command == "mcp-server":
        from mcp.ral_mcp_server import main as mcp_main

        mcp_argv = ["--repo-root", str(args.repo_root)]
        if args.db is not None:
            mcp_argv.extend(["--db", str(args.db)])
        if args.cache_dir is not None:
            mcp_argv.extend(["--cache-dir", str(args.cache_dir)])
        if args.sync_release:
            mcp_argv.append("--sync-release")
        if args.github_repo:
            mcp_argv.extend(["--github-repo", str(args.github_repo)])
        return mcp_main(mcp_argv)

    return 2


if __name__ == "__main__":
    raise SystemExit(main())

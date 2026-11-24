"""One-time local OAuth2 Desktop token minter for RAL_Ore_Warehouse (Blueprint Section 10 Option B)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Sequence

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SRC_DIR = _REPO_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

from ral.storage.drive_ore import DRIVE_FILE_SCOPE, DriveOreStore  # noqa: E402


def load_client_secrets_metadata(client_secrets_path: Path) -> dict[str, str]:
    """Validate OAuth2 client secrets JSON and extract client_id and client_secret."""
    raw = json.loads(client_secrets_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Client secrets JSON must be a dictionary")
    section = raw.get("installed") or raw.get("web")
    if not isinstance(section, dict):
        raise ValueError("Expected 'installed' or 'web' OAuth2 client block in secrets JSON")
    client_id = str(section.get("client_id") or "").strip()
    client_secret = str(section.get("client_secret") or "").strip()
    if not client_id or not client_secret:
        raise ValueError("Client secrets JSON is missing 'client_id' or 'client_secret'")
    return {"client_id": client_id, "client_secret": client_secret}


def mint_refresh_token(client_secrets_path: Path, *, port: int = 0) -> dict[str, str]:
    """
    Run local loopback OAuth2 authorization once using narrow `drive.file` scope.

    Lazy-imports `google_auth_oauthlib.flow.InstalledAppFlow` so test suites without
    oauthlib can inspect argument parsing and client metadata safely.
    """
    meta = load_client_secrets_metadata(client_secrets_path)
    from google_auth_oauthlib.flow import InstalledAppFlow

    flow = InstalledAppFlow.from_client_secrets_file(
        str(client_secrets_path),
        scopes=[DRIVE_FILE_SCOPE],
    )
    creds = flow.run_local_server(port=port, access_type="offline", prompt="consent")
    refresh_token = getattr(creds, "refresh_token", None)
    if not refresh_token:
        raise RuntimeError(
            "Google OAuth did not return a refresh_token. Revoke prior app access at "
            "https://myaccount.google.com/permissions and re-run with prompt='consent'."
        )
    return {
        "client_id": meta["client_id"],
        "client_secret": meta["client_secret"],
        "refresh_token": str(refresh_token),
        "scope": DRIVE_FILE_SCOPE,
        "creds_obj": creds,  # type: ignore[dict-item]
    }


def verify_root_folder_canary(creds: Any, root_folder_id: str) -> None:
    """Optionally run the 16-byte Drive canary immediately after local OAuth2 authorization."""
    from googleapiclient.discovery import build

    service = build("drive", "v3", credentials=creds, cache_discovery=False)
    store = DriveOreStore(service=service, root_folder_id=root_folder_id)
    canary = store.run_preflight_canary()
    if not canary.ok:
        raise RuntimeError(f"Drive preflight canary failed: {canary.error}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Authorize Google Drive once locally with narrow `drive.file` scope and "
            "print GitHub Actions secret values for RAL_Ore_Warehouse."
        )
    )
    parser.add_argument(
        "--client-secrets",
        required=True,
        type=Path,
        help="Path to downloaded GCP OAuth2 Desktop client_secret_*.json (keep outside Git!)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Local loopback callback port (0 picks an open ephemeral port)",
    )
    parser.add_argument(
        "--root-folder-id",
        type=str,
        default="",
        help="Optional RAL_Ore_Warehouse Drive folder ID to verify immediately via 16-byte canary",
    )
    parser.add_argument(
        "--output-env",
        type=Path,
        default=None,
        help="Optional local .env file path to write secrets (never commit to Git)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if not args.client_secrets.exists():
        print(f"ERROR: client secrets file not found: {args.client_secrets}", file=sys.stderr)
        return 1

    result = mint_refresh_token(args.client_secrets, port=args.port)
    creds = result.pop("creds_obj")

    if args.root_folder_id:
        verify_root_folder_canary(creds, args.root_folder_id)
        print("Drive 16-byte preflight canary PASSED.")

    lines = [
        f"RAL_GDRIVE_CLIENT_ID={result['client_id']}",
        f"RAL_GDRIVE_CLIENT_SECRET={result['client_secret']}",
        f"RAL_GDRIVE_REFRESH_TOKEN={result['refresh_token']}",
    ]
    if args.root_folder_id:
        lines.append(f"RAL_GDRIVE_ROOT_FOLDER_ID={args.root_folder_id}")

    if args.output_env is not None:
        args.output_env.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"Wrote credentials to {args.output_env} (DO NOT commit this file to Git).")
    else:
        print("\n=== Store these as GitHub Actions Repository Secrets ===")
        for line in lines:
            print(line)

    print(
        "\nIMPORTANT: Ensure your GCP OAuth Consent Screen Publishing Status is set to "
        "'In production' so your refresh token does not expire every 7 days."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

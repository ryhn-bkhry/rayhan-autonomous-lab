"""Rate-limit-aware GitHub REST & Blue-Green Release Swap client (Sections 1.1, 1.3, 5)."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import time
from typing import Any, Callable, Mapping, Sequence
from urllib.parse import quote

from ral.canonicalize import sha256_bytes
from ral.storage.index_builder import decompress_sqlite_zst

DEFAULT_API_BASE = "https://api.github.com"
DEFAULT_UPLOAD_BASE = "https://uploads.github.com"
INDEX_RELEASE_TAG = "nightly-index"
LATEST_POINTER_NAME = "ral-index-latest.json"
CRITICAL_ALERT_LABEL = "ral-critical-alert"

SNAPSHOT_NAME_RE = re.compile(r"^ral-index-\d{8}-[A-Za-z0-9._-]+\.sqlite\.zst$")
_RUN_DATE_RE = re.compile(r"^\d{8}$")
_RUN_ID_RE = re.compile(r"^[A-Za-z0-9._-]+$")


class GitHubAPIError(RuntimeError):
    """Raised when a GitHub REST API request returns an unexpected non-retryable error."""

    def __init__(self, status_code: int, message: str, response_text: str = "") -> None:
        super().__init__(f"GitHub API HTTP {status_code}: {message}")
        self.status_code = int(status_code)
        self.response_text = response_text


class GitHubBudgetExceededError(RuntimeError):
    """Raised when nightly GitHub API calls exceed the configured safety budget (<40 calls)."""


class ReleaseIntegrityError(RuntimeError):
    """Raised when Blue-Green release asset checksum verification fails before pointer swap."""


@dataclass(frozen=True)
class RateLimitInfo:
    """Parsed GitHub REST rate-limit response headers."""

    remaining: int | None
    reset_epoch: int | None
    retry_after: int | None


@dataclass(frozen=True)
class BlueGreenSwapResult:
    """Outcome of an atomic Blue-Green Release index swap."""

    release_tag: str
    snapshot_name: str
    snapshot_asset_id: int
    snapshot_sha256: str
    snapshot_size_bytes: int
    pointer: dict[str, Any]
    kept_snapshots: tuple[str, ...]
    pruned_snapshots: tuple[str, ...]
    api_calls_used: int


def _default_request_fn(method: str, url: str, **kwargs: Any) -> Any:
    import requests

    return requests.request(method, url, **kwargs)


def _header_ci(headers: Mapping[str, Any] | None, target: str) -> str | None:
    if not headers:
        return None
    target_lower = target.lower()
    for k, v in headers.items():
        if str(k).lower() == target_lower:
            return str(v)
    return None


def _parse_int(raw: str | None) -> int | None:
    if raw is None:
        return None
    try:
        return int(raw.strip())
    except ValueError:
        return None


def _parse_rate_limit(headers: Mapping[str, Any] | None) -> RateLimitInfo:
    return RateLimitInfo(
        remaining=_parse_int(_header_ci(headers, "X-RateLimit-Remaining")),
        reset_epoch=_parse_int(_header_ci(headers, "X-RateLimit-Reset")),
        retry_after=_parse_int(_header_ci(headers, "Retry-After")),
    )


def _strip_upload_template(upload_url: str) -> str:
    return re.sub(r"\{[^}]*\}", "", upload_url).rstrip("/")


class GitHubClient:
    """
    Centralized rate-limit-aware GitHub REST helper enforcing the nightly <40-call budget,
    `X-RateLimit-Remaining`/`Retry-After` backoff, and Blue-Green Release index swaps.
    """

    def __init__(
        self,
        owner: str,
        repo: str,
        *,
        token: str | None = None,
        api_base: str = DEFAULT_API_BASE,
        upload_base: str = DEFAULT_UPLOAD_BASE,
        request_fn: Callable[..., Any] | None = None,
        sleep_fn: Callable[[float], None] | None = None,
        time_fn: Callable[[], float] | None = None,
        max_calls_budget: int = 40,
        max_retries: int = 4,
        low_remaining_threshold: int = 50,
    ) -> None:
        if not owner or not repo:
            raise ValueError("owner and repo must be non-empty")
        self.owner = owner.strip()
        self.repo = repo.strip()
        self._token = token if token is not None else os.environ.get("GITHUB_TOKEN")
        self.api_base = api_base.rstrip("/")
        self.upload_base = upload_base.rstrip("/")
        self._request_fn = request_fn or _default_request_fn
        self._sleep = sleep_fn or time.sleep
        self._now = time_fn or time.time
        self.max_calls_budget = int(max_calls_budget)
        self.max_retries = int(max_retries)
        self.low_remaining_threshold = int(low_remaining_threshold)

        self.api_call_count: int = 0
        self.last_rate_limit: RateLimitInfo | None = None
        self.backoff_delays: list[float] = []

    def _base_headers(self, extra: Mapping[str, str] | None = None) -> dict[str, str]:
        headers: dict[str, str] = {
            "Accept": "application/vnd.github+json",
            "User-Agent": "RayhansAutonomousLab/2.1",
            "X-GitHub-Api-Version": "2022-11-28",
        }
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        if extra:
            headers.update(extra)
        return headers

    def _compute_backoff_delay(
        self,
        *,
        attempt: int,
        rl: RateLimitInfo,
        max_cap_s: float = 60.0,
    ) -> float:
        if rl.retry_after is not None and rl.retry_after >= 0:
            return min(max_cap_s, float(rl.retry_after))
        if rl.remaining == 0 and rl.reset_epoch is not None:
            wait = float(rl.reset_epoch) - float(self._now()) + 1.0
            if wait > 0.0:
                return min(max_cap_s, wait)
        return min(30.0, float(2**attempt))

    def _request(
        self,
        method: str,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        params: Mapping[str, Any] | None = None,
        json_body: Any = None,
        data: bytes | None = None,
        expected_statuses: Sequence[int] = (200,),
        allow_404: bool = False,
        timeout: int = 30,
    ) -> Any:
        merged_headers = self._base_headers(headers)
        for attempt in range(self.max_retries + 1):
            if self.api_call_count >= self.max_calls_budget:
                raise GitHubBudgetExceededError(
                    f"GitHub API call budget exhausted ({self.api_call_count}/{self.max_calls_budget})"
                )
            self.api_call_count += 1
            resp = self._request_fn(
                method,
                url,
                headers=merged_headers,
                params=params,
                json=json_body,
                data=data,
                timeout=timeout,
            )
            status = int(getattr(resp, "status_code", 0))
            resp_headers = getattr(resp, "headers", None) or {}
            rl = _parse_rate_limit(resp_headers)
            self.last_rate_limit = rl

            if allow_404 and status == 404:
                return None

            body_text = str(getattr(resp, "text", "") or "")
            is_rate_limited = (
                status == 429
                or (
                    status == 403
                    and (
                        rl.remaining == 0
                        or rl.retry_after is not None
                        or "rate limit" in body_text.lower()
                    )
                )
            )
            is_transient_5xx = 500 <= status < 600

            if (is_rate_limited or is_transient_5xx) and attempt < self.max_retries:
                delay = self._compute_backoff_delay(attempt=attempt, rl=rl)
                self.backoff_delays.append(delay)
                self._sleep(delay)
                continue

            if status not in expected_statuses:
                raise GitHubAPIError(status, f"{method} {url} failed", response_text=body_text)

            if rl.remaining is not None and 0 < rl.remaining < self.low_remaining_threshold:
                throttle = 1.0
                self.backoff_delays.append(throttle)
                self._sleep(throttle)

            return resp

        raise GitHubAPIError(500, f"{method} {url} exceeded max retries")

    def _repo_url(self, suffix: str) -> str:
        return f"{self.api_base}/repos/{self.owner}/{self.repo}{suffix}"

    def get_or_create_release(
        self,
        tag: str = INDEX_RELEASE_TAG,
        *,
        name: str | None = None,
        body: str = "Ephemeral compiled RAL SQLite index snapshots (Blue-Green swapped).",
    ) -> dict[str, Any]:
        """Fetch release by tag, creating it if absent."""
        resp = self._request(
            "GET",
            self._repo_url(f"/releases/tags/{quote(tag, safe='')}"),
            expected_statuses=(200,),
            allow_404=True,
        )
        if resp is not None:
            return dict(resp.json())

        created = self._request(
            "POST",
            self._repo_url("/releases"),
            json_body={
                "tag_name": tag,
                "name": name or f"RAL {tag}",
                "body": body,
                "draft": False,
                "prerelease": False,
            },
            expected_statuses=(200, 201),
        )
        return dict(created.json())

    def list_release_assets(self, release_id: int) -> list[dict[str, Any]]:
        """List all assets attached to a release."""
        resp = self._request(
            "GET",
            self._repo_url(f"/releases/{int(release_id)}/assets"),
            params={"per_page": 100},
            expected_statuses=(200,),
        )
        payload = resp.json()
        return [dict(a) for a in payload] if isinstance(payload, list) else []

    def upload_release_asset(
        self,
        upload_url_template: str,
        asset_name: str,
        payload: bytes,
        *,
        content_type: str = "application/octet-stream",
    ) -> dict[str, Any]:
        """Upload a binary or JSON asset to a GitHub Release."""
        base = _strip_upload_template(upload_url_template)
        resp = self._request(
            "POST",
            base,
            headers={"Content-Type": content_type},
            params={"name": asset_name},
            data=payload,
            expected_statuses=(200, 201),
            timeout=120,
        )
        return dict(resp.json())

    def download_asset_bytes(self, download_url: str) -> bytes:
        """Download raw asset bytes via release download or API asset URL."""
        headers: dict[str, str] = {}
        if "/releases/assets/" in download_url:
            headers["Accept"] = "application/octet-stream"
        resp = self._request(
            "GET",
            download_url,
            headers=headers,
            expected_statuses=(200,),
            timeout=120,
        )
        content = getattr(resp, "content", None)
        if content is not None:
            return bytes(content)
        return str(getattr(resp, "text", "")).encode("utf-8")

    def delete_release_asset(self, asset_id: int) -> None:
        """Delete a GitHub Release asset by ID."""
        self._request(
            "DELETE",
            self._repo_url(f"/releases/assets/{int(asset_id)}"),
            expected_statuses=(200, 204),
            allow_404=True,
        )

    def blue_green_swap_index(
        self,
        zst_path: Path,
        *,
        run_date: str,
        run_id: str,
        embedding_model_id: str = "",
        embedding_dim: int = 768,
        node_count: int = 0,
        tag: str = INDEX_RELEASE_TAG,
        keep_snapshots: int = 2,
        verify_download: bool = True,
        now: datetime | None = None,
    ) -> BlueGreenSwapResult:
        """
        Blue-Green Release Swap (Sections 1.1 & 5):
        1. Upload versioned `ral-index-YYYYMMDD-{run_id}.sqlite.zst` (NEVER `--clobber`).
        2. Verify size + SHA-256 checksum BEFORE touching `ral-index-latest.json`.
        3. Swap ~300-byte pointer `ral-index-latest.json`.
        4. Prune older `.sqlite.zst` snapshots keeping the last `keep_snapshots` (default 2).
        """
        clean_date = run_date.replace("-", "").strip()
        if not _RUN_DATE_RE.match(clean_date):
            raise ValueError(f"run_date must be YYYYMMDD or YYYY-MM-DD, got {run_date!r}")
        clean_run_id = run_id.strip()
        if not _RUN_ID_RE.match(clean_run_id):
            raise ValueError(f"Invalid run_id for release asset name: {run_id!r}")

        snapshot_name = f"ral-index-{clean_date}-{clean_run_id}.sqlite.zst"
        if not SNAPSHOT_NAME_RE.match(snapshot_name):
            raise ValueError(f"Snapshot name violates schema: {snapshot_name!r}")

        payload = Path(zst_path).read_bytes()
        local_sha256 = sha256_bytes(payload)
        local_size = len(payload)

        calls_before = self.api_call_count
        release = self.get_or_create_release(tag)
        release_id = int(release["id"])
        upload_url = str(
            release.get("upload_url")
            or f"{self.upload_base}/repos/{self.owner}/{self.repo}/releases/{release_id}/assets"
        )
        existing_assets = list(release["assets"]) if "assets" in release else self.list_release_assets(release_id)

        uploaded = self.upload_release_asset(
            upload_url,
            snapshot_name,
            payload,
            content_type="application/zstd",
        )
        uploaded_id = int(uploaded["id"])
        download_url = str(
            uploaded.get("browser_download_url")
            or uploaded.get("url")
            or ""
        )

        # Verify uploaded asset integrity BEFORE touching `ral-index-latest.json`.
        try:
            remote_size = uploaded.get("size")
            if remote_size is not None and int(remote_size) != local_size:
                raise ReleaseIntegrityError(
                    f"Uploaded size mismatch for {snapshot_name}: expected {local_size}, got {remote_size}"
                )
            remote_digest = uploaded.get("digest")
            if isinstance(remote_digest, str) and remote_digest.startswith("sha256:"):
                if remote_digest.removeprefix("sha256:").lower() != local_sha256:
                    raise ReleaseIntegrityError(
                        f"Uploaded digest mismatch for {snapshot_name}: {remote_digest} != sha256:{local_sha256}"
                    )
            if verify_download:
                if not download_url:
                    raise ReleaseIntegrityError(f"Uploaded asset {snapshot_name} returned no download URL")
                remote_bytes = self.download_asset_bytes(download_url)
                remote_sha = sha256_bytes(remote_bytes)
                if remote_sha != local_sha256:
                    raise ReleaseIntegrityError(
                        f"Downloaded checksum mismatch for {snapshot_name}: {remote_sha} != {local_sha256}"
                    )
        except Exception:
            try:
                self.delete_release_asset(uploaded_id)
            except Exception:
                pass
            raise

        ts_iso = (now or datetime.now(timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")
        pointer: dict[str, Any] = {
            "schema_version": "2.1",
            "asset_name": snapshot_name,
            "asset_id": uploaded_id,
            "download_url": download_url,
            "sha256": local_sha256,
            "size_bytes": local_size,
            "run_id": clean_run_id,
            "embedding_model_id": embedding_model_id,
            "embedding_dim": int(embedding_dim),
            "node_count": int(node_count),
            "updated_at": ts_iso,
        }
        pointer_bytes = (json.dumps(pointer, separators=(",", ":")) + "\n").encode("utf-8")

        for asset in existing_assets:
            if asset.get("name") == LATEST_POINTER_NAME:
                self.delete_release_asset(int(asset["id"]))

        self.upload_release_asset(
            upload_url,
            LATEST_POINTER_NAME,
            pointer_bytes,
            content_type="application/json",
        )

        # Prune older `.sqlite.zst` snapshots keeping the last `keep_snapshots` (default 2).
        all_assets = self.list_release_assets(release_id)
        snapshots = [
            a for a in all_assets if SNAPSHOT_NAME_RE.match(str(a.get("name") or ""))
        ]
        other_snapshots = sorted(
            [a for a in snapshots if int(a["id"]) != uploaded_id],
            key=lambda a: (str(a.get("created_at") or ""), str(a.get("name") or "")),
            reverse=True,
        )
        ordered = [uploaded] + other_snapshots
        keep_n = max(1, int(keep_snapshots))
        kept = ordered[:keep_n]
        to_prune = ordered[keep_n:]
        for old_asset in to_prune:
            self.delete_release_asset(int(old_asset["id"]))

        return BlueGreenSwapResult(
            release_tag=tag,
            snapshot_name=snapshot_name,
            snapshot_asset_id=uploaded_id,
            snapshot_sha256=local_sha256,
            snapshot_size_bytes=local_size,
            pointer=pointer,
            kept_snapshots=tuple(str(a["name"]) for a in kept),
            pruned_snapshots=tuple(str(a["name"]) for a in to_prune),
            api_calls_used=self.api_call_count - calls_before,
        )

    def fetch_latest_index_pointer(self, tag: str = INDEX_RELEASE_TAG) -> dict[str, Any] | None:
        """Fetch and parse `ral-index-latest.json` from the release tag, or return None."""
        resp = self._request(
            "GET",
            self._repo_url(f"/releases/tags/{quote(tag, safe='')}"),
            expected_statuses=(200,),
            allow_404=True,
        )
        if resp is None:
            return None
        release = resp.json()
        assets = release.get("assets") or self.list_release_assets(int(release["id"]))
        for asset in assets:
            if asset.get("name") == LATEST_POINTER_NAME:
                dl_url = str(asset.get("browser_download_url") or asset.get("url") or "")
                raw = self.download_asset_bytes(dl_url)
                parsed = json.loads(raw.decode("utf-8"))
                return dict(parsed) if isinstance(parsed, Mapping) else None
        return None

    def download_latest_index(
        self,
        output_sqlite_path: Path,
        *,
        tag: str = INDEX_RELEASE_TAG,
        cache_dir: Path | None = None,
    ) -> tuple[Path, dict[str, Any]] | None:
        """
        Download the active index snapshot pointed to by `ral-index-latest.json`, verify
        SHA-256 BEFORE decompressing, and write the SQLite DB to `output_sqlite_path`.
        """
        pointer = self.fetch_latest_index_pointer(tag=tag)
        if pointer is None:
            return None
        dl_url = str(pointer.get("download_url") or "")
        expected_sha = str(pointer.get("sha256") or "")
        asset_name = str(pointer.get("asset_name") or "ral-index-latest.sqlite.zst")
        raw_zst = self.download_asset_bytes(dl_url)

        stage_dir = Path(cache_dir) if cache_dir is not None else Path(output_sqlite_path).parent
        stage_dir.mkdir(parents=True, exist_ok=True)
        zst_path = stage_dir / asset_name
        zst_path.write_bytes(raw_zst)

        db_out = decompress_sqlite_zst(
            zst_path,
            Path(output_sqlite_path),
            expected_sha256=expected_sha,
        )
        return db_out, pointer

    def upsert_alert_issue(
        self,
        title: str,
        body: str,
        *,
        label: str = CRITICAL_ALERT_LABEL,
    ) -> dict[str, Any]:
        """
        Open or update a single Dead-Man's Switch alert Issue labeled `ral-critical-alert`
        without creating duplicate issues across consecutive failing nights.
        """
        resp = self._request(
            "GET",
            self._repo_url("/issues"),
            params={"labels": label, "state": "open", "per_page": 10},
            expected_statuses=(200,),
        )
        existing = resp.json()
        if isinstance(existing, list) and existing:
            issue_num = int(existing[0]["number"])
            upd = self._request(
                "PATCH",
                self._repo_url(f"/issues/{issue_num}"),
                json_body={"title": title, "body": body},
                expected_statuses=(200,),
            )
            return dict(upd.json())

        created = self._request(
            "POST",
            self._repo_url("/issues"),
            json_body={"title": title, "body": body, "labels": [label]},
            expected_statuses=(200, 201),
        )
        return dict(created.json())

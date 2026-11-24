"""Content-addressed Google Drive Ore warehouse with MD5 verification and 16-byte canary (Sections 1.1, 6.3, 10)."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any, Callable, Mapping

DRIVE_FILE_SCOPE = "https://www.googleapis.com/auth/drive.file"
DRIVE_FOLDER_MIME = "application/vnd.google-apps.folder"
DEFAULT_WAREHOUSE_NAME = "RAL_Ore_Warehouse"

_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_SAFE_EXT_RE = re.compile(r"^[A-Za-z0-9]+(?:\.[A-Za-z0-9]+)*$")


class OreIntegrityError(RuntimeError):
    """Raised when SHA-256 or remote Google Drive MD5 verification fails."""


class DriveAuthError(RuntimeError):
    """Raised when Google Drive credentials or client dependencies are missing/invalid."""


@dataclass(frozen=True)
class OreUploadResult:
    """Metadata returned by a verified content-addressed Ore upload."""

    file_id: str
    ore_sha256: str
    md5_checksum: str
    size_bytes: int
    ore_uri: str
    relative_path: str
    deduplicated: bool = False


@dataclass(frozen=True)
class CanaryResult:
    """Result of the 16-byte Google Drive preflight write/read/delete canary."""

    ok: bool
    canary_name: str
    canary_size_bytes: int
    local_md5: str
    remote_md5: str | None = None
    cleaned_up: bool = False
    error: str | None = None


class _InMemoryMediaBody:
    """Lightweight media wrapper used when googleapiclient is absent in unit tests."""

    def __init__(self, payload: bytes, mime_type: str) -> None:
        self.payload = payload
        self.mime_type = mime_type
        self._stream = io.BytesIO(payload)

    def size(self) -> int:
        return len(self.payload)

    def read(self, size: int = -1) -> bytes:
        return self._stream.read(size)

    def getbytes(self, begin: int = 0, length: int | None = None) -> bytes:
        if length is None:
            return self.payload[begin:]
        return self.payload[begin : begin + length]


def _default_media_factory(payload: bytes, mime_type: str) -> Any:
    try:
        from googleapiclient.http import MediaIoBaseUpload

        return MediaIoBaseUpload(io.BytesIO(payload), mimetype=mime_type, resumable=True)
    except ImportError:
        return _InMemoryMediaBody(payload, mime_type)


def _normalize_sha256(sha256_hex: str) -> str:
    cleaned = sha256_hex.removeprefix("sha256:").strip().lower()
    if not _SHA256_RE.match(cleaned):
        raise ValueError(f"Expected 64-char hex SHA-256, got {sha256_hex!r}")
    return cleaned


def _normalize_ext(ext: str) -> str:
    cleaned = ext.lstrip(".").strip()
    if not cleaned or ".." in cleaned or "/" in cleaned or "\\" in cleaned or not _SAFE_EXT_RE.match(cleaned):
        raise ValueError(f"Unsafe or empty Ore extension: {ext!r}")
    return cleaned


def build_ore_relative_path(sha256_hex: str, ext: str) -> str:
    """Build `by-sha256/{h[0:2]}/{h[2:4]}/{sha256}.{ext}` relative object path."""
    h = _normalize_sha256(sha256_hex)
    clean_ext = _normalize_ext(ext)
    return f"by-sha256/{h[0:2]}/{h[2:4]}/{h}.{clean_ext}"


def build_ore_uri(
    sha256_hex: str,
    ext: str,
    warehouse_name: str = DEFAULT_WAREHOUSE_NAME,
) -> str:
    """Build canonical `gdrive://{warehouse}/by-sha256/...` URI."""
    rel = build_ore_relative_path(sha256_hex, ext)
    return f"gdrive://{warehouse_name}/{rel}"


def compute_bytes_hashes(data: bytes) -> tuple[str, str]:
    """Return `(sha256_hex, md5_hex)` for an in-memory payload."""
    return hashlib.sha256(data).hexdigest(), hashlib.md5(data).hexdigest()


def compute_file_hashes(path: Path) -> tuple[str, str]:
    """Return `(sha256_hex, md5_hex)` for a file on disk using streaming chunks."""
    sha = hashlib.sha256()
    md5 = hashlib.md5()
    with path.open("rb") as handle:
        while chunk := handle.read(65536):
            sha.update(chunk)
            md5.update(chunk)
    return sha.hexdigest(), md5.hexdigest()


def _escape_drive_query(value: str) -> str:
    return value.replace("\\", "\\\\").replace("'", "\\'")


def _infer_ore_extension(path: Path) -> str:
    lower_name = path.name.lower()
    for multi in (".tar.gz", ".tar.bz2", ".tar.xz", ".tar.zst"):
        if lower_name.endswith(multi):
            return multi.lstrip(".")
    return path.suffix.lstrip(".")


class DriveOreStore:
    """
    Content-addressed Google Drive Ore archive implementing Verify-Before-Archive 2PC.

    Objects live at `by-sha256/{h[0:2]}/{h[2:4]}/{sha256}.{ext}` under `root_folder_id`.
    """

    def __init__(
        self,
        service: Any,
        root_folder_id: str,
        *,
        warehouse_name: str = DEFAULT_WAREHOUSE_NAME,
        media_factory: Callable[[bytes, str], Any] | None = None,
    ) -> None:
        if not root_folder_id or not root_folder_id.strip():
            raise ValueError("root_folder_id must be a non-empty Drive folder ID")
        self._service = service
        self.root_folder_id = root_folder_id.strip()
        self.warehouse_name = warehouse_name
        self._media_factory = media_factory or _default_media_factory
        self._folder_cache: dict[tuple[str, str], str] = {}

    def _find_item_in_parent(
        self,
        name: str,
        parent_id: str,
        *,
        mime_type: str | None = None,
    ) -> dict[str, Any] | None:
        clauses = [
            f"name = '{_escape_drive_query(name)}'",
            f"'{_escape_drive_query(parent_id)}' in parents",
            "trashed = false",
        ]
        if mime_type is not None:
            clauses.append(f"mimeType = '{_escape_drive_query(mime_type)}'")
        query = " and ".join(clauses)
        resp = (
            self._service.files()
            .list(
                q=query,
                spaces="drive",
                fields="files(id, name, md5Checksum, size)",
                supportsAllDrives=True,
                includeItemsFromAllDrives=True,
                pageSize=1,
            )
            .execute()
        )
        files = resp.get("files", [])
        return files[0] if files else None

    def _ensure_folder(self, name: str, parent_id: str) -> str:
        cache_key = (parent_id, name)
        if cache_key in self._folder_cache:
            return self._folder_cache[cache_key]

        existing = self._find_item_in_parent(name, parent_id, mime_type=DRIVE_FOLDER_MIME)
        if existing is not None:
            folder_id = str(existing["id"])
            self._folder_cache[cache_key] = folder_id
            return folder_id

        created = (
            self._service.files()
            .create(
                body={"name": name, "mimeType": DRIVE_FOLDER_MIME, "parents": [parent_id]},
                fields="id",
                supportsAllDrives=True,
            )
            .execute()
        )
        folder_id = str(created["id"])
        self._folder_cache[cache_key] = folder_id
        return folder_id

    def ensure_shard_folder(self, sha256_hex: str) -> str:
        """Ensure `by-sha256/{h[0:2]}/{h[2:4]}` directory hierarchy exists and return leaf ID."""
        h = _normalize_sha256(sha256_hex)
        by_sha_id = self._ensure_folder("by-sha256", self.root_folder_id)
        level1_id = self._ensure_folder(h[0:2], by_sha_id)
        return self._ensure_folder(h[2:4], level1_id)

    def _delete_file_safe(self, file_id: str) -> None:
        self._service.files().delete(fileId=file_id, supportsAllDrives=True).execute()

    def upload_ore_bytes(
        self,
        payload: bytes,
        ext: str,
        *,
        mime_type: str = "application/octet-stream",
        expected_sha256: str | None = None,
    ) -> OreUploadResult:
        """
        Upload raw Ore bytes to `by-sha256/{h[0:2]}/{h[2:4]}/{sha256}.{ext}` and verify remote MD5.

        If an object with matching name and matching MD5 already exists in the shard folder,
        returns `deduplicated=True` without re-uploading. Deletes the uploaded object and raises
        `OreIntegrityError` if remote MD5 verification fails.
        """
        clean_ext = _normalize_ext(ext)
        local_sha256, local_md5 = compute_bytes_hashes(payload)
        if expected_sha256 is not None and local_sha256 != _normalize_sha256(expected_sha256):
            raise OreIntegrityError(
                f"Local SHA-256 {local_sha256} does not match expected {expected_sha256}"
            )

        rel_path = build_ore_relative_path(local_sha256, clean_ext)
        ore_uri = build_ore_uri(local_sha256, clean_ext, warehouse_name=self.warehouse_name)
        filename = f"{local_sha256}.{clean_ext}"
        shard_folder_id = self.ensure_shard_folder(local_sha256)

        existing = self._find_item_in_parent(filename, shard_folder_id)
        if existing is not None:
            existing_md5 = (existing.get("md5Checksum") or "").lower()
            if existing_md5 == local_md5:
                return OreUploadResult(
                    file_id=str(existing["id"]),
                    ore_sha256=local_sha256,
                    md5_checksum=existing_md5,
                    size_bytes=int(existing.get("size") or len(payload)),
                    ore_uri=ore_uri,
                    relative_path=rel_path,
                    deduplicated=True,
                )
            raise OreIntegrityError(
                f"Remote object {filename} exists with mismatched MD5 {existing_md5!r} != {local_md5!r}"
            )

        media_body = self._media_factory(payload, mime_type)
        created = (
            self._service.files()
            .create(
                body={"name": filename, "parents": [shard_folder_id]},
                media_body=media_body,
                fields="id, md5Checksum, size",
                supportsAllDrives=True,
            )
            .execute()
        )
        file_id = str(created["id"])
        remote_md5 = created.get("md5Checksum")
        remote_size = created.get("size")

        if not remote_md5:
            meta = (
                self._service.files()
                .get(fileId=file_id, fields="id, md5Checksum, size", supportsAllDrives=True)
                .execute()
            )
            remote_md5 = meta.get("md5Checksum")
            remote_size = meta.get("size", remote_size)

        if not remote_md5 or str(remote_md5).lower() != local_md5:
            try:
                self._delete_file_safe(file_id)
            except Exception:
                pass
            raise OreIntegrityError(
                f"Remote MD5 mismatch for {filename}: expected {local_md5}, got {remote_md5!r}"
            )

        return OreUploadResult(
            file_id=file_id,
            ore_sha256=local_sha256,
            md5_checksum=str(remote_md5).lower(),
            size_bytes=int(remote_size if remote_size is not None else len(payload)),
            ore_uri=ore_uri,
            relative_path=rel_path,
            deduplicated=False,
        )

    def upload_ore_file(
        self,
        local_path: Path,
        ext: str | None = None,
        *,
        mime_type: str = "application/octet-stream",
        expected_sha256: str | None = None,
    ) -> OreUploadResult:
        """Read a staged local Ore file and perform content-addressed Drive 2PC upload."""
        path = Path(local_path)
        inferred_ext = ext if ext is not None else _infer_ore_extension(path)
        payload = path.read_bytes()
        return self.upload_ore_bytes(
            payload,
            inferred_ext,
            mime_type=mime_type,
            expected_sha256=expected_sha256,
        )

    def run_preflight_canary(self) -> CanaryResult:
        """
        Execute the 16-byte Drive write / MD5 + read-back / delete preflight canary.

        Never raises on Drive errors; returns `CanaryResult(ok=False, ...)` so Job 0 can
        trip `circuit_breaker="DRIVE_AUTH_FAIL"` cleanly.
        """
        payload = os.urandom(16)
        _, local_md5 = compute_bytes_hashes(payload)
        canary_name = f"_ral_canary_{os.urandom(6).hex()}.bin"
        canary_id: str | None = None
        remote_md5: str | None = None
        cleaned_up = False
        error: str | None = None

        try:
            created = (
                self._service.files()
                .create(
                    body={"name": canary_name, "parents": [self.root_folder_id]},
                    media_body=self._media_factory(payload, "application/octet-stream"),
                    fields="id, md5Checksum",
                    supportsAllDrives=True,
                )
                .execute()
            )
            canary_id = str(created["id"])
            remote_md5 = created.get("md5Checksum")
            if not remote_md5:
                meta = (
                    self._service.files()
                    .get(fileId=canary_id, fields="id, md5Checksum", supportsAllDrives=True)
                    .execute()
                )
                remote_md5 = meta.get("md5Checksum")

            if not remote_md5 or str(remote_md5).lower() != local_md5:
                error = f"Canary remote MD5 mismatch: expected {local_md5}, got {remote_md5!r}"
            else:
                downloaded = self._service.files().get_media(fileId=canary_id).execute()
                if downloaded != payload:
                    error = "Canary read-back payload did not match written 16 bytes"
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        finally:
            if canary_id is not None:
                try:
                    self._delete_file_safe(canary_id)
                    cleaned_up = True
                except Exception as del_exc:
                    cleanup_err = f"Canary cleanup failed: {type(del_exc).__name__}: {del_exc}"
                    error = f"{error}; {cleanup_err}" if error else cleanup_err

        return CanaryResult(
            ok=(error is None and cleaned_up),
            canary_name=canary_name,
            canary_size_bytes=len(payload),
            local_md5=local_md5,
            remote_md5=str(remote_md5).lower() if remote_md5 else None,
            cleaned_up=cleaned_up,
            error=error,
        )


def build_drive_service_from_env(environ: Mapping[str, str] | None = None) -> tuple[Any, str]:
    """
    Construct an authenticated Google Drive v3 client from environment variables (Section 10).

    Supports Option A (`RAL_GDRIVE_SA_JSON`) and Option B (`RAL_GDRIVE_CLIENT_ID`,
    `RAL_GDRIVE_CLIENT_SECRET`, `RAL_GDRIVE_REFRESH_TOKEN`), plus `RAL_GDRIVE_ROOT_FOLDER_ID`.
    """
    env = environ if environ is not None else os.environ
    root_folder_id = (env.get("RAL_GDRIVE_ROOT_FOLDER_ID") or "").strip()
    if not root_folder_id:
        raise DriveAuthError("Missing RAL_GDRIVE_ROOT_FOLDER_ID environment variable")

    try:
        from googleapiclient.discovery import build
    except ImportError as exc:
        raise DriveAuthError(
            "google-api-python-client is required for live Google Drive operations"
        ) from exc

    sa_json = (env.get("RAL_GDRIVE_SA_JSON") or "").strip()
    if sa_json:
        from google.oauth2.service_account import Credentials as SACredentials

        try:
            sa_info = json.loads(sa_json)
        except json.JSONDecodeError as exc:
            raise DriveAuthError("RAL_GDRIVE_SA_JSON is not valid JSON") from exc
        creds = SACredentials.from_service_account_info(sa_info, scopes=[DRIVE_FILE_SCOPE])
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return service, root_folder_id

    client_id = (env.get("RAL_GDRIVE_CLIENT_ID") or "").strip()
    client_secret = (env.get("RAL_GDRIVE_CLIENT_SECRET") or "").strip()
    refresh_token = (env.get("RAL_GDRIVE_REFRESH_TOKEN") or "").strip()
    if client_id and client_secret and refresh_token:
        from google.oauth2.credentials import Credentials as UserCredentials

        creds = UserCredentials(
            token=None,
            refresh_token=refresh_token,
            token_uri="https://oauth2.googleapis.com/token",
            client_id=client_id,
            client_secret=client_secret,
            scopes=[DRIVE_FILE_SCOPE],
        )
        service = build("drive", "v3", credentials=creds, cache_discovery=False)
        return service, root_folder_id

    raise DriveAuthError(
        "Missing Drive credentials: set RAL_GDRIVE_SA_JSON or "
        "(RAL_GDRIVE_CLIENT_ID, RAL_GDRIVE_CLIENT_SECRET, RAL_GDRIVE_REFRESH_TOKEN)"
    )

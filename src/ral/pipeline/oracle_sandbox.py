"""Air-gapped Docker execution harness for untrusted verification kernels (Section 6.1)."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
import re
import subprocess
import tempfile
import time
from typing import Callable, Mapping

from ral.pipeline.ast_linter import lint_verification_script
from ral.pipeline.oracles.base import OracleResult

_SAFE_SCRIPT_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+\.py$")
_SAFE_EXTRA_NAME_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

_SANDBOX_CONTAINER_ENV: tuple[str, ...] = (
    "HOME=/tmp",
    "HYPOTHESIS_STORAGE_DIRECTORY=/tmp/.hypothesis",
    "MPLCONFIGDIR=/tmp/matplotlib",
)

_SAFE_HOST_ENV_KEYS: frozenset[str] = frozenset(
    {
        "PATH",
        "SYSTEMROOT",
        "WINDIR",
        "COMSPEC",
        "DOCKER_HOST",
        "DOCKER_TLS_VERIFY",
        "DOCKER_CERT_PATH",
        "DOCKER_CONTEXT",
        "TMP",
        "TEMP",
    }
)


@dataclass(frozen=True)
class SandboxConfig:
    """Resource and isolation limits enforced on every sandbox container."""

    image: str = "ral-sandbox:latest"
    memory: str = "1g"
    memory_swap: str = "1g"
    cpus: str = "1"
    pids_limit: int = 64
    user: str = "65534:65534"
    tmpfs_spec: str = "/tmp:rw,noexec,nosuid,size=64m"
    default_timeout_s: int = 30
    kill_grace_s: int = 5
    docker_bin: str = "docker"


def scrubbed_host_env(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """
    Build a minimal host environment for invoking the Docker CLI.

    Drops all secret-bearing variables (RAL_*, GEMINI_*, GDRIVE_*, GITHUB_*, *_KEY, *_TOKEN).
    """
    source = environ if environ is not None else os.environ
    return {k: v for k, v in source.items() if k in _SAFE_HOST_ENV_KEYS}


def _validate_single_segment_name(filename: str, *, pattern: re.Pattern[str]) -> str:
    if not filename or "/" in filename or "\\" in filename or ".." in filename:
        raise ValueError(f"Unsafe sandbox filename: {filename!r}")
    if not pattern.match(filename):
        raise ValueError(f"Invalid sandbox filename: {filename!r}")
    return filename


class OracleSandbox:
    """Executes Python verification scripts inside an air-gapped read-only Docker container."""

    def __init__(
        self,
        config: SandboxConfig | None = None,
        runner: Callable[..., subprocess.CompletedProcess[str]] | None = None,
    ) -> None:
        self.config = config or SandboxConfig()
        self._runner = runner or subprocess.run

    def build_docker_command(
        self,
        script_dir: Path,
        script_name: str = "verify.py",
        timeout_s: int | None = None,
    ) -> list[str]:
        """Construct exact Section 6.1 air-gapped Docker invocation with zero secret env flags."""
        safe_name = _validate_single_segment_name(script_name, pattern=_SAFE_SCRIPT_NAME_RE)
        effective_timeout = int(timeout_s if timeout_s is not None else self.config.default_timeout_s)
        if effective_timeout <= 0:
            raise ValueError("timeout_s must be positive")

        resolved_dir = str(Path(script_dir).resolve())
        cfg = self.config
        cmd = [
            cfg.docker_bin,
            "run",
            "--rm",
            "--network=none",
            "--read-only",
            "--cap-drop=ALL",
            "--security-opt=no-new-privileges",
            f"--memory={cfg.memory}",
            f"--memory-swap={cfg.memory_swap}",
            f"--cpus={cfg.cpus}",
            f"--pids-limit={cfg.pids_limit}",
            f"--user={cfg.user}",
            "--tmpfs",
            cfg.tmpfs_spec,
        ]
        for env_entry in _SANDBOX_CONTAINER_ENV:
            cmd.extend(["-e", env_entry])
        cmd.extend(
            [
                "-v",
                f"{resolved_dir}:/work:ro",
                cfg.image,
                "timeout",
                "-k",
                str(cfg.kill_grace_s),
                str(effective_timeout),
                "python3",
                f"/work/{safe_name}",
            ]
        )
        return cmd

    def run_script(
        self,
        source_code: str,
        *,
        plugin_name: str = "sandbox",
        script_name: str = "verify.py",
        timeout_s: int | None = None,
        enforce_ast_lint: bool = True,
        min_assertions: int = 0,
        extra_files: Mapping[str, bytes] | None = None,
    ) -> OracleResult:
        """
        Pre-lint and execute `source_code` inside the air-gapped container.

        When `enforce_ast_lint` is True, scripts importing forbidden modules (`os`, `socket`,
        `subprocess`, etc.) or using forbidden builtins are rejected before spawning Docker.
        """
        t0 = time.monotonic()
        safe_name = _validate_single_segment_name(script_name, pattern=_SAFE_SCRIPT_NAME_RE)
        effective_timeout = int(timeout_s if timeout_s is not None else self.config.default_timeout_s)

        if enforce_ast_lint:
            violations = lint_verification_script(
                source_code,
                min_assertions=min_assertions,
                check_anti_cheat=(min_assertions > 0),
            )
            if violations:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                return OracleResult(
                    passed=False,
                    plugin_name=plugin_name,
                    stdout="",
                    stderr="AST_LINT_REJECTED: " + "; ".join(violations),
                    elapsed_ms=elapsed_ms,
                )

        if extra_files:
            for extra_name in extra_files:
                _validate_single_segment_name(extra_name, pattern=_SAFE_EXTRA_NAME_RE)
                if extra_name == safe_name:
                    raise ValueError(f"extra_files collides with script_name {safe_name!r}")

        with tempfile.TemporaryDirectory(prefix="ral_sandbox_") as tmpdir_str:
            tmpdir = Path(tmpdir_str)
            script_path = tmpdir / safe_name
            script_path.write_text(source_code, encoding="utf-8")

            if extra_files:
                for extra_name, content in extra_files.items():
                    (tmpdir / extra_name).write_bytes(content)

            if os.name != "nt":
                os.chmod(tmpdir, 0o755)
                os.chmod(script_path, 0o644)

            cmd = self.build_docker_command(
                tmpdir,
                script_name=safe_name,
                timeout_s=effective_timeout,
            )
            host_timeout = effective_timeout + self.config.kill_grace_s + 10
            try:
                proc = self._runner(
                    cmd,
                    capture_output=True,
                    text=True,
                    timeout=host_timeout,
                    env=scrubbed_host_env(),
                    check=False,
                )
            except subprocess.TimeoutExpired as exc:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                stdout = exc.stdout if isinstance(exc.stdout, str) else (exc.stdout or b"").decode("utf-8", "replace")
                stderr = exc.stderr if isinstance(exc.stderr, str) else (exc.stderr or b"").decode("utf-8", "replace")
                return OracleResult(
                    passed=False,
                    plugin_name=plugin_name,
                    stdout=stdout,
                    stderr=f"SANDBOX_TIMEOUT({effective_timeout}s): {stderr}".strip(),
                    elapsed_ms=elapsed_ms,
                )
            except OSError as exc:
                elapsed_ms = int((time.monotonic() - t0) * 1000)
                return OracleResult(
                    passed=False,
                    plugin_name=plugin_name,
                    stdout="",
                    stderr=f"SANDBOX_EXEC_ERROR: {type(exc).__name__}: {exc}",
                    elapsed_ms=elapsed_ms,
                )

            elapsed_ms = int((time.monotonic() - t0) * 1000)
            return OracleResult(
                passed=(proc.returncode == 0),
                plugin_name=plugin_name,
                stdout=proc.stdout or "",
                stderr=proc.stderr or "",
                elapsed_ms=elapsed_ms,
            )

    def is_docker_available(self) -> bool:
        """Return True if the Docker daemon responds to `docker info`."""
        try:
            proc = self._runner(
                [self.config.docker_bin, "info"],
                capture_output=True,
                text=True,
                timeout=10,
                env=scrubbed_host_env(),
                check=False,
            )
            return proc.returncode == 0
        except OSError:
            return False

    def is_image_available(self) -> bool:
        """Return True if the configured sandbox image exists locally."""
        try:
            proc = self._runner(
                [self.config.docker_bin, "image", "inspect", self.config.image],
                capture_output=True,
                text=True,
                timeout=10,
                env=scrubbed_host_env(),
                check=False,
            )
            return proc.returncode == 0
        except OSError:
            return False

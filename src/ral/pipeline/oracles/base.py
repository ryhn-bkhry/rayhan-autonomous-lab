"""Abstract OraclePlugin interface and plugin registry (Blueprint Section 6.2)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml


@dataclass(frozen=True)
class OracleResult:
    """Result returned by executing a verification or falsification kernel."""

    passed: bool
    plugin_name: str
    stdout: str
    stderr: str
    elapsed_ms: int


class OraclePlugin(ABC):
    """Abstract base contract for pluggable deterministic verification oracles."""

    name: str
    enabled_by_default: bool = True

    @abstractmethod
    def lint(self, source_code: str) -> list[str]:
        """Return list of static anti-cheating violations; empty means pass."""

    @abstractmethod
    def execute(self, source_code: str, timeout_s: int = 30) -> OracleResult:
        """Execute verification kernel inside air-gapped Docker sandbox."""


class OraclePluginRegistry:
    """Registry managing enabled oracle plugins backed by config/oracle_plugins.yaml."""

    def __init__(self) -> None:
        self._plugins: dict[str, OraclePlugin] = {}

    def register(self, plugin: OraclePlugin) -> None:
        """Register an OraclePlugin instance under its unique name."""
        if not getattr(plugin, "name", None):
            raise ValueError("OraclePlugin must define a non-empty 'name' attribute")
        self._plugins[plugin.name] = plugin

    def get(self, name: str) -> OraclePlugin:
        """Retrieve a registered plugin by name."""
        if name not in self._plugins:
            raise KeyError(f"OraclePlugin '{name}' is not registered")
        return self._plugins[name]

    def list_registered(self) -> list[str]:
        """Return sorted names of registered plugins."""
        return sorted(self._plugins.keys())

    @staticmethod
    def load_config(path: Path = Path("config/oracle_plugins.yaml")) -> dict[str, Any]:
        """Load plugin configuration YAML."""
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    @classmethod
    def is_enabled(
        cls,
        name: str,
        config: dict[str, Any] | None = None,
        config_path: Path = Path("config/oracle_plugins.yaml"),
    ) -> bool:
        """Check whether a plugin is enabled in config/oracle_plugins.yaml."""
        cfg = config if config is not None else cls.load_config(config_path)
        plugins = cfg.get("plugins", {})
        if name not in plugins:
            return False
        return bool(plugins[name].get("enabled", False))

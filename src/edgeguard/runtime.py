"""Explicit filesystem contract for compatibility and readiness execution."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RuntimePathContract:
    """All mutable runtime roots supplied by a wrapper instead of inferred by platform."""

    runtime_current_root: Path
    runtime_py311_root: Path
    checkout_root: Path
    evidence_root: Path
    log_root: Path
    cache_root: Path
    data_root: Path

    @classmethod
    def from_workspace(cls, workspace_root: Path) -> RuntimePathContract:
        """Build the deterministic local layout beneath one caller-selected workspace."""
        root = workspace_root.resolve()
        return cls(
            runtime_current_root=root / "runtime-current",
            runtime_py311_root=root / "runtime-py311",
            checkout_root=root / "checkouts",
            evidence_root=root / "evidence",
            log_root=root / "logs",
            cache_root=root / "cache",
            data_root=root / "data",
        ).validated()

    def validated(self, *, forbid_content: bool = False) -> RuntimePathContract:
        """Resolve, de-alias, and optionally reject Colab roots for local execution."""
        values = {name: Path(value).resolve() for name, value in self.as_dict().items()}
        if len(set(values.values())) != len(values):
            raise ValueError("runtime path contract entries must be distinct")
        for name, value in values.items():
            if value == Path(value.anchor):
                raise ValueError(f"runtime path {name} cannot be a filesystem root")
            if forbid_content and (value == Path("/content") or Path("/content") in value.parents):
                raise ValueError("local readiness paths cannot point to /content")
        return RuntimePathContract(
            runtime_current_root=values["runtime_current_root"],
            runtime_py311_root=values["runtime_py311_root"],
            checkout_root=values["checkout_root"],
            evidence_root=values["evidence_root"],
            log_root=values["log_root"],
            cache_root=values["cache_root"],
            data_root=values["data_root"],
        )

    def as_dict(self) -> dict[str, Path]:
        """Return named paths without platform-specific interpretation."""
        return {
            "runtime_current_root": self.runtime_current_root,
            "runtime_py311_root": self.runtime_py311_root,
            "checkout_root": self.checkout_root,
            "evidence_root": self.evidence_root,
            "log_root": self.log_root,
            "cache_root": self.cache_root,
            "data_root": self.data_root,
        }

    def receipt(self) -> dict[str, Any]:
        """Return a path-free layout receipt suitable for portable evidence."""
        return {
            "schema_version": "1.0",
            "record_type": "edgeguard_runtime_path_contract",
            "entries": {name: path.name for name, path in sorted(self.as_dict().items())},
        }

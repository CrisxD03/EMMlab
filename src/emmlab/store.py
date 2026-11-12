"""Content-addressed artifact store: every stage output keyed by hash(stage, inputs)."""

from __future__ import annotations

import hashlib
import importlib.metadata as md
import io
import json
import os
import platform
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

PACKAGES = ["emmlab", "numpy", "pandas", "scipy", "statsmodels", "pysubgroup", "pydantic", "pyarrow"]


def versions() -> dict[str, str]:
    out = {"python": platform.python_version()}
    for p in PACKAGES:
        try:
            out[p] = md.version(p)
        except md.PackageNotFoundError:
            out[p] = "?"
    return out


def stage_key(stage: str, inputs: dict[str, Any]) -> str:
    blob = json.dumps({"stage": stage, "inputs": inputs}, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:20]


def dumps_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, indent=1, default=_default)


def _default(o: Any) -> Any:
    import numpy as np
    if isinstance(o, np.bool_):
        return bool(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, tuple):
        return list(o)
    raise TypeError(type(o))


def parquet_bytes(df: pd.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.to_parquet(buf, index=False, engine="pyarrow", compression="zstd")
    return buf.getvalue()


@dataclass
class Store:
    root: Path | None  # None = in-memory only
    hits: int = 0
    misses: int = 0
    computed: list[str] = field(default_factory=list)
    _mem: dict[str, Any] = field(default_factory=dict)

    def path(self, key: str) -> Path:
        assert self.root is not None
        return self.root / key[:2] / key

    def cached(self, stage: str, inputs: dict[str, Any], compute: Callable[[], Any],
               kind: str = "json") -> tuple[Any, str]:
        """Return (value, key); compute and persist on a miss."""
        key = stage_key(stage, inputs)
        if key in self._mem:
            self.hits += 1
            return self._mem[key], key
        if self.root is not None:
            f = self.path(key) / f"{stage}.{'parquet' if kind == 'parquet' else 'json'}"
            if f.exists():
                self.hits += 1
                val = pd.read_parquet(f) if kind == "parquet" else json.loads(f.read_text())
                self._mem[key] = val
                return val, key
        self.misses += 1
        self.computed.append(stage)
        val = compute()
        self._mem[key] = val
        if self.root is not None:
            d = self.path(key)
            d.mkdir(parents=True, exist_ok=True)
            if kind == "parquet":
                (d / f"{stage}.parquet").write_bytes(parquet_bytes(val))
            else:
                (d / f"{stage}.json").write_text(dumps_json(val))
            (d / "inputs.json").write_text(dumps_json({"stage": stage, "inputs": inputs}))
        return val, key

    def write_manifest(self, key: str, manifest: dict[str, Any]) -> Path | None:
        if self.root is None:
            return None
        d = self.path(key)
        d.mkdir(parents=True, exist_ok=True)
        p = d / "manifest.json"
        p.write_text(dumps_json(manifest))
        return p


def default_store() -> Store:
    root = os.environ.get("EMMLAB_DATA")
    return Store(Path(root) / "artifacts" if root else Path.cwd() / "lab_data" / "artifacts")

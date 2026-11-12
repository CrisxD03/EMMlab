"""Comment-preserving YAML updates: merge a new plain spec into an existing document."""

from __future__ import annotations

import io
from typing import Any

from ruamel.yaml.comments import CommentedMap, CommentedSeq

from .spec import _yaml


def _merge(old: Any, new: Any) -> Any:
    if isinstance(old, CommentedMap) and isinstance(new, dict):
        for k in [k for k in old if k not in new]:
            del old[k]
        for k, v in new.items():
            old[k] = _merge(old[k], v) if k in old else v
        return old
    if isinstance(old, CommentedSeq) and isinstance(new, list):
        if len(old) == len(new):
            for i, v in enumerate(new):
                old[i] = _merge(old[i], v)
            return old
        return new
    if old == new:
        return old
    return new


def update_document(text: str, new: dict[str, Any]) -> str:
    """Return ``text`` with values replaced by ``new`` while keeping comments and key order."""
    y = _yaml()
    doc = y.load(io.StringIO(text)) if text.strip() else CommentedMap()
    doc = _merge(doc, new)
    buf = io.StringIO()
    y.dump(doc, buf)
    return buf.getvalue()

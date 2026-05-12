"""File system helpers for safe path handling and upload storage."""

from __future__ import annotations

import os
import secrets
from pathlib import Path


def ensure_dir(path: str | Path) -> Path:
    """Create directory if missing and return it as Path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def safe_filename(original_name: str) -> str:
    """Create a conservative filename from user input."""
    name = os.path.basename(original_name).strip().replace(" ", "_")
    name = "".join(ch for ch in name if ch.isalnum() or ch in ("-", "_", ".", "@"))
    return name or "upload.bin"


def unique_path(directory: str | Path, filename: str) -> Path:
    """Return a unique path in directory by adding a random suffix if needed."""
    d = Path(directory)
    base = safe_filename(filename)
    candidate = d / base
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    for _ in range(50):
        token = secrets.token_hex(4)
        cand = d / f"{stem}_{token}{suffix}"
        if not cand.exists():
            return cand
    # Fallback (very unlikely)
    return d / f"{stem}_{secrets.token_hex(8)}{suffix}"


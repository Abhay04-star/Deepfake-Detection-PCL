"""Hashing helpers (SHA-256)."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    """Compute SHA-256 of a file in a streaming manner.

    Args:
        path: Path to file.
        chunk_size: Read chunk size in bytes.

    Returns:
        Hex digest (64 hex chars).
    """
    p = Path(path)
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_hex_to_bytes32(hex_digest: str) -> bytes:
    """Convert a SHA-256 hex digest to 32 raw bytes (Solidity bytes32).

    Raises:
        ValueError: if digest is not 64 hex chars.
    """
    s = hex_digest.lower().strip()
    if len(s) != 64:
        raise ValueError("SHA-256 digest must be 64 hex characters.")
    return bytes.fromhex(s)


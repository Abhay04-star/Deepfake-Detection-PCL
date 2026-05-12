"""SHA-256 hashing utilities.

Requested API:
- `sha256_file(path)` → hex digest (64 hex chars)
"""

from __future__ import annotations

from pathlib import Path

from utils.hash_utils import sha256_file as _sha256_file, sha256_hex_to_bytes32 as _sha256_hex_to_bytes32


def sha256_file(path: str | Path) -> str:
    """Generate SHA-256 hash of a file (streaming)."""
    return _sha256_file(path)


def sha256_hex_to_bytes32(hex_digest: str) -> bytes:
    """Convert SHA-256 hex digest to raw 32 bytes (Solidity bytes32)."""
    return _sha256_hex_to_bytes32(hex_digest)


__all__ = ["sha256_file", "sha256_hex_to_bytes32"]


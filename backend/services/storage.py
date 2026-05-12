"""Upload storage service (filesystem) with hashing."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from werkzeug.datastructures import FileStorage

from utils.file_utils import ensure_dir, unique_path
from utils.hash_utils import sha256_file


@dataclass
class StorageService:
    upload_dir: Path
    work_dir: Path

    def __post_init__(self):
        self.upload_dir = ensure_dir(self.upload_dir)
        self.work_dir = ensure_dir(self.work_dir)

    def save_upload(self, file_storage: FileStorage) -> Path:
        """Save uploaded file to upload_dir and return path."""
        dst = unique_path(self.upload_dir, file_storage.filename or "upload.bin")
        file_storage.save(str(dst))
        return dst

    def compute_sha256(self, path: str | Path) -> str:
        return sha256_file(path)


def save_upload_and_hash(storage: StorageService, file_storage: FileStorage) -> dict[str, Any]:
    path = storage.save_upload(file_storage)
    digest = storage.compute_sha256(path)
    return {"path": str(path), "filename": path.name, "sha256": digest}


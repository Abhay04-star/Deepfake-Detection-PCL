"""Blockchain integration service.

This service supports a real Ethereum connection when environment variables are
provided, and otherwise falls back to a local mock registry so verification can
still be exercised during development.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from backend.config import AppConfig
from blockchain.web3_client import DeepfakeRegistryClient, Web3Config, load_web3_config_from_env


class MockBlockchainClient:
    def __init__(self, storage_path: Path):
        self.storage_path = storage_path
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self._hashes = self._load_hashes()

    def _load_hashes(self) -> set[str]:
        if not self.storage_path.exists():
            return set()
        try:
            raw = json.loads(self.storage_path.read_text(encoding='utf-8'))
            return set(raw.get('hashes', []))
        except Exception:
            return set()

    def _save_hashes(self) -> None:
        self.storage_path.write_text(json.dumps({'hashes': sorted(self._hashes)}, indent=2), encoding='utf-8')

    def store_hash(self, sha256_hex: str) -> str:
        self._hashes.add(sha256_hex)
        self._save_hashes()
        return 'mock://' + sha256_hex

    def verify_hash(self, sha256_hex: str) -> bool:
        return sha256_hex in self._hashes


@dataclass
class BlockchainService:
    enabled: bool
    client: DeepfakeRegistryClient | MockBlockchainClient | None = None
    error: str | None = None
    mock: bool = False

    @classmethod
    def from_env_optional(cls, cfg: AppConfig) -> "BlockchainService":
        try:
            web3_cfg = load_web3_config_from_env()
            client = DeepfakeRegistryClient(web3_cfg)
            return cls(enabled=True, client=client, mock=False)
        except Exception as err:  # noqa: BLE001 - optional blockchain integration
            storage_path = cfg.upload_dir.parent / 'blockchain_fallback.json'
            mock_client = MockBlockchainClient(storage_path=storage_path)
            return cls(
                enabled=True,
                client=mock_client,
                error=str(err),
                mock=True,
            )


def store_hash_on_chain(service: BlockchainService, sha256: str) -> dict[str, Any]:
    if service.client is None:
        return {"enabled": False, "stored": False, "error": service.error or "Blockchain unavailable."}
    try:
        tx = service.client.store_hash(sha256)
        return {"enabled": True, "stored": True, "tx_hash": tx, "mock": service.mock}
    except Exception as e:  # noqa: BLE001 - return error rather than crashing request
        return {"enabled": True, "stored": False, "error": str(e), "mock": service.mock}


def verify_hash_on_chain(service: BlockchainService, sha256: str) -> dict[str, Any]:
    if service.client is None:
        return {"enabled": False, "exists": False, "error": service.error or "Blockchain unavailable."}
    try:
        exists = service.client.verify_hash(sha256)
        return {"enabled": True, "exists": bool(exists), "mock": service.mock}
    except Exception as e:
        return {"enabled": True, "exists": False, "error": str(e), "mock": service.mock}


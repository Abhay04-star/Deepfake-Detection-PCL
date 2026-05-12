"""Service bootstrapper (dependency injection)."""

from __future__ import annotations

from dataclasses import dataclass

from backend.config import AppConfig
from backend.services.blockchain_service import BlockchainService
from backend.services.ml_service import MlService
from backend.services.storage import StorageService


@dataclass(frozen=True)
class Services:
    storage: StorageService
    ml: MlService
    blockchain: BlockchainService


def bootstrap_services(cfg: AppConfig) -> Services:
    storage = StorageService(upload_dir=cfg.upload_dir, work_dir=cfg.work_dir)
    ml = MlService(
        weights_path=cfg.model_weights_path,
        input_size=cfg.model_input_size,
        video_max_frames=cfg.video_max_frames,
        threshold=cfg.detection_threshold,
    )
    blockchain = BlockchainService.from_env_optional(cfg)
    return Services(storage=storage, ml=ml, blockchain=blockchain)


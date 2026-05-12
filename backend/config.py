"""Backend configuration loaded from environment.

No hardcoded machine-specific paths; uses sensible defaults relative to repo root.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class AppConfig:
    host: str
    port: int
    max_upload_mb: int
    upload_dir: Path
    work_dir: Path
    model_weights_path: Path
    video_max_frames: int
    model_input_size: int
    detection_threshold: float  # Threshold for fake/real classification (0.0-1.0)


def load_config() -> AppConfig:
    load_dotenv(dotenv_path=_repo_root() / ".env", override=False)

    host = os.getenv("BACKEND_HOST", "127.0.0.1").strip() or "127.0.0.1"
    port = int(os.getenv("BACKEND_PORT", "5000"))
    max_upload_mb = int(os.getenv("MAX_UPLOAD_MB", "200"))

    upload_dir = Path(os.getenv("UPLOAD_DIR", "").strip() or (_repo_root() / "data" / "uploads"))
    work_dir = Path(os.getenv("WORK_DIR", "").strip() or (_repo_root() / "data" / "work"))

    default_weights = _repo_root() / "ml_model" / "weights" / "resnet50_deepfake.keras"
    model_weights_path = Path(os.getenv("MODEL_WEIGHTS_PATH", "").strip() or default_weights)

    video_max_frames = int(os.getenv("VIDEO_MAX_FRAMES", "32"))
    model_input_size = int(os.getenv("MODEL_INPUT_SIZE", "224"))
    
    # Detection threshold: lower = more sensitive to fakes, higher = more conservative
    # Default 0.5 means p_fake >= 0.5 is classified as FAKE
    detection_threshold = float(os.getenv("DETECTION_THRESHOLD", "0.5"))

    return AppConfig(
        host=host,
        port=port,
        max_upload_mb=max_upload_mb,
        upload_dir=upload_dir,
        work_dir=work_dir,
        model_weights_path=model_weights_path,
        video_max_frames=video_max_frames,
        model_input_size=model_input_size,
        detection_threshold=detection_threshold,
    )


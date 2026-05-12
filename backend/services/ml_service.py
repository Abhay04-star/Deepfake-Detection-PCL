"""ML inference service.

Loads model lazily on first request to keep startup fast.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from ml_model.predict import Prediction, load_default_model, predict_on_frames
from utils.video_utils import extract_video_frames, is_video_path, read_image_bgr, select_key_frames


@dataclass
class MlService:
    weights_path: Path
    input_size: int
    video_max_frames: int
    threshold: float = 0.5  # Detection threshold for fake/real classification
    _model: tf.keras.Model | None = None

    def _get_model(self) -> tf.keras.Model:
        if self._model is None:
            self._model = load_default_model(self.weights_path, input_size=self.input_size)
        return self._model

    def predict_file(self, path: str | Path) -> dict[str, Any]:
        """Predict REAL/FAKE for an image or video file."""
        p = Path(path)
        model = self._get_model()

        if is_video_path(p):
            frames = extract_video_frames(p, max_frames=self.video_max_frames)
            key_count = min(len(frames.frames_bgr), max(4, self.video_max_frames // 2))
            key_frames = select_key_frames(
                frames_bgr=frames.frames_bgr,
                frame_indices=frames.frame_indices,
                key_frame_count=key_count,
                original_total_frames=frames.total_frames,
            )
            pred = predict_on_frames(
                model, 
                key_frames.frames_bgr, 
                input_size=self.input_size,
                threshold=self.threshold
            )
            return {
                "label": pred.label,
                "confidence": float(pred.confidence),
                "p_fake": float(pred.p_fake),
                "threshold": self.threshold,
                "frame_count": len(key_frames.frames_bgr),
                "total_frames": frames.total_frames,
                "frame_indices": key_frames.frame_indices,
            }

        img = read_image_bgr(p)
        pred = predict_on_frames(model, [img], input_size=self.input_size, threshold=self.threshold)
        return {
            "label": pred.label,
            "confidence": float(pred.confidence),
            "p_fake": float(pred.p_fake),
            "threshold": self.threshold,
            "frame_count": 1,
            "total_frames": 1,
            "frame_indices": [0],
        }


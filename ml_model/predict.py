"""Prediction helpers for deepfake detection.

This module:
- Loads a saved model (`.h5` full model or weights-only fallback)
- Accepts an image/frame (OpenCV BGR numpy array)
- Returns a label (REAL/FAKE) and confidence
- Supports threshold tuning to reduce false positives
- Video prediction with majority voting
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, List
from enum import Enum
import logging

import numpy as np
import tensorflow as tf

from ml_model.model import ModelConfig, load_model
from ml_model.preprocess import (
    batch_preprocess_frames, 
    resize_and_normalize,
    preprocess_with_face_detection,
    extract_frames_from_video,
)

logger = logging.getLogger(__name__)


Label = Literal["REAL", "FAKE"]


class AggregationMethod(Enum):
    """Methods for aggregating predictions across frames."""
    MEAN = "mean"           # Average probabilities
    MAJORITY = "majority"   # Majority voting
    MAX_CONFIDENCE = "max_confidence"  # Use frame with highest confidence


@dataclass(frozen=True)
class Prediction:
    """Prediction result with confidence and threshold information."""
    label: Label
    confidence: float
    p_fake: float
    threshold: float = 0.5
    all_frame_predictions: List[float] | None = None
    
    @property
    def is_real(self) -> bool:
        """Check if prediction is REAL."""
        return self.label == "REAL"
    
    @property
    def is_fake(self) -> bool:
        """Check if prediction is FAKE."""
        return self.label == "FAKE"
    
    def to_dict(self) -> dict:
        """Convert prediction to dictionary."""
        return {
            'label': self.label,
            'confidence': round(self.confidence, 4),
            'p_fake': round(self.p_fake, 4),
            'threshold': self.threshold,
            'is_real': self.is_real,
            'is_fake': self.is_fake,
        }


def predict_frame(
    model: tf.keras.Model,
    frame_bgr: np.ndarray,
    input_size: int = 224,
    threshold: float = 0.5,
    use_face_detection: bool = True,
) -> Prediction:
    """Predict REAL/FAKE on a single frame.

    Args:
        model: Trained model
        frame_bgr: OpenCV BGR image
        input_size: Target input size
        threshold: Classification threshold (default 0.5, higher = more conservative)
        use_face_detection: Whether to detect and crop face region
        
    Returns:
        Prediction with label, confidence, and probability
    """
    if frame_bgr is None:
        raise ValueError("frame_bgr is None")
    
    # Preprocess with optional face detection
    if use_face_detection:
        x = preprocess_with_face_detection(frame_bgr, size=input_size, use_face_detection=True)
    else:
        x = resize_and_normalize(frame_bgr, size=input_size)
    
    x = np.expand_dims(x, axis=0)  # (1,H,W,3)
    p = model.predict(x, verbose=0).reshape(-1)
    p_fake = float(p[0])
    
    # Classification with threshold
    # p_fake >= threshold => FAKE
    # p_fake < threshold => REAL
    label: Label = "FAKE" if p_fake >= threshold else "REAL"
    confidence = p_fake if label == "FAKE" else (1.0 - p_fake)
    
    return Prediction(
        label=label, 
        confidence=confidence, 
        p_fake=p_fake,
        threshold=threshold
    )


def predict_on_frames(
    model: tf.keras.Model,
    frames_bgr: list[np.ndarray],
    input_size: int = 224,
    threshold: float = 0.5,
    aggregation: AggregationMethod = AggregationMethod.MAJORITY,
    use_face_detection: bool = True,
) -> Prediction:
    """Predict REAL/FAKE on multiple frames with configurable aggregation.

    Args:
        model: Trained model
        frames_bgr: List of OpenCV BGR frames
        input_size: Target input size
        threshold: Classification threshold
        aggregation: Method to aggregate frame predictions
        use_face_detection: Whether to use face detection
        
    Returns:
        Aggregated prediction
    """
    if not frames_bgr:
        raise ValueError("No frames provided for prediction.")

    # Get predictions for all frames
    frame_predictions = []
    for frame in frames_bgr:
        pred = predict_frame(
            model, frame, input_size, 
            threshold=threshold, 
            use_face_detection=use_face_detection
        )
        frame_predictions.append(pred.p_fake)
    
    p_fake_array = np.array(frame_predictions)
    
    # Aggregate based on method
    if aggregation == AggregationMethod.MEAN:
        # Average probabilities
        p_fake = float(np.mean(p_fake_array))
        
    elif aggregation == AggregationMethod.MAJORITY:
        # Majority voting: count frames above/below threshold
        fake_votes = np.sum(p_fake_array >= threshold)
        real_votes = len(p_fake_array) - fake_votes
        
        # Majority decides, use average as confidence
        if fake_votes > real_votes:
            p_fake = float(np.mean(p_fake_array[p_fake_array >= threshold])) if fake_votes > 0 else threshold
        else:
            p_fake = float(np.mean(p_fake_array[p_fake_array < threshold])) if real_votes > 0 else threshold - 0.01
            
    elif aggregation == AggregationMethod.MAX_CONFIDENCE:
        # Use the prediction with highest confidence
        confidences = np.maximum(p_fake_array, 1 - p_fake_array)
        max_conf_idx = np.argmax(confidences)
        p_fake = float(p_fake_array[max_conf_idx])
        
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")
    
    # Final classification
    label: Label = "FAKE" if p_fake >= threshold else "REAL"
    confidence = p_fake if label == "FAKE" else (1.0 - p_fake)
    
    return Prediction(
        label=label,
        confidence=confidence,
        p_fake=p_fake,
        threshold=threshold,
        all_frame_predictions=frame_predictions
    )


def predict_video(
    model: tf.keras.Model,
    video_path: str | Path,
    input_size: int = 224,
    max_frames: int = 32,
    threshold: float = 0.5,
    aggregation: AggregationMethod = AggregationMethod.MAJORITY,
    use_face_detection: bool = True,
) -> Prediction:
    """Predict REAL/FAKE on a video file.
    
    Extracts frames, predicts on each, and aggregates results.
    
    Args:
        model: Trained model
        video_path: Path to video file
        input_size: Input image size
        max_frames: Maximum frames to extract
        threshold: Classification threshold
        aggregation: Aggregation method
        use_face_detection: Whether to use face detection
        
    Returns:
        Aggregated prediction
    """
    import cv2
    
    # Extract frames
    result = extract_frames_from_video(video_path, max_frames=max_frames)
    
    if not result.frames_bgr:
        raise ValueError(f"Could not extract frames from video: {video_path}")
    
    logger.info(f"Extracted {len(result.frames_bgr)} frames from video")
    
    # Predict on frames
    return predict_on_frames(
        model=model,
        frames_bgr=result.frames_bgr,
        input_size=input_size,
        threshold=threshold,
        aggregation=aggregation,
        use_face_detection=use_face_detection,
    )


def find_optimal_threshold(
    model: tf.keras.Model,
    images_bgr: list[np.ndarray],
    true_labels: list[int],
    thresholds: list[float] = None,
    use_face_detection: bool = True,
) -> dict:
    """Find optimal threshold for reducing false positives.
    
    Tests multiple thresholds and returns metrics for each.
    
    Args:
        model: Trained model
        images_bgr: List of test images
        true_labels: True labels (0=REAL, 1=FAKE)
        thresholds: List of thresholds to test (default: [0.5, 0.6, 0.7])
        use_face_detection: Whether to use face detection
        
    Returns:
        Dictionary with metrics for each threshold
    """
    if thresholds is None:
        thresholds = [0.5, 0.6, 0.7]
    
    results = {}
    
    for thresh in thresholds:
        predictions = []
        for img in images_bgr:
            pred = predict_frame(
                model, img, 
                threshold=thresh, 
                use_face_detection=use_face_detection
            )
            predictions.append(1 if pred.label == "FAKE" else 0)
        
        # Calculate metrics
        tp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 1)
        tn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 0)
        fp = sum(1 for p, t in zip(predictions, true_labels) if p == 1 and t == 0)
        fn = sum(1 for p, t in zip(predictions, true_labels) if p == 0 and t == 1)
        
        accuracy = (tp + tn) / len(true_labels) if len(true_labels) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        results[thresh] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'false_positives': fp,
            'false_negatives': fn,
            'fp_rate': fp / (fp + tn) if (fp + tn) > 0 else 0,
        }
    
    return results


def load_default_model(model_path: str | Path, input_size: int = 224) -> tf.keras.Model:
    """Load the model used for inference.

    `model_path` can be:
    - a full `.h5` model saved via `model.save(...)`
    - a weights file saved via `model.save_weights(...)`
    """
    cfg = ModelConfig(input_size=input_size)
    return load_model(weights_path=model_path, cfg=cfg)


def predict_image(
    model_path: str | Path,
    image_bgr: np.ndarray,
    input_size: int = 224,
    threshold: float = 0.5,
    use_face_detection: bool = True,
) -> Prediction:
    """Convenience helper: load model + predict on one image.
    
    Args:
        model_path: Path to trained model
        image_bgr: OpenCV BGR image
        input_size: Input size
        threshold: Classification threshold
        use_face_detection: Whether to use face detection
        
    Returns:
        Prediction result
    """
    model = load_default_model(model_path=model_path, input_size=input_size)
    return predict_frame(
        model=model, 
        frame_bgr=image_bgr, 
        input_size=input_size,
        threshold=threshold,
        use_face_detection=use_face_detection,
    )


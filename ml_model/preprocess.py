"""ML preprocessing utilities.

Independent preprocessing module that provides:
- Extract frames from a video using OpenCV
- Resize and normalize images for ResNet50 input
- Face detection using OpenCV DNN for focusing on facial regions
- Data augmentation for REAL images

All functions avoid hardcoded paths and operate on passed-in inputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, List
import logging

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExtractFramesResult:
    frames_bgr: list[np.ndarray]
    frame_indices: list[int]
    total_frames: int


def extract_frames_from_video(
    video_path: str | Path,
    max_frames: int = 32,
) -> ExtractFramesResult:
    """Extract up to `max_frames` evenly-spaced frames from a video.

    Returns:
        ExtractFramesResult where frames are OpenCV BGR arrays.
    """
    p = Path(video_path)
    cap = cv2.VideoCapture(str(p))
    if not cap.isOpened():
        raise ValueError(f"Unable to open video: {p}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    if total <= 0:
        # Fallback: sequential read until EOF or max_frames reached.
        frames: list[np.ndarray] = []
        indices: list[int] = []
        i = 0
        while len(frames) < max_frames:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            frames.append(frame)
            indices.append(i)
            i += 1
        cap.release()
        return ExtractFramesResult(frames_bgr=frames, frame_indices=indices, total_frames=i)

    sample_count = min(max_frames, total)
    idxs = np.linspace(0, max(total - 1, 0), num=sample_count, dtype=int).tolist()
    frames: list[np.ndarray] = []

    # Sample by seeking to each selected frame index.
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if ok and frame is not None:
            frames.append(frame)
        else:
            # Keep shape stable for aggregation; model will handle empty-ish content.
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))

    cap.release()
    return ExtractFramesResult(frames_bgr=frames, frame_indices=idxs, total_frames=total)


def resize_and_normalize(
    image_bgr: np.ndarray,
    size: int = 224,
) -> np.ndarray:
    """Resize and normalize an image to ResNet-compatible input.

    Args:
        image_bgr: OpenCV BGR image.
        size: Target height/width.

    Returns:
        Float32 RGB image shaped (size, size, 3) with values in [0, 1].
    """
    if image_bgr is None:
        raise ValueError("image_bgr is None")
    rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    resized = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_AREA)
    x = resized.astype(np.float32) / 255.0
    return x


def batch_preprocess_frames(
    frames_bgr: Iterable[np.ndarray],
    size: int = 224,
) -> np.ndarray:
    """Preprocess a batch of BGR frames into a model-ready tensor."""
    xs = [resize_and_normalize(f, size=size) for f in frames_bgr]
    if not xs:
        raise ValueError("No frames provided for preprocessing.")
    return np.stack(xs, axis=0)


# ============================================================================
# Face Detection
# ============================================================================

class FaceDetector:
    """Face detector using OpenCV DNN face detection model.
    
    Uses the OpenCV DNN face detector for reliable face detection
    before classification to focus on facial regions.
    """
    
    def __init__(self, confidence_threshold: float = 0.5):
        """Initialize face detector.
        
        Args:
            confidence_threshold: Minimum confidence for face detection (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self._net = None
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the DNN face detection model."""
        # Use OpenCV's built-in face detector (Caffe model)
        # These are typically available in OpenCV installations
        prototxt_path = cv2.data.haarcascades.replace(
            'haarcascades', 'dnn/face_detector/deploy.prototxt'
        ) if hasattr(cv2, 'data') else None
        
        model_path = cv2.data.haarcascades.replace(
            'haarcascades', 'dnn/face_detector/res10_300x300_ssd_iter_140000.caffemodel'
        ) if hasattr(cv2, 'data') else None
        
        # Fallback to Haar Cascade if DNN model not available
        self._cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # Try to load DNN model
        try:
            if prototxt_path and model_path:
                self._net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
                logger.info("Loaded DNN face detector")
            else:
                self._net = None
                logger.info("Using Haar Cascade face detector")
        except Exception as e:
            self._net = None
            logger.warning(f"Failed to load DNN face detector: {e}. Using Haar Cascade.")
    
    def detect_face(
        self, 
        image_bgr: np.ndarray,
        target_size: int = 224,
        padding: float = 0.3
    ) -> np.ndarray:
        """Detect and extract the largest face from an image.
        
        Args:
            image_bgr: Input BGR image
            target_size: Output size for the face crop
            padding: Padding factor around detected face (0.3 = 30% extra)
            
        Returns:
            Cropped and resized face image (target_size x target_size)
            If no face detected, returns the resized original image
        """
        if image_bgr is None:
            raise ValueError("image_bgr is None")
        
        h, w = image_bgr.shape[:2]
        
        # Try DNN detector first
        face_rect = self._detect_face_dnn(image_bgr) if self._net is not None else None
        
        # Fallback to Haar Cascade
        if face_rect is None:
            face_rect = self._detect_face_haar(image_bgr)
        
        if face_rect is None:
            # No face detected, return resized original
            logger.debug("No face detected, using full image")
            return cv2.resize(image_bgr, (target_size, target_size))
        
        # Extract face with padding
        x, y, fw, fh = face_rect
        
        # Add padding
        pad_x = int(fw * padding)
        pad_y = int(fh * padding)
        
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + fw + pad_x)
        y2 = min(h, y + fh + pad_y)
        
        # Extract and resize face
        face_crop = image_bgr[y1:y2, x1:x2]
        face_resized = cv2.resize(face_crop, (target_size, target_size), interpolation=cv2.INTER_AREA)
        
        return face_resized
    
    def _detect_face_dnn(self, image_bgr: np.ndarray) -> Tuple[int, int, int, int] | None:
        """Detect face using DNN model.
        
        Returns:
            (x, y, width, height) of largest face or None
        """
        h, w = image_bgr.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(
            cv2.resize(image_bgr, (300, 300)), 1.0, (300, 300), 
            (104.0, 177.0, 123.0)
        )
        
        self._net.setInput(blob)
        detections = self._net.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > self.confidence_threshold:
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x1, y1, x2, y2 = box.astype(int)
                fw, fh = x2 - x1, y2 - y1
                faces.append((x1, y1, fw, fh, confidence))
        
        if not faces:
            return None
        
        # Return largest face by area
        faces.sort(key=lambda f: f[2] * f[3], reverse=True)
        return faces[0][:4]
    
    def _detect_face_haar(self, image_bgr: np.ndarray) -> Tuple[int, int, int, int] | None:
        """Detect face using Haar Cascade.
        
        Returns:
            (x, y, width, height) of largest face or None
        """
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        faces = self._cascade.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50)
        )
        
        if len(faces) == 0:
            return None
        
        # Return largest face
        faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
        return tuple(faces[0])


# Global face detector instance
_face_detector: FaceDetector | None = None


def get_face_detector(confidence_threshold: float = 0.5) -> FaceDetector:
    """Get or create global face detector instance."""
    global _face_detector
    if _face_detector is None:
        _face_detector = FaceDetector(confidence_threshold)
    return _face_detector


def preprocess_with_face_detection(
    image_bgr: np.ndarray,
    size: int = 224,
    use_face_detection: bool = True
) -> np.ndarray:
    """Preprocess image with optional face detection.
    
    Args:
        image_bgr: Input BGR image
        size: Target output size
        use_face_detection: Whether to detect and crop face
        
    Returns:
        Preprocessed RGB image (size, size, 3) normalized to [0, 1]
    """
    if use_face_detection:
        detector = get_face_detector()
        processed = detector.detect_face(image_bgr, target_size=size)
    else:
        processed = cv2.resize(image_bgr, (size, size))
    
    # Convert to RGB and normalize
    rgb = cv2.cvtColor(processed, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype(np.float32) / 255.0
    return normalized


# ============================================================================
# Data Augmentation
# ============================================================================

def augment_real_image(
    image_rgb: np.ndarray,
    rotation_range: int = 15,
    brightness_range: float = 0.2,
    blur_probability: float = 0.3
) -> np.ndarray:
    """Apply augmentation to REAL images to increase diversity.
    
    Augmentations applied:
    - Random rotation
    - Random brightness adjustment
    - Random horizontal flip
    - Random slight blur
    
    Args:
        image_rgb: Input RGB image (H, W, 3) with values in [0, 1] or [0, 255]
        rotation_range: Max rotation angle in degrees
        brightness_range: Brightness adjustment factor
        blur_probability: Probability of applying blur
        
    Returns:
        Augmented RGB image
    """
    img = image_rgb.copy()
    
    # Ensure float32
    if img.dtype != np.float32:
        img = img.astype(np.float32)
    
    # Normalize to [0, 1] if needed
    if img.max() > 1.0:
        img = img / 255.0
    
    h, w = img.shape[:2]
    
    # Random rotation
    angle = np.random.uniform(-rotation_range, rotation_range)
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    
    # Random horizontal flip
    if np.random.random() > 0.5:
        img = cv2.flip(img, 1)
    
    # Random brightness adjustment
    brightness_factor = 1.0 + np.random.uniform(-brightness_range, brightness_range)
    img = np.clip(img * brightness_factor, 0, 1)
    
    # Random slight blur
    if np.random.random() < blur_probability:
        kernel_size = np.random.choice([3, 5])
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
    
    return img


def create_augmentation_layer():
    """Create TensorFlow/Keras augmentation layer for training.
    
    Returns:
        tf.keras.Sequential with augmentation layers
    """
    import tensorflow as tf
    
    return tf.keras.Sequential([
        tf.keras.layers.RandomRotation(factor=0.08),  # ~15 degrees
        tf.keras.layers.RandomFlip(mode='horizontal'),
        tf.keras.layers.RandomBrightness(factor=0.2),
        tf.keras.layers.RandomContrast(factor=0.1),
    ], name='augmentation')



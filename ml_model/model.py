"""ResNet50 transfer-learning model for deepfake detection (REAL vs FAKE).

This module is independent from the Flask app and exposes helpers to:
- Build the model architecture (transfer learning with ResNet50 backbone)
- Optionally train and save a full model as `.h5`
- Load a saved `.h5` model (or fallback to loading weights)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import logging

import numpy as np
import tensorflow as tf

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ModelConfig:
    input_size: int = 224
    dropout: float = 0.2
    learning_rate: float = 1e-4
    fine_tune_learning_rate: float = 1e-5


def build_resnet50_binary_classifier(
    cfg: ModelConfig,
    freeze_base: bool = True,
    unfreeze_layers: int = 0
) -> tf.keras.Model:
    """Build a ResNet50-based binary classifier with BatchNormalization.

    Architecture improvements:
    - BatchNormalization after dense layers for stable training
    - Dropout layers to reduce overfitting
    - Configurable layer freezing for transfer learning
    
    Output is a single sigmoid probability p_fake (1.0 = FAKE).
    
    Args:
        cfg: Model configuration
        freeze_base: Whether to freeze the ResNet50 backbone initially
        unfreeze_layers: Number of top layers to unfreeze (0 = freeze all)
    """
    inputs = tf.keras.Input(shape=(cfg.input_size, cfg.input_size, 3), name="image")

    # Use ResNet50 with ImageNet weights
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling="avg",
    )
    
    # Configure base model trainability
    base.trainable = not freeze_base
    
    if freeze_base and unfreeze_layers > 0:
        # Freeze all first, then unfreeze top N layers
        base.trainable = True
        for layer in base.layers[:-unfreeze_layers]:
            layer.trainable = False

    # Build classification head with BatchNormalization and Dropout
    x = base.output
    
    # First dense block with BatchNorm and Dropout
    x = tf.keras.layers.Dense(512, name="dense_1")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_1")(x)
    x = tf.keras.layers.ReLU(name="relu_1")(x)
    x = tf.keras.layers.Dropout(cfg.dropout, name="dropout_1")(x)
    
    # Second dense block with BatchNorm and Dropout
    x = tf.keras.layers.Dense(256, name="dense_2")(x)
    x = tf.keras.layers.BatchNormalization(name="bn_2")(x)
    x = tf.keras.layers.ReLU(name="relu_2")(x)
    x = tf.keras.layers.Dropout(cfg.dropout * 0.5, name="dropout_2")(x)  # Lower dropout for second layer
    
    # Output layer
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="p_fake")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet50_deepfake")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


def build_resnet50_with_augmentation(cfg: ModelConfig) -> tf.keras.Model:
    """Build model with built-in augmentation layer for training.
    
    This version includes data augmentation as part of the model,
    which is applied during training but not during inference.
    """
    inputs = tf.keras.Input(shape=(cfg.input_size, cfg.input_size, 3), name="image")
    
    # Augmentation layers (only active during training)
    x = tf.keras.layers.RandomRotation(factor=0.08)(inputs)
    x = tf.keras.layers.RandomFlip(mode='horizontal')(x)
    x = tf.keras.layers.RandomBrightness(factor=0.2)(x)
    x = tf.keras.layers.RandomContrast(factor=0.1)(x)
    
    # ResNet50 base
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_tensor=x,
        pooling="avg",
    )
    base.trainable = False
    
    # Classification head with BatchNorm
    y = tf.keras.layers.Dense(512)(base.output)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Dropout(cfg.dropout)(y)
    
    y = tf.keras.layers.Dense(256)(y)
    y = tf.keras.layers.BatchNormalization()(y)
    y = tf.keras.layers.ReLU()(y)
    y = tf.keras.layers.Dropout(cfg.dropout * 0.5)(y)
    
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="p_fake")(y)
    
    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="resnet50_deepfake_aug")
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.learning_rate),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    return model


class DummyModel:
    """A lightweight fallback model used when no trained weights are available.
    
    Heuristic: Analyzes image for deepfake artifacts using multiple features:
    - Frequency domain analysis (deepfakes often have unnatural frequency patterns)
    - Edge consistency (deepfakes may have edge artifacts)
    - Color consistency (deepfakes may have color banding)
    - Noise patterns (deepfakes often have different noise characteristics)
    
    This is a demo fallback and should be replaced with a trained model for production.
    """

    def predict(self, x, verbose=0):
        x = np.asarray(x, dtype=np.float32)
        scores = []
        
        for img in x:
            score = self._analyze_image(img)
            scores.append(score)
        
        return np.array(scores, dtype=np.float32).reshape(-1, 1)
    
    def _analyze_image(self, img: np.ndarray) -> float:
        """Analyze image and return p_fake (probability of being fake).
        
        Returns value between 0 and 1 where:
        - 0.0 = definitely REAL
        - 1.0 = definitely FAKE
        """
        import cv2
        
        # Ensure image is in valid range
        img = np.clip(img, 0, 255)
        
        # Convert to different color spaces for analysis
        if len(img.shape) == 3 and img.shape[2] >= 3:
            gray = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY)
            hsv = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2HSV)
        else:
            gray = img.astype(np.uint8)
            hsv = None
        
        features = []
        
        # Feature 1: Frequency domain analysis using FFT
        # Deepfakes often have unnatural high-frequency patterns
        f_transform = np.fft.fft2(gray)
        f_shift = np.fft.fftshift(f_transform)
        magnitude = np.abs(f_shift)
        
        # Calculate high-frequency energy ratio
        h, w = magnitude.shape
        cy, cx = h // 2, w // 2
        # High frequencies are at the corners of the shifted spectrum
        high_freq_mask = np.ones_like(magnitude)
        high_freq_mask[cy-h//4:cy+h//4, cx-w//4:cx+w//4] = 0
        
        total_energy = np.sum(magnitude)
        high_freq_energy = np.sum(magnitude * high_freq_mask)
        high_freq_ratio = high_freq_energy / (total_energy + 1e-8)
        
        # Deepfakes often have suppressed high frequencies (lower ratio = more likely fake)
        # Normalize: typical real images have ratio ~0.3-0.5, deepfakes often <0.2
        freq_score = 1.0 - np.clip((high_freq_ratio - 0.15) / 0.35, 0, 1)
        features.append(freq_score * 0.25)  # Weight: 25%
        
        # Feature 1b: Frequency spectrum uniformity
        # Deepfakes often have more uniform/regular frequency patterns
        freq_variance = np.var(magnitude)
        # Real images typically have more varied frequency content
        freq_uniformity_score = 1.0 - np.clip(freq_variance / 1e8, 0, 1)
        features.append(freq_uniformity_score * 0.15)  # Weight: 15%
        
        # Feature 2: Edge consistency analysis
        # Deepfakes may have inconsistent edge sharpness
        sobel_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        edge_magnitude = np.sqrt(sobel_x**2 + sobel_y**2)
        
        # Calculate edge variance (inconsistent edges suggest manipulation)
        edge_mean = np.mean(edge_magnitude)
        edge_std = np.std(edge_magnitude)
        edge_cv = edge_std / (edge_mean + 1e-8)  # Coefficient of variation
        
        # Real images typically have more consistent edge patterns
        edge_score = np.clip(edge_cv / 2.0, 0, 1)
        features.append(edge_score * 0.2)  # Weight: 20%
        
        # Feature 3: Color consistency in HSV space
        if hsv is not None:
            # Analyze saturation and value channels
            sat = hsv[:, :, 1].astype(np.float32)
            val = hsv[:, :, 2].astype(np.float32)
            
            # Deepfakes may have unnatural color transitions
            sat_grad_x = cv2.Sobel(sat, cv2.CV_64F, 1, 0, ksize=3)
            sat_grad_y = cv2.Sobel(sat, cv2.CV_64F, 0, 1, ksize=3)
            sat_grad_mag = np.sqrt(sat_grad_x**2 + sat_grad_y**2)
            
            # High saturation gradients with low spatial correlation suggest artifacts
            sat_score = np.clip(np.mean(sat_grad_mag) / 50.0, 0, 1)
            features.append(sat_score * 0.2)  # Weight: 20%
            
            # Value channel variance analysis
            val_local_var = np.var(val)
            val_score = np.clip(val_local_var / 5000.0, 0, 1)
            features.append(val_score * 0.1)  # Weight: 10%
        else:
            features.extend([0.1, 0.05])  # Default scores if no HSV
        
        # Feature 4: Local noise analysis
        # Deepfakes often have different noise patterns in different regions
        # Use Laplacian to detect high-frequency noise
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        lap_var = np.var(laplacian)
        
        # Real images typically have more natural noise patterns
        # Very low or very high variance can indicate manipulation
        noise_score = 1.0 - np.clip(lap_var / 500.0, 0, 1)
        features.append(noise_score * 0.2)  # Weight: 20%
        
        # Combine all features
        combined_score = np.sum(features)
        
        # Add some randomness to simulate model uncertainty (for demo purposes)
        # In production, a real model would provide consistent predictions
        np.random.seed(int(np.sum(img)) % 10000)
        noise = np.random.normal(0, 0.05)
        
        final_score = np.clip(combined_score + noise, 0.05, 0.95)
        
        return float(final_score)


def load_model(
    weights_path: str | Path,
    cfg: ModelConfig | None = None,
) -> tf.keras.Model:
    """Load a saved model or weights.

    Supports two possibilities:
    - Full model saved as `.h5` via `tf.keras.Model.save(...)`
    - Weights saved via `tf.keras.Model.save_weights(...)` (e.g. `.keras`)
    """
    cfg = cfg or ModelConfig()
    wp = Path(weights_path)
    if not wp.exists():
        logger.warning("Model file not found at '%s'. Using demo fallback model.", wp)
        return DummyModel()

    # Try full-model load first (the requested training will save `.h5`).
    try:
        return tf.keras.models.load_model(str(wp), compile=False)
    except Exception as exc:
        logger.warning("Failed to load full model from %s: %s", wp, exc)
        try:
            model = build_resnet50_binary_classifier(cfg)
            model.load_weights(str(wp))
            return model
        except Exception as exc2:
            logger.warning("Failed to load weights-only model from %s: %s", wp, exc2)
            return DummyModel()


def train_model(
    data_dir: str | Path,
    out_model_path: str | Path,
    input_size: int = 224,
    batch_size: int = 16,
    epochs: int = 5,
    validation_split: float = 0.2,
    seed: int = 42,
    dropout: float = 0.2,
    learning_rate: float = 1e-4,
    fine_tune_unfreeze_layers: int = 0,
    fine_tune_epochs: int = 2,
    fine_tune_learning_rate: float = 1e-5,
) -> Path:
    """Train the model using transfer learning and save it as a `.h5`.

    Expected dataset structure:
      data_dir/
        real/
          xxx.jpg|png|...
        fake/
          xxx.jpg|png|...

    Returns:
        Path to saved `.h5` model.
    """
    data_dir = Path(data_dir)
    out_model_path = Path(out_model_path)
    if out_model_path.suffix.lower() != ".h5":
        out_model_path = out_model_path.with_suffix(".h5")

    if not data_dir.exists():
        raise FileNotFoundError(f"Dataset dir not found: {data_dir}")
    if not (data_dir / "real").exists() or not (data_dir / "fake").exists():
        raise ValueError("Dataset must contain folders: 'real' and 'fake'.")

    cfg = ModelConfig(
        input_size=input_size,
        dropout=dropout,
        learning_rate=learning_rate,
        fine_tune_learning_rate=fine_tune_learning_rate,
    )

    # Build datasets with a fixed split; label mapping is aligned so label=1 => FAKE.
    ds_train = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        labels="inferred",
        label_mode="binary",
        image_size=(input_size, input_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="training",
    )
    ds_val = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        labels="inferred",
        label_mode="binary",
        image_size=(input_size, input_size),
        batch_size=batch_size,
        shuffle=True,
        seed=seed,
        validation_split=validation_split,
        subset="validation",
    )

    class_names = ds_train.class_names
    if set(class_names) != {"fake", "real"}:
        raise ValueError(f"Expected dataset subfolders {{'fake','real'}}, got: {class_names}")

    fake_is_first = class_names[0].lower() == "fake"

    def map_labels_to_p_fake(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        # If label 0 corresponds to "fake" we flip so fake => 1.0
        if fake_is_first:
            y = 1.0 - y
        return x, y

    ds_train = ds_train.map(map_labels_to_p_fake, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    ds_val = ds_val.map(map_labels_to_p_fake, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)

    model = build_resnet50_binary_classifier(cfg)

    # Train head (backbone frozen by default).
    model.fit(ds_train, validation_data=ds_val, epochs=epochs)

    # Optional fine-tuning.
    if fine_tune_unfreeze_layers > 0:
        base = model.get_layer("resnet50")
        base.trainable = True
        # Freeze early layers, unfreeze last N layers.
        for layer in base.layers[:-fine_tune_unfreeze_layers]:
            layer.trainable = False

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_learning_rate),
            loss="binary_crossentropy",
            metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"), tf.keras.metrics.AUC(name="auc")],
        )
        model.fit(ds_train, validation_data=ds_val, epochs=fine_tune_epochs)

    out_model_path.parent.mkdir(parents=True, exist_ok=True)
    # Save full model as requested.
    model.save(str(out_model_path))
    return out_model_path



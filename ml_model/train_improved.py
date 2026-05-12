"""Improved training script for deepfake detection with:
- Data augmentation
- Class balancing
- Early stopping and learning rate reduction
- Validation split
- Fine-tuning strategy

Usage:
    python -m ml_model.train_improved \
        --data_dir data \
        --out ml_model/weights/resnet50_deepfake_improved.h5 \
        --epochs 50 \
        --batch_size 32
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

from ml_model.model import ModelConfig, build_resnet50_binary_classifier
from ml_model.preprocess import create_augmentation_layer

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


def create_data_pipeline(
    data_dir: Path,
    input_size: int,
    batch_size: int,
    validation_split: float,
    seed: int = 42,
) -> Tuple[tf.data.Dataset, tf.data.Dataset, dict]:
    """Create training and validation datasets with augmentation.
    
    Args:
        data_dir: Path to dataset with 'real' and 'fake' subfolders
        input_size: Image size (height/width)
        batch_size: Batch size
        validation_split: Fraction of data for validation
        seed: Random seed for reproducibility
        
    Returns:
        (train_dataset, val_dataset, class_info_dict)
    """
    # Load datasets
    train_ds = tf.keras.utils.image_dataset_from_directory(
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
    
    val_ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        labels="inferred",
        label_mode="binary",
        image_size=(input_size, input_size),
        batch_size=batch_size,
        shuffle=False,
        seed=seed,
        validation_split=validation_split,
        subset="validation",
    )
    
    class_names = train_ds.class_names
    logger.info(f"Class names: {class_names}")
    
    if set(class_names) != {"fake", "real"}:
        raise ValueError(f"Expected folders {{fake, real}}, got: {class_names}")
    
    # Determine if fake is first (alphabetically)
    fake_is_first = class_names[0].lower() == "fake"
    
    # Count samples in each class
    fake_count = sum(1 for _, label in train_ds.unbatch() if 
                     (label.numpy() == 0) == fake_is_first)
    total_train = sum(1 for _ in train_ds.unbatch())
    real_count = total_train - fake_count
    
    logger.info(f"Training samples - REAL: {real_count}, FAKE: {fake_count}")
    
    # Compute class weights for imbalance handling
    labels = []
    for _, label in train_ds.unbatch().take(10000):  # Sample for efficiency
        labels.append(1 if (label.numpy() == 0) == fake_is_first else 0)
    
    if len(labels) > 0:
        class_weights = compute_class_weight(
            'balanced',
            classes=np.unique(labels),
            y=labels
        )
        class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
        logger.info(f"Class weights: {class_weight_dict}")
    else:
        class_weight_dict = None
    
    # Normalization function
    def normalize_and_map(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        # Map: fake=1, real=0
        if fake_is_first:
            y = 1.0 - y
        return x, y
    
    # Apply normalization
    train_ds = train_ds.map(normalize_and_map, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(normalize_and_map, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Add augmentation for training using tf.image operations
    def augment_image(x, y):
        # Apply random augmentation
        x = tf.image.random_flip_left_right(x)
        x = tf.image.random_brightness(x, max_delta=0.2)
        x = tf.image.random_contrast(x, lower=0.8, upper=1.2)
        # Random rotation simulation using tf.image (simplified)
        x = tf.clip_by_value(x, 0.0, 1.0)
        return x, y
    
    train_ds = train_ds.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    
    # Optimize performance
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)
    
    class_info = {
        'class_names': class_names,
        'fake_is_first': fake_is_first,
        'real_count': int(real_count),
        'fake_count': int(fake_count),
        'class_weights': class_weight_dict,
    }
    
    return train_ds, val_ds, class_info


def create_callbacks(
    checkpoint_path: Path,
    patience_early_stop: int = 10,
    patience_lr_reduce: int = 5,
    min_lr: float = 1e-7,
) -> list:
    """Create training callbacks.
    
    Args:
        checkpoint_path: Path to save best model
        patience_early_stop: Epochs to wait before early stopping
        patience_lr_reduce: Epochs to wait before reducing LR
        min_lr: Minimum learning rate
        
    Returns:
        List of Keras callbacks
    """
    callbacks = [
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_auc',
            mode='max',
            patience=patience_early_stop,
            restore_best_weights=True,
            verbose=1,
        ),
        
        # Reduce learning rate when plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            mode='min',
            factor=0.5,
            patience=patience_lr_reduce,
            min_lr=min_lr,
            verbose=1,
        ),
        
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            str(checkpoint_path),
            monitor='val_auc',
            mode='max',
            save_best_only=True,
            verbose=1,
        ),
        
        # TensorBoard logging
        tf.keras.callbacks.TensorBoard(
            log_dir=str(checkpoint_path.parent / 'logs'),
            histogram_freq=1,
        ),
    ]
    
    return callbacks


def train_model_improved(
    data_dir: Path,
    output_path: Path,
    input_size: int = 224,
    batch_size: int = 32,
    epochs: int = 50,
    validation_split: float = 0.2,
    learning_rate: float = 1e-4,
    dropout: float = 0.5,
    fine_tune_at: int = 100,
    fine_tune_epochs: int = 30,
    fine_tune_lr: float = 1e-5,
    seed: int = 42,
) -> dict:
    """Train deepfake detection model with improved pipeline.
    
    Training strategy:
    1. Train only the classification head (frozen backbone)
    2. Fine-tune top layers of ResNet50
    
    Args:
        data_dir: Path to dataset
        output_path: Path to save trained model
        input_size: Input image size
        batch_size: Batch size
        epochs: Epochs for initial training
        validation_split: Validation data fraction
        learning_rate: Initial learning rate
        dropout: Dropout rate
        fine_tune_at: Layer to start fine-tuning from
        fine_tune_epochs: Epochs for fine-tuning
        fine_tune_lr: Learning rate for fine-tuning
        seed: Random seed
        
    Returns:
        Training history dictionary
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Create data pipeline
    logger.info("Creating data pipeline...")
    train_ds, val_ds, class_info = create_data_pipeline(
        data_dir=data_dir,
        input_size=input_size,
        batch_size=batch_size,
        validation_split=validation_split,
        seed=seed,
    )
    
    # Save class info
    with open(output_path.parent / 'class_info.json', 'w') as f:
        json.dump(class_info, f, indent=2, default=str)
    
    cfg = ModelConfig(
        input_size=input_size,
        dropout=dropout,
        learning_rate=learning_rate,
        fine_tune_learning_rate=fine_tune_lr,
    )
    
    # Phase 1: Train with frozen backbone
    logger.info("Phase 1: Training classification head (frozen backbone)...")
    model = build_resnet50_binary_classifier(cfg, freeze_base=True)
    
    callbacks = create_callbacks(
        checkpoint_path=output_path.parent / 'best_phase1.h5',
        patience_early_stop=10,
        patience_lr_reduce=5,
    )
    
    history1 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        class_weight=class_info['class_weights'],
        callbacks=callbacks,
        verbose=1,
    )
    
    # Load best weights from phase 1
    model.load_weights(str(output_path.parent / 'best_phase1.h5'))
    
    # Phase 2: Fine-tune top layers
    logger.info(f"Phase 2: Fine-tuning top {fine_tune_at} layers...")
    
    # Unfreeze all layers first
    model.trainable = True
    
    # Get all layers except the classification head (last 7 layers)
    all_layers = model.layers
    # Freeze all early layers, keep top N unfrozen
    for layer in all_layers[:-fine_tune_at]:
        layer.trainable = False
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=fine_tune_lr),
        loss="binary_crossentropy",
        metrics=[
            tf.keras.metrics.BinaryAccuracy(name="accuracy"),
            tf.keras.metrics.AUC(name="auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )
    
    callbacks2 = create_callbacks(
        checkpoint_path=output_path.parent / 'best_phase2.h5',
        patience_early_stop=15,
        patience_lr_reduce=7,
    )
    
    history2 = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=fine_tune_epochs,
        class_weight=class_info['class_weights'],
        callbacks=callbacks2,
        verbose=1,
    )
    
    # Load best weights from phase 2
    model.load_weights(str(output_path.parent / 'best_phase2.h5'))
    
    # Save final model
    model.save(str(output_path))
    logger.info(f"Saved model to: {output_path}")
    
    # Combine histories
    combined_history = {
        'phase1': {k: [float(v) for v in vals] for k, vals in history1.history.items()},
        'phase2': {k: [float(v) for v in vals] for k, vals in history2.history.items()},
        'class_info': class_info,
    }
    
    # Save history
    with open(output_path.parent / 'training_history.json', 'w') as f:
        json.dump(combined_history, f, indent=2)
    
    return combined_history


def main():
    parser = argparse.ArgumentParser(description="Train improved deepfake detection model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset")
    parser.add_argument("--out", type=str, required=True, help="Output model path (.h5)")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=50, help="Initial training epochs")
    parser.add_argument("--validation_split", type=float, default=0.2, help="Validation split")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--dropout", type=float, default=0.5, help="Dropout rate")
    parser.add_argument("--fine_tune_at", type=int, default=100, help="Layers to fine-tune")
    parser.add_argument("--fine_tune_epochs", type=int, default=30, help="Fine-tune epochs")
    parser.add_argument("--fine_tune_lr", type=float, default=1e-5, help="Fine-tune learning rate")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    history = train_model_improved(
        data_dir=Path(args.data_dir),
        output_path=Path(args.out),
        input_size=args.input_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        validation_split=args.validation_split,
        learning_rate=args.learning_rate,
        dropout=args.dropout,
        fine_tune_at=args.fine_tune_at,
        fine_tune_epochs=args.fine_tune_epochs,
        fine_tune_lr=args.fine_tune_lr,
        seed=args.seed,
    )
    
    print("Training complete!")
    print(f"Model saved to: {args.out}")


if __name__ == "__main__":
    main()

"""Evaluation script for deepfake detection model.

Provides:
- Confusion matrix visualization
- Precision, Recall, F1-score calculation
- Threshold tuning (0.5, 0.6, 0.7)
- False positive analysis
- ROC and Precision-Recall curves

Usage:
    python -m ml_model.evaluate \
        --model_path ml_model/weights/resnet50_deepfake_improved.h5 \
        --data_dir data/test \
        --output_dir results
"""

from __future__ import annotations

import argparse
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple, Dict

import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    precision_recall_fscore_support,
    roc_curve,
    precision_recall_curve,
    auc,
)
import matplotlib.pyplot as plt

from ml_model.model import load_model, ModelConfig
from ml_model.preprocess import preprocess_with_face_detection

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


@dataclass
class ThresholdMetrics:
    """Metrics for a specific threshold."""
    threshold: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    false_positives: int
    false_negatives: int
    true_positives: int
    true_negatives: int
    
    @property
    def fp_rate(self) -> float:
        """False positive rate (REAL predicted as FAKE)."""
        total_real = self.true_negatives + self.false_positives
        return self.false_positives / total_real if total_real > 0 else 0
    
    @property
    def fn_rate(self) -> float:
        """False negative rate (FAKE predicted as REAL)."""
        total_fake = self.true_positives + self.false_negatives
        return self.false_negatives / total_fake if total_fake > 0 else 0


def load_test_dataset(
    data_dir: Path,
    input_size: int = 224,
    batch_size: int = 32,
    use_face_detection: bool = True,
) -> Tuple[List[np.ndarray], List[int], List[str]]:
    """Load test dataset.
    
    Args:
        data_dir: Path to test data with 'real' and 'fake' subfolders
        input_size: Image size
        batch_size: Batch size (for dataset loading)
        use_face_detection: Whether to apply face detection
        
    Returns:
        (images, labels, file_paths)
        labels: 0 = REAL, 1 = FAKE
    """
    images = []
    labels = []
    file_paths = []
    
    # Load real images
    real_dir = data_dir / "real"
    if real_dir.exists():
        for img_path in real_dir.glob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                img = tf.keras.preprocessing.image.load_img(
                    str(img_path), target_size=(input_size, input_size)
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                
                if use_face_detection:
                    # Convert to BGR for face detection
                    img_bgr = img_array[:, :, ::-1].astype(np.uint8)
                    img_array = preprocess_with_face_detection(
                        img_bgr, size=input_size, use_face_detection=True
                    )
                    img_array = (img_array * 255).astype(np.uint8)
                
                images.append(img_array)
                labels.append(0)  # REAL = 0
                file_paths.append(str(img_path))
    
    # Load fake images
    fake_dir = data_dir / "fake"
    if fake_dir.exists():
        for img_path in fake_dir.glob("*"):
            if img_path.suffix.lower() in ['.jpg', '.jpeg', '.png', '.bmp', '.webp']:
                img = tf.keras.preprocessing.image.load_img(
                    str(img_path), target_size=(input_size, input_size)
                )
                img_array = tf.keras.preprocessing.image.img_to_array(img)
                
                if use_face_detection:
                    img_bgr = img_array[:, :, ::-1].astype(np.uint8)
                    img_array = preprocess_with_face_detection(
                        img_bgr, size=input_size, use_face_detection=True
                    )
                    img_array = (img_array * 255).astype(np.uint8)
                
                images.append(img_array)
                labels.append(1)  # FAKE = 1
                file_paths.append(str(img_path))
    
    logger.info(f"Loaded {len(images)} test images ({labels.count(0)} REAL, {labels.count(1)} FAKE)")
    
    return images, labels, file_paths


def predict_batch(
    model: tf.keras.Model,
    images: List[np.ndarray],
    batch_size: int = 32,
) -> np.ndarray:
    """Predict probabilities for a batch of images.
    
    Args:
        model: Trained model
        images: List of preprocessed images
        batch_size: Batch size for prediction
        
    Returns:
        Array of p_fake probabilities
    """
    predictions = []
    
    for i in range(0, len(images), batch_size):
        batch = images[i:i + batch_size]
        # Normalize to [0, 1]
        batch = np.array(batch) / 255.0
        preds = model.predict(batch, verbose=0)
        predictions.extend(preds.flatten())
    
    return np.array(predictions)


def evaluate_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    threshold: float,
) -> ThresholdMetrics:
    """Evaluate model at a specific threshold.
    
    Args:
        y_true: True labels (0=REAL, 1=FAKE)
        y_pred_proba: Predicted probabilities of FAKE
        threshold: Classification threshold
        
    Returns:
        ThresholdMetrics object
    """
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Calculate confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Calculate metrics
    accuracy = (tp + tn) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    return ThresholdMetrics(
        threshold=threshold,
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1_score=f1,
        false_positives=int(fp),
        false_negatives=int(fn),
        true_positives=int(tp),
        true_negatives=int(tn),
    )


def find_optimal_threshold(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    metric: str = "f1",
) -> Tuple[float, float]:
    """Find optimal threshold based on a metric.
    
    Args:
        y_true: True labels
        y_pred_proba: Predicted probabilities
        metric: Metric to optimize ('f1', 'precision', 'recall', 'accuracy')
        
    Returns:
        (optimal_threshold, best_score)
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_score = 0
    best_threshold = 0.5
    
    for thresh in thresholds:
        metrics = evaluate_threshold(y_true, y_pred_proba, thresh)
        score = getattr(metrics, metric, metrics.f1_score)
        
        if score > best_score:
            best_score = score
            best_threshold = thresh
    
    return best_threshold, best_score


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
    title: str = "Confusion Matrix",
):
    """Plot and save confusion matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=['REAL', 'FAKE'],
        yticklabels=['REAL', 'FAKE'],
        title=title,
        ylabel='True Label',
        xlabel='Predicted Label'
    )
    
    # Add text annotations
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], 'd'),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved confusion matrix to: {output_path}")


def plot_roc_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path,
):
    """Plot ROC curve."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ROC curve to: {output_path}")


def plot_precision_recall_curve(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path,
):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_pred_proba)
    pr_auc = auc(recall, precision)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision, color='blue', lw=2, label=f'PR curve (AUC = {pr_auc:.3f})')
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title('Precision-Recall Curve')
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved PR curve to: {output_path}")


def plot_threshold_analysis(
    y_true: np.ndarray,
    y_pred_proba: np.ndarray,
    output_path: Path,
):
    """Plot metrics vs threshold."""
    thresholds = np.arange(0.1, 0.9, 0.01)
    metrics_list = []
    
    for thresh in thresholds:
        metrics = evaluate_threshold(y_true, y_pred_proba, thresh)
        metrics_list.append(metrics)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Accuracy
    axes[0, 0].plot(thresholds, [m.accuracy for m in metrics_list], 'b-', label='Accuracy')
    axes[0, 0].set_xlabel('Threshold')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].set_title('Accuracy vs Threshold')
    axes[0, 0].grid(True, alpha=0.3)
    
    # Precision and Recall
    axes[0, 1].plot(thresholds, [m.precision for m in metrics_list], 'g-', label='Precision')
    axes[0, 1].plot(thresholds, [m.recall for m in metrics_list], 'r-', label='Recall')
    axes[0, 1].plot(thresholds, [m.f1_score for m in metrics_list], 'm-', label='F1-Score')
    axes[0, 1].set_xlabel('Threshold')
    axes[0, 1].set_ylabel('Score')
    axes[0, 1].set_title('Precision, Recall, F1 vs Threshold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # False Positive Rate (REAL predicted as FAKE)
    axes[1, 0].plot(thresholds, [m.fp_rate for m in metrics_list], 'r-', label='False Positive Rate')
    axes[1, 0].set_xlabel('Threshold')
    axes[1, 0].set_ylabel('False Positive Rate')
    axes[1, 0].set_title('False Positive Rate (REAL→FAKE) vs Threshold')
    axes[1, 0].grid(True, alpha=0.3)
    
    # False Negative Rate (FAKE predicted as REAL)
    axes[1, 1].plot(thresholds, [m.fn_rate for m in metrics_list], 'orange', label='False Negative Rate')
    axes[1, 1].set_xlabel('Threshold')
    axes[1, 1].set_ylabel('False Negative Rate')
    axes[1, 1].set_title('False Negative Rate (FAKE→REAL) vs Threshold')
    axes[1, 1].grid(True, alpha=0.3)
    
    fig.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved threshold analysis to: {output_path}")


def evaluate_model(
    model_path: Path,
    data_dir: Path,
    output_dir: Path,
    input_size: int = 224,
    use_face_detection: bool = True,
) -> Dict:
    """Complete model evaluation.
    
    Args:
        model_path: Path to trained model
        data_dir: Path to test dataset
        output_dir: Directory to save results
        input_size: Input image size
        use_face_detection: Whether to use face detection
        
    Returns:
        Evaluation results dictionary
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    logger.info(f"Loading model from: {model_path}")
    cfg = ModelConfig(input_size=input_size)
    model = load_model(model_path, cfg)
    
    # Load test data
    logger.info(f"Loading test data from: {data_dir}")
    images, labels, file_paths = load_test_dataset(
        data_dir, input_size, use_face_detection=use_face_detection
    )
    
    y_true = np.array(labels)
    
    # Predict
    logger.info("Running predictions...")
    y_pred_proba = predict_batch(model, images)
    
    # Evaluate at different thresholds
    thresholds = [0.5, 0.6, 0.7]
    results = {}
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    
    for thresh in thresholds:
        metrics = evaluate_threshold(y_true, y_pred_proba, thresh)
        results[f'threshold_{thresh}'] = {
            'accuracy': metrics.accuracy,
            'precision': metrics.precision,
            'recall': metrics.recall,
            'f1_score': metrics.f1_score,
            'false_positives': metrics.false_positives,
            'false_negatives': metrics.false_negatives,
            'fp_rate': metrics.fp_rate,
            'fn_rate': metrics.fn_rate,
        }
        
        print(f"\nThreshold: {thresh}")
        print(f"  Accuracy:  {metrics.accuracy:.4f}")
        print(f"  Precision: {metrics.precision:.4f}")
        print(f"  Recall:    {metrics.recall:.4f}")
        print(f"  F1-Score:  {metrics.f1_score:.4f}")
        print(f"  FP (REAL→FAKE): {metrics.false_positives} ({metrics.fp_rate:.2%})")
        print(f"  FN (FAKE→REAL): {metrics.false_negatives} ({metrics.fn_rate:.2%})")
        
        # Plot confusion matrix
        y_pred = (y_pred_proba >= thresh).astype(int)
        plot_confusion_matrix(
            y_true, y_pred,
            output_dir / f'confusion_matrix_thresh_{thresh}.png',
            title=f'Confusion Matrix (Threshold={thresh})'
        )
    
    # Find optimal thresholds
    opt_f1, f1_score = find_optimal_threshold(y_true, y_pred_proba, 'f1')
    opt_precision, precision_score = find_optimal_threshold(y_true, y_pred_proba, 'precision')
    opt_recall, recall_score = find_optimal_threshold(y_true, y_pred_proba, 'recall')
    
    print("\n" + "-"*60)
    print("OPTIMAL THRESHOLDS")
    print("-"*60)
    print(f"Best F1-Score:     Threshold={opt_f1:.2f}, F1={f1_score:.4f}")
    print(f"Best Precision:    Threshold={opt_precision:.2f}, Precision={precision_score:.4f}")
    print(f"Best Recall:       Threshold={opt_recall:.2f}, Recall={recall_score:.4f}")
    
    results['optimal_thresholds'] = {
        'f1': {'threshold': opt_f1, 'score': f1_score},
        'precision': {'threshold': opt_precision, 'score': precision_score},
        'recall': {'threshold': opt_recall, 'score': recall_score},
    }
    
    # Classification report at default threshold
    print("\n" + "-"*60)
    print("CLASSIFICATION REPORT (Threshold=0.5)")
    print("-"*60)
    y_pred_default = (y_pred_proba >= 0.5).astype(int)
    print(classification_report(y_true, y_pred_default, target_names=['REAL', 'FAKE']))
    
    # Generate plots
    logger.info("Generating plots...")
    plot_roc_curve(y_true, y_pred_proba, output_dir / 'roc_curve.png')
    plot_precision_recall_curve(y_true, y_pred_proba, output_dir / 'pr_curve.png')
    plot_threshold_analysis(y_true, y_pred_proba, output_dir / 'threshold_analysis.png')
    
    # Save results
    with open(output_dir / 'evaluation_results.json', 'w') as f:
        json.dump(results, f, indent=2)
    
    logger.info(f"Results saved to: {output_dir}")
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Evaluate deepfake detection model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to model")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to test data")
    parser.add_argument("--output_dir", type=str, default="results", help="Output directory")
    parser.add_argument("--input_size", type=int, default=224, help="Input image size")
    parser.add_argument("--no_face_detection", action="store_true", help="Disable face detection")
    
    args = parser.parse_args()
    
    results = evaluate_model(
        model_path=Path(args.model_path),
        data_dir=Path(args.data_dir),
        output_dir=Path(args.output_dir),
        input_size=args.input_size,
        use_face_detection=not args.no_face_detection,
    )
    
    print("\nEvaluation complete!")


if __name__ == "__main__":
    main()

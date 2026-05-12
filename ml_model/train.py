"""Optional training script (transfer learning).

This is a template: you must provide a dataset structured as:

data/
  real/
    img1.jpg
    ...
  fake/
    img1.jpg
    ...

Then run:
  python -m ml_model.train --data_dir data --out ml_model/weights/resnet50_deepfake.keras
"""

from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from ml_model.model import ModelConfig, build_resnet50_binary_classifier


def build_dataset(data_dir: Path, input_size: int, batch_size: int = 16):
    ds = tf.keras.utils.image_dataset_from_directory(
        str(data_dir),
        labels="inferred",
        label_mode="binary",
        image_size=(input_size, input_size),
        batch_size=batch_size,
        shuffle=True,
        seed=42,
    )
    # Map labels: expects subfolders alphabetical. If your folders are `fake` and `real`,
    # alphabetical order is fake(0), real(1) which is opposite of our convention.
    class_names = ds.class_names
    if set(class_names) != {"fake", "real"}:
        raise ValueError(f"Expected folders {{fake, real}}, got: {class_names}")

    # Convert uint8 [0,255] -> float32 [0,1], and flip labels so fake=1, real=0.
    def _map(x, y):
        x = tf.cast(x, tf.float32) / 255.0
        # If fake is index 0 and real is index 1, y=0 means fake; flip to make fake=1.
        y = 1.0 - y
        return x, y

    ds = ds.map(_map, num_parallel_calls=tf.data.AUTOTUNE).prefetch(tf.data.AUTOTUNE)
    return ds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--input_size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch_size", type=int, default=16)
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)

    ds = build_dataset(data_dir, input_size=args.input_size, batch_size=args.batch_size)
    cfg = ModelConfig(input_size=args.input_size)
    model = build_resnet50_binary_classifier(cfg)

    # Warm-up head training
    model.fit(ds, epochs=args.epochs)

    # Optional fine-tuning: unfreeze top layers
    base = model.get_layer("resnet50")
    base.trainable = True
    for layer in base.layers[:-40]:
        layer.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.BinaryAccuracy(name="acc"), tf.keras.metrics.AUC(name="auc")],
    )
    model.fit(ds, epochs=max(1, args.epochs // 2))

    model.save_weights(str(out))
    print(f"Saved weights to: {out}")


if __name__ == "__main__":
    main()


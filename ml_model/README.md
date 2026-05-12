# ML Model (ResNet50 Transfer Learning)

## What it does

- Uses **ResNet50 pretrained on ImageNet** as a frozen backbone
- Adds a small dense head for **binary classification**
- Output is sigmoid \(p_\text{fake}\) where:
  - `p_fake >= 0.5` → `FAKE`
  - `p_fake < 0.5` → `REAL`

## Weights

Default expected weights path:

- `ml_model/weights/resnet50_deepfake.keras`

Override via `.env`:

- `MODEL_WEIGHTS_PATH=...`

## Train (template)

Prepare folders:

```
data/
  fake/
  real/
```

Run:

```bash
python -m ml_model.train --data_dir data --out ml_model/weights/resnet50_deepfake.keras
```


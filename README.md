# LineFixer: U-Net for Line Art Repair

This project trains a U-Net to convert broken single-pixel line drawings into repaired line art.

## Project Layout

- `linefixer/`
  - `models/unet.py` — U-Net architecture
  - `data/dataset.py` — Dataset for `(broken -> fixed)` pairs
  - `losses/combined.py` — Combined BCE + Dice + L1 loss
  - `utils/patch_infer.py` — Patch-based inference utilities (`fix_line_art`)
  - `utils/visualize.py` — Training curves, batch visualization, original vs fixed comparison
- `scripts/`
  - `generate_data.py` — Generate training data
- `run_training.py` — Train the model end-to-end
- `run_inference_dir.py` — Run inference over `test/borken/**` and save to `test/fixed/**`
- `requirements.txt` — Python dependencies
- `checkpoints/` — Model checkpoints will be saved here (created on demand)
- `results/` — Training history and plots (created on demand)
- Data folders (expected):
  - `train/` and `val/` with subfolders:
    - `broken/` (inputs)
    - `fixed/` (targets)
  - `test/borken/**` and `test/fixed/**` (for batch inference and comparison)

## Installation

## Training

Assumes your data structure:

```
train/
  broken/*.png
  fixed/*.png
val/
  broken/*.png
  fixed/*.png
```

Run training:

```bash
python run_training.py \
  --train_dir train \
  --val_dir val \
  --batch_size 16 \
  --epochs 100 \
  --lr 1e-3 \
  --save_every 5
```

Artifacts:
- Best model at `checkpoints/best_model.pth`
- Periodic checkpoints at `checkpoints/checkpoint_epoch_*.pth`
- Training history JSON at `results/history.json`

## Inference (Batch over test set)

Assumes `test/borken/**` contains category subfolders with PNGs. Outputs will be saved to `test/fixed/**` with the same structure.

```bash
python run_inference_dir.py \
  --test_root test \
  --checkpoint checkpoints/best_model.pth \
  --image_size 64 \
  --overlap 8 \
  --display_every 0
```

- `--overlap` can be increased (e.g., 16 or 32) for smoother patch blending on large images.

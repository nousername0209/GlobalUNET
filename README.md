# GlobalUNET

Reproducible semantic segmentation benchmark comparing U-Net and U-Net++ on PASCAL VOC 2012 and Cityscapes.

## What This Runs

The project runs four experiments with the same optimizer, scheduler, seed control, image size, batch size, and augmentation policy:

| Dataset | Model | Script |
| --- | --- | --- |
| PASCAL VOC 2012 | U-Net | `experiments/train_voc2012_unet.py` |
| PASCAL VOC 2012 | U-Net++ | `experiments/train_voc2012_unetpp.py` |
| Cityscapes | U-Net | `experiments/train_cityscapes_unet.py` |
| Cityscapes | U-Net++ | `experiments/train_cityscapes_unetpp.py` |

Each run saves:

- mIoU
- Dice/F1
- pixel accuracy
- trainable parameter count
- inference time per image
- per-epoch history CSV
- final metrics CSV

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

On Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Data

Use `data/` as the default dataset root.

### PASCAL VOC 2012

VOC can be downloaded automatically by adding `--download-voc` to the VOC commands.

Expected path after download:

```text
data/
  VOCdevkit/
    VOC2012/
```

### Cityscapes

Cityscapes requires manual download from the official Cityscapes site. Put the extracted folders under `data/`:

```text
data/
  leftImg8bit/
    train/
    val/
  gtFine/
    train/
    val/
```

## Run Experiments

The commands below use the same training policy for every model. Change shared options such as `--epochs`, `--batch-size`, `--image-size`, and `--seed` in the same way for all four commands when making a fair comparison.

```bash
python -m experiments.train_voc2012_unet --data-root data --download-voc --epochs 50 --batch-size 4 --image-size 256 --seed 42
python -m experiments.train_voc2012_unetpp --data-root data --download-voc --epochs 50 --batch-size 4 --image-size 256 --seed 42
python -m experiments.train_cityscapes_unet --data-root data --epochs 50 --batch-size 4 --image-size 256 --seed 42
python -m experiments.train_cityscapes_unetpp --data-root data --epochs 50 --batch-size 4 --image-size 256 --seed 42
```

CPU-only smoke test:

```bash
python -m experiments.train_voc2012_unet --data-root data --download-voc --epochs 1 --batch-size 1 --image-size 128 --device cpu --no-amp
```

## Export Summary CSV

Each experiment writes:

```text
runs/<dataset>_<model>_seed<seed>/history.csv
runs/<dataset>_<model>_seed<seed>/metrics.csv
```

Combine all final metrics:

```bash
python summarize_results.py --results-dir runs --output-csv runs/summary.csv
```

The summary CSV contains one row per experiment with:

```text
dataset, model, seed, epochs, batch_size, image_size, optimizer, scheduler,
learning_rate, weight_decay, miou, dice_f1, pixel_accuracy,
parameter_count, inference_time_ms_per_image, run_dir
```

## Reproducibility Notes

- All scripts use `seed=42` by default.
- Python, NumPy, PyTorch CPU, and PyTorch CUDA seeds are fixed.
- CuDNN benchmarking is disabled and deterministic algorithms are requested.
- Both models use AdamW and cosine annealing.
- Both datasets use the same paired segmentation augmentation policy: horizontal flip, resize, and color jitter for training; resize only for validation.
- VOC uses 21 classes and ignore index `255`.
- Cityscapes maps official label IDs to the standard 19 train IDs and ignores all other labels with index `255`.

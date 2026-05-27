# GlobalUNET

Reproducible graduation-research benchmark for semantic segmentation.

## Research Framing

U-Net was originally proposed for biomedical image segmentation. U-Net++ was proposed as an improved architecture with redesigned nested skip pathways.

This project does not exactly reproduce the original medical-image papers or their training pipelines. Instead, it evaluates whether the claimed structural improvement of U-Net++ generalizes to two non-medical global benchmarks:

- PASCAL VOC 2012
- Cityscapes

The public `zhixuhao/unet` repository is used only as a reference for a simple U-Net architecture style. A public PyTorch Nested U-Net / U-Net++ implementation style is used only as a reference for nested skip pathways and optional deep supervision. Both models are reimplemented in one unified PyTorch codebase so the training and evaluation policy is shared.

## Project Structure

```text
GlobalUNET/
  README.md
  requirements.txt
  AGENTS.md
  train.py
  evaluate.py
  benchmark.py
  configs/
  models/
  datasets/
  utils/
  scripts/
  tests/
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Dataset Preparation

By default, configs expect datasets under `data/`.

PASCAL VOC 2012:

```text
data/
  VOCdevkit/
    VOC2012/
```

If you want torchvision to download VOC, set `dataset.download: true` in the VOC configs.

Cityscapes requires manual download from the official Cityscapes website:

```text
data/
  leftImg8bit/
    train/
    val/
  gtFine/
    train/
    val/
```

Cityscapes labels are mapped to the standard 19 train IDs. Other labels are ignored with index `255`.

## Training

```bash
python train.py --config configs/voc_unet.yaml
python train.py --config configs/voc_unetpp.yaml
python train.py --config configs/cityscapes_unet.yaml
python train.py --config configs/cityscapes_unetpp.yaml
```

Shell shortcuts:

```bash
bash scripts/run_voc_unet.sh
bash scripts/run_voc_unetpp.sh
bash scripts/run_cityscapes_unet.sh
bash scripts/run_cityscapes_unetpp.sh
```

## Evaluation

```bash
python evaluate.py --config configs/voc_unet.yaml
python evaluate.py --config configs/voc_unetpp.yaml
python evaluate.py --config configs/cityscapes_unet.yaml
python evaluate.py --config configs/cityscapes_unetpp.yaml
```

Each run writes outputs to:

```text
runs/<experiment_name>/
  checkpoints/best.pt
  checkpoints/last.pt
  logs/history.csv
  train_summary.csv
  metrics.csv
```

## Benchmark Summary

```bash
python benchmark.py --results-root runs --output-csv runs/summary.csv --output-md runs/summary.md
```

The summary reports mIoU, Dice/F1, pixel accuracy, parameter count, inference latency per image, training time per epoch, and GPU memory usage when CUDA is available.

## Fairness Controls

The comparison is controlled through YAML configs:

- same framework: PyTorch
- same image preprocessing policy
- same augmentation policy per dataset
- same optimizer: AdamW
- same scheduler: CosineAnnealingLR
- same seed control
- same standard train/val split usage
- same batch size unless memory forces a documented change
- same logging and evaluation pipeline
- same cross-entropy loss for multiclass segmentation

U-Net++ deep supervision is configurable and disabled by default:

```yaml
model:
  deep_supervision: false
```

If deep supervision is enabled, report it as a separate setting rather than mixing it with the baseline U-Net++ result.

## Tests

```bash
pytest tests
```

The tests check model forward shapes, dataset transform output format, Cityscapes label mapping, and metric correctness on toy tensors.

## Assumptions

- This is a unified benchmark implementation, not a line-by-line reproduction of the original U-Net or U-Net++ training pipelines.
- The reference repositories guide architecture choices only.
- Dataset splits follow each dataset's standard train/val protocol.
- If GPU memory requires changing batch size for one model, document the change in the relevant config and final report.

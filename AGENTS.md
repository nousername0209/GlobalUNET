# Agent Instructions

This repository is a reproducible graduation-research benchmark comparing U-Net and U-Net++ on PASCAL VOC 2012 and Cityscapes.

## How To Validate Changes

Run smoke tests:

```bash
pytest tests
```

Run command help checks:

```bash
python train.py --help
python evaluate.py --help
python benchmark.py --help
```

Full training requires the datasets and a PyTorch environment.

## Experiment Scripts

Use these configs:

- `configs/voc_unet.yaml`
- `configs/voc_unetpp.yaml`
- `configs/cityscapes_unet.yaml`
- `configs/cityscapes_unetpp.yaml`

Use these shell wrappers:

- `scripts/run_voc_unet.sh`
- `scripts/run_voc_unetpp.sh`
- `scripts/run_cityscapes_unet.sh`
- `scripts/run_cityscapes_unetpp.sh`

## Fairness Rules

Do not silently change model capacity, loss, optimizer, scheduler, augmentation, preprocessing, split usage, seed, batch size, or logging policy for only one model.

If a change is necessary for memory or dataset compatibility, document it in the affected YAML config, `README.md`, and the final result report.

Keep U-Net++ deep supervision configurable. The default comparison should use `deep_supervision: false`. If enabled, treat it as a separate experimental condition.

## Reference Policy

Use `zhixuhao/unet` only as a U-Net architecture reference. Use public UNet++ / Nested U-Net repositories only as architecture references for nested skip pathways and deep supervision. Do not import incompatible external training pipelines.

Both models must stay inside this single PyTorch training/evaluation framework.

#!/usr/bin/env bash
set -euo pipefail
python train.py --config configs/cityscapes_unet.yaml
python evaluate.py --config configs/cityscapes_unet.yaml

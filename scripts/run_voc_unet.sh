#!/usr/bin/env bash
set -euo pipefail
python train.py --config configs/voc_unet.yaml
python evaluate.py --config configs/voc_unet.yaml

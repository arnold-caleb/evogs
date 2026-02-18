#!/bin/bash
# NOTE: This is a legacy alias for train_anchors.slurm
# Please use train_anchors.slurm instead.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec bash "${SCRIPT_DIR}/train_anchors.slurm" "$@"

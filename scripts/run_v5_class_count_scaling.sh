#!/usr/bin/env bash
set -Eeuo pipefail

# Generic entry point. Select the GPU with GPU=<index>.
exec bash /mnt/zzbnew/rnamodel/dengjingye/bacteria/new0715/scripts/run_v5_class_count_scaling_cuda0.sh "$@"

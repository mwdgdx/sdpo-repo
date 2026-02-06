#!/bin/bash
# Run SDPO training. Usage: ./verl_training.sh <experiment_name> <config_name> <data_path> [hydra_overrides...]

set -e
export PYTHONBUFFERED=1

export EXPERIMENT=${1:-"sdpo_local"}
CONFIG_NAME=${2:-"sdpo"}
export TASK=${3:-"datasets/tooluse"}

shift 3 || true
EXTRA_ARGS=("$@")

export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export PYTHONPATH="${PROJECT_ROOT}/verl:${PYTHONPATH:-}"

echo "Experiment: $EXPERIMENT"
echo "Config: $CONFIG_NAME"
echo "Task (data path): $TASK"
echo "Project root: $PROJECT_ROOT"
echo "Extra args: ${EXTRA_ARGS[*]}"

python -m verl.trainer.main_ppo --config-name "$CONFIG_NAME" "${EXTRA_ARGS[@]}"

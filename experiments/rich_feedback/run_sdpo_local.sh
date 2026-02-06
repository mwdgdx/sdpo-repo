#!/bin/bash
# Local SDPO rich-feedback run: small model + LoRA, save to Hugging Face.
# Usage: ./run_sdpo_local.sh [data_path]
# Set HF_USER and PUSH_TO_HUB_ID (or pass trainer.push_to_hub_id=...) to push to HF.

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export PROJECT_ROOT="${PROJECT_ROOT:-$(cd "$SCRIPT_DIR/../.." && pwd)}"
cd "$PROJECT_ROOT"

DATA_PATH="${1:-datasets/tooluse}"
CONFIG_NAME="sdpo"

# Small model (override in ARGS if needed)
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-1.5B-Instruct}"
# Your Hugging Face repo for LoRA adapter, e.g. YOUR_HF_USER/sdpo-lora-lcb
PUSH_TO_HUB_ID="${PUSH_TO_HUB_ID:-}"

TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR=1e-5
ALPHA=1.0
EXPERIMENT="SDPO-rich-feedback-local-$(date +%Y%m%d-%H%M)"

export PROJECT_ROOT
export N_GPUS_PER_NODE=1
export USER="${USER:-$(whoami)}"

ARGS=(
  data.train_batch_size=$TRAIN_BATCH_SIZE
  trainer.group_name=SDPO-rich-feedback
  actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE
  actor_rollout_ref.model.path=$MODEL_PATH
  actor_rollout_ref.actor.optim.lr=$LR
  actor_rollout_ref.actor.ppo_mini_batch_size=32
  actor_rollout_ref.actor.self_distillation.distillation_topk=20
  algorithm.rollout_correction.rollout_is=token
  actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=True
  actor_rollout_ref.actor.self_distillation.alpha=$ALPHA
  actor_rollout_ref.actor.self_distillation.ema_update_rate=0.01
  actor_rollout_ref.actor.optim.lr_warmup_steps=0
  actor_rollout_ref.rollout.val_kwargs.n=4
)

if [ -n "$PUSH_TO_HUB_ID" ]; then
  ARGS+=( "trainer.push_to_hub_id=$PUSH_TO_HUB_ID" )
fi

echo "----------------------------------------------------------------"
echo "SDPO Rich Feedback (local)"
echo "  Data: $DATA_PATH"
echo "  Model: $MODEL_PATH (LoRA)"
echo "  Push to HF: ${PUSH_TO_HUB_ID:-none}"
echo "----------------------------------------------------------------"

bash "$PROJECT_ROOT/training/verl_training.sh" "$EXPERIMENT" "$CONFIG_NAME" "$DATA_PATH" "${ARGS[@]}"

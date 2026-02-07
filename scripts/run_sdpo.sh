#!/bin/bash
# SDPO Training Script for 8x H100
# Usage: ./scripts/run_sdpo.sh [OPTIONS]
#
# Fixed Configuration: 8x H100 80GB, Full Fine-tuning, Qwen3-4B
#
# Environment Variables (optional overrides):
#   MODEL_PATH          - Hugging Face model path (default: Qwen/Qwen3-4B)
#   TASK                - Task name, corresponds to datasets/<TASK>/ directory (default: lcb_v6)
#   TOTAL_EPOCHS        - Total training epochs (default: 30)
#   EXPERIMENT_NAME     - Experiment name for logging (default: sdpo_$TASK)
#   SAVE_FREQ           - Checkpoint save frequency in iterations (default: 10)
#   TEST_FREQ           - Validation frequency in iterations (default: 5)
#   LEARNING_RATE       - Learning rate (default: 1e-5)
#   TRAIN_BATCH_SIZE    - Training batch size (default: 32)
#
#   WandB (Logging):
#   WANDB_API_KEY       - WandB API key for metrics tracking
#   WANDB_PROJECT       - WandB project name (default: SDPO)
#
#   Hugging Face (Upload):
#   HF_TOKEN            - Hugging Face token for uploading checkpoints
#   HF_REPO_ID          - Hugging Face repo ID for uploading

set -e

# ============================================================================
# Auto-start in tmux session (so training survives SSH disconnect)
# ============================================================================
if [ -z "$TMUX" ] && [ "$NO_TMUX" != "true" ] && command -v tmux &> /dev/null; then
    SESSION_NAME="sdpo_training"
    
    # Check if session already exists
    if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
        echo "=============================================="
        echo "tmux session '$SESSION_NAME' already exists!"
        echo "=============================================="
        echo ""
        echo "Options:"
        echo "  1. Attach to existing session:"
        echo "     tmux attach -t $SESSION_NAME"
        echo ""
        echo "  2. Kill existing session and start new:"
        echo "     tmux kill-session -t $SESSION_NAME"
        echo "     bash $0 $@"
        echo ""
        echo "  3. Run without tmux (not recommended):"
        echo "     NO_TMUX=true bash $0 $@"
        echo ""
        exit 1
    fi
    
    echo "=============================================="
    echo "Starting training in tmux session: $SESSION_NAME"
    echo "=============================================="
    echo ""
    echo "Training will continue even if SSH disconnects!"
    echo ""
    echo "Useful commands:"
    echo "  - Detach (keep running): Ctrl+B, then D"
    echo "  - Reattach later:        tmux attach -t $SESSION_NAME"
    echo "  - Kill session:          tmux kill-session -t $SESSION_NAME"
    echo ""
    sleep 2
    
    exec tmux new-session -s "$SESSION_NAME" "NO_TMUX=true bash $0 $@; echo ''; echo 'Training finished. Press Enter to close.'; read"
fi

# ============================================================================
# Get Workspace Root (auto-detect)
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script is in sdpo/scripts/, so parent is sdpo/
SDPO_ROOT="$(dirname "$SCRIPT_DIR")"
VERL_DIR="${SDPO_ROOT}/verl"

# ============================================================================
# Load .env file if exists
# ============================================================================
ENV_FILE="${SDPO_ROOT}/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading configuration from ${ENV_FILE}..."
    source "$ENV_FILE"
fi

# ============================================================================
# Fixed Configuration: 8x H100 Full Fine-tuning
# ============================================================================
echo "=============================================="
echo "Configuration: 8x H100 80GB (Full Fine-tuning)"
echo "=============================================="

export N_GPUS=8
export LORA_RANK=0          # Full fine-tuning (no LoRA)
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.60}"

# ============================================================================
# Default Configuration
# ============================================================================

# Set HuggingFace cache
export HF_HOME="${HF_HOME:-${SDPO_ROOT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_HOME"

# Model: Qwen3-4B for full fine-tuning
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-4B}"
export TASK="${TASK:-lcb_v6}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-30}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-sdpo_${TASK}}"
export PROJECT_NAME="${PROJECT_NAME:-${WANDB_PROJECT:-SDPO}}"

# Checkpoint settings
export SAVE_FREQ="${SAVE_FREQ:-10}"
export TEST_FREQ="${TEST_FREQ:-5}"
export MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-2}"

# Training settings
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"

# WandB configuration
export WANDB_PROJECT="${WANDB_PROJECT:-$PROJECT_NAME}"

# Upload settings
if [ -n "$HF_TOKEN" ] && [ -n "$HF_REPO_ID" ]; then
    UPLOAD_CHECKPOINT="${UPLOAD_CHECKPOINT:-true}"
else
    UPLOAD_CHECKPOINT="${UPLOAD_CHECKPOINT:-false}"
fi

# Data paths - based on TASK
export TRAIN_DATA_PATH="${SDPO_ROOT}/datasets/${TASK}/train.parquet"
export VAL_DATA_PATH="${SDPO_ROOT}/datasets/${TASK}/test.parquet"

# Checkpoint directory
export CKPT_DIR="${SDPO_ROOT}/checkpoints"

# Reward function path
export REWARD_FN_PATH="${VERL_DIR}/verl/utils/reward_score/feedback/__init__.py"

# ============================================================================
# Print Configuration
# ============================================================================
echo ""
echo "SDPO_ROOT:        $SDPO_ROOT"
echo "VERL_DIR:         $VERL_DIR"
echo ""
echo "Model & Training:"
echo "  MODEL_PATH:       $MODEL_PATH"
echo "  N_GPUS:           $N_GPUS"
echo "  TOTAL_EPOCHS:     $TOTAL_EPOCHS"
echo "  TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE"
echo "  MICRO_BATCH_SIZE: $MICRO_BATCH_SIZE"
echo "  LEARNING_RATE:    $LEARNING_RATE"
echo "  LORA_RANK:        $LORA_RANK (Full fine-tuning)"
echo "  GPU_MEM_UTIL:     $GPU_MEM_UTIL"
echo ""
echo "Data:"
echo "  TASK:             $TASK"
echo "  TRAIN_DATA_PATH:  $TRAIN_DATA_PATH"
echo "  VAL_DATA_PATH:    $VAL_DATA_PATH"
echo ""
echo "Output:"
echo "  EXPERIMENT_NAME:  $EXPERIMENT_NAME"
echo "  CKPT_DIR:         $CKPT_DIR"
echo "  SAVE_FREQ:        $SAVE_FREQ (save every N iterations)"
echo "  MAX_CKPT_TO_KEEP: $MAX_CKPT_TO_KEEP (older auto-deleted)"
echo "  HF_HOME:          $HF_HOME"
echo ""
echo "Logging:"
echo "  WANDB_PROJECT:     $WANDB_PROJECT"
echo "  WANDB_API_KEY:     ${WANDB_API_KEY:+[SET]}${WANDB_API_KEY:-[NOT SET]}"
echo ""
echo "Post-training:"
echo "  UPLOAD_CHECKPOINT: $UPLOAD_CHECKPOINT"
echo "  HF_REPO_ID:        ${HF_REPO_ID:-<not set>}"
echo "=============================================="

# Warn if no WandB credentials
if [ -z "$WANDB_API_KEY" ]; then
    echo ""
    echo "⚠️  WARNING: WANDB_API_KEY not set!"
    echo "   Training will use console logging only."
    echo "   To enable WandB tracking:"
    echo "     export WANDB_API_KEY=your_wandb_key"
    echo ""
    export USE_WANDB="false"
else
    export USE_WANDB="true"
fi

# Warn if no HF credentials
if [ "$UPLOAD_CHECKPOINT" = "false" ]; then
    echo ""
    echo "⚠️  WARNING: HF_TOKEN or HF_REPO_ID not set!"
    echo "   Checkpoints will NOT be uploaded to Hugging Face."
    echo ""
fi

# ============================================================================
# Validate Data Files
# ============================================================================
if [ ! -f "$TRAIN_DATA_PATH" ]; then
    echo "ERROR: Training data not found at $TRAIN_DATA_PATH"
    echo "Available datasets:"
    ls -la "${SDPO_ROOT}/datasets/" 2>/dev/null || echo "  (datasets directory not found)"
    exit 1
fi

if [ ! -f "$VAL_DATA_PATH" ]; then
    echo "WARNING: Validation data not found at $VAL_DATA_PATH"
    echo "Training will proceed without validation"
fi

# ============================================================================
# Create checkpoint directory
# ============================================================================
mkdir -p "$CKPT_DIR"

# ============================================================================
# Run Training
# ============================================================================
echo ""
echo "Starting SDPO training..."
echo ""

cd "$VERL_DIR"

# Build training command with config overrides
EXTRA_ARGS=""

# Logger override
if [ "$USE_WANDB" = "false" ]; then
    EXTRA_ARGS="$EXTRA_ARGS trainer.logger=[console]"
fi

# Training parameters overrides
EXTRA_ARGS="$EXTRA_ARGS data.train_batch_size=$TRAIN_BATCH_SIZE"
EXTRA_ARGS="$EXTRA_ARGS actor_rollout_ref.actor.ppo_mini_batch_size=$TRAIN_BATCH_SIZE"
EXTRA_ARGS="$EXTRA_ARGS actor_rollout_ref.actor.optim.lr=$LEARNING_RATE"

TRAINING_SUCCESS=false
python -m verl.trainer.main_ppo \
    --config-path "${VERL_DIR}/verl/trainer/config" \
    --config-name sdpo \
    $EXTRA_ARGS \
    "$@" && TRAINING_SUCCESS=true

echo ""
if [ "$TRAINING_SUCCESS" = "true" ]; then
    echo "✅ Training completed successfully!"
else
    echo "❌ Training failed!"
fi
echo "Checkpoints saved to: $CKPT_DIR/$EXPERIMENT_NAME"

# ============================================================================
# Upload to Hugging Face
# ============================================================================
if [ "$UPLOAD_CHECKPOINT" = "true" ] && [ "$TRAINING_SUCCESS" = "true" ]; then
    echo ""
    echo "=============================================="
    echo "Uploading checkpoint to Hugging Face..."
    echo "=============================================="
    
    # Find the latest checkpoint
    LATEST_CKPT=$(ls -td "${CKPT_DIR}/${EXPERIMENT_NAME}"/global_step_* 2>/dev/null | head -1)
    
    if [ -n "$LATEST_CKPT" ] && [ -d "$LATEST_CKPT" ]; then
        echo "Latest checkpoint: $LATEST_CKPT"
        echo "Uploading to: $HF_REPO_ID"
        
        python "${SCRIPT_DIR}/push_to_hub.py" \
            --checkpoint_path "$LATEST_CKPT" \
            --repo_id "$HF_REPO_ID" \
            --token "$HF_TOKEN" \
            --commit_message "SDPO training: ${EXPERIMENT_NAME} - $(basename $LATEST_CKPT)"
        
        echo "✅ Upload completed!"
    else
        echo "❌ No checkpoint found to upload at ${CKPT_DIR}/${EXPERIMENT_NAME}/"
    fi
fi

echo ""
echo "=============================================="
echo "All done!"
echo "=============================================="

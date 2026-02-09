#!/bin/bash
# =============================================================================
# Run Original SDPO Locally: Data Processing + Training
# =============================================================================
# This script runs the original SDPO code locally (no SLURM)
# All operations happen in the original SDPO directory
# Training runs in tmux to survive SSH disconnection
# 
# Usage:
#   ./run_original_sdpo.sh              # Full run: data processing + training
#   ./run_original_sdpo.sh --skip-data  # Skip data processing
#   ./run_original_sdpo.sh --dry-run    # Print commands only
#   ./run_original_sdpo.sh --no-tmux    # Don't use tmux (run directly)
#
# Prerequisites:
#   - Original SDPO code at ~/sdpo-origin (or set SDPO_ORIGIN_DIR)
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION (exactly from original run_sdpo.sh)
# =============================================================================

SDPO_ORIGIN_DIR="${SDPO_ORIGIN_DIR:-$HOME/sdpo-origin}"
TMUX_SESSION_NAME="sdpo-training"

# Training config - exactly from experiments/rich_feedback/run_sdpo.sh
CONFIG_NAME="sdpo"
DATA_PATH="datasets/lcb_v6"
MODEL_PATH="Qwen/Qwen3-8B"
TRAIN_BATCH_SIZE=32
ROLLOUT_BATCH_SIZE=8
LR="1e-6"
ALPHA="1.0"
PPO_MINI_BATCH_SIZE=1
DISTILLATION_TOPK=20
TEACHER_UPDATE_RATE="0.01"
LR_WARMUP_STEPS=0
VAL_ROLLOUT_N=4
DONT_REPROMPT_ON_SELF_SUCCESS="True"
ROLLOUT_IS="token"
WANDB_GROUP_NAME="SDPO-rich-feedback"

# Flags
SKIP_DATA_PROCESSING=false
DRY_RUN=false
USE_TMUX=true
IN_TMUX_WORKER=false

# =============================================================================
# PARSE ARGUMENTS
# =============================================================================

while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-data)
            SKIP_DATA_PROCESSING=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --no-tmux)
            USE_TMUX=false
            shift
            ;;
        --tmux-worker)
            # Internal flag: we're running inside tmux
            IN_TMUX_WORKER=true
            shift
            ;;
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-data    Skip data processing (use existing data)"
            echo "  --dry-run      Print commands without executing"
            echo "  --no-tmux      Don't use tmux (run directly)"
            echo "  --help         Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  SDPO_ORIGIN_DIR    Original SDPO directory (default: ~/sdpo-origin)"
            echo "  WANDB_API_KEY      WandB API key"
            echo ""
            echo "tmux commands:"
            echo "  tmux attach -t $TMUX_SESSION_NAME    # Reconnect to training"
            echo "  Ctrl+B then D                        # Detach from tmux"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

run_cmd() {
    if [ "$DRY_RUN" = "true" ]; then
        echo "[DRY-RUN] $@"
    else
        log "Running: $@"
        "$@"
    fi
}

# =============================================================================
# SETUP
# =============================================================================

log "=========================================="
log "Running Original SDPO (Local)"
log "=========================================="
log "SDPO_ORIGIN_DIR: $SDPO_ORIGIN_DIR"
log ""
log "Config (from run_sdpo.sh):"
log "  MODEL_PATH: $MODEL_PATH"
log "  TRAIN_BATCH_SIZE: $TRAIN_BATCH_SIZE"
log "  ROLLOUT_BATCH_SIZE: $ROLLOUT_BATCH_SIZE"
log "  LR: $LR"
log "  PPO_MINI_BATCH_SIZE: $PPO_MINI_BATCH_SIZE"
log "  ALPHA: $ALPHA"
log "=========================================="

# Check original SDPO directory exists
if [ ! -d "$SDPO_ORIGIN_DIR" ]; then
    echo "ERROR: SDPO_ORIGIN_DIR not found: $SDPO_ORIGIN_DIR"
    echo "Set SDPO_ORIGIN_DIR environment variable to your original SDPO path"
    exit 1
fi

# Change to original SDPO directory
cd "$SDPO_ORIGIN_DIR"
log "Changed to directory: $(pwd)"

# Setup environment variables
export PYTHONPATH="$SDPO_ORIGIN_DIR:$PYTHONPATH"
export USER="${USER:-root}"
export WANDB_ENTITY="${WANDB_ENTITY:-}"  # Override to use personal account
export VLLM_USE_V1=1
export PYTHONBUFFERED=1
unset VLLM_ATTENTION_BACKEND

log "Environment configured"

# =============================================================================
# FIX ORIGINAL VERL_TRAINING.SH (comment out WANDB_ENTITY)
# =============================================================================

VERL_TRAINING_SH="$SDPO_ORIGIN_DIR/training/verl_training.sh"
if grep -q '^export WANDB_ENTITY=' "$VERL_TRAINING_SH" 2>/dev/null; then
    log "Fixing WANDB_ENTITY in verl_training.sh..."
    sed -i 's/^export WANDB_ENTITY=/#export WANDB_ENTITY=/' "$VERL_TRAINING_SH"
fi

# =============================================================================
# DEPENDENCIES & LOGIN
# =============================================================================

log "Installing dependencies..."
pip install -q word2number latex2sympy2 "math-verify[antlr4_9_3]==0.8.0" 2>/dev/null || true
pip install -q -e "$SDPO_ORIGIN_DIR" 2>/dev/null || true
pip install -q --upgrade wandb 2>/dev/null || true

# Check WandB login
WANDB_USER=$(wandb whoami 2>/dev/null | head -1 || echo "")
if [ -z "$WANDB_USER" ] || echo "$WANDB_USER" | grep -q "not logged in"; then
    if [ -z "$WANDB_API_KEY" ]; then
        echo ""
        echo "=========================================="
        echo "WandB login required!"
        echo "=========================================="
        echo "Run: wandb login"
        echo "Or set: export WANDB_API_KEY='your_key'"
        echo "Get key from: https://wandb.ai/authorize"
        echo "=========================================="
        wandb login
    fi
else
    log "WandB logged in as: $WANDB_USER"
fi

# Create symlink for path compatibility
if [ ! -L "/users/$USER/SDPO" ] && [ ! -d "/users/$USER/SDPO" ]; then
    log "Creating symlink for path compatibility..."
    mkdir -p "/users/$USER" 2>/dev/null || sudo mkdir -p "/users/$USER" 2>/dev/null || true
    ln -sf "$SDPO_ORIGIN_DIR" "/users/$USER/SDPO" 2>/dev/null || sudo ln -sf "$SDPO_ORIGIN_DIR" "/users/$USER/SDPO" 2>/dev/null || true
fi

# =============================================================================
# STEP 1: DATA PROCESSING
# =============================================================================

if [ "$SKIP_DATA_PROCESSING" = "false" ]; then
    log "=========================================="
    log "STEP 1: Data Processing"
    log "=========================================="
    
    mkdir -p datasets/lcb_v6
    
    log "Step 1.1: Downloading LiveCodeBench v6..."
    run_cmd python data/load_dataset.py \
        --dataset_name livecodebench/code_generation_lite-v6 \
        --output_path datasets/lcb_v6.json
    
    log "Step 1.2: Splitting tests..."
    run_cmd python data/split_tests.py \
        --json_path datasets/lcb_v6.json \
        --output_dir datasets/lcb_v6
    
    log "Step 1.3: Preprocessing to parquet..."
    run_cmd python data/preprocess.py \
        --data_source datasets/lcb_v6
    
    log "Data processing completed!"
else
    log "Skipping data processing (--skip-data)"
    if [ ! -f "datasets/lcb_v6/train.parquet" ]; then
        echo "ERROR: Data not found. Run without --skip-data first."
        exit 1
    fi
fi

# =============================================================================
# STEP 2: TRAINING
# =============================================================================

log "=========================================="
log "STEP 2: Training"
log "=========================================="

MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="SDPO-original-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-alpha${ALPHA}-${MODEL_NAME}"

export EXPERIMENT="$EXP_NAME"
export TASK="$DATA_PATH"

log "Experiment: $EXP_NAME"

# Build training command
TRAIN_CMD="bash training/verl_training.sh '$EXP_NAME' '$CONFIG_NAME' '$DATA_PATH' \
    'data.train_batch_size=$TRAIN_BATCH_SIZE' \
    'trainer.group_name=$WANDB_GROUP_NAME' \
    'actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE' \
    'actor_rollout_ref.model.path=$MODEL_PATH' \
    'actor_rollout_ref.actor.optim.lr=$LR' \
    'actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE' \
    'actor_rollout_ref.actor.self_distillation.distillation_topk=$DISTILLATION_TOPK' \
    'algorithm.rollout_correction.rollout_is=$ROLLOUT_IS' \
    'actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=$DONT_REPROMPT_ON_SELF_SUCCESS' \
    'actor_rollout_ref.actor.self_distillation.alpha=$ALPHA' \
    'actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE' \
    'actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS' \
    'actor_rollout_ref.rollout.val_kwargs.n=$VAL_ROLLOUT_N'"

# Check if we should use tmux
if [ "$USE_TMUX" = "true" ] && [ "$IN_TMUX_WORKER" = "false" ] && [ -z "$TMUX" ]; then
    # Not in tmux, start a new session
    log "Starting training in tmux session: $TMUX_SESSION_NAME"
    log ""
    log "=========================================="
    log "Training will run in background!"
    log "=========================================="
    log "To view training:  tmux attach -t $TMUX_SESSION_NAME"
    log "To detach:         Ctrl+B then D"
    log "To kill:           tmux kill-session -t $TMUX_SESSION_NAME"
    log "=========================================="
    
    # Kill existing session if any
    tmux kill-session -t "$TMUX_SESSION_NAME" 2>/dev/null || true
    
    # Create new tmux session and run training
    tmux new-session -d -s "$TMUX_SESSION_NAME" "cd '$SDPO_ORIGIN_DIR' && \
        export PYTHONPATH='$SDPO_ORIGIN_DIR:\$PYTHONPATH' && \
        export USER='$USER' && \
        export WANDB_ENTITY='' && \
        export VLLM_USE_V1=1 && \
        export EXPERIMENT='$EXP_NAME' && \
        export TASK='$DATA_PATH' && \
        $TRAIN_CMD; \
        echo ''; echo 'Training finished! Press Enter to close.'; read"
    
    log "Training started in tmux. Use 'tmux attach -t $TMUX_SESSION_NAME' to view."
else
    # Already in tmux or --no-tmux flag
    log "Starting training directly..."
    if [ "$DRY_RUN" = "true" ]; then
        echo "[DRY-RUN] $TRAIN_CMD"
    else
        eval "$TRAIN_CMD"
    fi
    
    log "=========================================="
    log "Training completed!"
    log "=========================================="
fi

#!/bin/bash
# =============================================================================
# Run Original SDPO Locally: Data Processing + Training
# =============================================================================
# This script runs the original SDPO code locally (no SLURM)
# All operations happen in the original SDPO directory
# 
# Usage:
#   ./run_original_sdpo.sh              # Full run: data processing + training
#   ./run_original_sdpo.sh --skip-data  # Skip data processing
#   ./run_original_sdpo.sh --dry-run    # Print commands only
#
# Prerequisites:
#   - Run setup_env.sh first to configure environment
#   - Original SDPO code at ~/SDPO (or set SDPO_ORIGIN_DIR)
# =============================================================================

set -e

# =============================================================================
# CONFIGURATION (exactly from original run_sdpo.sh)
# =============================================================================

SDPO_ORIGIN_DIR="${SDPO_ORIGIN_DIR:-$HOME/sdpo-origin}"

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
        --help)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --skip-data    Skip data processing (use existing data)"
            echo "  --dry-run      Print commands without executing"
            echo "  --help         Show this help message"
            echo ""
            echo "Environment variables:"
            echo "  SDPO_ORIGIN_DIR    Original SDPO directory (default: ~/SDPO)"
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

# Setup Python path (as per original README)
export PYTHONPATH="$SDPO_ORIGIN_DIR:$PYTHONPATH"
log "PYTHONPATH set"

# =============================================================================
# STEP 1: DATA PROCESSING (following original README.md)
# =============================================================================

if [ "$SKIP_DATA_PROCESSING" = "false" ]; then
    log "=========================================="
    log "STEP 1: Data Processing (from README.md)"
    log "=========================================="
    
    # Create datasets directory
    mkdir -p datasets
    mkdir -p datasets/lcb_v6
    
    # 1.1 Download LiveCodeBench v6 (from README.md)
    log "Step 1.1: python data/load_dataset.py ..."
    run_cmd python data/load_dataset.py \
        --dataset_name livecodebench/code_generation_lite-v6 \
        --output_path datasets/lcb_v6.json
    
    # 1.2 Split tests (from README.md)
    log "Step 1.2: python data/split_tests.py ..."
    run_cmd python data/split_tests.py \
        --json_path datasets/lcb_v6.json \
        --output_dir datasets/lcb_v6
    
    # 1.3 Preprocess to parquet (from README.md)
    log "Step 1.3: python data/preprocess.py ..."
    run_cmd python data/preprocess.py \
        --data_source datasets/lcb_v6
    
    log "Data processing completed!"
    log "Data location: $SDPO_ORIGIN_DIR/datasets/lcb_v6/"
else
    log "Skipping data processing (--skip-data flag)"
    
    # Check if data exists
    if [ ! -f "datasets/lcb_v6/train.parquet" ]; then
        echo "ERROR: Training data not found at $SDPO_ORIGIN_DIR/datasets/lcb_v6/train.parquet"
        echo "Run without --skip-data to generate data first"
        exit 1
    fi
    log "Using existing data at: $SDPO_ORIGIN_DIR/datasets/lcb_v6/"
fi

# =============================================================================
# STEP 2: TRAINING (local execution using verl_training.sh)
# =============================================================================

log "=========================================="
log "STEP 2: Training (Local)"
log "=========================================="

# Environment setup (from original verl_training.sh)
unset VLLM_ATTENTION_BACKEND
export VLLM_USE_V1=1
export PYTHONBUFFERED=1
export USER="${USER:-root}"  # Fix for RunPod where USER may not be set
ulimit -c 0

# Experiment name (matching original naming convention)
MODEL_NAME=$(echo "$MODEL_PATH" | tr '/' '-')
EXP_NAME="SDPO-local-train${TRAIN_BATCH_SIZE}-rollout${ROLLOUT_BATCH_SIZE}-lr${LR}-alpha${ALPHA}-${MODEL_NAME}"

export EXPERIMENT="$EXP_NAME"
export TASK="$DATA_PATH"

log "Experiment: $EXP_NAME"
log "Task: $TASK"

# Run training using verl_training.sh (local, no sbatch)
# This is exactly what run_sdpo.sh does, but without sbatch
log "Starting training..."
run_cmd bash training/verl_training.sh "$EXP_NAME" "$CONFIG_NAME" "$DATA_PATH" \
    "data.train_batch_size=$TRAIN_BATCH_SIZE" \
    "trainer.group_name=$WANDB_GROUP_NAME" \
    "actor_rollout_ref.rollout.n=$ROLLOUT_BATCH_SIZE" \
    "actor_rollout_ref.model.path=$MODEL_PATH" \
    "actor_rollout_ref.actor.optim.lr=$LR" \
    "actor_rollout_ref.actor.ppo_mini_batch_size=$PPO_MINI_BATCH_SIZE" \
    "actor_rollout_ref.actor.self_distillation.distillation_topk=$DISTILLATION_TOPK" \
    "algorithm.rollout_correction.rollout_is=$ROLLOUT_IS" \
    "actor_rollout_ref.actor.self_distillation.dont_reprompt_on_self_success=$DONT_REPROMPT_ON_SELF_SUCCESS" \
    "actor_rollout_ref.actor.self_distillation.alpha=$ALPHA" \
    "actor_rollout_ref.actor.self_distillation.teacher_update_rate=$TEACHER_UPDATE_RATE" \
    "actor_rollout_ref.actor.optim.lr_warmup_steps=$LR_WARMUP_STEPS" \
    "actor_rollout_ref.rollout.val_kwargs.n=$VAL_ROLLOUT_N"

log "=========================================="
log "Training completed!"
log "=========================================="

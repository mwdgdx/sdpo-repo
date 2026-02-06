#!/bin/bash
# SDPO Training Script for RunPod
# Usage: ./scripts/run_sdpo.sh [OPTIONS]
#
# Presets (or run interactively to select with arrow keys):
#   PRESET=8xa6000  - 8x A6000 48GB, LoRA fine-tuning (Recommended)
#   PRESET=8xa100   - 8x A100 80GB, full fine-tuning (Best quality)
#   PRESET=4xa6000  - 4x A6000 48GB, LoRA fine-tuning (Budget)
#
# To skip interactive menu: INTERACTIVE=false ./run_sdpo.sh
#
# Environment Variables:
#   MODEL_PATH          - Hugging Face model path (default: Qwen/Qwen3-8B)
#   TASK                - Task name, corresponds to datasets/<TASK>/ directory (default: lcb_v6)
#   N_GPUS              - Number of GPUs (default: 4)
#   TOTAL_EPOCHS        - Total training epochs (default: 30)
#   EXPERIMENT_NAME     - Experiment name for logging (default: sdpo_$TASK)
#   SAVE_FREQ           - Checkpoint save frequency in iterations (default: 10)
#   TEST_FREQ           - Validation frequency in iterations (default: 5)
#
#   Model/Training:
#   LORA_RANK           - LoRA rank, 0 for full fine-tuning (default: 32)
#   LORA_ALPHA          - LoRA alpha scaling factor (default: 32)
#   LEARNING_RATE       - Learning rate (default: 1e-5)
#   TRAIN_BATCH_SIZE    - Training batch size (default: 16)
#   MICRO_BATCH_SIZE    - Micro batch size per GPU (default: 1)
#   GPU_MEM_UTIL        - vLLM GPU memory utilization (default: 0.55)
#
#   WandB (Logging):
#   WANDB_API_KEY       - WandB API key for metrics tracking
#   WANDB_PROJECT       - WandB project name (default: SDPO)
#
#   Hugging Face (Upload):
#   HF_TOKEN            - Hugging Face token for uploading checkpoints (REQUIRED for auto-upload)
#   HF_REPO_ID          - Hugging Face repo ID for uploading (REQUIRED for auto-upload)
#
#   Post-training:
#   AUTO_SHUTDOWN       - Set to "true" to shutdown after training (default: true)
#   UPLOAD_CHECKPOINT   - Set to "true" to upload final checkpoint (default: true if HF_TOKEN set)

set -e

# ============================================================================
# Hardware selection BEFORE tmux (so user can interact)
# ============================================================================
if [ -z "$PRESET" ] && [ "$INTERACTIVE" != "false" ]; then
    echo ""
    echo "=============================================="
    echo "Select Hardware Configuration"
    echo "=============================================="
    echo ""
    echo "  [0] 8x A6000 48GB  │ LoRA │ batch=32 │ (Recommended)"
    echo "  [1] 8x A100 80GB   │ Full │ batch=32 │ (Best quality)"
    echo "  [2] 4x A6000 48GB  │ LoRA │ batch=16 │ (Budget)"
    echo ""
    read -p "Enter choice [0-2] (default: 0): " hw_choice
    hw_choice="${hw_choice:-0}"
    export PRESET="$hw_choice"
    echo ""
fi

# ============================================================================
# Auto-start in tmux session (so training survives SSH disconnect)
# ============================================================================
# Skip tmux if: already in tmux, NO_TMUX=true, or tmux not installed
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
    
    # Start tmux session with PRESET already set
    exec tmux new-session -s "$SESSION_NAME" "PRESET=$PRESET NO_TMUX=true bash $0 $@; echo ''; echo 'Training finished. Press Enter to close.'; read"
fi

# ============================================================================
# Get Workspace Root (auto-detect)
# ============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script is in sdpo/scripts/, so parent is sdpo/
SDPO_ROOT="$(dirname "$SCRIPT_DIR")"
VERL_DIR="${SDPO_ROOT}/verl"

# ============================================================================
# Load .env file if exists (created by setup_runpod.sh)
# ============================================================================
ENV_FILE="${SDPO_ROOT}/.env"
if [ -f "$ENV_FILE" ]; then
    echo "Loading configuration from ${ENV_FILE}..."
    source "$ENV_FILE"
fi

# ============================================================================
# Interactive Menu Function (arrow key selection)
# ============================================================================
select_option() {
    local options=("$@")
    local num_options=${#options[@]}
    local selected=0
    local key
    
    # Hide cursor
    tput civis 2>/dev/null || true
    
    # Function to print menu
    print_menu() {
        # Move cursor up to redraw
        if [ "$1" = "redraw" ]; then
            for ((i=0; i<num_options; i++)); do
                tput cuu1 2>/dev/null || printf "\033[1A"
            done
        fi
        
        for ((i=0; i<num_options; i++)); do
            if [ $i -eq $selected ]; then
                printf "  \033[1;32m❯ %s\033[0m\n" "${options[$i]}"
            else
                printf "    %s\n" "${options[$i]}"
            fi
        done
    }
    
    print_menu
    
    while true; do
        # Read single keypress
        IFS= read -rsn1 key
        
        case "$key" in
            $'\x1b')  # Escape sequence (arrow keys)
                read -rsn2 key
                case "$key" in
                    '[A')  # Up arrow
                        ((selected--))
                        [ $selected -lt 0 ] && selected=$((num_options-1))
                        print_menu redraw
                        ;;
                    '[B')  # Down arrow
                        ((selected++))
                        [ $selected -ge $num_options ] && selected=0
                        print_menu redraw
                        ;;
                esac
                ;;
            '')  # Enter key
                break
                ;;
        esac
    done
    
    # Show cursor
    tput cnorm 2>/dev/null || true
    
    return $selected
}

# ============================================================================
# Hardware Presets
# ============================================================================
PRESET="${PRESET:-}"
INTERACTIVE="${INTERACTIVE:-true}"

# Apply preset from environment or interactive selection
apply_preset() {
    case "$1" in
        0|"8xa6000"|"8xA6000")
            echo "✓ Selected: 8x A6000 48GB (LoRA Fine-tuning)"
            export N_GPUS="${N_GPUS:-8}"
            export LORA_RANK="${LORA_RANK:-32}"
            export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
            export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
            export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.50}"
            ;;
        1|"8xa100"|"8xA100")
            echo "✓ Selected: 8x A100 80GB (Full Fine-tuning)"
            export N_GPUS="${N_GPUS:-8}"
            export LORA_RANK="${LORA_RANK:-0}"
            export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-32}"
            export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
            export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.55}"
            ;;
        2|"4xa6000"|"4xA6000")
            echo "✓ Selected: 4x A6000 48GB (LoRA Fine-tuning)"
            export N_GPUS="${N_GPUS:-4}"
            export LORA_RANK="${LORA_RANK:-32}"
            export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
            export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
            export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.50}"
            ;;
        *)
            # Default: 4x A6000
            export N_GPUS="${N_GPUS:-4}"
            export LORA_RANK="${LORA_RANK:-32}"
            export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
            export MICRO_BATCH_SIZE="${MICRO_BATCH_SIZE:-1}"
            export GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.50}"
            ;;
    esac
}

# Apply preset (should already be set from menu above or environment)
if [ -n "$PRESET" ]; then
    apply_preset "$PRESET"
else
    # Fallback default
    apply_preset "8xa6000"
fi

# ============================================================================
# Default Configuration (applied after preset selection)
# ============================================================================

# Set HuggingFace cache to workspace to avoid filling root partition
export HF_HOME="${HF_HOME:-${SDPO_ROOT}/.cache/huggingface}"
export TRANSFORMERS_CACHE="${HF_HOME}"
export HF_DATASETS_CACHE="${HF_HOME}/datasets"
mkdir -p "$HF_HOME"

export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export TASK="${TASK:-lcb_v6}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-30}"
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-sdpo_${TASK}}"
export PROJECT_NAME="${PROJECT_NAME:-${WANDB_PROJECT:-SDPO}}"
# Checkpoint frequency: save every N iterations (higher = less disk usage)
# With 30 epochs, save_freq=10 means ~3 checkpoints per epoch
export SAVE_FREQ="${SAVE_FREQ:-10}"
# Validation frequency
export TEST_FREQ="${TEST_FREQ:-5}"
# Max checkpoints to keep (older ones auto-deleted)
export MAX_CKPT_TO_KEEP="${MAX_CKPT_TO_KEEP:-2}"

# These are set by preset selection above, but can be overridden
export LORA_ALPHA="${LORA_ALPHA:-32}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"

# WandB configuration
export WANDB_PROJECT="${WANDB_PROJECT:-$PROJECT_NAME}"
# WANDB_API_KEY should be set in .env or environment

# Auto-upload and shutdown settings
AUTO_SHUTDOWN="${AUTO_SHUTDOWN:-true}"
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
echo "=============================================="
echo "SDPO Training Configuration"
echo "=============================================="
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
if [ "$LORA_RANK" -gt 0 ] 2>/dev/null; then
    echo "  LORA_RANK:        $LORA_RANK (LoRA enabled)"
    echo "  LORA_ALPHA:       $LORA_ALPHA"
else
    echo "  LORA_RANK:        $LORA_RANK (Full fine-tuning)"
fi
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
echo "  AUTO_SHUTDOWN:     $AUTO_SHUTDOWN"
echo "=============================================="

# Warn if no WandB credentials
if [ -z "$WANDB_API_KEY" ]; then
    echo ""
    echo "⚠️  WARNING: WANDB_API_KEY not set!"
    echo "   Training will use console logging only."
    echo "   To enable WandB tracking:"
    echo "     export WANDB_API_KEY=your_wandb_key"
    echo ""
    # Set logger to console only
    export USE_WANDB="false"
else
    export USE_WANDB="true"
fi

# Warn if no HF credentials
if [ "$UPLOAD_CHECKPOINT" = "false" ]; then
    echo ""
    echo "⚠️  WARNING: HF_TOKEN or HF_REPO_ID not set!"
    echo "   Checkpoints will NOT be uploaded to Hugging Face."
    echo "   Set these environment variables to enable auto-upload:"
    echo "     export HF_TOKEN=hf_xxxxxxx"
    echo "     export HF_REPO_ID=your-username/model-name"
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

# Data paths are read from environment variables in user_runpod.yaml
# No need to override here - TRAIN_DATA_PATH and VAL_DATA_PATH are already exported

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

# ============================================================================
# Auto Shutdown
# ============================================================================
if [ "$AUTO_SHUTDOWN" = "true" ]; then
    echo ""
    echo "=============================================="
    echo "Auto-shutdown enabled. Shutting down in 30 seconds..."
    echo "Press Ctrl+C to cancel."
    echo "=============================================="
    sleep 30
    
    # Try different shutdown methods
    if command -v runpodctl &> /dev/null; then
        echo "Using runpodctl to stop pod..."
        runpodctl stop pod
    elif [ -f /etc/rp_pod_id ]; then
        echo "Stopping RunPod instance..."
        # RunPod specific shutdown
        curl -s -X POST "https://api.runpod.io/v2/${RUNPOD_POD_ID}/stop" \
            -H "Authorization: Bearer ${RUNPOD_API_KEY}" 2>/dev/null || true
    fi
    
    # Fallback: system shutdown (requires root)
    echo "Attempting system shutdown..."
    sudo shutdown -h now 2>/dev/null || poweroff 2>/dev/null || echo "Could not shutdown automatically. Please shutdown manually."
fi

echo ""
echo "=============================================="
echo "All done!"
echo "=============================================="

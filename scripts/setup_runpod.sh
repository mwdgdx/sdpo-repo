#!/bin/bash
# Setup script for RunPod
# This script installs dependencies, configures credentials, and prepares the environment
#
# ============================================================================
# RunPod Setup Instructions (IMPORTANT!)
# ============================================================================
# 1. Create Pod with:
#    - Image: verlai/verl:vllm012.latest
#    - Command/Entrypoint: sleep infinity  <-- CRITICAL! Prevents vLLM auto-start
#    - GPU: 4x or 8x A6000/A100
#
# 2. SSH into the Pod and run:
#    cd /workspace
#    git clone <your-repo> sdpo
#    cd sdpo
#    ./scripts/setup_runpod.sh
#    ./scripts/run_sdpo.sh
#
# ============================================================================
# Expected directory structure:
#   /workspace/
#   └── sdpo/                 # git repo cloned here
#       ├── scripts/          # This setup script and run script
#       ├── verl/             # verl with SDPO modifications
#       ├── datasets/         # Pre-prepared datasets
#       │   └── lcb_v6/
#       │       ├── train.parquet
#       │       └── test.parquet
#       └── checkpoints/      # Will be created for saving checkpoints

set -e

echo "=============================================="
echo "SDPO RunPod Setup"
echo "=============================================="

# Get script directory and workspace root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# Script is in sdpo/scripts/, so parent is sdpo/
SDPO_ROOT="$(dirname "$SCRIPT_DIR")"
VERL_DIR="${SDPO_ROOT}/verl"

echo "VERL_DIR:  $VERL_DIR"
echo "SDPO_ROOT: $SDPO_ROOT"

# ============================================================================
# WandB Configuration (Interactive)
# ============================================================================
echo ""
echo "=============================================="
echo "Weights & Biases (WandB) Configuration"
echo "=============================================="
echo ""
echo "WandB tracks training metrics and visualizations."
echo "Get your API key from: https://wandb.ai/authorize"
echo ""

if [ -n "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY is already set (from environment)"
else
    echo "Enter your WandB API key (or press Enter to skip):"
    read -r -s WANDB_API_KEY_INPUT
    echo ""
    if [ -n "$WANDB_API_KEY_INPUT" ]; then
        export WANDB_API_KEY="$WANDB_API_KEY_INPUT"
        echo "✓ WANDB_API_KEY set"
    else
        echo "⚠ WANDB_API_KEY not set - will use console logging only"
    fi
fi

if [ -n "$WANDB_PROJECT" ]; then
    echo "WANDB_PROJECT is already set: $WANDB_PROJECT"
else
    echo "Enter WandB project name (default: SDPO):"
    read -r WANDB_PROJECT_INPUT
    if [ -n "$WANDB_PROJECT_INPUT" ]; then
        export WANDB_PROJECT="$WANDB_PROJECT_INPUT"
    else
        export WANDB_PROJECT="SDPO"
    fi
    echo "✓ WANDB_PROJECT: $WANDB_PROJECT"
fi

# ============================================================================
# Hugging Face Configuration (Interactive)
# ============================================================================
echo ""
echo "=============================================="
echo "Hugging Face Configuration"
echo "=============================================="
echo ""
echo "To automatically upload checkpoints after training, you need:"
echo "  1. HF_TOKEN  - Your Hugging Face access token"
echo "  2. HF_REPO_ID - Target repository (e.g., username/model-name)"
echo ""

# Check if already set
if [ -n "$HF_TOKEN" ]; then
    echo "HF_TOKEN is already set (from environment)"
else
    echo "Enter your Hugging Face token (or press Enter to skip):"
    echo "(Get token from: https://huggingface.co/settings/tokens)"
    read -r -s HF_TOKEN_INPUT
    echo ""
    if [ -n "$HF_TOKEN_INPUT" ]; then
        export HF_TOKEN="$HF_TOKEN_INPUT"
        echo "✓ HF_TOKEN set"
    else
        echo "⚠ HF_TOKEN not set - checkpoints won't be uploaded automatically"
    fi
fi

if [ -n "$HF_REPO_ID" ]; then
    echo "HF_REPO_ID is already set: $HF_REPO_ID"
else
    echo ""
    echo "Enter your Hugging Face repo ID (e.g., your-username/sdpo-qwen3-8b):"
    read -r HF_REPO_ID_INPUT
    if [ -n "$HF_REPO_ID_INPUT" ]; then
        export HF_REPO_ID="$HF_REPO_ID_INPUT"
        echo "✓ HF_REPO_ID set: $HF_REPO_ID"
    else
        echo "⚠ HF_REPO_ID not set - checkpoints won't be uploaded automatically"
    fi
fi

# ============================================================================
# Hardware Preset Selection
# ============================================================================
echo ""
echo "=============================================="
echo "Hardware Configuration"
echo "=============================================="
echo ""
echo "Select your hardware preset:"
echo "  1) 4x A6000 48GB - LoRA fine-tuning (DEFAULT)"
echo "  2) 8x A100 80GB - Full fine-tuning"
echo "  3) 4x A100 80GB - LoRA fine-tuning"
echo "  4) 4x 24GB GPU (A10/4090) - LoRA + smaller batch"
echo "  5) Custom - I'll set my own parameters"
echo ""
read -r -p "Enter choice [1-5, default=1]: " PRESET_CHOICE

case "$PRESET_CHOICE" in
    2)
        PRESET="8xa100"
        N_GPUS=8
        LORA_RANK=0
        TRAIN_BATCH_SIZE=32
        ;;
    3)
        PRESET="4xa100"
        N_GPUS=4
        LORA_RANK=32
        TRAIN_BATCH_SIZE=16
        ;;
    4)
        PRESET="4x24g"
        N_GPUS=4
        LORA_RANK=32
        TRAIN_BATCH_SIZE=8
        ;;
    5)
        PRESET=""
        echo ""
        read -r -p "Number of GPUs [default=4]: " N_GPUS
        N_GPUS="${N_GPUS:-4}"
        read -r -p "LoRA rank (0=full fine-tuning) [default=32]: " LORA_RANK
        LORA_RANK="${LORA_RANK:-32}"
        read -r -p "Training batch size [default=16]: " TRAIN_BATCH_SIZE
        TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-16}"
        ;;
    *)
        # Default: 4x A6000 48GB
        PRESET="4xa6000"
        N_GPUS=4
        LORA_RANK=32
        TRAIN_BATCH_SIZE=16
        ;;
esac

echo ""
echo "✓ Hardware config: N_GPUS=$N_GPUS, LORA_RANK=$LORA_RANK, BATCH_SIZE=$TRAIN_BATCH_SIZE"

# Save to .env file for persistence
ENV_FILE="${SDPO_ROOT}/.env"
echo ""
echo "Saving configuration to ${ENV_FILE}..."
cat > "$ENV_FILE" << EOF
# SDPO Environment Configuration
# Generated by setup_runpod.sh on $(date)

# Hardware Preset (8xa100, 4xa100, 4x24g, or empty for custom)
export PRESET="${PRESET}"

# WandB credentials
export WANDB_API_KEY="${WANDB_API_KEY}"
export WANDB_PROJECT="${WANDB_PROJECT:-SDPO}"

# Hugging Face credentials
export HF_TOKEN="${HF_TOKEN}"
export HF_REPO_ID="${HF_REPO_ID}"

# Training configuration
export MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
export TASK="${TASK:-lcb_v6}"
export N_GPUS="${N_GPUS}"
export TOTAL_EPOCHS="${TOTAL_EPOCHS:-30}"
export LORA_RANK="${LORA_RANK}"
export TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE}"
export LEARNING_RATE="${LEARNING_RATE:-1e-5}"
EOF

echo "✓ Configuration saved to ${ENV_FILE}"
echo ""
echo "To load these settings in a new terminal, run:"
echo "  source ${ENV_FILE}"

# ============================================================================
# Install Dependencies
# ============================================================================
# NOTE: If using verlai/verl:vllm012.latest Docker image:
#   - vllm, ray, flash_attn, transformers are ALREADY installed
#   - verl is UNINSTALLED (need to reinstall with our SDPO modifications)
#   - IMPORTANT: Set Pod Command to "sleep infinity" to prevent vLLM server auto-start!
echo ""
echo "=============================================="
echo "Installing dependencies..."
echo "=============================================="

# Check if running in verl Docker image
if python -c "import vllm" 2>/dev/null; then
    echo "✓ vllm already installed (verl Docker image detected)"
    echo ""
    echo "⚠️  REMINDER: Make sure you set Pod Command to 'sleep infinity'"
    echo "   Otherwise vLLM server will auto-start and eat all GPU memory!"
    echo ""
else
    echo "vllm not found, installing heavy dependencies..."
    pip install vllm ray[default] --quiet
    pip install flash-attn --no-build-isolation --quiet || echo "flash-attn installation failed"
fi

# Install verl with local SDPO modifications (ALWAYS needed!)
echo "Installing verl with SDPO modifications..."
cd "$VERL_DIR"
pip install --no-deps -e . --quiet

# Install dependencies that may be missing
echo "Installing additional dependencies..."
pip install --quiet \
    hydra-core \
    omegaconf \
    peft \
    datasets \
    wandb \
    huggingface_hub \
    accelerate

echo ""
echo "✓ Dependencies ready!"

# Verify critical imports
echo ""
echo "Verifying installations..."
python -c "import verl; print('✓ verl')" || echo "❌ verl import failed"
python -c "import vllm; print('✓ vllm')" || echo "❌ vllm import failed"
python -c "import ray; print('✓ ray')" || echo "❌ ray import failed"
python -c "import hydra; print('✓ hydra')" || echo "❌ hydra import failed"
python -c "import peft; print('✓ peft')" || echo "❌ peft import failed"
python -c "import wandb; print('✓ wandb')" || echo "❌ wandb import failed"

# ============================================================================
# Create checkpoint directory
# ============================================================================
echo ""
echo "Creating checkpoint directory..."
mkdir -p "${SDPO_ROOT}/checkpoints"

# ============================================================================
# Verify Data
# ============================================================================
echo ""
echo "=============================================="
echo "Checking datasets..."
echo "=============================================="
if [ -d "${SDPO_ROOT}/datasets/lcb_v6" ]; then
    echo "✓ Found lcb_v6 dataset:"
    ls -lh "${SDPO_ROOT}/datasets/lcb_v6/"
else
    echo "⚠ WARNING: lcb_v6 dataset not found at ${SDPO_ROOT}/datasets/lcb_v6/"
    echo "Available directories:"
    ls -la "${SDPO_ROOT}/" 2>/dev/null || echo "  (sdpo root not found)"
fi

# ============================================================================
# Summary
# ============================================================================
echo ""
echo "=============================================="
echo "✅ Setup Complete!"
echo "=============================================="
echo ""
echo "Hardware Configuration:"
echo "  PRESET:         ${PRESET:-custom}"
echo "  N_GPUS:         $N_GPUS"
echo "  LORA_RANK:      $LORA_RANK $([ "$LORA_RANK" -gt 0 ] 2>/dev/null && echo '(LoRA)' || echo '(Full FT)')"
echo "  BATCH_SIZE:     $TRAIN_BATCH_SIZE"
echo ""
echo "Credentials:"
echo "  WANDB_API_KEY:  ${WANDB_API_KEY:+[SET]}${WANDB_API_KEY:-[NOT SET]}"
echo "  WANDB_PROJECT:  ${WANDB_PROJECT:-SDPO}"
echo "  HF_TOKEN:       ${HF_TOKEN:+[SET]}${HF_TOKEN:-[NOT SET]}"
echo "  HF_REPO_ID:     ${HF_REPO_ID:-[NOT SET]}"
echo ""
echo "Directory structure:"
echo "  ${SDPO_ROOT}/"
echo "  ├── verl/            # Code"
echo "  ├── datasets/lcb_v6/ # Training data"
echo "  ├── checkpoints/     # Checkpoints will be saved here"
echo "  └── .env             # Your configuration"
echo ""
echo "=============================================="
echo "Ready to train! Run:"
echo "=============================================="
echo ""
echo "  source ${ENV_FILE}  # Load your credentials"
echo "  ${SCRIPT_DIR}/run_sdpo.sh"
echo ""
if [ -n "$HF_TOKEN" ] && [ -n "$HF_REPO_ID" ]; then
    echo "After training completes:"
    echo "  - Checkpoint will be uploaded to: https://huggingface.co/${HF_REPO_ID}"
    echo "  - Machine will auto-shutdown to save costs 💰"
else
    echo "⚠ Note: HF credentials not fully configured."
    echo "  Checkpoints will be saved locally but NOT uploaded."
    echo "  To fix, edit ${ENV_FILE} and add your credentials."
fi
echo ""

#!/bin/bash

# Global Constants
TOTAL_STEPS=24000
LOG_EVERY=1000
CKPT_EVERY=2000
SKIP_FIRST_CKPT=1

IMAGE_SIZE=256
MODEL="DiT-XL/2"

BATCH_SIZE=8
BATCH_SIZE=4
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
VAE="ema"
NUM_WORKERS=4
CFG_SCALE=1.0
NUM_SAMPLE_STEPS=50
NSAMPLE=10000

FID_DEVICE="cuda:0"
NPROC_PER_NODE=2
is_port_in_use() {
  local port=$1
  (echo >/dev/tcp/127.0.0.1/$port) &>/dev/null
}

PORT=$(shuf -i 20000-40000 -n 1)  # 🔧 Initialize first
while is_port_in_use $PORT; do
  PORT=$(shuf -i 20000-40000 -n 1)
done
export PORT
echo "Using MASTER_PORT=$PORT"

# Set paths and dataset details
resolve_dataset_config() {
  case "$DATASET" in
    food-101_processed|caltech-101_processed|caltech-101) NUM_CLASSES=101 ;;
    *) echo "Unknown dataset: $DATASET"; exit 1 ;;
  esac
  DATA_DIR_ZIP="$DATASETS_DIR/food-101.zip"
  REAL_DATA_DIR="$DATASETS_DIR/$DATASET"
  # Normal Generated Directory for Ours and MG
  RESULTS_DIR="$RESULTS_PRE_DIR/$DATASET/$EXPERIMENT_NAME/"
  GENERATED_DIR="$RESULTS_DIR/samples"
  # Generated Directory for baselines
  GENERATED_DIR_CG1="$RESULTS_DIR/samples"
  RESULTS_FILE_CG1="$RESULTS_DIR/results_cg1"
  GENERATED_DIR_CG1_5="$RESULTS_DIR/samples"
  RESULTS_FILE_CG1_5="$RESULTS_DIR/results_cg1_5"
  GENERATED_DIR_DoG1_5="$RESULTS_DIR/samples"
  RESULTS_FILE_DoG1_5="$RESULTS_DIR/results_dog1_5"
  # Till here...
  CHECKPOINT_DIR="$RESULTS_DIR/checkpoints"
  LOG_FILE="$RESULTS_DIR/training_log.txt"
  RESULTS_FILE="$RESULTS_DIR/results"
}

# ==== PATHS ====
resolve_server_paths() {
    case "$SERVER" in
        bool|dse)
            SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
            if [ -f "$SCRIPT_ROOT/train_DogFit.py" ]; then
                CODE_PRE_DIR="$SCRIPT_ROOT"
                if [ -d "$SCRIPT_ROOT/../../dl" ]; then
                    ENV_PATH="$(cd "$SCRIPT_ROOT/../.." && pwd)/dl"
                elif [ -d "/home/dse_413/DL_proj/dl" ]; then
                    ENV_PATH="/home/dse_413/DL_proj/dl"
                elif [ -d "$SCRIPT_ROOT/../dlom" ]; then
                    ENV_PATH="$(cd "$SCRIPT_ROOT/.." && pwd)/dlom"
                else
                    ENV_PATH="/DATA/agniva/om/DL_proj/dlom"
                fi
            elif [ -d "/home/dse_413/DL_proj/dl" ]; then
                CODE_PRE_DIR="/home/dse_413/DL_proj/Om/DogFit"
                ENV_PATH="/home/dse_413/DL_proj/dl"
            elif [ -d "/home/dse_413/DL_proj/Om" ]; then
                CODE_PRE_DIR="/home/dse_413/DL_proj/Om/DogFit"
                ENV_PATH="/home/dse_413/DL_proj/Om/dlom"
            elif [ -d "/DATA/agniva/om/DL_proj" ]; then
                CODE_PRE_DIR="/DATA/agniva/om/DL_proj/DogFit"
                ENV_PATH="/DATA/agniva/om/DL_proj/dlom"
            else
                CODE_PRE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
                ENV_PATH="$(dirname "$CODE_PRE_DIR")/dlom"
            fi
            DATASETS_DIR="$CODE_PRE_DIR/datasets" # Can keep this the way it is
            RESULTS_PRE_DIR="$CODE_PRE_DIR/results" # Can keep this the way it is
            if [ -n "$CUDA_DEVICES" ]; then
                NPROC_PER_NODE=$(awk -F',' '{print NF}' <<< "$CUDA_DEVICES")
            fi
            ;;
        *)
            echo "Unknown server: $SERVER" >&2
            exit 1
            ;;
    esac
}

create_environment() {
  echo ">>> Setting up environment..."

  # Set TORCH_HOME to local torch_cache if available
  if [ -d "$CODE_PRE_DIR/../torch_cache" ]; then
    export TORCH_HOME="$(cd "$CODE_PRE_DIR/../torch_cache" && pwd)"
    echo "Using offline TORCH_HOME=$TORCH_HOME"
  elif [ -d "$CODE_PRE_DIR/torch_cache" ]; then
    export TORCH_HOME="$(cd "$CODE_PRE_DIR/torch_cache" && pwd)"
    echo "Using offline TORCH_HOME=$TORCH_HOME"
  fi

  # 1. If an environment is already active, just use it!
  if [ -n "$VIRTUAL_ENV" ]; then
    echo "Using currently active environment: $VIRTUAL_ENV"
  elif [ -f "$ENV_PATH/bin/activate" ]; then
    echo "Activating environment from $ENV_PATH"
    source "$ENV_PATH/bin/activate"
  elif [ -f "$CODE_PRE_DIR/../dlom/bin/activate" ]; then
    echo "Activating environment from $CODE_PRE_DIR/../dlom"
    source "$CODE_PRE_DIR/../dlom/bin/activate"
  elif [ -d "$ENV_PATH" ] && command -v conda &>/dev/null; then
    source "$(conda info --base)/etc/profile.d/conda.sh"
    conda activate "$ENV_PATH"
  else
    echo "Using existing environment in PATH ($(which python))"
  fi

  # Ensure dgm-eval is accessible
  if [ ! -d "dgm-eval" ]; then
    if [ -d "$CODE_PRE_DIR/../dgm-eval" ]; then
      ln -s "$CODE_PRE_DIR/../dgm-eval" dgm-eval
    fi
  fi
}


prepare_dataset() {
  if [ -d "$REAL_DATA_DIR" ] && [ "$(ls -A "$REAL_DATA_DIR" 2>/dev/null)" ]; then
    echo ">>> Dataset found at: $REAL_DATA_DIR"
  else
    echo ">>> [ERROR] Dataset not found at: $REAL_DATA_DIR"
    echo ">>> Please place or symlink the extracted images into: $REAL_DATA_DIR"
    exit 1
  fi
}

# ====================== HELPER ======================
log_and_run() {
    echo ">>> $1"
    shift
    "$@" 2>&1 | tee -a "$LOG_FILE"
    return "${PIPESTATUS[0]}"
}

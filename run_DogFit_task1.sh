#!/bin/bash
set -o pipefail
export OMP_NUM_THREADS=4

# ====================== DEFAULT CONFIGURATION ======================

CUDA_DEVICES="0,1"
CUDA_DEVICES="0"
DATASET="food-101_processed"
SERVER="bool"
EXPERIMENT_PRENAME=""
USE_GUIDANCE_CUTOFF=0
MG_HIGH=1
LATE_START=1000

GUIDANCE_CONTROL=0
W_MAX=1.0
W_MIN=1.0
SAMPLE_GUIDANCE=0

W_TRAIN_DOG=1.5
DROPOUT_RATIO=0 
CONTROL_DISTRIBUTION="uniform"
ZERO_NORM_VARIANCE="011"
SAMPLE_BATCH_SIZE=32

# Task 1: Self-Paced Adaptive Guidance Scheduling Parameters
ADAPTIVE_GUIDANCE=1
ADAPTIVE_GATE_TYPE="sigmoid"
ADAPTIVE_TAU=1.0
ADAPTIVE_TEMP=1.0
ADAPTIVE_BINS=10
ADAPTIVE_DECAY=0.99
ADAPTIVE_GATE_MIN=0.0
ADAPTIVE_GATE_MAX=1.0
ADAPTIVE_OVERRIDE_CUTOFF=1

EVAL_ONLY=0
EVAL_STEP=""

# Load all logic
source scripts/config.sh

# ====================== ARGUMENT PARSING ======================

while [[ "$#" -gt 0 ]]; do
  case $1 in
    --cuda_devices) CUDA_DEVICES="$2"; shift ;;
    --dataset) DATASET="$2"; shift ;;
    --server) SERVER="$2"; shift ;;
    --experiment_prename) EXPERIMENT_PRENAME="$2"; shift ;;
    --latestart) LATE_START="$2"; shift ;;
    --mghigh) MG_HIGH="$2"; shift ;;
    --wtraindog) W_TRAIN_DOG="$2"; shift ;;
    --model_name) MODEL="$2"; shift ;;
    --guidance_control) GUIDANCE_CONTROL="$2"; shift ;;
    --w_max) W_MAX="$2"; shift ;;
    --w_min) W_MIN="$2"; shift ;;
    --sample_guidance) SAMPLE_GUIDANCE="$2"; shift ;;
    --control_distribution) CONTROL_DISTRIBUTION="$2"; shift ;;
    --batch_size) BATCH_SIZE="$2"; shift ;;
    --sample_batch_size) SAMPLE_BATCH_SIZE="$2"; shift ;;
    --nsample) NSAMPLE="$2"; shift ;;
    --eval_only) EVAL_ONLY=1 ;;
    --eval_step) EVAL_STEP="$2"; shift ;;
    --zero_norm_variance) ZERO_NORM_VARIANCE="$2"; shift ;;
    --port) PORT="$2"; shift ;;

    # Task 1 Adaptive options
    --adaptive_gate_type) ADAPTIVE_GATE_TYPE="$2"; shift ;;
    --adaptive_tau) ADAPTIVE_TAU="$2"; shift ;;
    --adaptive_temp) ADAPTIVE_TEMP="$2"; shift ;;
    --adaptive_bins) ADAPTIVE_BINS="$2"; shift ;;
    --adaptive_decay) ADAPTIVE_DECAY="$2"; shift ;;
    --adaptive_override_cutoff) ADAPTIVE_OVERRIDE_CUTOFF="$2"; shift ;;

    *) echo "Unknown parameter passed: $1"; exit 1 ;;
  esac
  shift
done

EXPERIMENT_NAME="$EXPERIMENT_PRENAME/dogfit_TASK1_self_paced_LATE_START${LATE_START}_W${W_TRAIN_DOG}_control${GUIDANCE_CONTROL}_gate_${ADAPTIVE_GATE_TYPE}_tau${ADAPTIVE_TAU}"

resolve_server_paths
resolve_dataset_config

train_model() {
    log_and_run "Training model with Task 1 Self-Paced Adaptive Guidance..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --master_port=$PORT --nproc_per_node=$NPROC_PER_NODE train_DogFit_task1.py \
        --data-path "$REAL_DATA_DIR" \
        --results-dir "$RESULTS_DIR" \
        --model "$MODEL" \
        --image-size "$IMAGE_SIZE" \
        --num-classes "$NUM_CLASSES" \
        --total-steps "$TOTAL_STEPS" \
        --log-every "$LOG_EVERY" \
        --ckpt-every "$CKPT_EVERY" \
        --global-batch-size "$BATCH_SIZE" \
        --vae "$VAE" \
        --num-workers "$NUM_WORKERS" \
        --w-dog "$W_TRAIN_DOG" \
        --guidance-cutoff "$USE_GUIDANCE_CUTOFF" \
        --mg-high "$MG_HIGH" \
        --dropout-ratio "$DROPOUT_RATIO" \
        --late-start-iter "$LATE_START" \
        --guidance-control "$GUIDANCE_CONTROL" \
        --w-max "$W_MAX" \
        --w-min "$W_MIN" \
        --control-distribution "$CONTROL_DISTRIBUTION" \
        --zero-norm-variance "$ZERO_NORM_VARIANCE" \
        --adaptive-guidance "$ADAPTIVE_GUIDANCE" \
        --adaptive-bins "$ADAPTIVE_BINS" \
        --adaptive-gate-type "$ADAPTIVE_GATE_TYPE" \
        --adaptive-tau "$ADAPTIVE_TAU" \
        --adaptive-temp "$ADAPTIVE_TEMP" \
        --adaptive-decay "$ADAPTIVE_DECAY"
}

run_sampling() {
    log_and_run "Sampling images..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES torchrun --master_port=$PORT --nproc_per_node=$NPROC_PER_NODE sample.py \
        --model "$MODEL" \
        --vae "$VAE" \
        --sample-dir "$GENERATED_DIR/$PADDED_STEP" \
        --ckpt "$CHECKPOINT_DIR/$PADDED_CKPT" \
        --per-proc-batch-size "$SAMPLE_BATCH_SIZE" \
        --num-fid-samples "$NSAMPLE" \
        --image-size "$IMAGE_SIZE" \
        --num-classes "$NUM_CLASSES" \
        --cfg-scale "$CFG_SCALE" \
        --num-sampling-steps "$NUM_SAMPLE_STEPS" \
        --dropout-ratio "$DROPOUT_RATIO" \
        --guidance-control "$GUIDANCE_CONTROL" \
        --w-dgft "$SAMPLE_GUIDANCE" \
        --zero-norm-variance "$ZERO_NORM_VARIANCE"
}

calculate_fid() {
    log_and_run "Calculating FID DINO..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR/$PADDED_STEP" \
        --model dinov2 \
        --device "$FID_DEVICE" \
        --nsample "$NSAMPLE" \
        --clean_resize \
        --metrics fd prdc kd \
        --save \
        --output_dir "$RESULTS_FILE"

    log_and_run "Calculating FID..." \
    env CUDA_VISIBLE_DEVICES=$CUDA_DEVICES python -m dgm_eval "$REAL_DATA_DIR" "$GENERATED_DIR/$PADDED_STEP" \
        --model inception \
        --device "$FID_DEVICE" \
        --nsample "$NSAMPLE" \
        --clean_resize \
        --metrics fd prdc kd \
        --save \
        --output_dir "$RESULTS_FILE"
}

main() {
    echo ">>> Logging to: $LOG_FILE"
    if [[ "$EVAL_ONLY" -eq 0 ]]; then
      rm -f "$LOG_FILE"
    fi
    mkdir -p "$(dirname "$LOG_FILE")"

    create_environment
    prepare_dataset

    if [ "$EVAL_ONLY" -eq 0 ]; then
        train_model || { echo ">>> [ERROR] Training failed! Exiting."; exit 1; }
        echo ">>> Training completed successfully!"
        exit 0
    fi

    if [ -z "$EVAL_STEP" ]; then
        STEPS_TO_EVAL=("$TOTAL_STEPS")
    else
        STEPS_TO_EVAL=("$EVAL_STEP")
    fi

    for i in "${STEPS_TO_EVAL[@]}"; do
        printf -v PADDED_STEP "%07d" "$i"
        printf -v PADDED_CKPT "%07d.pt" "$i"

        if (( $(echo "$GUIDANCE_CONTROL > 0" | bc -l) )); then
            RESULTS_BASE="$RESULTS_FILE"
            GENERATED_BASE="$GENERATED_DIR"
            if (( $(echo "$SAMPLE_GUIDANCE == 0" | bc -l) )); then
                GUIDANCE_VALUES=(1.0 1.5 2.0)
                for SG in "${GUIDANCE_VALUES[@]}"; do
                    SAMPLE_GUIDANCE=$SG
                    RESULTS_FILE="${RESULTS_BASE}_w_dgft${SAMPLE_GUIDANCE}"
                    GENERATED_DIR="${GENERATED_BASE}_w_dgft${SAMPLE_GUIDANCE}"
                    run_sampling
                    calculate_fid
                done
            else
                RESULTS_FILE="${RESULTS_BASE}_w_dgft${SAMPLE_GUIDANCE}"
                GENERATED_DIR="${GENERATED_BASE}_w_dgft${SAMPLE_GUIDANCE}"
                run_sampling
                calculate_fid
            fi
        else
            run_sampling
            calculate_fid
        fi
    done

    echo ">>> Evaluation completed successfully!"
}

main

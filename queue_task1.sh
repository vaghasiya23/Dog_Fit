#!/bin/bash
# ==============================================================================
# Multi-Run Queue Manager for Task 1
# Supports sequential (--lane all) or 2 concurrent lanes (--lane 1 / --lane 2):
#   Lane 1: run1_sigmoid_tau1.0 -> run3_sigmoid_tau1.2
#   Lane 2: run2_gaussian_tau1.0 -> run4_control_tau1.0
# Sampling configured for 2,000 images on w in {1.0, 1.5, 2.0} (when control > 0)
# ==============================================================================

set -eo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source /home/dse_413/DL_proj/dl/bin/activate

LANE="all"
while [[ "$#" -gt 0 ]]; do
    case $1 in
        --lane) LANE="$2"; shift ;;
        *) echo "Unknown argument: $1"; exit 1 ;;
    esac
    shift
done

LOG_DIR="queue_logs"
mkdir -p "$LOG_DIR"

pick_free_port() {
    local port="$1"
    while (echo >/dev/tcp/127.0.0.1/"$port") &>/dev/null; do
        port=$((port + 1))
    done
    echo "$port"
}

echo "=========================================================="
echo " Starting DogFit Task 1 Queue Manager (lane=${LANE})"
echo " Sampling configured for 2,000 images on w in {1.0, 1.5, 2.0}"
echo "=========================================================="

run_experiment() {
    local RUN_NAME="$1"
    local GATE_TYPE="$2"
    local TAU="$3"
    local CONTROL="$4"
    local W_MAX="$5"
    local CUDA_DEV="$6"
    local BASE_PORT="$7"
    local LOG_FILE="$LOG_DIR/${RUN_NAME}.log"
    local TRAIN_PORT
    TRAIN_PORT=$(pick_free_port "$BASE_PORT")

    echo ""
    echo "=========================================================="
    echo " [QUEUE lane=${LANE}] Starting: ${RUN_NAME} on GPU ${CUDA_DEV} (port=${TRAIN_PORT})"
    echo " Gate: ${GATE_TYPE}, Tau: ${TAU}, Control: ${CONTROL}"
    echo " Log: ${LOG_FILE}"
    echo "=========================================================="

    ./run_DogFit_task1.sh \
        --cuda_devices "${CUDA_DEV}" \
        --dataset "food-101_processed" \
        --latestart 1000 \
        --adaptive_gate_type "${GATE_TYPE}" \
        --adaptive_tau "${TAU}" \
        --guidance_control "${CONTROL}" \
        --w_max "${W_MAX}" \
        --w_min "1.0" \
        --port "${TRAIN_PORT}" \
        2>&1 | tee "${LOG_FILE}"

    echo " [QUEUE lane=${LANE}] Finished Training: ${RUN_NAME}"

    if [ "${CONTROL}" -gt 0 ]; then
        # Control mode supports variable guidance w at inference
        for W_VAL in 1.0 1.5 2.0; do
            local EVAL_PORT
            EVAL_PORT=$(pick_free_port "$((BASE_PORT + 20))")
            echo " [QUEUE lane=${LANE}] Sampling 2,000 images at w=${W_VAL} (port=${EVAL_PORT})..."
            ./run_DogFit_task1.sh \
                --cuda_devices "${CUDA_DEV}" \
                --dataset "food-101_processed" \
                --latestart 1000 \
                --adaptive_gate_type "${GATE_TYPE}" \
                --adaptive_tau "${TAU}" \
                --guidance_control "${CONTROL}" \
                --eval_only \
                --eval_step 24000 \
                --sample_guidance "${W_VAL}" \
                --sample_batch_size 64 \
                --nsample 2000 \
                --port "${EVAL_PORT}" \
                2>&1 | tee -a "${LOG_FILE}"
        done
    else
        local EVAL_PORT
        EVAL_PORT=$(pick_free_port "$((BASE_PORT + 20))")
        echo " [QUEUE lane=${LANE}] Sampling & evaluating 2,000 images (control=0, port=${EVAL_PORT})..."
        ./run_DogFit_task1.sh \
            --cuda_devices "${CUDA_DEV}" \
            --dataset "food-101_processed" \
            --latestart 1000 \
            --adaptive_gate_type "${GATE_TYPE}" \
            --adaptive_tau "${TAU}" \
            --guidance_control "${CONTROL}" \
            --eval_only \
            --eval_step 24000 \
            --sample_batch_size 64 \
            --nsample 2000 \
            --port "${EVAL_PORT}" \
            2>&1 | tee -a "${LOG_FILE}"
    fi

    echo "=========================================================="
    echo " [QUEUE lane=${LANE}] All steps complete for: ${RUN_NAME}"
    echo "=========================================================="
}

case "${LANE}" in
    1)
        # Lane 1: Run 1 (Sigmoid, Tau=1.0) -> Run 3 (Sigmoid, Tau=1.2)
        run_experiment "run1_sigmoid_tau1.0" "sigmoid" "1.0" "0" "1.0" "0" "29610"
        run_experiment "run3_sigmoid_tau1.2" "sigmoid" "1.2" "0" "1.0" "0" "29615"
        ;;
    2)
        # Lane 2: Run 2 (Gaussian, Tau=1.0) -> Run 4 (Guidance Control Mode, w in [1.0, 1.5])
        run_experiment "run2_gaussian_tau1.0" "gaussian" "1.0" "0" "1.0" "0" "29710"
        run_experiment "run4_control_tau1.0" "sigmoid" "1.0" "1" "1.5" "0" "29715"
        ;;
    all)
        run_experiment "run1_sigmoid_tau1.0" "sigmoid" "1.0" "0" "1.0" "0" "29610"
        run_experiment "run2_gaussian_tau1.0" "gaussian" "1.0" "0" "1.0" "0" "29615"
        run_experiment "run3_sigmoid_tau1.2" "sigmoid" "1.2" "0" "1.0" "0" "29620"
        run_experiment "run4_control_tau1.0" "sigmoid" "1.0" "1" "1.5" "0" "29625"
        ;;
    *)
        echo "Invalid --lane '${LANE}' (expected 1, 2, or all)"
        exit 1
        ;;
esac

echo ""
echo "=========================================================="
echo " ALL QUEUED EXPERIMENTS COMPLETED (lane=${LANE})!"
echo "=========================================================="

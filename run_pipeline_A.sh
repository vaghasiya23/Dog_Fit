#!/bin/bash
# ==============================================================================
# Pipeline A: Runs Run 1 (Sigmoid, Tau=1.0) then Run 3 (Sigmoid, Tau=1.2)
# With 2,000 image sampling on w in {1.0, 1.5, 2.0}
# ==============================================================================

set -eo pipefail
cd /DATA/agniva/om/DL_proj/clean_repo/DogFit
mkdir -p queue_logs

run_job() {
    local RUN_NAME="$1"
    local GATE="$2"
    local TAU="$3"
    local LOG="queue_logs/${RUN_NAME}.log"

    echo "=== [Pipeline A] Starting ${RUN_NAME} ==="
    ./run_DogFit_task1.sh --cuda_devices "0" \
        --adaptive_gate_type "${GATE}" \
        --adaptive_tau "${TAU}" \
        --guidance_control 0 \
        2>&1 | tee "${LOG}"

    for W_VAL in 1.0 1.5 2.0; do
        echo "=== [Pipeline A] Sampling 2k images for w=${W_VAL} ==="
        ./run_DogFit_task1.sh --cuda_devices "0" \
            --adaptive_gate_type "${GATE}" \
            --adaptive_tau "${TAU}" \
            --guidance_control 0 \
            --eval_only \
            --eval_step 24000 \
            --sample_guidance "${W_VAL}" \
            --nsample 2000 \
            2>&1 | tee -a "${LOG}"
    done
}

# Run 1
run_job "run1_sigmoid_tau1.0" "sigmoid" "1.0"

# Followed in sequence by Run 3
run_job "run3_sigmoid_tau1.2" "sigmoid" "1.2"

echo "=== [Pipeline A] ALL DONE! ==="

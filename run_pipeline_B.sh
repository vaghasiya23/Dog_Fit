#!/bin/bash
# ==============================================================================
# Pipeline B: Runs Run 2 (Gaussian, Tau=1.0) then Run 4 (Control Mode)
# With 2,000 image sampling on w in {1.0, 1.5, 2.0}
# ==============================================================================

set -eo pipefail
cd /DATA/agniva/om/DL_proj/clean_repo/DogFit
mkdir -p queue_logs

run_job() {
    local RUN_NAME="$1"
    local GATE="$2"
    local TAU="$3"
    local CONTROL="$4"
    local W_MAX="$5"
    local LOG="queue_logs/${RUN_NAME}.log"

    echo "=== [Pipeline B] Starting ${RUN_NAME} ==="
    ./run_DogFit_task1.sh --cuda_devices "0" \
        --adaptive_gate_type "${GATE}" \
        --adaptive_tau "${TAU}" \
        --guidance_control "${CONTROL}" \
        --w_max "${W_MAX}" \
        --w_min "1.0" \
        2>&1 | tee "${LOG}"

    for W_VAL in 1.0 1.5 2.0; do
        echo "=== [Pipeline B] Sampling 2k images for w=${W_VAL} ==="
        ./run_DogFit_task1.sh --cuda_devices "0" \
            --adaptive_gate_type "${GATE}" \
            --adaptive_tau "${TAU}" \
            --guidance_control "${CONTROL}" \
            --w_max "${W_MAX}" \
            --w_min "1.0" \
            --eval_only \
            --eval_step 24000 \
            --sample_guidance "${W_VAL}" \
            --nsample 2000 \
            2>&1 | tee -a "${LOG}"
    done
}

# Run 2: Gaussian band gating
run_job "run2_gaussian_tau1.0" "gaussian" "1.0" "0" "1.0"

# Followed in sequence by Run 4: Guidance control
run_job "run4_control_tau1.0" "sigmoid" "1.0" "1" "1.5"

echo "=== [Pipeline B] ALL DONE! ==="

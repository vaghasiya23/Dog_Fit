#!/bin/bash
# ==============================================================================
# Task 2 Dual Anchor on Control 1 (Food-101)
# Full FP32 precision, late start 12000, 2000 samples across w in {1, 1.5, 2, 3, 4, 5}
# ==============================================================================

set -eo pipefail
cd /home/dse_413/DL_proj/Om/DogFit

source /home/dse_413/DL_proj/dl/bin/activate
mkdir -p queue_logs
QUEUE_LOG="queue_logs/task2_control1.log"

echo "==================================================================" | tee "$QUEUE_LOG"
echo "[TASK 2] Starting Task 2 Dual Anchor (Control 1) at $(date)" | tee -a "$QUEUE_LOG"
echo "==================================================================" | tee -a "$QUEUE_LOG"

./run_DogFit_task2_fixed.sh \
    --dataset "food-101_processed" \
    --latestart "12000" \
    --mghigh "1" \
    --batch_size "4" \
    --guidance_control "1" \
    --nsample "2000" 2>&1 | tee -a "$QUEUE_LOG"

echo "[TASK 2] Task 2 Control 1 completed at $(date)!" | tee -a "$QUEUE_LOG"

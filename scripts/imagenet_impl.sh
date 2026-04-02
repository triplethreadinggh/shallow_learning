#!/bin/bash
# imagenet_impl.sh
# Run ImageNet CNN training in the background on Lovelace

EPOCHS=1000
TRAIN_RATIO=0.10
VAL_RATIO=0.05

LOG_FILE="logs/imagenet_training.log"
mkdir -p logs

echo "Starting ImageNet CNN training..."
echo "  Epochs:      ${EPOCHS}"
echo "  Train ratio: ${TRAIN_RATIO}"
echo "  Val ratio:   ${VAL_RATIO}"
echo "  Log file:    ${LOG_FILE}"

nohup python3 scripts/imagenet_impl.py \
    --epochs      ${EPOCHS} \
    --train_ratio ${TRAIN_RATIO} \
    --val_ratio   ${VAL_RATIO} \
    > ${LOG_FILE} 2>&1 &

echo "Training running in background with PID $!"
echo "Monitor progress with: tail -f ${LOG_FILE}"

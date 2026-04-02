#!/bin/bash
# acc_impl.sh
# Train ACCNet on the ACC dataset
 
DATA_DIR="/data/CPE_487-587/ACCDataset"
EPOCHS=5
LR=0.001
BATCH=256
KEYWORD="acc"
 
echo "Training ACCNet..."
python3 scripts/acc_impl.py \
    --data ${DATA_DIR} \
    --epochs ${EPOCHS} \
    --lr ${LR} \
    --batch ${BATCH} \
    --keyword ${KEYWORD}
 
echo "Done. Check scripts/output_acc/ for metrics and ONNX model."

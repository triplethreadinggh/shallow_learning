#!/bin/bash

# multiclass_impl.sh
# Run multiclass training 5 times and then aggregate results

DATA_FILE="data/Android_Malware.csv"
EPOCHS=1000
LR=0.001
KEYWORD="hw02"

echo "Running multiclass_impl.py 5 times with keyword=${KEYWORD}"

for i in {1..5}
do
    echo "Run $i / 5"
    python3 scripts/multiclass_impl.py \
        --data ${DATA_FILE} \
        --epochs ${EPOCHS} \
        --lr ${LR} \
        --keyword ${KEYWORD}
done

echo "All training runs completed."

echo "Running multiclass_eval.py to aggregate results..."
python3 scripts/multiclass_eval.py ${KEYWORD}

echo "Done."


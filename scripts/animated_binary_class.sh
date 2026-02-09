#!/bin/bash

# Define the script name
SCRIPT_NAME="binaryclassification_animate_impl.py"

echo "Starting the binary classification training task..."

# Check if the file exists before running
if [ -f "$SCRIPT_NAME" ]; then
    python3 "$SCRIPT_NAME"
else
    echo "Error: $SCRIPT_NAME not found!"
    exit 1
fi

echo "Process finished at $(date)"

#!/usr/bin/env python3

import sys
import glob
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import os

if len(sys.argv) < 2:
    print("Usage: python3 scripts/multiclass_eval.py <keyword>")
    sys.exit(1)

keyword = sys.argv[1]

files = glob.glob(f"data/metrics_*{keyword}*.csv")

if not files:
    print(f"No metric files found with keyword '{keyword}'")
    sys.exit(1)

# Load and concatenate all CSVs
dfs = []
for f in files:
    df = pd.read_csv(f)
    df["source_file"] = os.path.basename(f)
    dfs.append(df)

metrics_df = pd.concat(dfs, ignore_index=True)

train_df = metrics_df[metrics_df["Dataset"] == "Train"]
test_df  = metrics_df[metrics_df["Dataset"] == "Test"]

metrics = ["Accuracy", "F1", "Precision", "Recall"]

plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
train_df[metrics].boxplot()
plt.title("Training Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)

plt.subplot(1, 2, 2)
test_df[metrics].boxplot()
plt.title("Test Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)

plt.tight_layout()

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
out_file = f"data/multiclass_metrics_{keyword}_{timestamp}.png"
plt.savefig(out_file)
plt.show()

print(f"Boxplot saved to {out_file}")


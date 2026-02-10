#!/usr/bin/env python3

import sys
import getopt
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.utils.class_weight import compute_class_weight
from datetime import datetime
import os
import csv
import numpy as np

from shallow_learning.deepl import SimpleNN, ClassTrainer
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

def main(argv):
    data_file = "data/Android_Malware.csv"
    epochs = 1000
    learning_rate = 0.001
    unique_keyword = "hw02"

    try:
        opts, args = getopt.getopt(argv, "hd:e:", ["help","data=","epochs=","lr=","keyword="])
    except getopt.GetoptError:
        print('Usage: multiclass_impl.py --data <datafile> -e <epochs> --lr <learning_rate> --kw <keyword>')
        sys.exit(2)

    for opt, arg in opts:
        if opt in ('-h', '--help'):
            print('Usage: multiclass_impl.py --data <datafile> -e <epochs> --lr <learning_rate> --keyword <keyword>')
            sys.exit()
        elif opt in ("-d","--data"):
            data_file = arg
        elif opt in ("-e","--epochs"):
            epochs = int(arg)
        elif opt in ("--lr",):
            learning_rate = float(arg)
        elif opt in ("--keyword",):
            unique_keyword = arg

    df = pd.read_csv(data_file, low_memory=False)
    
    # Strip whitespace from column names
    df.columns = df.columns.str.strip()
    
    # Drop unhelpful columns
    drop_cols = ['Flow ID', 'Source IP', 'Source Port', 'Destination IP', 
                 'Destination Port', 'Protocol', 'Timestamp']
    df = df.drop(columns=drop_cols, errors='ignore')
    
    # Strip whitespace and ensure 'Label' is string
    df['Label'] = df['Label'].astype(str).str.strip()
    
    # Convert feature columns to numeric
    feature_cols = df.columns.drop('Label')  # exclude label
    df[feature_cols] = df[feature_cols].apply(pd.to_numeric, errors='coerce')
    
    # Fill NaNs in features with 0
    df[feature_cols] = df[feature_cols].fillna(0)
    
    # Features and labels
    X = df[feature_cols].values
    y = df['Label'].values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Encode labels to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    num_classes = len(le.classes_)

    classes, counts = np.unique(y, return_counts=True)
    for cls, cnt in zip(le.classes_, counts):
        print(f"Class {cls}: {cnt} samples ({cnt/len(y)*100:.2f}%)")
    
    print("Classes found:", le.classes_)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    num_features = X_train.shape[1]

    #class_weights = compute_class_weight(
    #    class_weight='balanced',
    #    classes=np.unique(y_encoded),
    #    y=y_encoded
    #)
    #print("Class weights:", class_weights)

    model = SimpleNN(in_features=num_features, num_classes=num_classes)
    trainer = ClassTrainer(X_train, y_train, model, eta=learning_rate, epochs=epochs) #class_weights=class_weights

    trainer.train()

    trainer.evaluation(X_test, y_test)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs("data", exist_ok=True)
    csv_file = os.path.join("data", f"metrics_{unique_keyword}_{timestamp}.csv")
    #csv_file = f"metrics_{unique_keyword}_{timestamp}.csv"
    y_train_pred = trainer.predict(X_train)
    y_test_pred = trainer.predict(X_test)

    metrics = [
        ['Dataset','Accuracy','F1','Precision','Recall'],
        ['Train',
         round(trainer.accuracy_vector[-1].item(),4),
         round(f1_score(y_train, y_train_pred, average='weighted'),4),
         round(precision_score(y_train, y_train_pred, average='weighted'),4),
         round(recall_score(y_train, y_train_pred, average='weighted'),4)],
        ['Test',
         round(accuracy_score(y_test, y_test_pred),4),
         round(f1_score(y_test, y_test_pred, average='weighted'),4),
         round(precision_score(y_test, y_test_pred, average='weighted'),4),
         round(recall_score(y_test, y_test_pred, average='weighted'),4)]
    ]

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerows(metrics)

    print(f"Metrics saved to {csv_file}")


if __name__ == "__main__":
    main(sys.argv[1:])


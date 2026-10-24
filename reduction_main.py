"""Reduction experiments: the best kNN configuration against the three
sample reduction techniques, evaluated on kNN, SVM and Random Forest.
"""

import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from data_loader import load_fold
from reduction_knn import REDUCTIONS, reduce_training_set
from knn import kNNAlgorithm
from svm import svmAlgorithm
from rf import rfAlgorithm

BEST_KNN = {
    'pima-indians': ('euclidean', 7, 'majority', 'OG'),
    'mx':           ('euclidean', 7, 'majority', 'OG'),
}

datasets = ['pima-indians', 'mx']
results = []

for name in datasets:
    print(f"--- Processing dataset: {name} ---")
    dist, k, voting, retention = BEST_KNN[name]

    for i in range(10):
        x_train, y_train, x_test, y_test, types = load_fold(name, i)
        x_test = np.asarray(x_test)
        original_size = len(x_train)

        for red_name in REDUCTIONS:

            # --- Perform Reduction ---
            start_red = time.time()
            x_train_red, y_train_red = reduce_training_set(x_train, y_train, red_name, dist)
            time_red = time.time() - start_red

            storage_pct = len(x_train_red) / original_size * 100.0

            # --- Evaluate k-NN ---
            start_knn = time.time()
            pred_knn, final_size = kNNAlgorithm(
                x_train_red, y_train_red, x_test, y_test,
                k=k, distance=dist, voting=voting, retention=retention, types=types
            )
            time_knn = time.time() - start_knn

            results.append({
                'dataset': name, 'fold': i, 'reduction': red_name, 'model': 'kNN',
                'accuracy': accuracy_score(y_test, pred_knn),
                'time_reduction': time_red, 'time_inference': time_knn,
                # storage_pct: left after reduction
                # final_storage_pct: what's left after OG forgets after inference
                'storage_pct': storage_pct,
                'final_storage_pct': final_size / original_size * 100.0
            })

            # --- Evaluate SVM ---
            # Same configuration as the unreduced baseline in main.py
            # (RBF kernel, default C and gamma), so that the only thing
            # that changes is the training set.
            start_svm = time.time()
            pred_svm = svmAlgorithm(x_train_red, y_train_red, x_test,
                                    kernel='rbf')
            time_svm = time.time() - start_svm

            results.append({
                'dataset': name, 'fold': i, 'reduction': red_name, 'model': 'SVM',
                'accuracy': accuracy_score(y_test, pred_svm),
                'time_reduction': time_red, 'time_inference': time_svm, 'storage_pct': storage_pct
            })

            # --- Evaluate Random Forest ---
            # Same configuration as the unreduced baseline in main.py
            # (100 trees, gini).
            start_rf = time.time()
            pred_rf = rfAlgorithm(x_train_red, y_train_red, x_test,
                                  n_estimators=100, criterion='gini')
            time_rf = time.time() - start_rf

            results.append({
                'dataset': name, 'fold': i, 'reduction': red_name, 'model': 'RF',
                'accuracy': accuracy_score(y_test, pred_rf),
                'time_reduction': time_red, 'time_inference': time_rf, 'storage_pct': storage_pct
            })

df_results = pd.DataFrame(results)
df_results.to_csv(Path(__file__).parent / 'reduction_results.csv', index=False)

print("\n=== Results Summary (Means) ===")
summary = df_results.groupby(['dataset', 'reduction', 'model'])[['accuracy', 'time_inference', 'storage_pct']].mean()
print(summary.round(4))
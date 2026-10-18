"""Reduction experiments: the best kNN configuration against the three
sample reduction techniques, evaluated on kNN, SVM and Random Forest.
"""

import time
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

from data_loader import load_fold
from fcnn import condense
from main import kNNAlgorithm

# Chosen from the Friedman and Nemenyi analysis over the 120 kNN
# configurations (see results/analysis_output.txt).
#
# k = 7 wins accuracy, time and storage in both datasets, significantly.
# Voting is never significant, so the simpler scheme is used.
# OG wins time and storage and is not significantly worse in accuracy.
# IVDM is significantly worse than the other four metrics on mx, and on
# pima-indians the four are statistically tied, so euclidean is used in
# both: it is the fastest and it keeps the two datasets comparable.
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
        # 1. Load the original fold
        x_train, y_train, x_test, y_test, types = load_fold(name, i)
        original_size = len(x_train)

        # ==========================================
        # REDUCTION 1: FCNN (Condensation)
        # ==========================================
        start_red = time.time()
        # Apply FCNN to the original training set
        x_train_fcnn, y_train_fcnn = condense(x_train, y_train, metric=dist)
        time_fcnn = time.time() - start_red

        # Calculate storage percentage retained
        storage_fcnn = len(x_train_fcnn) / original_size * 100.0

        # --- Evaluate k-NN with FCNN-reduced data ---
        start_knn = time.time()
        pred_knn, _ = kNNAlgorithm(
            x_train_fcnn, y_train_fcnn, x_test, y_test,
            k=k, distance=dist, voting=voting, retention=retention, types=types
        )
        time_knn = time.time() - start_knn

        results.append({
            'dataset': name, 'fold': i,
            'reduction': 'FCNN', 'model': 'kNN',
            'accuracy': accuracy_score(y_test, pred_knn),
            'time_reduction': time_fcnn,
            'time_inference': time_knn,
            'storage_pct': storage_fcnn
        })

        # --- Evaluate SVM with FCNN-reduced data ---
        svm_model = SVC(kernel='rbf', random_state=42)
        start_svm = time.time()
        svm_model.fit(x_train_fcnn, y_train_fcnn)
        pred_svm = svm_model.predict(x_test)
        time_svm = time.time() - start_svm

        results.append({
            'dataset': name, 'fold': i,
            'reduction': 'FCNN', 'model': 'SVM',
            'accuracy': accuracy_score(y_test, pred_svm),
            'time_reduction': time_fcnn,
            'time_inference': time_svm,
            'storage_pct': storage_fcnn
        })

        # --- Evaluate Random Forest with FCNN-reduced data ---
        rf_model = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)
        start_rf = time.time()
        rf_model.fit(x_train_fcnn, y_train_fcnn)
        pred_rf = rf_model.predict(x_test)
        time_rf = time.time() - start_rf

        results.append({
            'dataset': name, 'fold': i,
            'reduction': 'FCNN', 'model': 'RF',
            'accuracy': accuracy_score(y_test, pred_rf),
            'time_reduction': time_fcnn,
            'time_inference': time_rf,
            'storage_pct': storage_fcnn
        })

        # ==========================================
        # REDUCTION 2: RENN (Edition) - COMING SOON
        # ==========================================
        # RENN code will go here

        # ==========================================
        # REDUCTION 3: DROP3 (Hybrid) - COMING SOON
        # ==========================================
        # DROP3 code will go here

# Save and display results
df_results = pd.DataFrame(results)
df_results.to_csv('reduction_results.csv', index=False)

print("\n=== Results Summary (Means) ===")
summary = df_results.groupby(['dataset', 'reduction', 'model'])[['accuracy', 'time_inference', 'storage_pct']].mean()
print(summary.round(4))


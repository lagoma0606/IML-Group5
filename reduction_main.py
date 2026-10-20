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
from eennth import reduce_eennth
from main import kNNAlgorithm

BEST_KNN = {
    'pima-indians': ('euclidean', 7, 'majority', 'OG'),
    'mx':           ('euclidean', 7, 'majority', 'OG'),
}

# Map of reduction methods to ensure uniform loop execution
REDUCTION_METHODS = {
    'FCNN': condense,
    'EENN': reduce_eennth,
    # 'DROP3': reduce_drop3  # TODO: Hybrid DROP3 implementation goes here
}

datasets = ['pima-indians', 'mx']
results = []

for name in datasets:
    print(f"--- Processing dataset: {name} ---")
    dist, k, voting, retention = BEST_KNN[name]

    for i in range(10):
        x_train, y_train, x_test, y_test, types = load_fold(name, i)
        original_size = len(x_train)

        for red_name, red_func in REDUCTION_METHODS.items():
            
            # --- Perform Reduction ---
            start_red = time.time()
            x_train_red, y_train_red = red_func(x_train, y_train, metric=dist)
            time_red = time.time() - start_red

            storage_pct = len(x_train_red) / original_size * 100.0

            # --- Evaluate k-NN ---
            start_knn = time.time()
            pred_knn, _ = kNNAlgorithm(
                x_train_red, y_train_red, x_test, y_test,
                k=k, distance=dist, voting=voting, retention=retention, types=types
            )
            time_knn = time.time() - start_knn

            results.append({
                'dataset': name, 'fold': i, 'reduction': red_name, 'model': 'kNN',
                'accuracy': accuracy_score(y_test, pred_knn),
                'time_reduction': time_red, 'time_inference': time_knn, 'storage_pct': storage_pct
            })

            # --- Evaluate SVM ---
            svm_model = SVC(kernel='rbf', random_state=42)
            start_svm = time.time()
            svm_model.fit(x_train_red, y_train_red)
            pred_svm = svm_model.predict(x_test)
            time_svm = time.time() - start_svm

            results.append({
                'dataset': name, 'fold': i, 'reduction': red_name, 'model': 'SVM',
                'accuracy': accuracy_score(y_test, pred_svm),
                'time_reduction': time_red, 'time_inference': time_svm, 'storage_pct': storage_pct
            })

            # --- Evaluate Random Forest ---
            rf_model = RandomForestClassifier(n_estimators=100, criterion='gini', random_state=42)
            start_rf = time.time()
            rf_model.fit(x_train_red, y_train_red)
            pred_rf = rf_model.predict(x_test)
            time_rf = time.time() - start_rf

            results.append({
                'dataset': name, 'fold': i, 'reduction': red_name, 'model': 'RF',
                'accuracy': accuracy_score(y_test, pred_rf),
                'time_reduction': time_red, 'time_inference': time_rf, 'storage_pct': storage_pct
            })

df_results = pd.DataFrame(results)
df_results.to_csv('reduction_results.csv', index=False)

print("\n=== Results Summary (Means) ===")
summary = df_results.groupby(['dataset', 'reduction', 'model'])[['accuracy', 'time_inference', 'storage_pct']].mean()
print(summary.round(4))
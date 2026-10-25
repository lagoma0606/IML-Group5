"""Reduction experiments: the best kNN configuration against the three
sample reduction techniques, evaluated on kNN, SVM and Random Forest.
"""

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score

from data_loader import load_fold
from reduction_knn import REDUCTIONS, reduce_training_set, reduction_kNNAlgorithm
from svm import svmAlgorithm
from rf import rfAlgorithm

BEST_KNN = {
    'pima-indians': ('euclidean', 7, 'majority', 'OG'),
    'mx':           ('euclidean', 7, 'majority', 'OG'),
}

ALL_DATASETS = ['pima-indians', 'mx']
ALL_REDUCTIONS = list(REDUCTIONS)
ALL_MODELS = ['knn', 'svm', 'rf']

# ------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------
# Run with no arguments to reproduce the report. The options exist so that
# a single experiment can be repeated without editing this file.
parser = argparse.ArgumentParser(
    description='Run the instance reduction experiments and write one row '
                'per run to a CSV file. With no arguments every experiment '
                'is run.')
parser.add_argument('--dataset', nargs='+', choices=ALL_DATASETS,
                    help='datasets to run (default: both)')
parser.add_argument('--reduction', nargs='+', choices=ALL_REDUCTIONS,
                    help='reduction techniques to apply (default: all three)')
parser.add_argument('--model', nargs='+', choices=ALL_MODELS,
                    help='models trained on the reduced set: knn, svm, rf '
                         '(default: all three)')
parser.add_argument('--folds', type=int, default=10, choices=range(1, 11),
                    metavar='N',
                    help='run the first N folds, 1 to 10 (default: 10)')
parser.add_argument('--output',
                    help='output CSV file name (default: '
                         'reduction_results.csv for a full run, '
                         'reduction_results_partial.csv otherwise)')
args = parser.parse_args()

datasets = args.dataset or ALL_DATASETS
reductions = args.reduction or ALL_REDUCTIONS
models = args.model or ALL_MODELS
folds = range(args.folds)

# A narrowed run must not overwrite the full results the report is built on.
full_run = (args.dataset is None and args.reduction is None
            and args.model is None and args.folds == 10)
OUTPUT_FILE = args.output or ('reduction_results.csv' if full_run
                              else 'reduction_results_partial.csv')

print('datasets   :', datasets)
print('reductions :', reductions)
print('models     :', models)
print('folds      :', args.folds)
print('writing    :', OUTPUT_FILE)
print()

results = []

for name in datasets:
    print(f"--- Processing dataset: {name} ---")
    dist, k, voting, retention = BEST_KNN[name]

    for i in folds:
        x_train, y_train, x_test, y_test, types = load_fold(name, i)
        x_test = np.asarray(x_test)
        original_size = len(x_train)

        for red_name in reductions:

            if 'knn' in models:
                # --- Reduce the training set, then classify with the kNN ---
                (pred_knn, final_size, x_train_red, y_train_red,
                 time_red, time_knn) = reduction_kNNAlgorithm(
                    x_train, y_train, x_test, y_test,
                    k=k, distance=dist, voting=voting, retention=retention,
                    types=types, reduction=red_name
                )
            else:
                # The kNN was not asked for: only the reduction is needed,
                # timed the same way as inside reduction_kNNAlgorithm.
                start = time.time()
                x_train_red, y_train_red = reduce_training_set(
                    x_train, y_train, red_name, dist)
                time_red = time.time() - start

            storage_pct = len(x_train_red) / original_size * 100.0

            if 'knn' in models:
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
            if 'svm' in models:
                start_svm = time.time()
                pred_svm = svmAlgorithm(x_train_red, y_train_red, x_test,
                                        kernel='rbf')
                time_svm = time.time() - start_svm

                results.append({
                    'dataset': name, 'fold': i, 'reduction': red_name, 'model': 'SVM',
                    'accuracy': accuracy_score(y_test, pred_svm),
                    'time_reduction': time_red, 'time_inference': time_svm,
                    'storage_pct': storage_pct
                })

            # --- Evaluate Random Forest ---
            # Same configuration as the unreduced baseline in main.py
            # (100 trees, gini).
            if 'rf' in models:
                start_rf = time.time()
                pred_rf = rfAlgorithm(x_train_red, y_train_red, x_test,
                                      n_estimators=100, criterion='gini')
                time_rf = time.time() - start_rf

                results.append({
                    'dataset': name, 'fold': i, 'reduction': red_name, 'model': 'RF',
                    'accuracy': accuracy_score(y_test, pred_rf),
                    'time_reduction': time_red, 'time_inference': time_rf,
                    'storage_pct': storage_pct
                })

df_results = pd.DataFrame(results)
df_results.to_csv(Path(__file__).parent / OUTPUT_FILE, index=False)

print("\n=== Results Summary (Means) ===")
summary = df_results.groupby(['dataset', 'reduction', 'model'])[
    ['accuracy', 'time_inference', 'storage_pct']].mean()
print(summary.round(4))
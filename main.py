import argparse

import pandas as pd
from sklearn.metrics import accuracy_score
import time
from data_loader import load_fold
from itertools import product
from knn import kNNAlgorithm
from svm import svmAlgorithm, svmTunedAlgorithm
from rf import rfAlgorithm, rfTunedAlgorithm
from pathlib import Path
# ------------------------------------------------------------------
# Experiment grid
# ------------------------------------------------------------------
# These are the full grids the report was produced with. The command line
# below narrows them; with no arguments every experiment runs.
ALL_DATASETS = ['pima-indians', 'mx']
ALL_DISTANCES = ['euclidean', 'manhattan', 'clark', 'canberra', 'ivdm']
ALL_K = [3, 5, 7]
ALL_VOTINGS = ['majority', 'inverse_distance']
ALL_RETENTIONS = ['NR', 'AR', 'DC', 'OG']
ALL_KERNELS = ['linear', 'rbf']
ALL_MODELS = ['knn', 'svm', 'rf', 'tuned']

# Random forest: every number of trees against every criterion.
ALL_RF = [(n, c) for n in [10, 50, 100, 200] for c in ['gini', 'entropy']]

# ------------------------------------------------------------------
# Command line
# ------------------------------------------------------------------
# Run with no arguments to reproduce the report. The options exist so that
# a single experiment can be repeated without editing this file.
parser = argparse.ArgumentParser(
    description='Run the Work 2 experiments and write one row per run to a '
                'CSV file. With no arguments every experiment is run.')
parser.add_argument('--dataset', nargs='+', choices=ALL_DATASETS,
                    help='datasets to run (default: both)')
parser.add_argument('--model', nargs='+', choices=ALL_MODELS,
                    help='models to run: knn, svm, rf, tuned (default: all)')
parser.add_argument('--distance', nargs='+', choices=ALL_DISTANCES,
                    help='kNN distance metrics (default: all five)')
parser.add_argument('--k', nargs='+', type=int, choices=ALL_K,
                    help='kNN numbers of neighbours (default: 3 5 7)')
parser.add_argument('--voting', nargs='+', choices=ALL_VOTINGS,
                    help='kNN voting schemes (default: both)')
parser.add_argument('--retention', nargs='+', choices=ALL_RETENTIONS,
                    help='kNN retention policies (default: all four)')
parser.add_argument('--kernel', nargs='+', choices=ALL_KERNELS,
                    help='SVM kernels (default: both)')
parser.add_argument('--folds', type=int, default=10, choices=range(1, 11),
                    metavar='N',
                    help='run the first N folds, 1 to 10 (default: 10)')
parser.add_argument('--output',
                    help='output CSV file name (default: results.csv for a '
                         'full run, results_partial.csv otherwise)')
args = parser.parse_args()

datasets = args.dataset or ALL_DATASETS
models = args.model or ALL_MODELS
distances = args.distance or ALL_DISTANCES
k_values = args.k or ALL_K
votings = args.voting or ALL_VOTINGS
retentions = args.retention or ALL_RETENTIONS
svm_kernels = args.kernel or ALL_KERNELS
config = ALL_RF
folds = range(args.folds)

knn_configs = list(product(distances, k_values, votings, retentions))

# A model that was not asked for gets an empty grid, so its loop does
# nothing and no rows for it reach the CSV.
if 'rf' not in models:
    config = []
if 'svm' not in models:
    svm_kernels = []
if 'knn' not in models:
    knn_configs = []
run_tuned = 'tuned' in models

# A narrowed run must not overwrite the full results the report is built on.
full_run = (args.dataset is None and args.model is None
            and args.distance is None and args.k is None
            and args.voting is None and args.retention is None
            and args.kernel is None and args.folds == 10)
OUTPUT_FILE = args.output or ('results.csv' if full_run
                              else 'results_partial.csv')

print('datasets :', datasets)
print('models   :', models)
print('kNN      :', len(knn_configs), 'configurations')
print('folds    :', args.folds)
print('writing  :', OUTPUT_FILE)
print()

# ------------------------------------------------------------------
# Hyperparameter grids for the tuned SVM and RF live in svm.py (C and
# gamma, tuned together) and rf.py. The search uses cross validation
# inside the training set of each fold (cv=5), so the test set does not
# take part in choosing the parameters.
# ------------------------------------------------------------------

results = []

for name in datasets:
    print("processing dataset:", name)
    for i in folds:
        # All loading and preprocessing lives in data_loader.
        # main.py only asks for a fold that is ready to use.
        x_train, y_train, x_test, y_test, types = load_fold(name, i)

        # --------------------------------------------------------------
        # Random Forest
        # n_estimators = number of trees: more trees give more stability
        # but cost more time.
        # Gini measures the probability of misclassifying if labels were
        # assigned at random; entropy measures the bits needed to know
        # which class an element belongs to.
        # random_state fixes the random sequence for reproducibility.
        # --------------------------------------------------------------
        for n_estimators, c in config:
            # Stopwatch: starts before training, stops after predicting
            start = time.time()
            prediction = rfAlgorithm(x_train, y_train, x_test,
                                     n_estimators=n_estimators,
                                     criterion=c)
            end_time = time.time() - start

            accuracy = accuracy_score(y_test, prediction)
            results.append({
                'dataset': name,
                'model': 'RF',
                'config': str(n_estimators) + '_' + c,
                'n_estimators': n_estimators,
                'criterion': c,
                'fold': i,
                'accuracy': accuracy,
                'time': end_time
            })

        # --------------------------------------------------------------
        # SVM
        # The kernel decides the shape of the boundary between classes:
        # linear draws a straight line, rbf allows curved boundaries.
        # --------------------------------------------------------------
        for kernel in svm_kernels:
            start = time.time()
            prediction = svmAlgorithm(x_train, y_train, x_test, kernel=kernel)
            end_time = time.time() - start

            accuracy = accuracy_score(y_test, prediction)
            results.append({
                'dataset': name,
                'model': 'svm',
                'config': kernel,
                'fold': i,
                'accuracy': accuracy,
                'time': end_time
            })

        # --------------------------------------------------------------
        # SVM and RF with hyperparameters tuned per fold.
        # Recorded as separate models so their runtime is not mixed with
        # the fixed configurations: a grid search trains dozens of
        # models, so its time is not comparable with a single fit.
        # --------------------------------------------------------------
        if run_tuned:
            start = time.time()
            prediction, best_svm = svmTunedAlgorithm(x_train, y_train, x_test)
            end_time = time.time() - start

            accuracy = accuracy_score(y_test, prediction)
            results.append({
                'dataset': name,
                'model': 'svm-tuned',
                'config': str(best_svm),
                'fold': i,
                'accuracy': accuracy,
                'time': end_time
            })

            start = time.time()
            prediction, best_rf = rfTunedAlgorithm(x_train, y_train, x_test)
            end_time = time.time() - start

            accuracy = accuracy_score(y_test, prediction)
            results.append({
                'dataset': name,
                'model': 'RF-tuned',
                'config': str(best_rf),
                'fold': i,
                'accuracy': accuracy,
                'time': end_time
            })


        # --------------------------------------------------------------
        # kNN: 120 configurations per fold
        # --------------------------------------------------------------
        for distance, k, voting, retention in knn_configs:
            start = time.time()
            prediction, case_base_size = kNNAlgorithm(
                x_train, y_train, x_test,y_test,
                k=k,
                distance=distance,
                voting=voting,
                retention=retention,
                types=types
            )
            end_time = time.time() - start

            accuracy = accuracy_score(y_test, prediction)
            n_correct = accuracy_score(y_test, prediction, normalize=False)
            results.append({
                'dataset': name,
                'model': 'kNN',
                'config': distance + '_k' + str(k) + '_' + voting + '_' + retention,
                'distance': distance,
                'k': k,
                'voting': voting,
                'retention': retention,
                'fold': i,
                'accuracy': accuracy,
                # Storage: grows under AR, shrinks under OG. The original
                # size is stored too, so the retained percentage can be
                # computed later.
                'case_base_size': case_base_size,
                'fold_train_size': len(x_train),
                'storage_pct': 100.0 * case_base_size / len(x_train),
                'n_test': len(y_test),
                'n_correct': n_correct,
                'n_incorrect': len(y_test) - n_correct,
                'time': end_time
            })

df_results = pd.DataFrame(results)
df_results.to_csv(Path(__file__).parent / OUTPUT_FILE, index=False)

# A narrowed run leaves some of these empty, so each summary is printed
# only when there is something to summarise.
for label, rows, by in [
        ('RF', df_results[df_results['model'] == 'RF'],
         ['dataset', 'n_estimators', 'criterion']),
        ('svm', df_results[df_results['model'] == 'svm'],
         ['dataset', 'config']),
        ('kNN', df_results[df_results['model'] == 'kNN'],
         ['dataset', 'distance', 'k']),
        ('svm-tuned', df_results[df_results['model'] == 'svm-tuned'],
         ['dataset', 'config']),
        ('RF-tuned', df_results[df_results['model'] == 'RF-tuned'],
         ['dataset', 'config'])]:
    if not rows.empty:
        print(rows.groupby(by)[['accuracy', 'time']].mean())

# Final case base size per retention policy.
knn = df_results[df_results['model'] == 'kNN']
if not knn.empty:
    print(knn.groupby(['dataset', 'retention'])[
        ['case_base_size', 'fold_train_size', 'storage_pct']].mean())
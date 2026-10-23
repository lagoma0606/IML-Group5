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
# Random forest configuration
# ------------------------------------------------------------------
config = [
    (10, 'gini'),
    (10, 'entropy'),
    (50, 'gini'),
    (50, 'entropy'),
    (100, 'gini'),
    (100, 'entropy'),
    (200, 'gini'),
    (200, 'entropy')
]

datasets = ['pima-indians','mx']

# ------------------------------------------------------------------
# KNN configuration
# ------------------------------------------------------------------
distances = ['euclidean', 'manhattan', 'clark', 'canberra', 'ivdm']
k_values = [3, 5, 7]
votings = ['majority', 'inverse_distance']
retentions = ['NR', 'AR', 'DC', 'OG']

knn_configs = list(product(distances, k_values, votings, retentions))

# ------------------------------------------------------------------
# Hyperparameter grids for the tuned SVM and RF live in svm.py (C and
# gamma, tuned together) and rf.py. The search uses cross validation
# inside the training set of each fold (cv=5), so the test set does not
# take part in choosing the parameters.
# ------------------------------------------------------------------

results = []

for name in datasets:
    print("processing dataset:", name)
    for i in range(10):
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
        for kernel in ['linear', 'rbf']:
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
df_results.to_csv(Path(__file__).parent / 'results.csv', index=False)

rf = df_results[df_results['model'] == 'RF']
svm = df_results[df_results['model'] == 'svm']
knn = df_results[df_results['model'] == 'kNN']

print(rf.groupby(['dataset', 'n_estimators', 'criterion'])[['accuracy', 'time']].mean())
print(svm.groupby(['dataset', 'config'])[['accuracy', 'time']].mean())
print(knn.groupby(['dataset', 'distance', 'k'])[['accuracy', 'time']].mean())

# Final case base size per retention policy.
print(knn.groupby(['dataset', 'retention'])[['case_base_size', 'fold_train_size', 'storage_pct']].mean())

svm_tuned = df_results[df_results['model'] == 'svm-tuned']
rf_tuned = df_results[df_results['model'] == 'RF-tuned']

print(svm_tuned.groupby(['dataset', 'config'])[['accuracy', 'time']].mean())
print(rf_tuned.groupby(['dataset', 'config'])[['accuracy', 'time']].mean())
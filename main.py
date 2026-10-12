import pandas as pd
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
import time
from data_loader import load_fold
from itertools import product
import numpy as np
from distance import get_distance
from ivdm import IVDM
from voting import VOTING


def kNNAlgorithm(x_train, y_train, x_test, k, distance, voting, retention, types):
    # Convert inputs to uniform NumPy arrays/lists
    X_tr = x_train.to_numpy() if isinstance(x_train, pd.DataFrame) else np.asarray(x_train)
    Y_tr = y_train.to_numpy() if isinstance(y_train, pd.Series) else np.asarray(y_train)
    X_te = x_test.to_numpy() if isinstance(x_test, pd.DataFrame) else np.asarray(x_test)

    # 1. Distance metric selection
    if distance == 'ivdm':
        categorical_indices = [idx for idx, t in enumerate(types) if t != 'numeric']
        n_classes = len(np.unique(Y_tr))
        ivdm_model = IVDM(n_bins =max(5, n_classes), categorical_features=categorical_indices)
        ivdm_model.fit(X_tr, Y_tr)

        def calc_dist(a, b):
            return ivdm_model.get_distance(a, b)
    else:
        dist_func = get_distance(distance)
        def calc_dist(a, b):
            return dist_func(a, b)

    vote_func = VOTING[voting]

    # 2. Pre-inference retention pruning (OG)
    x_train = [row for row in X_tr]
    y_train = list(Y_tr)

    if retention == 'OG':
        pruned_x, pruned_y = [], []
        for i, sample in enumerate(x_train):
            rest_x = [x_train[j] for j in range(len(x_train)) if j != i]
            rest_y = [y_train[j] for j in range(len(y_train)) if j != i]

            if len(rest_x) > 0:
                dists = [calc_dist(sample, r_x) for r_x in rest_x]
                nn_idx = int(np.argmin(dists))
                if rest_y[nn_idx] != y_train[i]:
                    pruned_x.append(sample)
                    pruned_y.append(y_train[i])

        if len(pruned_x) > 0:
            x_train, y_train = pruned_x, pruned_y

    # 3. Predict & update case base (NR, AR, DC)
    predictions = []

    for test_pt in X_te:
        dists = [calc_dist(test_pt, train_pt) for train_pt in x_train]

        k_indices = np.argsort(dists)[:k]
        k_neighbors = [(dists[idx], y_train[idx]) for idx in k_indices]

        pred_label = vote_func(k_neighbors)
        predictions.append(pred_label)

        if retention == 'AR':
            x_train.append(np.asarray(test_pt))
            y_train.append(pred_label)

        elif retention == 'DC':
            if dists[k_indices[0]] > np.mean(dists):
                x_train.append(np.asarray(test_pt))
                y_train.append(pred_label)
    case_base_size = len(x_train)
    return predictions, case_base_size


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

datasets = ['pima-indians', 'mx']

# ------------------------------------------------------------------
# KNN configuration
# ------------------------------------------------------------------
distances = ['euclidean', 'manhattan', 'camberra', 'ivdm']
k_values = [3, 5, 7]
votings = ['majority', 'inverse_distance']
retentions = ['NR', 'AR', 'DC', 'OG']

knn_configs = list(product(distances, k_values, votings, retentions))

# ------------------------------------------------------------------
# Hyperparameter search. C and gamma interact with each other, so they
# are tuned together rather than one at a time. The search uses cross
# validation inside the training set of each fold (cv=5), so the test
# set does not take part in choosing the parameters.
# ------------------------------------------------------------------
param_grid_svm = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
}

param_grid_rf = {
    'n_estimators': [100, 300],
    'max_depth': [None, 5, 10],
    'max_features': ['sqrt', 'log2', None],
}

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
            model = RandomForestClassifier(
                n_estimators=n_estimators,
                criterion=c,
                random_state=42
            )
            # Stopwatch: starts before training, stops after predicting
            start = time.time()
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)
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
            model = SVC(kernel=kernel)
            start = time.time()
            model.fit(x_train, y_train)
            prediction = model.predict(x_test)
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
        grid_svm = GridSearchCV(
            SVC(kernel='rbf', random_state=42),
            param_grid_svm, cv=5, scoring='accuracy'
        )
        start = time.time()
        grid_svm.fit(x_train, y_train)
        prediction = grid_svm.predict(x_test)
        end_time = time.time() - start

        accuracy = accuracy_score(y_test, prediction)
        results.append({
            'dataset': name,
            'model': 'svm-tuned',
            'config': str(grid_svm.best_params_),
            'fold': i,
            'accuracy': accuracy,
            'time': end_time
        })

        grid_rf = GridSearchCV(
            RandomForestClassifier(criterion='gini', random_state=42),
            param_grid_rf, cv=5, scoring='accuracy', n_jobs=-1
        )
        start = time.time()
        grid_rf.fit(x_train, y_train)
        prediction = grid_rf.predict(x_test)
        end_time = time.time() - start

        accuracy = accuracy_score(y_test, prediction)
        results.append({
            'dataset': name,
            'model': 'RF-tuned',
            'config': str(grid_rf.best_params_),
            'fold': i,
            'accuracy': accuracy,
            'time': end_time
        })


        # --------------------------------------------------------------
        # kNN: 96 configurations per fold
        # --------------------------------------------------------------
        for distance, k, voting, retention in knn_configs:
            start = time.time()
            prediction, case_base_size = kNNAlgorithm(
                x_train, y_train, x_test,
                k=k,
                distance=distance,
                voting=voting,
                retention=retention,
                types=types
            )
            end_time = time.time() - start

            accuracy = accuracy_score(y_test, prediction)
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
                'time': end_time
            })

df_results = pd.DataFrame(results)
df_results.to_csv('results.csv', index=False)

rf = df_results[df_results['model'] == 'RF']
svm = df_results[df_results['model'] == 'svm']
knn = df_results[df_results['model'] == 'kNN']

print(rf.groupby(['dataset', 'n_estimators', 'criterion'])[['accuracy', 'time']].mean())
print(svm.groupby(['dataset', 'config'])[['accuracy', 'time']].mean())
print(knn.groupby(['dataset', 'distance', 'k'])[['accuracy', 'time']].mean())

# Final case base size per retention policy.
# While the kNN is a placeholder all four give the same number; with the
# real one, NR should leave it unchanged, AR grow it and OG shrink it.
print(knn.groupby(['dataset', 'retention'])[['case_base_size', 'fold_train_size']].mean())

svm_tuned = df_results[df_results['model'] == 'svm-tuned']
rf_tuned = df_results[df_results['model'] == 'RF-tuned']

print(svm_tuned.groupby(['dataset', 'config'])[['accuracy', 'time']].mean())
print(rf_tuned.groupby(['dataset', 'config'])[['accuracy', 'time']].mean())

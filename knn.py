"""The kNN classifier.

Kept apart from main.py so importing it does not launch an experiment.
"""

import numpy as np
import pandas as pd

from distance import get_distance
from ivdm import IVDM
from voting import VOTING


def kNNAlgorithm(x_train, y_train, x_test, y_test, k, distance, voting, retention, types):
    # Convert inputs to uniform NumPy arrays/lists
    X_tr = x_train.to_numpy() if isinstance(x_train, pd.DataFrame) else np.asarray(x_train)
    Y_tr = y_train.to_numpy() if isinstance(y_train, pd.Series) else np.asarray(y_train)
    X_te = x_test.to_numpy() if isinstance(x_test, pd.DataFrame) else np.asarray(x_test)
    Y_te = y_test.to_numpy() if isinstance(y_test, pd.Series) else np.asarray(y_test)

    # 1. Distance metric selection
    if distance == 'ivdm':
        categorical_indices = [idx for idx, t in enumerate(types) if t != 'numeric']
        n_classes = len(np.unique(Y_tr))
        ivdm_model = IVDM(n_bins=max(5, n_classes), categorical_features=categorical_indices)
        ivdm_model.fit(X_tr, Y_tr)

        # IVDM is the Euclidean distance between the mapped probability
        # vectors, so map once instead of recomputing on every comparison.
        X_tr = ivdm_model.transform(X_tr)
        X_te = ivdm_model.transform(X_te)

        def calc_dist(a, b):
            return float(np.sqrt(np.sum((a - b) ** 2)))
    else:
        dist_func = get_distance(distance)

        def calc_dist(a, b):
            return dist_func(a, b)

    vote_func = VOTING[voting]

    # 2. The case base. A list, not an array: the policies add and remove.
    x_train = [row for row in X_tr]
    y_train = list(Y_tr)

    # OG: born neutral, forgotten once goodness drops back below 0.5.
    INITIAL_GOODNESS = 0.5
    ALPHA = 0.2
    goodness = [INITIAL_GOODNESS] * len(x_train)

    # 3. Predict & update case base (NR, AR, DC, OG)
    predictions = []

    for test_pt, true_label in zip(X_te, Y_te):
        dists = [calc_dist(test_pt, train_pt) for train_pt in x_train]

        k_indices = np.argsort(dists)[:k]
        k_neighbors = [(dists[idx], y_train[idx]) for idx in k_indices]

        pred_label = vote_func(k_neighbors)
        predictions.append(pred_label)

        # Store the true label, not the predicted one, or errors spread.
        if retention == 'AR':
            x_train.append(np.asarray(test_pt))
            y_train.append(true_label)
            goodness.append(INITIAL_GOODNESS)

        elif retention == 'DC':
            if pred_label != true_label:
                x_train.append(np.asarray(test_pt))
                y_train.append(true_label)
                goodness.append(INITIAL_GOODNESS)

        elif retention == 'OG':
            # Only the retrieved neighbours update: goodness tracks use.
            to_delete = []
            for idx in k_indices:
                if y_train[idx] == true_label:
                    reward = 1.0
                else:
                    reward = 0.0
                goodness[idx] += ALPHA * (reward - goodness[idx])
                if goodness[idx] < INITIAL_GOODNESS:
                    to_delete.append(idx)

            # Highest index first, or the pending ones shift.
            # Never drop below k, or the next query cannot retrieve k.
            for idx in sorted(to_delete, reverse=True):
                if len(x_train) > k:
                    del x_train[idx]
                    del y_train[idx]
                    del goodness[idx]

    case_base_size = len(x_train)
    return predictions, case_base_size
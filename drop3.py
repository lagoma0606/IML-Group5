"""DROP3 Hybrid Instance Reduction Algorithm.

Reference:
    D. R. Wilson and T. R. Martinez, "Reduction Techniques for Instance-Based
    Learning Algorithms," Machine Learning, vol. 38, no. 3, pp. 257-286, 2000.
"""

import numpy as np
from distance import get_distance


def _compute_distance_matrix(X, metric):
    """Precompute pairwise distance matrix for efficiency."""
    n = len(X)
    dist_func = get_distance(metric)
    D = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = dist_func(X[i], X[j])
            D[i, j] = d
            D[j, i] = d
    return D


def _get_k_nearest_neighbors(D, mask, target_idx, k):
    """Find indices of the k nearest neighbors of target_idx within the subset enabled by mask."""
    valid_indices = np.where(mask)[0]
    valid_indices = valid_indices[valid_indices != target_idx]

    if len(valid_indices) == 0:
        return np.array([], dtype=int)

    dists = D[target_idx, valid_indices]
    actual_k = min(k, len(valid_indices))
    nearest_rel_idx = np.argpartition(dists, actual_k - 1)[:actual_k]
    nearest_rel_idx = nearest_rel_idx[np.argsort(dists[nearest_rel_idx])]

    return valid_indices[nearest_rel_idx]


def _enn_filter(D, y, k=3):
    """Step 1 of DROP3: ENN Noise Filtering.

    Removes any instance misclassified by its k nearest neighbors in T.
    """
    n = len(y)
    keep_mask = np.ones(n, dtype=bool)
    full_mask = np.ones(n, dtype=bool)

    for i in range(n):
        nn = _get_k_nearest_neighbors(D, full_mask, i, k)
        if len(nn) > 0:
            nn_labels = y[nn]
            classes, counts = np.unique(nn_labels, return_counts=True)
            pred_label = classes[np.argmax(counts)]
            if pred_label != y[i]:
                keep_mask[i] = False

    return keep_mask


def _find_associates(D, y, active_mask, k=3):
    """Find associates for all active instances.

    An instance 'a' is an associate of instance 'p' if 'p' is one of 'a's k-NNs.
    Returns dict: { instance_index: set_of_associate_indices }
    """
    active_indices = np.where(active_mask)[0]
    associates = {i: set() for i in active_indices}

    for a in active_indices:
        nn = _get_k_nearest_neighbors(D, active_mask, a, k)
        for p in nn:
            if p in associates:
                associates[p].add(a)

    return associates


def _classify(D, y, active_mask, target_idx, k=1):
    """Classify target_idx using current active instances in active_mask."""
    nn = _get_k_nearest_neighbors(D, active_mask, target_idx, k)
    if len(nn) == 0:
        return y[target_idx]

    nn_labels = y[nn]
    classes, counts = np.unique(nn_labels, return_counts=True)
    return classes[np.argmax(counts)]


def drop3(X, y, metric='euclidean', k=1):
    """DROP3 algorithm (Wilson & Martinez, 2000).

    Steps:
      1. ENN noise filter (k_enn=3) to remove border/noise instances.
      2. Sort remaining instances by distance to their nearest enemy (farthest first).
      3. For each instance p, remove p if accuracy on its associates does not decrease.

    Parameters
    ----------
    X : array-like of shape (n_samples, n_features)
    y : array-like of shape (n_samples,)
    metric : str, default='euclidean'
    k : int, default=1
        Number of neighbors for decision rule during reduction.

    Returns
    -------
    X_red, y_red : arrays of reduced dataset.
    """
    X_arr = np.asarray(X)
    y_arr = np.asarray(y)
    n = len(y_arr)

    # Precompute distance matrix
    D = _compute_distance_matrix(X_arr, metric)

    # 1. ENN Pass (k=3 as specified by DROP3)
    active_mask = _enn_filter(D, y_arr, k=3)

    # 2. Distance to nearest enemy calculation for sorting
    active_indices = np.where(active_mask)[0]
    dist_to_enemy = np.zeros(n)

    for i in active_indices:
        enemy_mask = active_mask & (y_arr != y_arr[i])
        enemy_indices = np.where(enemy_mask)[0]
        if len(enemy_indices) > 0:
            dist_to_enemy[i] = np.min(D[i, enemy_indices])
        else:
            dist_to_enemy[i] = np.inf

    # Sort remaining instances by distance to nearest enemy (descending order)
    sorted_active = active_indices[np.argsort(-dist_to_enemy[active_indices])]

    # 3. Iterative Removal Pass
    for p in sorted_active:
        associates_map = _find_associates(D, y_arr, active_mask, k=k)
        associates = associates_map.get(p, set())

        # No associates means no evidence either way, so p stays.
        if not associates:
            continue

        # Accuracy of associates with p present

        # Accuracy of associates with p present
        acc_with_p = sum(
            1 for a in associates if _classify(D, y_arr, active_mask, a, k=k) == y_arr[a]
        )

        # Temporarily remove p
        active_mask[p] = False

        # Accuracy of associates without p
        acc_without_p = sum(
            1 for a in associates if _classify(D, y_arr, active_mask, a, k=k) == y_arr[a]
        )

        # If removing p leads to equal or better performance on its associates, keep p removed
        if acc_without_p < acc_with_p:
            active_mask[p] = True  # Revert removal

    return X_arr[active_mask], y_arr[active_mask]


def reduce_drop3(x_train, y_train, metric='euclidean'):
    """Wrapper function matching the reduction_main interface."""
    return drop3(x_train, y_train, metric=metric, k=3)
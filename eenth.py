import numpy as np
from distance import get_distance

def wilson_prob_keep(D, yi, k=7, threshold=0.5):
    """Core EENN boolean mask generator."""
    n = len(yi)
    kk = min(k, n - 1)
    if kk < 1:
        return np.ones(n, dtype=bool)

    diag_idx = np.arange(n)
    original_diag = D[diag_idx, diag_idx].copy()
    D[diag_idx, diag_idx] = np.inf

    try:
        nn_idx = np.argpartition(D, kk - 1, axis=1)[:, :kk]
        nn_dists = np.take_along_axis(D, nn_idx, axis=1)
    finally:
        D[diag_idx, diag_idx] = original_diag

    weights = 1.0 / (1.0 + nn_dists)
    nn_labels = yi[nn_idx]
    
    n_classes = np.max(yi) + 1
    probs = np.zeros((n, n_classes), dtype=float)
    
    for i in range(kk):
        np.add.at(probs, (np.arange(n), nn_labels[:, i]), weights[:, i])

    total_weights = probs.sum(axis=1, keepdims=True)
    probs /= np.maximum(total_weights, 1e-12)

    pred_labels = np.argmax(probs, axis=1)
    max_probs = np.max(probs, axis=1)

    return (pred_labels == yi) & (max_probs > threshold)

def reduce_eennth(x_train, y_train, metric):
    """Wrapper to make EENN match the fcnn.condense signature."""
    dist_func = get_distance(metric)
    n_tr = len(x_train)
    D_train = np.zeros((n_tr, n_tr))
    
    for i in range(n_tr):
        for j in range(i + 1, n_tr):
            D_train[i, j] = D_train[j, i] = dist_func(x_train[i], x_train[j])
            
    unique_labels, y_mapped = np.unique(y_train, return_inverse=True)
    
    keep_mask = wilson_prob_keep(D_train, y_mapped, k=7, threshold=0.5)
    
    x_tr_arr = np.array(x_train)
    y_tr_arr = np.array(y_train)
    
    return x_tr_arr[keep_mask], y_tr_arr[keep_mask]
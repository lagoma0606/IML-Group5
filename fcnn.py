"""Fast Condensed Nearest Neighbor (FCNN).

Reference: F. Angiulli, "Fast Nearest Neighbor Condensation for Large Data
Sets Classification", IEEE TKDE 19(11), 1450-1464, 2007.
"""
import numpy as np
from scipy.spatial.distance import cdist


def distance_matrix(X, metric='euclidean'):
    """Full (n x n) distance matrix between the rows of X.

    metric: 'euclidean', 'manhattan', 'canberra' or 'clark'.
    IVDM is handled by the caller: map X with IVDM.transform and use 'euclidean'.
    """
    X = np.asarray(X, dtype=float)
    if metric == 'euclidean':
        return cdist(X, X, 'euclidean')
    if metric == 'manhattan':
        return cdist(X, X, 'cityblock')
    if metric == 'canberra':
        return cdist(X, X, 'canberra')
    if metric == 'clark':
        n = len(X)
        D = np.zeros((n, n))
        absX = np.abs(X)
        for i in range(n):
            den = absX[i] + absX
            diff = X[i] - X
            with np.errstate(divide='ignore', invalid='ignore'):
                term = np.where(den != 0, (diff / den) ** 2, 0.0)
            D[i] = np.sqrt(term.sum(axis=1))
        return D
    raise ValueError(f"unknown metric: {metric}")


def _distance_to_point(X, p, metric):
    """Distance from every row of X to the single point p."""
    X = np.asarray(X, dtype=float)
    p = np.asarray(p, dtype=float)
    if metric == 'euclidean':
        return np.linalg.norm(X - p, axis=1)
    if metric == 'manhattan':
        return np.abs(X - p).sum(axis=1)
    if metric == 'canberra':
        return cdist(X, p[None, :], 'canberra')[:, 0]
    if metric == 'clark':
        den = np.abs(p) + np.abs(X)
        with np.errstate(divide='ignore', invalid='ignore'):
            term = np.where(den != 0, ((p - X) / den) ** 2, 0.0)
        return np.sqrt(term.sum(axis=1))
    raise ValueError(f"unknown metric: {metric}")


def _centroid_seeds(X, y, metric):
    """Centroids(T): For each class, the training point nearest to the geometric centre (mean vector)
    of the class. Ties go to the lowest index (lexicographically smallest point once the rows are sorted)."""
    seeds = []
    for c in np.unique(y):
        members = np.where(y == c)[0]
        centroid = X[members].mean(axis=0)
        d = _distance_to_point(X[members], centroid, metric)
        seeds.append(int(members[np.argmin(d)]))
    return seeds


def _fcnn_sorted(X, y, D, metric):
    """FCNN1 assumes the rows are already in lexicographic order, so
    that 'lowest index' means 'lexicographically smallest point'.

    FCNN condensation. Returns the sorted indices of the kept instances.

    X: training instances, y: their classes, D: (n x n) distance matrix
    between them, metric: the metric that produced D, used only to locate
    the point nearest to each class centroid.
    """
    y = np.asarray(y)
    n = len(y)

    X = np.asarray(X, dtype=float)

    # Initial prototypes: the training point nearest to each class centroid
    S = _centroid_seeds(X, y, metric)

    in_S = np.zeros(n, dtype=bool)
    in_S[S] = True

    # Nearest prototype of every training point
    proto = np.sort(S)
    nn = proto[np.argmin(D[:, proto], axis=1)]
    nn_dist = D[np.arange(n), nn]

    while True:
        enemy = y != y[nn]

        # Nearest enemy of each prototype (only inside its own cell)
        additions = set()
        for p in S:
            cell = np.where((nn == p) & enemy)[0]
            if cell.size:
                additions.add(int(cell[np.argmin(D[p, cell])]))
        additions = sorted(q for q in additions if not in_S[q])

        # No new prototypes: S is consistent with the training set
        if not additions:
            break

        S.extend(additions)
        in_S[additions] = True

        # The new prototypes belong to themselves
        add = np.asarray(additions)
        nn[add] = add
        nn_dist[add] = 0.0

        # Only the new prototypes can have become the nearest one
        Dn = D[:, add]
        j = np.argmin(Dn, axis=1)
        d = Dn[np.arange(n), j]

        closer = (d < nn_dist) | ((d == nn_dist) & (add[j] < nn))
        closer &= ~in_S
        nn[closer] = add[j[closer]]
        nn_dist[closer] = d[closer]

    return np.array(sorted(S))

def fcnn(X, y, D, metric='euclidean'):
    """FCNN1 rule (Angiulli 2007, Fig. 5). Returns the sorted indices of the
    kept instances, referring to the original row order of X.

    k=1 gives the basic rule.
    k>1 uses the extension of Section 3.3: S is consistent with the k-NN rule.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    keys = [y] + [X[:, j] for j in range(X.shape[1] - 1, -1, -1)]
    order = np.lexsort(keys)
    Xs, ys, Ds = X[order], y[order], D[np.ix_(order, order)]

    idx = _fcnn_sorted(Xs, ys, Ds, metric)

    return np.sort(order[idx])


def condense(X, y, metric='euclidean'):
    """Convenience wrapper: returns the reduced (X, y)."""
    X = np.asarray(X)
    y = np.asarray(y)
    idx = fcnn(X, y, distance_matrix(X, metric), metric)
    return X[idx], y[idx]
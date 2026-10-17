"""Condensation: Fast Condensed Nearest Neighbor (FCNN).

Reference: F. Angiulli, "Fast Nearest Neighbor Condensation for Large Data
Sets Classification", IEEE TKDE 19(11), 1450-1464, 2007.

Public interface (shared by every reduction module of the project):

    idx = fcnn(X, y, D, metric, k) # indices of the instances kept
    X_red, y_red = condense(X, y, metric='euclidean', k=1)

`D` is the (n x n) matrix of distances between the training instances;
`X` is only needed to compute the class centroids. For IVDM pass the
instances mapped with IVDM.transform and metric='euclidean'.
"""
import numpy as np
from scipy.spatial.distance import cdist


def distance_matrix(X, metric='euclidean'):
    """Full (n x n) distance matrix between the rows of X.

    metric: 'euclidean', 'manhattan', 'canberra' or 'clark'. IVDM is
    handled by the caller: map X with IVDM.transform and use 'euclidean'.
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
    """Centroids(T): for each class, the training point nearest to the
    geometric centre (mean vector) of the class. Ties go to the lowest
    index (lexicographically smallest point once the rows are sorted)."""
    seeds = []
    for c in np.unique(y):
        members = np.where(y == c)[0]
        centroid = X[members].mean(axis=0)
        d = _distance_to_point(X[members], centroid, metric)
        seeds.append(int(members[np.argmin(d)]))
    return seeds


def _fcnn_sorted(X, y, D, metric):
    """FCNN1 core. Assumes the rows are already in lexicographic order, so
    that 'lowest index' means 'lexicographically smallest point'.

    FCNN condensation. Returns the sorted indices of the kept instances.

    X: training instances, y: their classes, D: (n x n) distance matrix
    between them (see distance_matrix), metric: the metric that produced D,
    used only to locate the point nearest to each class centroid.

    1. S starts with one prototype per class: the training point nearest to
       the centroid (mean vector) of the class, as in Angiulli's
       centroids(T).
    2. Every training point is assigned to its nearest prototype in S.
       The "enemies" of a prototype p are the points assigned to p whose
       class differs from p's.
    3. For each prototype that has enemies, its nearest enemy is added to S.
    4. Repeat until no prototype has enemies left, i.e. S classifies the
       whole training set correctly with 1-NN.

    Ties (equal distances) are resolved in favour of the lexicographically
    smallest point, as in the Appendix of the paper, which is what makes
    the result independent of the order of the training set.
    """
    y = np.asarray(y)
    n = len(y)

    X = np.asarray(X, dtype=float)

    # 1. Initial prototypes: the training point nearest to each class centroid
    S = _centroid_seeds(X, y, metric)

    in_S = np.zeros(n, dtype=bool)
    in_S[S] = True

    # 2. Nearest prototype of every training point
    proto = np.sort(S)
    nn = proto[np.argmin(D[:, proto], axis=1)]
    nn_dist = D[np.arange(n), nn]

    while True:
        enemy = y != y[nn]

        # 3. Nearest enemy of each prototype (only inside its own cell)
        additions = set()
        for p in S:
            cell = np.where((nn == p) & enemy)[0]
            if cell.size:
                additions.add(int(cell[np.argmin(D[p, cell])]))
        additions = sorted(q for q in additions if not in_S[q])

        # 4. No new prototypes: S is consistent with the training set
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
        # Strictly closer, or tied but lexicographically smaller (paper,
        # Appendix, point b). Points already in S stay their own neighbour.
        closer = (d < nn_dist) | ((d == nn_dist) & (add[j] < nn))
        closer &= ~in_S
        nn[closer] = add[j[closer]]
        nn_dist[closer] = d[closer]

    return np.array(sorted(S))


def _fcnn_k_sorted(X, y, D, metric, k):
    """k-NN extension of FCNN1 (Angiulli 2007, Section 3.3). Rows must be in
    lexicographic order, as for _fcnn_sorted.

    S is built until it is k-training-set-consistent: every point of T - S
    is classified correctly by S through the k-NN rule (plain majority
    vote). The Voronoi cells are still those of the 1-NN rule, but the
    enemies of a prototype p are now the points q of its cell (p excluded)
    whose class differs from the k-NN prediction of S:

        Voren_k(p, S, T) = { q in Vor(p, S, T) - {p} : l(q) != NN_k(q, S) }

    and the representative added for p is, as in FCNN1, the member of
    Voren_k closest to p. While S has fewer than k points, all of S votes.
    Ties between neighbours at equal distance go to the lowest index, and
    ties between classes in the vote to the smallest label, so the result
    is order independent as in the 1-NN case.
    """
    y = np.asarray(y)
    n = len(y)
    X = np.asarray(X, dtype=float)
    classes = np.unique(y)

    S = _centroid_seeds(X, y, metric)
    in_S = np.zeros(n, dtype=bool)
    in_S[S] = True

    while True:
        proto = np.sort(S)
        kk = min(k, len(proto))

        # Neighbours of every training point inside S (stable sort: ties go
        # to the lowest index because proto is ascending)
        order = np.argsort(D[:, proto], axis=1, kind='stable')[:, :kk]
        neigh = proto[order]                      # (n, kk) training indices
        nn1 = neigh[:, 0]                         # 1-NN: defines the cells

        # k-NN vote: class with most members, smallest label on ties
        votes = np.stack([(y[neigh] == c).sum(axis=1) for c in classes], axis=1)
        pred = classes[np.argmax(votes, axis=1)]
        enemy = (pred != y) & ~in_S               # misclassified, not in S

        additions = set()
        for p in proto:
            cell = np.where((nn1 == p) & enemy)[0]
            if cell.size:
                additions.add(int(cell[np.argmin(D[p, cell])]))
        additions = sorted(q for q in additions if not in_S[q])

        if not additions:
            break
        S.extend(additions)
        in_S[additions] = True

    return np.array(sorted(S))


def fcnn(X, y, D, metric='euclidean', k=1):
    """FCNN1 rule (Angiulli 2007, Fig. 5). Returns the sorted indices of the
    kept instances, referring to the original row order of X.

    k=1 gives the basic rule (S consistent with the 1-NN rule). k>1 uses
    the extension of Section 3.3: S is consistent with the k-NN rule.

    The rows are first put in lexicographic order (feature values from left
    to right, class label last). Every tie in the algorithm (nearest
    prototype, nearest enemy, class centroid) is then solved by taking the
    lowest index, i.e. the lexicographically smallest point, as the paper
    prescribes. The subset returned is therefore the same whatever the
    order of the training set.
    """
    X = np.asarray(X, dtype=float)
    y = np.asarray(y)
    keys = [y] + [X[:, j] for j in range(X.shape[1] - 1, -1, -1)]
    order = np.lexsort(keys)          # last key is the primary one
    Xs, ys, Ds = X[order], y[order], D[np.ix_(order, order)]
    if k == 1:
        idx = _fcnn_sorted(Xs, ys, Ds, metric)
    else:
        idx = _fcnn_k_sorted(Xs, ys, Ds, metric, k)
    return np.sort(order[idx])


def condense(X, y, metric='euclidean', k=1):
    """Convenience wrapper: returns the reduced (X, y)."""
    X = np.asarray(X)
    y = np.asarray(y)
    idx = fcnn(X, y, distance_matrix(X, metric), metric, k)
    return X[idx], y[idx]
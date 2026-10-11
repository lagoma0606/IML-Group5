"""Voting schemes for the kNN.

Both functions have the same signature, so the kNN core can swap them:

    vote(neighbors) -> predicted class

where `neighbors` is a list of (distance, class_label) tuples, one per
retrieved neighbour (the K nearest ones). The list does not need to be sorted.
"""
import math

EPS = 1e-12  # distances <= EPS are treated as "exactly zero"


def _break_tie(tied_labels, neighbors):
    """Tie-breaking rule (deterministic, always gives a single winner).

    1. Among the tied classes, the winner is the one that owns the nearest neighbour.
    2. If neighbours of different tied classes are at the same minimum distance,
       the smallest class label wins.
    """
    nearest = min(d for d, label in neighbors if label in tied_labels)
    candidates = [label for d, label in neighbors if label in tied_labels and d == nearest]
    return min(candidates)


def _winner(scores, neighbors):
    """Class with the highest score; ties are solved with _break_tie."""
    top = max(scores.values())
    tied = {label for label, s in scores.items() if math.isclose(s, top, rel_tol=1e-9, abs_tol=0.0)}
    if len(tied) == 1:
        return tied.pop()
    return _break_tie(tied, neighbors)


def majority_vote(neighbors):
    """Majority class among the neighbours (each neighbour counts as one vote)."""
    scores = {}
    for _, label in neighbors:
        scores[label] = scores.get(label, 0) + 1
    return _winner(scores, neighbors)


def inverse_distance_vote(neighbors, p=1):
    """Inverse distance weighted vote: each neighbour votes with weight 1 / distance**p.

    Zero-distance case: if one or more neighbours are at distance 0 (identical to
    the query), their weight would be infinite, so only those neighbours vote
    (one vote each, i.e. majority among them).
    """
    zero = [(d, label) for d, label in neighbors if d <= EPS]
    if zero:
        return majority_vote(zero)

    scores = {}
    for d, label in neighbors:
        scores[label] = scores.get(label, 0.0) + 1.0 / d ** p
    return _winner(scores, neighbors)


# Used to iterate over configurations.
VOTING = {
    "majority": majority_vote,
    "inverse_distance": inverse_distance_vote,
}
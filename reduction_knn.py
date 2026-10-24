"""reduction_kNNAlgorithm: the kNN with a training set reduction step before it.

The reduction techniques live in their own modules (fcnn.py, ennth.py,
drop3.py) and the classifier in knn.py. This module only chains them:
first reduce the TrainMatrix, then classify the TestMatrix with the kNN.
"""

from fcnn import condense
from ennth import reduce_ennth
from drop3 import reduce_drop3
from knn import kNNAlgorithm

# One technique of each family required by the assignment.
# 'none' keeps the whole training set, so the unreduced kNN can be run
# through exactly the same function.
REDUCTIONS = {
    'FCNN': condense,       # condensation
    'ENNTh': reduce_ennth,  # edition
    'DROP3': reduce_drop3,  # hybrid
}

# Distances the reduction techniques can compute. IVDM is not included:
# it needs a model fitted on the training set, which the reduction
# modules do not build.
SUPPORTED_DISTANCES = ['euclidean', 'manhattan', 'clark', 'canberra']


def reduce_training_set(x_train, y_train, reduction, distance):
    """Apply one reduction technique to the training set.

    reduction: 'FCNN', 'ENNTh', 'DROP3' or 'none'.
    distance: the metric the technique uses to find neighbours. It should be
    the same one the kNN uses afterwards, so that the instances kept are the
    ones that matter for that metric.

    Returns (x_reduced, y_reduced).
    """
    if reduction == 'none':
        return x_train, y_train
    if reduction not in REDUCTIONS:
        raise ValueError(f"unknown reduction: {reduction}. "
                         f"Valid values: {list(REDUCTIONS)} or 'none'")
    if distance not in SUPPORTED_DISTANCES:
        raise ValueError(f"the reduction techniques do not support the "
                         f"distance '{distance}'. Valid values: {SUPPORTED_DISTANCES}")
    return REDUCTIONS[reduction](x_train, y_train, metric=distance)


def reduction_kNNAlgorithm(x_train, y_train, x_test, y_test, k, distance,
                           voting, retention, types, reduction):
    """Reduce the TrainMatrix, then classify the TestMatrix with the kNN.

    Takes the same parameters as kNNAlgorithm plus `reduction`
    ('FCNN', 'ENNTh', 'DROP3' or 'none'). The reduction uses the same
    distance as the kNN.

    Returns (predictions, reduced_size, case_base_size):
      predictions    predicted class of each test instance
      reduced_size   training instances kept by the reduction
      case_base_size instances left at the end of classification, after
                     the retention policy has added or forgotten cases

    Storage as defined in the assignment is
    100 * reduced_size / len(x_train).
    """
    x_red, y_red = reduce_training_set(x_train, y_train, reduction, distance)

    predictions, case_base_size = kNNAlgorithm(
        x_red, y_red, x_test, y_test,
        k=k, distance=distance, voting=voting, retention=retention,
        types=types,
    )
    return predictions, len(y_red), case_base_size
import numpy as np

def _minkowski_distance(x: np.ndarray, y: np.ndarray, r=1) -> float:
    """
    Calculate the minkowski distance distance between two vectors.
        r=1: Manhattan distance
        r=2: Euclidean distance
    """
    if r < 1:
        raise ValueError("r must be >= 1")

    return float(np.sum(np.abs(x - y) ** r) ** (1 / r))
    


def _clark_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the Clark distance between two vectors."""
    denominator = x + y

    # Avoid division by zero when both features are zero.
    mask = denominator != 0

    if not np.any(mask):
        return 0.0

    return float(
        np.sqrt(
            np.sum(((x[mask] - y[mask]) / denominator[mask]) ** 2)
        )
    )


def _camberra_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Calculate the Canberra distance between two vectors."""
    denominator = np.abs(x) + np.abs(y)

    # Terms with a zero denominator contribute zero.
    mask = denominator != 0

    if not np.any(mask):
        return 0.0

    return float(
        np.sum(
            np.abs(x[mask] - y[mask]) / denominator[mask]
        )
    )


_distance_methods = {
    "minkowski": _minkowski_distance,
    "clark": _clark_distance,
    "camberra": _camberra_distance,
    "euclidean": lambda x, y: _minkowski_distance(x, y, 2),
    "manhattan": lambda x, y: _minkowski_distance(x, y)
}


def get_distance(distance_name: str):
    """Return a distance function by name, or None if it does not exist."""
    return _distance_methods.get(distance_name)
import numpy as np


class IVDM:
    """
    Interpolated Value Difference Metric (IVDM).

    Parameters
    ----------
    n_bins : int, default=5
        Number of equal-width bins for continuous features.
        If None, uses max(5, number_of_classes).

    categorical_features : tuple or list, default=()
        Indices of categorical features.
        All other features are treated as continuous.

    Attributes
    ----------
    classes_ : ndarray
        Unique target classes observed during fitting.
    """

    def __init__(self, n_bins=None, categorical_features=()):
        if n_bins is not None and n_bins < 1:
            raise ValueError("n_bins must be at least 1")

        self.n_bins = n_bins or max(5, len(self.classes_))
        self.categorical_features = set(categorical_features)
        

    def _input_validation(self, X, y):
        """Check that the input data is valid for fitting."""
        
        if X.ndim != 2:
            raise ValueError("X must be a 2D array")
        
        if y.ndim != 1:
            raise ValueError("y must be a 1D array")

        if len(X) != len(y):
            raise ValueError("X and y must contain the same number of samples")

        if len(X) == 0:
            raise ValueError("Training data cannot be empty")

        if len(self.categorical_features) > 0:
            if min(self.categorical_features) < 0 or max(
                self.categorical_features
            ) >= X.shape[1]:
                raise ValueError("Categorical feature index out of range")

        if not np.all(np.isfinite(X[:, [
            i for i in range(X.shape[1])
            if i not in self.categorical_features
        ]].astype(float))):
            raise ValueError("Continuous features must be finite")
        

    def fit(self, X, y):
        """Learn class-conditional probabilities from training data."""
        X = np.asarray(X)
        y = np.asarray(y)
        
        self._input_validation(X, y)

        self.n_features_in_ = X.shape[1]
        self.classes_, y_encoded = np.unique(y, return_inverse=True)
        
        self.feature_models_ = []

        for j in range(self.n_features_in_):
            if j in self.categorical_features:
                model = self._fit_categorical(X[:, j], y_encoded)
            else:
                model = self._fit_continuous(
                    X[:, j].astype(float),
                    y_encoded,
                )

            self.feature_models_.append(model)

        return self

    def _fit_categorical(self, feature, y_encoded):
        """Estimate P(class | categorical feature value)."""
        probabilities = {}

        for value in np.unique(feature):
            mask = feature == value

            counts = np.bincount(
                y_encoded[mask],
                minlength=len(self.classes_),
            )

            probabilities[value] = counts / counts.sum()

        return {
            "type": "categorical",
            "probabilities": probabilities,
        }

    def _fit_continuous(self, feature, y_encoded):
        """Estimate class probabilities at equal-width bin midpoints."""
        minimum = feature.min()
        maximum = feature.max()

        if minimum == maximum:
            counts = np.bincount(
                y_encoded,
                minlength=len(self.classes_),
            )

            return {
                "type": "constant",
                "value": minimum,
                "probability": counts / counts.sum(),
            }

        n_bins = self.n_bins_
        edges = np.linspace(minimum, maximum, n_bins + 1)
        width = (maximum - minimum) / n_bins

        bin_indices = np.floor((feature - minimum) / width).astype(int)
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        counts = np.zeros((n_bins, len(self.classes_)), dtype=float)

        for bin_index, class_index in zip(bin_indices, y_encoded):
            counts[bin_index, class_index] += 1

        totals = counts.sum(axis=1, keepdims=True)

        probabilities = np.divide(
            counts,
            totals,
            out=np.zeros_like(counts),
            where=totals != 0,
        )

        midpoints = (edges[:-1] + edges[1:]) / 2

        # Zero-probability points beyond the first and last bins
        # allow interpolation at the edges of the observed range.
        midpoints = np.concatenate((
            [midpoints[0] - width],
            midpoints,
            [midpoints[-1] + width],
        ))

        probabilities = np.vstack((
            np.zeros((1, len(self.classes_))),
            probabilities,
            np.zeros((1, len(self.classes_))),
        ))

        return {
            "type": "continuous",
            "midpoints": midpoints,
            "probabilities": probabilities,
        }

    def _get_probability(self, value, model):
        """Estimate the class-probability vector for one feature value."""
        if model["type"] == "categorical":
            return model["probabilities"].get(
                value,
                np.zeros(len(self.classes_)),
            )

        if model["type"] == "constant":
            return model["probability"]

        midpoints = model["midpoints"]
        probabilities = model["probabilities"]

        # Locate the neighboring midpoints.
        right = np.searchsorted(midpoints, value, side="right")
        right = np.clip(right, 1, len(midpoints) - 1)
        left = right - 1

        x_left = midpoints[left]
        x_right = midpoints[right]

        p_left = probabilities[left]
        p_right = probabilities[right]

        # Linear interpolation between neighboring probabilities.
        weight = (value - x_left) / (x_right - x_left)

        return p_left + weight * (p_right - p_left)

    def get_distance(self, x, y):
        """Calculate the IVDM distance between two feature vectors."""
        if not hasattr(self, "feature_models_"):
            raise RuntimeError("Call fit(X, y) before computing distances")

        x = np.asarray(x)
        y = np.asarray(y)

        if x.ndim != 1 or y.ndim != 1:
            raise ValueError("x and y must be one-dimensional vectors")

        if len(x) != self.n_features_in_ or len(y) != self.n_features_in_:
            raise ValueError("Feature vectors have an incorrect number of features")

        distance = 0.0

        for j, model in enumerate(self.feature_models_):
            p_x = self._get_probability(x[j], model)
            p_y = self._get_probability(y[j], model)

            # Squared L2 distance between class-probability vectors.
            distance += np.sum((p_x - p_y) ** 2)

        return float(np.sqrt(distance))
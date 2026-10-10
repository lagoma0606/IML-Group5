import pandas as pd

def to_text(value):
    """loadarff returns nominal values as bytes; normalise them to str."""
    if isinstance(value, bytes):
        return value.decode('utf-8')
    return str(value)


class Preprocessor:

    def __init__(self, zero_as_missing=None):
        """
        zero_as_missing: names of columns where a 0 means 'not measured'
        rather than a real value."""

        # 'or []' leaves an empty list when nothing is passed, so the loops
        # can iterate over it without any prior check
        self.zero_as_missing = zero_as_missing or []

        # Filled in fit, consulted in transform
        self.numeric_columns = []
        self.categorical_columns = []
        self.medians = {}
        self.min_max = {}

        # Label -> integer maps. Learned once so that train and test use the
        # same correspondence; if each set assigned numbers on its own they
        # could come out inverted and the predictions would look wrong
        # without being wrong.
        self.class_map = {}
        self.categorical_maps = {}

    def fit(self, df, meta):
        """
        Learns from the training fold. Does not modify df.

        Stores: which columns are numeric, which are categorical, the median
        of the columns holding false zeros, the min/max of every numeric
        column, and the label maps.
        """
        names = meta.names()
        types = meta.types()

        # The type is read from meta rather than inferred from the values:
        # on Mx the attributes are nominal but their values look like numbers
        # (0 and 1), so inspecting the data would lead to the wrong answer.
        # 'class' is excluded because it is the target, not an attribute.
        for n, t in zip(names, types):
            if n == 'class':
                continue
            if t == 'numeric':
                self.numeric_columns.append(n)
            else:
                self.categorical_columns.append(n)

        # Zeros are excluded from the calculation: including them would drag
        # the median down.
        for col in self.zero_as_missing:
            no_zero = df[col][df[col] != 0]
            self.medians[col] = no_zero.median()

        # Working copy: fit must only learn, never modify what it receives.
        df_train = df.copy()
        for col in self.zero_as_missing:
            df_train[col] = df_train[col].replace(0, self.medians[col])

        # min/max is computed AFTER imputing. Done before, the false zeros
        # would define the minimum.
        for col in self.numeric_columns:
            self.min_max[col] = (df_train[col].min(), df_train[col].max())

        # Class labels come from the .arff header, not from the data, so the
        # map is complete even if a given fold lacks one of the classes.
        # meta['class'] returns (type, possible_values).
        _, class_values = meta['class']
        for i, v in enumerate(class_values):
            self.class_map[to_text(v)] = i

        # Same for the categorical attributes of Mx. They are mapped to
        # integers so scikit-learn can consume them, while the attribute
        # types travel separately so that HEOM and IVDM can still treat them
        # as categories rather than quantities.
        for col in self.categorical_columns:
            _, col_values = meta[col]
            self.categorical_maps[col] = {
                to_text(v): i for i, v in enumerate(col_values)
            }

    def transform(self, df):
        """
        Applies the statistics learned in fit to the given DataFrame.

        Computes nothing new: everything comes from self. That is why the
        test set is transformed with the training numbers, which is what
        keeps both sets on the same scale and makes the accuracy measurement
        honest.

        Returns (X, y): attributes and target separated, which is the format
        the classifiers expect.
        """
        df_out = df.copy()

        # Fills the gaps with the median learned from the training fold
        for col in self.zero_as_missing:
            df_out[col] = df_out[col].replace(0, self.medians[col])

        # Min-max normalisation to [0,1]:
        # Needed because kNN and SVM add up differences across all
        # columns.Random Forest does not need it, but it is applied
        # anyway so that all three algorithms receive identical data and the
        # comparison stays clean.
        for col in self.numeric_columns:
            minimum = self.min_max[col][0]
            maximum = self.min_max[col][1]
            df_out[col] = (df_out[col] - minimum) / (maximum - minimum)

        # Nominal attributes arrive from loadarff as bytes (b'1'), not text.
        # Without decoding, any comparison against a normal string fails
        # silently. They are then mapped to the integer codes learned in fit.
        for col in self.categorical_columns:
            df_out[col] = (df_out[col]
                           .map(to_text)
                           .map(self.categorical_maps[col]))

        y = df_out['class'].map(to_text).map(self.class_map)
        x = df_out.drop(columns=['class'])

        # Test values may fall outside [0,1] when they exceed the range seen
        # in training. That is expected and correct.
        return x, y

    

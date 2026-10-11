from scipy.io import arff
import pandas as pd
from preprocessor import Preprocessor

# Pre-dataset configuration: columns where a 0 means 'not measured'.
# It lives here so main.py does not need to know anything about the
# datasets themselves.
ZERO_AS_MISSING = {
    'pima-indians': ['plas', 'pres', 'skin', 'insu', 'mass'],
    'mx': [],
}


def load_fold(dataset, i):
    """
    Returns fold i of the given dataset, already preprocessed and ready
    to classify.

    This function is the only one that knows how the data is preprocessed.
    If the implementation changes, only its body changes: the rest of the
    project does not notice.
    """
    train_path = f"data/{dataset}/{dataset}.fold.{i:06d}.train.arff"
    test_path = f"data/{dataset}/{dataset}.fold.{i:06d}.test.arff"

    data_train, meta_train = arff.loadarff(train_path)
    data_test, meta_test = arff.loadarff(test_path)

    df_train = pd.DataFrame(data_train)
    df_test = pd.DataFrame(data_test)

    # A fresh Preprocessor per fold: each training set has its own
    # medians and min/max. Reusing the object would let one fold's
    # statistics contaminate the next.
    p = Preprocessor(ZERO_AS_MISSING[dataset])

    # Fit on the training set only: this is what prevents data leakage.
    p.fit(df_train, meta_train)

    # Transform both with the same statistics learned from the training
    # set. The test set never gets a fit of its own; if it did, it would
    # be normalised on its own scale and stop being comparable.
    x_train, y_train = p.transform(df_train)
    x_test, y_test = p.transform(df_test)
    # Attribute types, without the class, in the same order as the
    # columns of x_train. HEOM and IVDM need them to tell numeric from
    # categorical attributes.
    types = []
    for n, t in zip(meta_train.names(), meta_train.types()):
        if n != 'class':
            types.append(t)

    return x_train, y_train, x_test, y_test, types
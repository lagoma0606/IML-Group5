"""SVM classifiers.
svmAlgorithm: SVM with a fixed configuration (the two kernels compared).
svmTunedAlgorithm: RBF SVM whose C and gamma are chosen by grid search.
"""

from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC

# C and gamma interact so they are searched together on a logarithmic grid.
PARAM_GRID_SVM = {
    'C': [0.1, 1, 10, 100],
    'gamma': [0.001, 0.01, 0.1, 1],
}


def svmAlgorithm(x_train, y_train, x_test, kernel='rbf', C=1.0, gamma='scale'):
    """Train an SVM on TrainMatrix and classify every row of TestMatrix.

    kernel: 'linear' or 'rbf'.
    C, gamma: scikit-learn defaults, so that both kernels are compared
    under the same conditions. gamma is ignored by the linear kernel.

    Returns the predicted class of each test instance.
    """
    model = SVC(kernel=kernel, C=C, gamma=gamma)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def svmTunedAlgorithm(x_train, y_train, x_test, param_grid=None, cv=5,
                      random_state=42):
    """RBF SVM with C and gamma tuned by grid search on the training set.

    The cross validation runs inside TrainMatrix only, so the test set
    takes no part in choosing the parameters.

    Returns (predictions, best_params).
    """
    if param_grid is None:
        param_grid = PARAM_GRID_SVM
    grid = GridSearchCV(SVC(kernel='rbf', random_state=random_state),
                        param_grid, cv=cv, scoring='accuracy')
    grid.fit(x_train, y_train)
    return grid.predict(x_test), grid.best_params_
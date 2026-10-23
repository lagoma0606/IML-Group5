"""Random Forest classifiers.
rfAlgorithm: Random Forest with fixed n_estimators and criterion.
rfTunedAlgorithm: Random Forest tuned by grid search.
"""

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

PARAM_GRID_RF = {
    'n_estimators': [100, 300],
    'max_depth': [None, 5, 10],
    'max_features': ['sqrt', 'log2', None],
}


def rfAlgorithm(x_train, y_train, x_test, n_estimators=100, criterion='gini',
                random_state=42):
    """Train a Random Forest on TrainMatrix and classify every row of TestMatrix.

    n_estimators: number of trees.
    criterion: 'gini' or 'entropy', the impurity measure used to split.
    random_state: fixed for reproducibility.

    Returns the predicted class of each test instance.
    """
    model = RandomForestClassifier(n_estimators=n_estimators,
                                   criterion=criterion,
                                   random_state=random_state)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def rfTunedAlgorithm(x_train, y_train, x_test, param_grid=None, cv=5,
                     random_state=42):
    """Random Forest (gini) tuned by grid search on the training set.

    Returns (predictions, best_params).
    """
    if param_grid is None:
        param_grid = PARAM_GRID_RF
    grid = GridSearchCV(RandomForestClassifier(criterion='gini',
                                               random_state=random_state),
                        param_grid, cv=cv, scoring='accuracy', n_jobs=-1)
    grid.fit(x_train, y_train)
    return grid.predict(x_test), grid.best_params_
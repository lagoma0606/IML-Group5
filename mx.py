# Packages we will use
import numpy as np
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import scale
from sklearn.preprocessing import LabelEncoder

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# mx dataset: 11 binary nominal features (a00-a10) and a binary class.
# There are no missing values and all the values are 0 or 1, so we do not need imputation or scaling.
# We only convert the values from bytes (b'0', b'1') to integers.
folder_mx = "mx"
datasets_mx = {}
for i in range(10):
    for kind in ["train", "test"]:
        name = f"mx.fold.{i:06d}.{kind}"
        data, meta = arff.loadarff(f"{folder_mx}/{name}.arff")
        df = pd.DataFrame(data)
        datasets_mx[name] = df.apply(lambda col: col.str.decode("utf-8")).astype(int)

# Now we train an SVM (Support Vector Machine) with the preprocessed data.
# Each fold has its own train and test set: we train with the train set and evaluate with the test set.
# C and gamma interact with each other, so we tune them together using GridSearchCV.
# The search uses cross-validation (cv=5) only inside the train set of each fold, so the test set is not used to
# choose the hyperparameters.

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [0.01, 0.1, 1, 10],
}

accuracies = []
for i in range(10):
    base = f"mx.fold.{i:06d}"
    train = datasets_mx[base + ".train"]
    test = datasets_mx[base + ".test"]

    X_cols = [c for c in train.columns if c != "class"]
    X_train, y_train = train[X_cols], train["class"]
    X_test, y_test = test[X_cols], test["class"]

    grid = GridSearchCV(SVC(kernel="rbf", random_state=42), param_grid, cv=5, scoring="accuracy")
    grid.fit(X_train, y_train)           # tries the different combinations using only the train set

    y_pred = grid.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    accuracies.append(acc)
    print(f"Fold {i}: best params = {grid.best_params_}, test accuracy = {acc:.4f}")

print(f"Mean accuracy: {np.mean(accuracies):.4f} (std {np.std(accuracies):.4f})")
# The SVM reaches an accuracy of 1.0000 in all the folds. This is consistent with the nature of the dataset: the 11-bit
# multiplexer is a deterministic function without noise, and each train set contains about 90% of the 2048 possible
# input combinations, so the rbf kernel can generalize to the remaining ones. The best parameters (C=1, gamma=1)
# are the same in every fold and lie inside the grid, so the grid is sufficient.

# Now we train a Random Forest (RF) with the preprocessed data.
# As with the SVM, we train with the train set of each fold and evaluate with the test set.
# The hyperparameters n_estimators, max_depth and max_features are tuned with GridSearchCV (cv=5) only inside the train
# set of each fold, so the test set is not used to choose them.

param_grid_rf = {
    "n_estimators": [100, 300],
    "max_depth": [None, 5, 10],
    "max_features": ["sqrt", "log2", None],
}

rf_accuracies = []
importances_per_fold = []
for i in range(10):
    base = f"mx.fold.{i:06d}"
    train = datasets_mx[base + ".train"]
    test = datasets_mx[base + ".test"]

    X_cols = [c for c in train.columns if c != "class"]
    X_train, y_train = train[X_cols], train["class"]
    X_test, y_test = test[X_cols], test["class"]

    grid_rf = GridSearchCV(
        RandomForestClassifier(criterion="gini", random_state=42),
        param_grid_rf, cv=5, scoring="accuracy", n_jobs=-1)
    grid_rf.fit(X_train, y_train)          # tries the 18 combinations using only the train set

    y_pred = grid_rf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    rf_accuracies.append(acc)

    # Feature importances of the best model of this fold
    importances_per_fold.append(grid_rf.best_estimator_.feature_importances_)
    print(f"Fold {i}: best params = {grid_rf.best_params_}, test accuracy = {acc:.4f}")

print(f"RF - mean accuracy: {np.mean(rf_accuracies):.4f} (std {np.std(rf_accuracies):.4f})")

# Mean importance of each feature over the 10 folds (step 5 of the workflow)
mean_importances = pd.Series(np.mean(importances_per_fold, axis=0), index=X_cols)
print(mean_importances.sort_values(ascending=False).round(3))

# The RF also reaches an accuracy of 1.0000 in all the folds, so SVM and RF cannot be
# distinguished with this metric. The feature importances show that the RF has found the
# structure of the 11-bit multiplexer: the three address bits (a00, a01, a02) have an importance
# of 0.169 each (about 51% of the total), while each of the eight data bits (a03-a10) has about 0.06.
# The best parameters were the same in every fold (max_depth=None, max_features=None): the trees
# need to see all the features and grow without limit to learn the function.
# n_estimators=100 was probably selected because of a tie with 300 (GridSearchCV keeps the first one).

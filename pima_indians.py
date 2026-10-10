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

folder = "pima-indians"

archives = [
    "pima-indians.fold.000000.test.arff",
    "pima-indians.fold.000000.train.arff",
    "pima-indians.fold.000001.test.arff",
    "pima-indians.fold.000001.train.arff",
    "pima-indians.fold.000002.test.arff",
    "pima-indians.fold.000002.train.arff",
    "pima-indians.fold.000003.test.arff",
    "pima-indians.fold.000003.train.arff",
    "pima-indians.fold.000004.test.arff",
    "pima-indians.fold.000004.train.arff",
    "pima-indians.fold.000005.test.arff",
    "pima-indians.fold.000005.train.arff",
    "pima-indians.fold.000006.test.arff",
    "pima-indians.fold.000006.train.arff",
    "pima-indians.fold.000007.test.arff",
    "pima-indians.fold.000007.train.arff",
    "pima-indians.fold.000008.test.arff",
    "pima-indians.fold.000008.train.arff",
    "pima-indians.fold.000009.test.arff",
    "pima-indians.fold.000009.train.arff",
]

datasets = {}
for archive in archives:
    data, meta = arff.loadarff(folder + "/" + archive)
    df = pd.DataFrame(data)
    datasets[archive.replace(".arff", "")] = df

for name, df in datasets.items():
    print(name, df.shape, df.dtypes)
# Each dataset has 9 features: 8 numeric features (float64) and the class label (object).

# There are no explicit missing values (NaN), but columns such as plas, pres, skin, insu and mass contain zeros,
# which are physiologically impossible and likely encode missing data. We treat them as missing.
cols_cero = ["plas", "pres", "skin", "insu", "mass"]
for name, df in datasets.items():
    print(name, (df[cols_cero] == 0).sum().to_dict())

# We are going to replace the missing values with the median of each feature.
# To do so, first we are going to convert the 0 that represent the missing values for NaN.
for name, df in datasets.items():
    df[cols_cero] = df[cols_cero].replace(0, np.nan)

# Now we replace the NaN values with the median of the train set of each fold. The median is computed only with the
# train set and applied to both train and test, so no information from the test set is used.
for i in range(10):
    base = f"pima-indians.fold.{i:06d}"
    train = datasets[base + ".train"]
    test = datasets[base + ".test"]

    X_cols = [c for c in train.columns if c != "class"]
    median = train[X_cols].median()   # NaN values are ignored automatically

    train[X_cols] = train[X_cols].fillna(median)
    test[X_cols] = test[X_cols].fillna(median)

# Now we standardize the numeric features: z = (x - mean) / std
# The train set is standardized with sklearn.preprocessing.scale, which uses its own mean and std.
# The test set is standardized with the mean and std of the train set of the same fold, so no information from the test
# set is used.
for i in range(10):
    base = f"pima-indians.fold.{i:06d}"
    train = datasets[base + ".train"]
    test = datasets[base + ".test"]

    X_cols = [c for c in train.columns if c != "class"]

    # statistics of the train set (ddof=0, same as scale)
    mean = train[X_cols].mean()
    std = train[X_cols].std(ddof=0)  # ddof=0 is the population std

    train[X_cols] = scale(train[X_cols])              # uses train mean and std
    test[X_cols] = (test[X_cols] - mean) / std        # same statistics applied to test

# The input features are all numeric, so no categorical encoding is needed for them.
# The only categorical column is the class label, which has two values, so we use label encoding
# (0 = tested_negative, 1 = tested_positive).
for i in range(10):
    base = f"pima-indians.fold.{i:06d}"
    train = datasets[base + ".train"]
    test = datasets[base + ".test"]

    encoder = LabelEncoder()
    train["class"] = encoder.fit_transform(train["class"])
    test["class"] = encoder.transform(test["class"])

# Now we train an SVM (Support Vector Machine) with the preprocessed data.
# Each fold has its own train and test set: we train with the train set and evaluate with the test set.
# C and gamma interact with each other, so we tune them together using GridSearchCV.
# The search uses cross-validation (cv=5) only inside the train set of each fold, so the test set is not used to
# choose the hyperparameters.

param_grid = {
    "C": [0.1, 1, 10, 100],
    "gamma": [0.001, 0.01, 0.1, 1],
}

accuracies = []
for i in range(10):
    base = f"pima-indians.fold.{i:06d}"
    train = datasets[base + ".train"]
    test = datasets[base + ".test"]

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
# We extended the grid (C up to 1000, gamma down to 0.0001) to check whether the optimum was
# at the border. The mean accuracy did not improve (0.7476 vs 0.7515), and the difference is much smaller than the std
# between folds (~0.03), so the original grid is enough.

# Now we train a Random Forest (RF) with the preprocessed data.
# As with the SVM, we train with the train set of each fold and evaluate with the test set.
# Random Forest does not need feature scaling, so the standardized data can be used as it is.
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
    base = f"pima-indians.fold.{i:06d}"
    train = datasets[base + ".train"]
    test = datasets[base + ".test"]

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

# The RF reaches a mean accuracy of 0.7621 (std 0.0446), versus 0.7515 (std 0.0294) for the SVM.
# The difference (~0.01) is much smaller than the std between folds, so both models perform equivalently.
# The RF is more variable across folds (0.6753 to 0.8158).
# Feature importances: "plas" (glucose) is by far the most important variable (0.326), followed by
# "mass" (BMI), "age" and "pedi". "insu" and "skin" have the lowest importance, which is consistent with
# the high proportion of values imputed with the median in these columns.





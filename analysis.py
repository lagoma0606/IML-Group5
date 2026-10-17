import pandas as pd
import numpy as np
from scipy.stats import wilcoxon, friedmanchisquare, studentized_range

df = pd.read_csv('results/baseline.csv')
knn = df[df['model'] == 'kNN']


# ------------------------------------------------------------------
# 1. Ranking of the 120 kNN configurations
# ------------------------------------------------------------------
# One row per configuration, averaged over the 10 folds. The groupby in
# main.py averaged over voting and retention as well, hiding their effect.
means = knn.groupby(['dataset', 'config'])[['accuracy', 'time', 'storage_pct']].mean()

for name in ['pima-indians', 'mx']:
    print('---', name, '---')
    print(means.loc[name].sort_values('accuracy', ascending=False).head(10).round(4))
    print()


# ------------------------------------------------------------------
# 2. Friedman and Nemenyi over the kNN configurations
# ------------------------------------------------------------------
def critical_difference(k, n, alpha=0.05):
    """Nemenyi critical difference between two mean ranks.

    k = number of methods compared, n = number of datasets (folds here).
    The studentized range replaces the published table, which stops well
    before k = 120.
    """
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


def friedman_with_ranks(pivot, label, higher_is_better=True):
    """Friedman test over the columns of pivot, plus Nemenyi mean ranks.

    pivot holds one row per fold and one column per method. Accuracy is
    better when high; time and storage are better when low, which is what
    higher_is_better switches.
    """
    columns = []
    for c in pivot.columns:
        columns.append(pivot[c].values)

    stat, p = friedmanchisquare(*columns)

    # Ranked within each fold, 1 = best. Ranking inside the fold removes
    # the fold's own difficulty from the comparison. Ties take average
    # ranks, as Demsar requires.
    ranks = pivot.rank(axis=1, ascending=not higher_is_better)
    mean_ranks = ranks.mean().sort_values()

    k = pivot.shape[1]
    n = pivot.shape[0]
    cd = critical_difference(k, n)

    print('---', label, '---')
    print('methods:', k, ' folds:', n, ' Friedman p-value:', round(p, 6))
    print('critical difference:', round(cd, 3))
    print(mean_ranks.round(3))

    # Anything within one critical difference of the best mean rank is
    # statistically indistinguishable from it.
    best = mean_ranks.iloc[0]
    tied = mean_ranks[mean_ranks <= best + cd]
    print('indistinguishable from the best:', len(tied), 'of', k)
    print()


# (metric, higher is better)
METRICS = [('accuracy', True), ('time', False), ('storage_pct', False)]

for name in ['pima-indians', 'mx']:
    subset = knn[knn['dataset'] == name]

    for metric, higher_is_better in METRICS:
        label = name + ' - ' + metric

        # All 120 configurations at once.
        pivot = subset.pivot_table(index='fold', columns='config', values=metric)
        friedman_with_ranks(pivot, label + ' - all configurations', higher_is_better)

        # Decomposed per parameter. Each fold contributes one value per
        # level, averaged over the other three parameters.
        for param in ['distance', 'k', 'retention']:
            pivot = subset.pivot_table(index='fold', columns=param, values=metric)
            friedman_with_ranks(pivot, label + ' - ' + param, higher_is_better)

        # Voting has only two levels, so Wilcoxon applies instead.
        pivot = subset.pivot_table(index='fold', columns='voting', values=metric)
        stat, p = wilcoxon(pivot['majority'].values,
                           pivot['inverse_distance'].values)
        print('---', label, '- voting ---')
        print('Wilcoxon p-value:', round(p, 6))
        print(pivot.mean().round(4))
        print()


# ------------------------------------------------------------------
# 3. SVM kernels
# ------------------------------------------------------------------
# Each fold is treated as a dataset, as the lecturer indicated.
svm = df[(df['model'] == 'svm') & (df['dataset'] == 'pima-indians')]

linear = svm[svm['config'] == 'linear'].sort_values('fold')['accuracy'].values
rbf = svm[svm['config'] == 'rbf'].sort_values('fold')['accuracy'].values

print(linear)
print(rbf)

# sort_values('fold') is essential: Wilcoxon is a paired test, it compares
# fold 0 against fold 0, fold 1 against fold 1. If the two vectors arrived
# in a different order, the pairing would be wrong.
stat, p = wilcoxon(linear, rbf)
print("p-value:", p)
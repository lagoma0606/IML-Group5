from pathlib import Path

import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare
from plots import critical_difference

df = pd.read_csv(Path(__file__).parent / 'results.csv')
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
    print('methods:', k, ' folds:', n, ' Friedman p-value:', format(p, '.3g'))
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
# We evaluate Linear vs RBF vs Tuned RBF simultaneously using the Friedman test.
# The friedman_with_ranks function will apply the Nemenyi post-hoc test (critical difference)
# to determine if the tuned model actually outperforms the others significantly.

for name in ['pima-indians', 'mx']:
    svm = df[(df['model'] == 'svm') & (df['dataset'] == name)]

    # sort_values('fold') is essential to ensure the vectors are correctly paired for the test
    linear = svm[svm['config'] == 'linear'].sort_values('fold')['accuracy'].values
    rbf = svm[svm['config'] == 'rbf'].sort_values('fold')['accuracy'].values

    # The tuned SVM (grid search) is stored as its own model, so it is read separately
    tuned = df[(df['model'] == 'svm-tuned') & (df['dataset'] == name)]
    tuned_accuracy = tuned.sort_values('fold')['accuracy'].values

    # Create a pivot dataframe with the 3 variants and run Friedman + Nemenyi
    pivot = pd.DataFrame({'linear': linear, 'rbf': rbf, 'tuned': tuned_accuracy})

    friedman_with_ranks(pivot, name + ' - SVM linear vs rbf vs tuned - accuracy', True)

    print('mean accuracy:')
    print(pivot.mean().round(4).to_string())


# ------------------------------------------------------------------
# 4. Random Forest
# ------------------------------------------------------------------
# main.py stores the configuration as '<n_estimators>_<criterion>', so the
# two parameters have to be read back out of that string before they can be
# compared one at a time.
rf = df[df['model'] == 'RF'].copy()

n_estimators = []
criterion = []
for value in rf['config']:
    parts = value.split('_')
    n_estimators.append(int(parts[0]))
    criterion.append(parts[1])

rf['n_estimators'] = n_estimators
rf['criterion'] = criterion

# Random Forest holds no case base, so storage does not apply to it.
RF_METRICS = [('accuracy', True), ('time', False)]

for name in ['pima-indians', 'mx']:
    subset = rf[rf['dataset'] == name]

    for metric, higher_is_better in RF_METRICS:
        label = name + ' - RF - ' + metric

        # All 8 configurations at once.
        pivot = subset.pivot_table(index='fold', columns='config', values=metric)
        friedman_with_ranks(pivot, label + ' - all configurations', higher_is_better)

        # Four numbers of trees, so Friedman applies.
        pivot = subset.pivot_table(index='fold', columns='n_estimators',
                                   values=metric)
        friedman_with_ranks(pivot, label + ' - n_estimators', higher_is_better)

        # Two criteria, so Wilcoxon applies instead.
        pivot = subset.pivot_table(index='fold', columns='criterion',
                                   values=metric)
        gini = pivot['gini'].values
        entropy = pivot['entropy'].values

        print('---', label, '- criterion ---')
        if (gini == entropy).all():
            print('every fold tied, Wilcoxon does not apply')
        else:
            stat, p = wilcoxon(gini, entropy)
            print('Wilcoxon p-value:', format(p, '.4g'))
        print(pivot.mean().round(4))
        print()


# ------------------------------------------------------------------
# 5. The best kNN against SVM and Random Forest
# ------------------------------------------------------------------
# Three methods, so Friedman applies.
#
# Where several configurations were statistically indistinguishable, the
# simplest and fastest was kept: on pima-indians the distance made no
# significant difference (p = 0.112) and on mx the four non-IVDM metrics tie
# at mean rank 2.50, so the Euclidean one was chosen.
#
# (label, model in results.csv, configuration)
BEST = [
    ('kNN', 'kNN', 'euclidean_k7_majority_OG'),
    ('SVM', 'svm', 'rbf'),
    ('RF', 'RF', '100_gini'),
]

# Storage does not apply here: only the kNN keeps a case base.
CROSS_METRICS = [('accuracy', True), ('time', False)]

for name in ['pima-indians', 'mx']:
    for metric, higher_is_better in CROSS_METRICS:
        columns = {}
        for label, model, config in BEST:
            rows = df[(df['dataset'] == name)
                      & (df['model'] == model)
                      & (df['config'] == config)].sort_values('fold')
            columns[label] = rows[metric].values

        pivot = pd.DataFrame(columns)

        friedman_with_ranks(pivot, name + ' - kNN vs SVM vs RF - ' + metric,
                            higher_is_better)
        print('mean', metric + ':')
        print(pivot.mean().round(4).to_string())
        print()

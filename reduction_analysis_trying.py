"""Statistics and figures for the reduction experiments.

Run after reduction_main.py. Reads both CSVs, because every comparison
here is against the unreduced baseline, which lives in results.csv.

For each dataset and model it tests, over the 10 folds:
  accuracy    - higher is better
  time        - training + prediction only (kNN: the whole classification)
  total_time  - the same plus the time spent reducing the training set
and for storage it compares the three techniques with each other.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon

from plots import critical_difference, mean_ranks, COLOURS, MARKERS, GRID

BASE_DIR = Path(__file__).parent
FIGURES_DIR = BASE_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

DATASETS = ['pima-indians', 'mx']
TECHNIQUES = ['FCNN', 'ENNTh', 'DROP3']

# The kNN the reduction is applied to, and the two baselines section 14
# compares the reduced training sets against.
BEST_KNN = 'euclidean_k7_majority_OG'

# The two files label the models differently: main.py writes 'svm',
# reduction_main.py writes 'SVM'. Each entry is
# (label in results.csv, label in reduction_results.csv, baseline config).
MODELS = [
    ('kNN', 'kNN', BEST_KNN),
    ('svm', 'SVM', 'rbf'),
    ('RF', 'RF', '100_gini'),
]

# (name, higher is better). Time and storage are better when low.
METRICS = [('accuracy', True), ('time', False), ('total_time', False)]


def baseline_values(df, dataset, model, config, metric):
    """One value per fold of the unreduced model, in fold order."""
    rows = df[(df['dataset'] == dataset)
              & (df['model'] == model)
              & (df['config'] == config)].sort_values('fold')
    # total_time of the baseline is its plain time: nothing was reduced.
    column = 'time' if metric in ('time', 'total_time') else metric
    return rows[column].values


def reduced_values(df, dataset, model, technique, metric):
    """One value per fold of the model trained on the reduced set."""
    rows = df[(df['dataset'] == dataset)
              & (df['model'] == model)
              & (df['reduction'] == technique)].sort_values('fold')
    if metric == 'accuracy':
        return rows['accuracy'].values
    if metric == 'time':
        return rows['time_inference'].values
    if metric == 'total_time':
        return (rows['time_reduction'] + rows['time_inference']).values
    raise ValueError(metric)


def paired_p(x, y):
    """Wilcoxon p-value of two paired vectors (fold against fold).

    Wilcoxon raises an error when every difference is zero, which is a
    legitimate outcome here (identical results), so it returns 1.
    """
    if (x == y).all():
        return 1.0
    return wilcoxon(x, y).pvalue


def compare(baseline, reduction, dataset, model, metric, higher_is_better,
            rows_out):
    """No reduction against the three techniques, on one model and metric."""
    base_label, reduced_label, config = model
    columns = {'none': baseline_values(baseline, dataset, base_label,
                                       config, metric)}
    for technique in TECHNIQUES:
        columns[technique] = reduced_values(reduction, dataset, reduced_label,
                                            technique, metric)

    for name in columns:
        if len(columns[name]) != 10:
            raise SystemExit(f'{dataset} {reduced_label} {metric}: {name} has '
                             f'{len(columns[name])} folds, expected 10')

    pivot = pd.DataFrame(columns)
    stat, p = friedmanchisquare(pivot['none'], pivot['FCNN'],
                                pivot['ENNTh'], pivot['DROP3'])
    ranks = mean_ranks(pivot, higher_is_better=higher_is_better)
    cd = critical_difference(4, 10)

    print('---', dataset, reduced_label, metric, '---')
    print('Friedman p-value:', format(p, '.3g'), ' CD:', round(cd, 2))
    print('mean', metric + ':')
    print(pivot.mean().round(4).to_string())
    print('mean rank:')
    print(ranks.round(2).to_string())
    print('within CD of the best:', int((ranks <= ranks.iloc[0] + cd).sum()),
          'of 4')

    # Each technique against the unreduced model (paired Wilcoxon).
    print('Wilcoxon against no reduction:')
    for technique in TECHNIQUES:
        pw = paired_p(pivot[technique].values, pivot['none'].values)
        print('  ', technique, 'p =', round(pw, 4))
        rows_out.append({
            'dataset': dataset, 'model': reduced_label, 'metric': metric,
            'technique': technique,
            'mean_none': pivot['none'].mean(),
            'mean_reduced': pivot[technique].mean(),
            'p_wilcoxon': pw, 'friedman_p': p,
        })
    print()


def storage_tests(baseline, reduction):
    """Storage is a property of the reduction, so it is the same for kNN,
    SVM and RF. It is compared among the three techniques (testing it
    against 100% would be trivial). For kNN the storage left after OG
    forgetting is also compared with OG alone, which is what the
    unreduced kNN already stores less than 100% of.
    """
    for dataset in DATASETS:
        rows = reduction[(reduction['dataset'] == dataset)
                         & (reduction['model'] == 'kNN')]
        pivot = rows.pivot_table(index='fold', columns='reduction',
                                 values='storage_pct')[TECHNIQUES]

        stat, p = friedmanchisquare(*[pivot[t] for t in TECHNIQUES])
        ranks = mean_ranks(pivot, higher_is_better=False)
        cd = critical_difference(3, 10)

        print('---', dataset, 'storage (% of training fold kept) ---')
        print('mean +- std:')
        print(pd.DataFrame({'mean': pivot.mean(),
                            'std': pivot.std()}).round(2).to_string())
        print('Friedman p-value:', format(p, '.3g'), ' CD:', round(cd, 2))
        print('mean rank (1 = keeps the fewest):')
        print(ranks.round(2).to_string())
        print('Wilcoxon between techniques:')
        for i in range(len(TECHNIQUES)):
            for j in range(i + 1, len(TECHNIQUES)):
                a, b = TECHNIQUES[i], TECHNIQUES[j]
                print('  ', a, 'vs', b, 'p =',
                      round(paired_p(pivot[a].values, pivot[b].values), 4))

        # kNN only: storage after OG, with and without reduction.
        base = baseline[(baseline['dataset'] == dataset)
                        & (baseline['model'] == 'kNN')
                        & (baseline['config'] == BEST_KNN)].sort_values('fold')
        print('kNN, final storage after OG (baseline stores',
              round(base['storage_pct'].mean(), 2), '%):')
        for technique in TECHNIQUES:
            final = rows[rows['reduction'] == technique].sort_values('fold')
            print('  ', technique, round(final['final_storage_pct'].mean(), 2),
                  '% , Wilcoxon vs OG alone p =',
                  round(paired_p(final['final_storage_pct'].values,
                                 base['storage_pct'].values), 4))
        print()


def storage_summary(reduction):
    """How much each technique removed, and what it cost to run."""
    knn = reduction[reduction['model'] == 'kNN']
    summary = knn.groupby(['dataset', 'reduction'])[
        ['storage_pct', 'final_storage_pct', 'time_reduction', 'time_inference']
    ].mean()

    print('--- storage and cost ---')
    print(summary.round(2).to_string())
    print()


def tradeoff_figure(baseline, reduction, dataset):
    """What each technique costs in accuracy for what it saves in storage."""
    base = baseline[(baseline['dataset'] == dataset)
                    & (baseline['model'] == 'kNN')
                    & (baseline['config'] == BEST_KNN)]

    labels = ['none']
    x = [base['storage_pct'].mean()]
    y = [base['accuracy'].mean()]

    for technique in TECHNIQUES:
        rows = reduction[(reduction['dataset'] == dataset)
                         & (reduction['model'] == 'kNN')
                         & (reduction['reduction'] == technique)]
        labels.append(technique)
        x.append(rows['storage_pct'].mean())
        y.append(rows['accuracy'].mean())

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for i in range(len(labels)):
        ax.scatter(x[i], y[i], s=70, color=COLOURS[i], marker=MARKERS[i],
                   edgecolor='white', linewidth=0.8, zorder=2)
        ax.annotate(labels[i], xy=(x[i], y[i]), xytext=(8, 0),
                    textcoords='offset points', va='center',
                    fontsize=8.5, color=COLOURS[i])

    ax.set_xlabel('Stored cases (% of training fold)')
    ax.set_ylabel('Mean accuracy over 10 folds')
    ax.margins(x=0.22, y=0.22)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title('Storage against accuracy — ' + dataset,
                 fontsize=10, loc='left')

    fig.savefig(FIGURES_DIR / ('reduction_tradeoff_' + dataset + '.png'))
    plt.close(fig)


def main():
    baseline = pd.read_csv(BASE_DIR / 'results.csv')
    reduction = pd.read_csv(BASE_DIR / 'reduction_results.csv')

    rows_out = []
    for dataset in DATASETS:
        for model in MODELS:
            for metric, higher_is_better in METRICS:
                compare(baseline, reduction, dataset, model, metric,
                        higher_is_better, rows_out)

    # One tidy table with every Wilcoxon result, ready for the report.
    tests = pd.DataFrame(rows_out)
    tests.to_csv(BASE_DIR / 'reduction_tests.csv', index=False)

    storage_tests(baseline, reduction)
    storage_summary(reduction)

    for dataset in DATASETS:
        tradeoff_figure(baseline, reduction, dataset)

    print('Tests written to', BASE_DIR / 'reduction_tests.csv')
    print('Figures written to', FIGURES_DIR)


if __name__ == '__main__':
    main()
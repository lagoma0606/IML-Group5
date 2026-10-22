"""Statistics and figures for the reduction experiments.

Run after reduction_main.py. Reads both CSVs, because every comparison
here is against the unreduced baseline, which lives in results.csv.
"""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import friedmanchisquare

from plots import critical_difference, mean_ranks, COLOURS, MARKERS, GRID

BASE_DIR = Path(__file__).parent
FIGURES_DIR = BASE_DIR / 'figures'
FIGURES_DIR.mkdir(exist_ok=True)

DATASETS = ['pima-indians', 'mx']
TECHNIQUES = ['FCNN', 'EENN', 'DROP3']

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


def accuracy_per_fold(df, dataset, model, config):
    """Accuracy of one model on one dataset, one value per fold, in order."""
    rows = df[(df['dataset'] == dataset)
              & (df['model'] == model)
              & (df['config'] == config)]
    return rows.sort_values('fold')['accuracy'].values


def compare(baseline, reduction, dataset, model):
    """No reduction against the three techniques, on one model."""
    base_label, reduced_label, config = model
    columns = {'none': accuracy_per_fold(baseline, dataset, base_label, config)}
    for technique in TECHNIQUES:
        rows = reduction[(reduction['dataset'] == dataset)
                         & (reduction['model'] == reduced_label)
                         & (reduction['reduction'] == technique)]
        columns[technique] = rows.sort_values('fold')['accuracy'].values

    for name in columns:
        if len(columns[name]) != 10:
            raise SystemExit(f'{dataset} {reduced_label}: {name} has '
                             f'{len(columns[name])} folds, expected 10')

    pivot = pd.DataFrame(columns)
    stat, p = friedmanchisquare(pivot['none'], pivot['FCNN'],
                                pivot['EENN'], pivot['DROP3'])
    ranks = mean_ranks(pivot, higher_is_better=True)
    cd = critical_difference(4, 10)

    print('---', dataset, reduced_label, '---')
    print('Friedman p-value:', format(p, '.3g'), ' CD:', round(cd, 2))
    print('mean accuracy:')
    print(pivot.mean().round(4).to_string())
    print('mean rank:')
    print(ranks.round(2).to_string())
    print('within CD of the best:', int((ranks <= ranks.iloc[0] + cd).sum()),
          'of 4')
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

    for dataset in DATASETS:
        for model in MODELS:
            compare(baseline, reduction, dataset, model)

    storage_summary(reduction)

    for dataset in DATASETS:
        tradeoff_figure(baseline, reduction, dataset)

    print('Figures written to', FIGURES_DIR)


if __name__ == '__main__':
    main()
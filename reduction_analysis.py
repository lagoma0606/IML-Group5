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

# The kNN the reduction is applied to, and the two baselines the reduced
# training sets are compared against.
BEST_KNN = 'euclidean_k7_majority_OG'

# The two files label the models differently: main.py writes 'svm',
# reduction_main.py writes 'SVM'. Each entry is
# (label in results.csv, label in reduction_results.csv, baseline config).
MODELS = [
    ('kNN', 'kNN', BEST_KNN),
    ('svm', 'SVM', 'rbf'),
    ('RF', 'RF', '100_gini'),
]

# The assignment asks for accuracy, efficiency and storage, so all three are
# tested rather than accuracy alone. Each entry is
# (label, column in results.csv, column in reduction_results.csv, higher is better).
#
# Two details about the column names:
#
# time_inference is the time of one run on the reduced training set, measured
# the same way as 'time' in results.csv: for kNN it is the whole
# classification; for SVM and RF it is training plus prediction. It leaves
# out the cost of applying the reduction technique. That cost is
# time_reduction, and 'time including reduction' adds the two, which is what
# decides whether reducing pays off.
#
# final_storage_pct is what remains after both the reduction and the OG
# retention policy. The baseline in results.csv has already had OG applied to
# it, so final_storage_pct is the column that compares like with like;
# storage_pct would leave OG out of one side of the comparison only.
METRICS = [
    ('accuracy', 'accuracy', 'accuracy', True),
    ('time', 'time', 'time_inference', False),
    ('time including reduction', 'time', 'time_total', False),
    ('storage', 'storage_pct', 'final_storage_pct', False),
]


def technique_labels(reduction):
    """The reduction names exactly as reduction_main.py wrote them.

    Read from the file rather than written out here, so that renaming a
    technique in reduction_main.py cannot silently leave this script
    comparing against a column that no longer exists.
    """
    labels = []
    for value in sorted(reduction['reduction'].unique()):
        labels.append(value)
    return labels


def values_per_fold(rows, column):
    """One value per fold, in fold order, so the folds line up across methods."""
    return rows.sort_values('fold')[column].values


def compare(baseline, reduction, dataset, model, metric, techniques):
    """No reduction against every technique, on one model and one metric."""
    base_label, reduced_label, config = model
    label, base_column, reduced_column, higher_is_better = metric

    base_rows = baseline[(baseline['dataset'] == dataset)
                         & (baseline['model'] == base_label)
                         & (baseline['config'] == config)]
    # The unreduced model: the model's own name plus '(without reduction)'.
    columns = {reduced_label + ' (without reduction)':
               values_per_fold(base_rows, base_column)}

    for technique in techniques:
        rows = reduction[(reduction['dataset'] == dataset)
                         & (reduction['model'] == reduced_label)
                         & (reduction['reduction'] == technique)]
        columns[technique] = values_per_fold(rows, reduced_column)

    for name in columns:
        if len(columns[name]) != 10:
            raise SystemExit(f'{dataset} {reduced_label} {label}: {name} has '
                             f'{len(columns[name])} folds, expected 10')

    pivot = pd.DataFrame(columns)

    series = []
    for name in pivot.columns:
        series.append(pivot[name].values)

    stat, p = friedmanchisquare(*series)
    ranks = mean_ranks(pivot, higher_is_better=higher_is_better)
    cd = critical_difference(pivot.shape[1], pivot.shape[0])

    print('---', dataset, reduced_label, '-', label, '---')
    print('Friedman p-value:', format(p, '.3g'), ' CD:', round(cd, 2))
    print('mean', label + ':')
    print(pivot.mean().round(4).to_string())
    print('mean rank:')
    print(ranks.round(2).to_string())
    print('within CD of the best:', int((ranks <= ranks.iloc[0] + cd).sum()),
          'of', pivot.shape[1])
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

    # The kNN with OG and no reduction, so it is not at 100%.
    labels = ['kNN (without reduction)']
    x = [base['storage_pct'].mean()]
    y = [base['accuracy'].mean()]

    for technique in technique_labels(reduction):
        rows = reduction[(reduction['dataset'] == dataset)
                         & (reduction['model'] == 'kNN')
                         & (reduction['reduction'] == technique)]
        labels.append(technique)
        x.append(rows['final_storage_pct'].mean())
        y.append(rows['accuracy'].mean())

    fig, ax = plt.subplots(figsize=(5.4, 3.6))
    for i in range(len(labels)):
        ax.scatter(x[i], y[i], s=70, color=COLOURS[i], marker=MARKERS[i],
                   edgecolor='white', linewidth=0.8, zorder=2)
        # The unreduced kNN sits close to ENNTh, so its label goes above the
        # point, centred, and the others to the right.
        if i == 0:
            offset, align, valign = (0, 9), 'center', 'bottom'
        else:
            offset, align, valign = (8, 0), 'left', 'center'
        ax.annotate(labels[i], xy=(x[i], y[i]), xytext=offset,
                    textcoords='offset points', ha=align, va=valign,
                    fontsize=8.5, color=COLOURS[i])

    ax.set_xlabel('Stored cases after reduction and OG (% of training fold)')
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

    # Reduction plus run time: the real cost of a reduced model.
    reduction['time_total'] = (reduction['time_reduction']
                               + reduction['time_inference'])

    techniques = technique_labels(reduction)

    for dataset in DATASETS:
        for model in MODELS:
            for metric in METRICS:
                # Only kNN keeps a case base, so only kNN has a storage figure.
                if metric[0] == 'storage' and model[1] != 'kNN':
                    continue
                compare(baseline, reduction, dataset, model, metric,
                        techniques)

    storage_summary(reduction)

    for dataset in DATASETS:
        tradeoff_figure(baseline, reduction, dataset)

    print('Figures written to', FIGURES_DIR)


if __name__ == '__main__':
    main()
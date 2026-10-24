"""Table of the reduction experiments for SVM and Random Forest, as a CSV.

Run after reduction_main.py. It reads results.csv (the models trained with
the full training set) and reduction_results.csv (the same models trained
with the sets reduced by each technique) and writes reduction_table.csv, one
row per condition, in the same layout as the other Friedman tables.

Each Friedman test compares four conditions at once (full set plus the three
techniques) over the 10 folds, so its p-value is repeated in every row of the
block. Mean rank and 'Within CD of best' describe each condition on its own.
"""

from pathlib import Path

import pandas as pd
from scipy.stats import friedmanchisquare

from plots import critical_difference, mean_ranks

BASE_DIR = Path(__file__).parent

DATASETS = ['pima-indians', 'mx']

# (label in results.csv, label in reduction_results.csv, baseline config).
# To add the kNN too, append ('kNN', 'kNN', 'euclidean_k7_majority_OG').
MODELS = [
    ('svm', 'SVM', 'rbf'),
    ('RF', 'RF', '100_gini'),
]

# (name in the table, column in results.csv, column in reduction_results.csv,
#  higher is better). time_total is created in main(): reduction + running time.
METRICS = [
    ('accuracy', 'accuracy', 'accuracy', True),
    ('time', 'time', 'time_inference', False),
    ('time including reduction', 'time', 'time_total', False),
]

FULL = 'Full'

# Excel in Spanish settings expects ';' as separator. If the file opens in a
# single column, change this to ';'.
CSV_SEP = ','


def values_per_fold(rows, column):
    """One value per fold, in fold order, so the folds line up across conditions."""
    return rows.sort_values('fold')[column].values


def block(baseline, reduction, dataset, model, metric, techniques):
    """Rows of one Friedman test: one dataset, one model, one metric."""
    base_label, reduced_label, config = model
    name, base_column, reduced_column, higher_is_better = metric

    base_rows = baseline[(baseline['dataset'] == dataset)
                         & (baseline['model'] == base_label)
                         & (baseline['config'] == config)]
    columns = {FULL: values_per_fold(base_rows, base_column)}

    for technique in techniques:
        rows = reduction[(reduction['dataset'] == dataset)
                         & (reduction['model'] == reduced_label)
                         & (reduction['reduction'] == technique)]
        columns[technique] = values_per_fold(rows, reduced_column)

    for level in columns:
        if len(columns[level]) != 10:
            raise SystemExit(f'{dataset} {reduced_label} {name}: {level} has '
                             f'{len(columns[level])} folds, expected 10')

    pivot = pd.DataFrame(columns)

    series = []
    for level in pivot.columns:
        series.append(pivot[level].values)
    stat, p = friedmanchisquare(*series)

    ranks = mean_ranks(pivot, higher_is_better=higher_is_better)
    cd = critical_difference(pivot.shape[1], pivot.shape[0])
    best = ranks.iloc[0]

    out = []
    for level in ranks.index:          # already sorted from best to worst rank
        if ranks[level] <= best + cd:
            verdict = 'yes'
        else:
            verdict = 'no'
        out.append({
            'Dataset': dataset,
            'Model': reduced_label,
            'Metric': name,
            'Level': level,
            'Mean': round(pivot[level].mean(), 4),
            'Mean rank': round(ranks[level], 2),
            'Friedman p': float(format(p, '.3g')),
            'CD': round(cd, 2),
            'Within CD of best': verdict,
        })
    return out


def main():
    baseline = pd.read_csv(BASE_DIR / 'results.csv')
    reduction = pd.read_csv(BASE_DIR / 'reduction_results.csv')

    # Reduction plus running time: the real cost of a reduced model.
    reduction['time_total'] = (reduction['time_reduction']
                               + reduction['time_inference'])

    # Read from the file, so a renamed technique cannot be silently missed.
    techniques = sorted(reduction['reduction'].unique())

    rows = []
    for dataset in DATASETS:
        for model in MODELS:
            for metric in METRICS:
                for row in block(baseline, reduction, dataset, model, metric,
                                 techniques):
                    rows.append(row)

    table = pd.DataFrame(rows)
    path = BASE_DIR / 'reduction_table.csv'
    table.to_csv(path, index=False, sep=CSV_SEP)

    print(table.to_string(index=False))
    print()
    print(len(table), 'rows written to', path)


if __name__ == '__main__':
    main()
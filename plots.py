"""Figures and tables for the report. Run after main.py."""

from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import friedmanchisquare, studentized_range

BASE_DIR = Path(__file__).parent
RESULTS_FILE = BASE_DIR / 'results.csv'
FIGURES_DIR = BASE_DIR / 'figures'
TABLE_FILE = BASE_DIR / 'analysis_table.md'
FIGURES_DIR.mkdir(exist_ok=True)

DISTANCES = ['euclidean', 'manhattan', 'clark', 'canberra', 'ivdm']
RETENTIONS = ['NR', 'AR', 'DC', 'OG']
DATASETS = ['pima-indians', 'mx']
PARAMS = ['distance', 'k', 'retention']

# (metric, higher is better)
METRICS = [('accuracy', True), ('time', False), ('storage_pct', False)]

COLOURS = ['#0072B2', '#D55E00', '#009E73', '#E69F00', '#CC79A7']
MARKERS = ['o', 's', '^', 'D', 'v']

INK = '#1a1a1a'
MUTED = '#6b6b6b'
GRID = '#dddddd'

plt.rcParams.update({
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'font.size': 9,
    'axes.edgecolor': MUTED,
    'xtick.color': MUTED,
    'ytick.color': MUTED,
    'axes.spines.top': False,
    'axes.spines.right': False,
})


def style_of(distance):
    i = DISTANCES.index(distance)
    return COLOURS[i], MARKERS[i]


def trim(value):
    """The k column arrives as 3.0, not 3."""
    text = str(value)
    if text.endswith('.0'):
        text = text[:-2]
    return text


# ----------------------------------------------------------------------
# Statistics
# ----------------------------------------------------------------------
def critical_difference(k, n, alpha=0.05):
    """Nemenyi critical difference (Demsar 2006), k methods over n datasets.

    scipy replaces the published table, which stops well before k = 120.
    Checked: k = 10 gives 3.164, the tabulated value.
    """
    q = studentized_range.ppf(1 - alpha, k, np.inf) / np.sqrt(2)
    return q * np.sqrt(k * (k + 1) / (6.0 * n))


def mean_ranks(pivot, higher_is_better=True):
    """Average rank per column, ranked inside each fold. 1 = best.

    Ranking inside the fold cancels out how hard that fold was.
    """
    ranks = pivot.rank(axis=1, ascending=not higher_is_better)
    return ranks.mean().sort_values()


def friedman_p(pivot):
    columns = []
    for c in pivot.columns:
        columns.append(pivot[c].values)
    stat, p = friedmanchisquare(*columns)
    return p


# ----------------------------------------------------------------------
# Figures
# ----------------------------------------------------------------------
def critical_difference_diagram(ranks, cd, title, filename):
    """Demsar's diagram: mean ranks on an axis, further left is better.

    A thick bar joins methods the Nemenyi test cannot tell apart.
    """
    names = list(ranks.index)
    values = list(ranks.values)
    k = len(names)
    low, high = 1, k

    fig, ax = plt.subplots(figsize=(7, 1.1 + 0.32 * k))
    ax.set_xlim(low - 0.2, high + 0.2)
    ax.set_ylim(-0.6 * k - 1.2, 1.9)
    ax.axis('off')

    ax.plot([low, high], [0, 0], color=INK, linewidth=1.2)
    for tick in range(low, high + 1):
        ax.plot([tick, tick], [0, 0.16], color=INK, linewidth=1.0)
        ax.text(tick, 0.32, str(tick), ha='center', va='bottom',
                color=MUTED, fontsize=8)

    # A bar one CD long, to compare against by eye.
    ax.plot([low, low + cd], [1.15, 1.15], color=INK, linewidth=2.0)
    ax.plot([low, low], [1.05, 1.25], color=INK, linewidth=1.0)
    ax.plot([low + cd, low + cd], [1.05, 1.25], color=INK, linewidth=1.0)
    ax.text(low + cd / 2.0, 1.35, 'CD = ' + str(round(cd, 2)),
            ha='center', va='bottom', fontsize=8, color=INK)

    # Half the labels to each side, or they collide.
    for i in range(k):
        row = -0.6 * (i + 1)
        rank = values[i]
        label = trim(names[i]) + '  (' + str(round(rank, 2)) + ')'

        if i < (k + 1) // 2:
            side_x, align, nudge = low - 0.15, 'right', -0.05
        else:
            side_x, align, nudge = high + 0.15, 'left', 0.05

        ax.plot([rank, rank], [0, row], color=MUTED, linewidth=0.9)
        ax.plot([rank, side_x], [row, row], color=MUTED, linewidth=0.9)
        ax.text(side_x + nudge, row, label, ha=align, va='center',
                fontsize=8.5, color=INK)

    # The list is sorted, so each run reaches as far right as it can.
    runs = []
    for i in range(k):
        j = i
        while j + 1 < k and values[j + 1] - values[i] <= cd:
            j += 1
        if j > i:
            runs.append((i, j))

    # Drop any run that sits inside a longer one.
    keep = []
    for a, b in runs:
        inside = False
        for c, d in runs:
            if (c, d) != (a, b) and c <= a and b <= d:
                inside = True
        if not inside:
            keep.append((a, b))

    level = -0.28
    for a, b in keep:
        left, right = values[a], values[b]
        # An exact tie gives a zero-width bar, which draws nothing.
        if right - left < 0.08:
            left, right = left - 0.06, right + 0.06
        ax.plot([left, right], [level, level], color=INK, linewidth=3.4,
                solid_capstyle='round')
        level -= 0.17

    ax.set_title(title, fontsize=10, color=INK, loc='left', pad=14)
    fig.savefig(FIGURES_DIR / filename)
    plt.close(fig)


def k_effect(knn, dataset):
    subset = knn[knn['dataset'] == dataset]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))

    # On mx four metrics are identical, so four lines and four labels land
    # on top of each other. Group them and draw one.
    groups = {}
    order = []
    for distance in DISTANCES:
        rows = subset[subset['distance'] == distance]
        means = rows.groupby('k')['accuracy'].mean().sort_index()
        key = tuple(np.round(means.values, 6))
        if key not in groups:
            groups[key] = {'names': [], 'means': means}
            order.append(key)
        groups[key]['names'].append(distance)

    ends = []
    for key in order:
        names = groups[key]['names']
        means = groups[key]['means']
        colour, marker = style_of(names[0])

        ax.plot(means.index, means.values, color=colour, marker=marker,
                markersize=6, linewidth=2)
        ends.append({'y': means.values[-1], 'x': means.index[-1],
                     'label': ' = '.join(names), 'colour': colour})

    # Pushes overlapping labels apart.
    ends.sort(key=lambda e: e['y'])
    gap = (ax.get_ylim()[1] - ax.get_ylim()[0]) * 0.05
    for i in range(1, len(ends)):
        if ends[i]['y'] - ends[i - 1]['y'] < gap:
            ends[i]['y'] = ends[i - 1]['y'] + gap

    for end in ends:
        ax.annotate(end['label'], xy=(end['x'], end['y']),
                    xytext=(7, 0), textcoords='offset points',
                    va='center', fontsize=7.5, color=end['colour'])

    ax.set_xlabel('k (number of neighbours)')
    ax.set_ylabel('Mean accuracy over 10 folds')
    ax.set_xticks([3, 5, 7])
    ax.margins(x=0.42)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_title('Effect of k — ' + dataset, fontsize=10, loc='left')

    fig.savefig(FIGURES_DIR / ('k_effect_' + dataset + '.png'))
    plt.close(fig)


def accuracy_vs_time(knn, dataset):
    """Accuracy against runtime, one point per configuration."""
    subset = knn[knn['dataset'] == dataset]
    means = subset.groupby(['config', 'distance'])[['accuracy', 'time']].mean()
    means = means.reset_index()

    fig, ax = plt.subplots(figsize=(5.4, 3.6))

    for distance in DISTANCES:
        rows = means[means['distance'] == distance]
        colour, marker = style_of(distance)
        ax.scatter(rows['time'], rows['accuracy'], s=34, color=colour,
                   marker=marker, edgecolor='white', linewidth=0.6,
                   label=distance, zorder=2)

    ax.set_xlabel('Mean runtime per fold (seconds)')
    ax.set_ylabel('Mean accuracy over 10 folds')
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title('Accuracy against cost, 120 configurations — ' + dataset,
                 fontsize=10, loc='left')

    fig.savefig(FIGURES_DIR / ('accuracy_vs_time_' + dataset + '.png'))
    plt.close(fig)


def storage_by_retention(knn):
    """Final case base per policy, as a percentage of the training fold."""
    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    width = 0.36
    positions = np.arange(len(RETENTIONS))

    for d, dataset in enumerate(DATASETS):
        subset = knn[knn['dataset'] == dataset]
        heights = []
        for policy in RETENTIONS:
            rows = subset[subset['retention'] == policy]
            heights.append(rows['storage_pct'].mean())

        bars = ax.bar(positions + (d - 0.5) * (width + 0.02), heights, width,
                      color=COLOURS[d], label=dataset, zorder=2)

        for bar, value in zip(bars, heights):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    str(round(value, 1)), ha='center', va='bottom',
                    fontsize=8, color=INK)

    ax.axhline(100, color=INK, linewidth=1.0, linestyle='--', zorder=3)
    ax.text(len(RETENTIONS) - 0.4, 101.5, 'original size',
            fontsize=8, color=MUTED, ha='right')

    ax.set_xticks(positions)
    ax.set_xticklabels(RETENTIONS)
    ax.set_ylabel('Final case base (% of training fold)')
    ax.set_ylim(0, 125)
    ax.grid(axis='y', color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    # Outside the axes: inside it covers the NR bars.
    ax.legend(frameon=False, loc='lower left', bbox_to_anchor=(0, 1.02),
              ncol=2, handlelength=1.2)
    ax.set_title('Storage by retention policy', fontsize=10, loc='left', pad=26)

    fig.savefig(FIGURES_DIR / 'storage_by_retention.png')
    plt.close(fig)


# ----------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------
def summary_tables(knn):
    """One row per level of each parameter, as markdown ready to paste."""
    lines = []

    for metric, higher_is_better in METRICS:
        lines.append('### ' + metric)
        lines.append('')
        lines.append('| Dataset | Parameter | Level | Mean rank '
                     '| Friedman p | CD | Within CD of best |')
        lines.append('| --- | --- | --- | --- | --- | --- | --- |')

        for dataset in DATASETS:
            subset = knn[knn['dataset'] == dataset]

            for param in PARAMS:
                pivot = subset.pivot_table(index='fold', columns=param,
                                           values=metric)
                ranks = mean_ranks(pivot, higher_is_better)
                p = friedman_p(pivot)
                cd = critical_difference(pivot.shape[1], pivot.shape[0])
                best = ranks.iloc[0]

                for level in ranks.index:
                    if ranks[level] <= best + cd:
                        verdict = 'yes'
                    else:
                        verdict = 'no'

                    lines.append('| ' + dataset
                                 + ' | ' + param
                                 + ' | ' + trim(level)
                                 + ' | ' + str(round(ranks[level], 2))
                                 + ' | ' + format(p, '.3g')
                                 + ' | ' + str(round(cd, 2))
                                 + ' | ' + verdict + ' |')
        lines.append('')

    TABLE_FILE.write_text('\n'.join(lines), encoding='utf-8')


# ----------------------------------------------------------------------
def main():
    if not RESULTS_FILE.exists():
        raise SystemExit('results.csv not found. Run main.py first.')

    knn = pd.read_csv(RESULTS_FILE)
    knn = knn[knn['model'] == 'kNN']

    for dataset in DATASETS:
        k_effect(knn, dataset)
        accuracy_vs_time(knn, dataset)

    storage_by_retention(knn)

    # One diagram, on k: the only parameter whose effect is significant.
    pivot = knn[knn['dataset'] == 'pima-indians'].pivot_table(
        index='fold', columns='k', values='accuracy')
    critical_difference_diagram(
        mean_ranks(pivot, True),
        critical_difference(pivot.shape[1], pivot.shape[0]),
        'Critical difference, accuracy by k — pima-indians',
        'cd_pima-indians_k.png')

    summary_tables(knn)

    print('Figures written to', FIGURES_DIR)
    print('Tables written to', TABLE_FILE)


if __name__ == '__main__':
    main()
# IML Group 5 — Work 2

A k-Nearest Neighbour (K-IBL) classifier written from scratch, compared
against SVM and Random Forest on two datasets with 10-fold cross-validation,
with and without instance reduction of the training set.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate           # Windows
source .venv/bin/activate        # macOS / Linux
pip install -r requirements.txt
```

Python 3.11, as the assignment requires.

## Running the code

Clone the repository and run the scripts. Nothing else has to be configured:
every path in the project is resolved relative to the file that uses it, not
to the directory a script happens to be launched from, so the scripts work
from anywhere once the repository is on disk.

```bash
git clone <repository url>
cd IML-Group5
python main.py
```

The `data/` folder travels with the repository and holds the 40 `.arff`
files the course distributed, 20 per dataset. Nothing is downloaded at run
time and no path needs editing.

The order matters, because each step reads what an earlier one wrote:

| # | Command | Reads | Writes | Time |
| --- | --- | --- | --- | --- |
| 1 | `python main.py` | `data/` | `results.csv` | ~80 min |
| 2 | `python reduction_main.py` | `data/` | `reduction_results.csv` | ~2.5 h |
| 3 | `python plots.py` | `results.csv` | `analysis_table.md`, `figures/` | seconds |
| 4 | `python analysis.py` | `results.csv` | prints to the terminal | seconds |
| 5 | `python reduction_analysis.py` | both CSV files | prints, plus two figures | seconds |
| 6 | `python reduction_table.py` | both CSV files | `reduction_table.csv` | seconds |

Steps 3 to 6 only read the CSV files, so they can be repeated as often as
needed without re-running the experiments. To keep the printed analyses:

```bash
python analysis.py > analysis_output.txt
python reduction_analysis.py > reduction_output_analysis.txt
```

**All experiments must be run on the same machine.** Accuracy and storage are
deterministic and comparable anywhere; runtime is not, and runtime is one of
the three metrics being evaluated. For the same reason the machine must not
be allowed to sleep during a run: a suspended laptop records the sleep as
classification time.

## The four classifiers

The assignment names four functions. Each lives in its own module:

| Function | File |
| --- | --- |
| `kNNAlgorithm` | `knn.py` |
| `svmAlgorithm`, `svmTunedAlgorithm` | `svm.py` |
| `rfAlgorithm`, `rfTunedAlgorithm` | `rf.py` |
| `reduction_kNNAlgorithm` | `reduction_knn.py` |

`main.py`, for each of the 10 folds of each dataset:

- **kNN**, 120 configurations: 5 distances (euclidean, manhattan, clark,
  canberra, ivdm) × 3 values of k (3, 5, 7) × 2 voting schemes (majority,
  inverse distance) × 4 retention policies (NR, AR, DC, OG)
- **Random Forest**, 8 configurations: 4 tree counts × 2 criteria
- **SVM**, 2 kernels: linear and rbf
- **SVM and RF tuned per fold** by a grid search whose inner cross-validation
  sees the training fold only, recorded as separate methods so that the cost
  of the search is never mixed with the cost of a single fit

`reduction_main.py` takes the best kNN configuration and applies three
reduction techniques to the training set before classifying: FCNN
(condensation), ENNTh (edition) and DROP3 (hybrid). Each reduced training set
is then also given to the SVM and the Random Forest.

## Layout

```
.
├── data/
│   ├── pima-indians/        20 .arff files (10 train + 10 test)
│   └── mx/                  20 .arff files
│
├── preprocessor.py          Preprocessor class: fit / transform
├── data_loader.py           Loads and prepares one fold
│
├── distance.py              The distance metrics
├── ivdm.py                  IVDM, as a transformation plus Euclidean
├── voting.py                Majority and inverse-distance voting
├── knn.py                   kNNAlgorithm: the K-IBL classifier
├── svm.py                   svmAlgorithm, svmTunedAlgorithm
├── rf.py                    rfAlgorithm, rfTunedAlgorithm
│
├── fcnn.py                  FCNN condensation
├── ennth.py                 ENNTh edition
├── drop3.py                 DROP3 hybrid reduction
├── reduction_knn.py         reduction_kNNAlgorithm and the reduction map
│
├── main.py                  Experiment harness      -> results.csv
├── reduction_main.py        Reduction experiments   -> reduction_results.csv
│
├── analysis.py              Statistical analysis, printed
├── reduction_analysis.py    Statistical analysis of the reduction
├── reduction_table.py       The reduction results as a CSV table
├── plots.py                 Figures and analysis_table.md
│
└── requirements.txt
```

Generating results and analysing them are deliberately kept apart: the two
harnesses take hours, the four analysis scripts take seconds, so the same
results can be re-analysed as often as needed without relaunching anything.

## Datasets

| Dataset | Instances | Attributes | Notes |
| --- | --- | --- | --- |
| Pima-Indians | 768 | 8 numeric | Zeros in `plas`, `pres`, `skin`, `insu`, `mass` encode missing values |
| Mx | 2,048 | 11 nominal | 11-bit multiplexer: 3 address bits select one of 8 data bits |

The 10 folds come pre-generated with the datasets and must not be modified,
so no additional partitioning is performed.

## Preprocessing

- **Missing values.** On Pima, zeros in `plas`, `pres`, `skin`, `insu` and
  `mass` are physiologically impossible and are treated as missing, then
  imputed with the median of the training fold. `preg` is left alone, since
  zero previous pregnancies is a valid reading.
- **Normalisation.** Numeric attributes are scaled to [0,1] with min-max, as
  the assignment requires. Without it, `insu` (range 0–846) would swamp
  `pedi` (range 0.08–2.42) in every distance computation.
- **Categorical attributes.** Mapped to integer codes taken from the ARFF
  header rather than from the values observed in a fold, so every legal value
  is covered even when a fold happens not to contain one of them.
- **No data leakage.** Every statistic — medians, minima, maxima, label maps
  — is learned from the training fold only and then applied to both folds. A
  fresh `Preprocessor` is built per fold.

## Statistical analysis

Following Demšar (2006), the paired t-test is avoided: it assumes normality,
it is sensitive to outliers, and it misbehaves under multiple comparisons.
Instead:

- **Two methods** → Wilcoxon signed-rank test
- **Three or more** → Friedman test, followed by the Nemenyi post-hoc test
  and its critical difference

The assignment fixes the number of datasets at two, while Demšar's framework
compares classifiers over many. Following the instruction given in the
course, each cross-validation fold is therefore treated as a separate
dataset, giving N = 10.

This is a known weakness rather than a neutral choice. Any two training folds
share about eight ninths of their instances, and Dietterich (1998) reports
that a paired test over 10-fold cross-validation carries somewhat elevated
Type I error for that reason, so the p-values should be read as optimistic.
What the overlap does not disturb is the pairing: every method is evaluated
on the same ten folds, so the mean ranks remain a fair comparison between
methods. The report states this explicitly.

## References

- Demšar, J. (2006). *Statistical comparisons of classifiers over multiple
  data sets.* Journal of Machine Learning Research, 7, 1–30.
- Dietterich, T. G. (1998). *Approximate statistical tests for comparing
  supervised classification learning algorithms.* Neural Computation, 10,
  1895–1923.
- Angiulli, F. (2007). *Fast nearest neighbor condensation for large data
  sets classification.* IEEE TKDE, 19(11), 1450–1464.
- Wilson, D. R. and Martinez, T. R. (1997). *Improved heterogeneous distance
  functions.* Journal of Artificial Intelligence Research, 6, 1–34.
- Wilson, D. R. and Martinez, T. R. (2000). *Reduction techniques for
  instance-based learning algorithms.* Machine Learning, 38(3), 257–286.
# IML Practice — Work 2

k-Nearest Neighbour written from scratch, compared against SVM and
Random Forest on two datasets with 10-fold cross-validation, with and
without instance reduction of the training set.

## Team

Group 5:

- Azel Akhmetova
- Daniel Gómez Vargas
- Marcos Jaimes Martínez
- Ángela Matute de Imaña

## Environment

- Python 3.11
- Tested on Windows 11 (PowerShell). The commands below also work on macOS
  and Linux; only the virtual-environment activation differs.
- No special hardware is required. A full run of the two experiment scripts
  takes about two hours on a current laptop.

## Project structure

```
code/
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
├── figures/                 Generated plots (created automatically)
├── README.md
└── requirements.txt
```

The four classifiers the assignment names each live in their own module:

| Function | File |
| --- | --- |
| `kNNAlgorithm` | `knn.py` |
| `svmAlgorithm`, `svmTunedAlgorithm` | `svm.py` |
| `rfAlgorithm`, `rfTunedAlgorithm` | `rf.py` |
| `reduction_kNNAlgorithm` | `reduction_knn.py` |

### How the files fit together

    main.py
     ├── data_loader.load_fold()  → Preprocessor (fit on train, transform train and test)
     ├── knn.kNNAlgorithm()       → distance.py, ivdm.IVDM, voting.VOTING
     ├── svm.svmAlgorithm(), svm.svmTunedAlgorithm()
     └── rf.rfAlgorithm(), rf.rfTunedAlgorithm()

    reduction_main.py
     ├── data_loader.load_fold()
     ├── reduction_knn.reduction_kNNAlgorithm()
     │    ├── reduce_training_set() → fcnn.condense / ennth.reduce_ennth / drop3.reduce_drop3
     │    └── knn.kNNAlgorithm()
     ├── svm.svmAlgorithm()        (on the reduced training set)
     └── rf.rfAlgorithm()          (on the reduced training set)

    analysis.py, plots.py, reduction_analysis.py, reduction_table.py
     └── read results.csv and reduction_results.csv; analysis.py and the two
         reduction scripts reuse the critical-difference helpers in plots.py

For each fold, the entry point loads and preprocesses the data once, then runs
every configuration on it. The two experiment scripts write the CSV files, and
the analysis scripts only read them.

Generating results and analysing them are deliberately kept apart: the two
experiment scripts take hours, the four analysis scripts take seconds, so
the same results can be re-analysed without relaunching anything.

Every path is resolved relative to the file that uses it, never to the
directory a script is launched from, so the scripts work from any working
directory once the archive is unzipped. No absolute path appears anywhere.

### Classes with state

Two classes keep state between calls. Both are created and used internally
by the scripts, so they never need to be called by hand to run the
experiments.

- **`Preprocessor`** (`preprocessor.py`). A new one is created for every
  fold by `data_loader.load_fold()`:

  ```
  p = Preprocessor(zero_as_missing=['plas', 'pres', 'skin', 'insu', 'mass'])
  p.fit(df_train, meta_train)            # learns medians, min/max and label maps from the training fold only
  x_train, y_train = p.transform(df_train)
  x_test, y_test = p.transform(df_test)  # the test fold uses the statistics of the training fold
  ```

- **`IVDM`** (`ivdm.py`). Created inside `kNNAlgorithm` when
  `distance='ivdm'`:

  ```
  ivdm = IVDM(n_bins=max(5, n_classes), categorical_features=categorical_indices)
  ivdm.fit(X_train, y_train)             # learns the class-probability tables from the training fold
  X_train_mapped = ivdm.transform(X_train)
  X_test_mapped = ivdm.transform(X_test)
  ```

  The IVDM distance is then the Euclidean distance between the mapped
  vectors.

## Installation

Open a terminal in the `code` folder.

```bash
cd <path-to-submission>/code

python -m venv .venv

# Windows PowerShell
.venv\Scripts\Activate.ps1
# macOS or Linux
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Run the experiments with the environment active, and `deactivate` when
finished.

## Entry points

There are six executable files, all in the `code` folder. Every command is
run from the `code` folder.

| File | Purpose | Command |
| --- | --- | --- |
| `code/main.py` | kNN (120 configurations), SVM and Random Forest experiments over the 10 folds of both datasets | `python main.py` (optional flags in *Parameters*) |
| `code/reduction_main.py` | Instance reduction experiments: FCNN, ENNTh and DROP3 with the best kNN, the SVM and the Random Forest | `python reduction_main.py` (optional flags in *Parameters*) |
| `code/analysis.py` | Friedman, Nemenyi and Wilcoxon tests on `results.csv` | `python analysis.py` |
| `code/plots.py` | Figures and summary tables from `results.csv` | `python plots.py` |
| `code/reduction_analysis.py` | Friedman and Nemenyi tests and figures for the reduction experiments | `python reduction_analysis.py` |
| `code/reduction_table.py` | Table of the SVM and RF reduction results | `python reduction_table.py` |

The other `.py` files are modules imported by these six and are not run
directly.

## Execution order

The order matters: each analysis step reads a file an earlier step wrote.

| # | Command | Reads | Writes | Time |
| --- | --- | --- | --- | --- |
| 1 | `python main.py` | `data/` | `results.csv` | ~80 min |
| 2 | `python reduction_main.py` | `data/` | `reduction_results.csv` | 20–45 min |
| 3 | `python plots.py` | `results.csv` | `analysis_table.md`, `figures/` | seconds |
| 4 | `python analysis.py` | `results.csv` | printed to the terminal | seconds |
| 5 | `python reduction_analysis.py` | both CSV files | printed, plus 2 figures | seconds |
| 6 | `python reduction_table.py` | both CSV files | `reduction_table.csv` | seconds |

Steps 1 and 2 are independent of each other and can be run in either order.
Steps 3 to 6 only read the two CSV files and can be repeated as often as
needed. The times were measured on our laptops and depend on the machine; do
not let the computer sleep during steps 1 and 2, since a suspended machine
records the sleep as running time.

To keep the printed analyses as files:

```bash
# macOS or Linux
python analysis.py > analysis_output.txt
python reduction_analysis.py > reduction_output_analysis.txt

# Windows PowerShell (writes the files in UTF-8)
python analysis.py | Out-File -Encoding utf8 analysis_output.txt
python reduction_analysis.py | Out-File -Encoding utf8 reduction_output_analysis.txt
```

Do not redirect the output of `reduction_table.py`: it writes
`reduction_table.csv` itself, and redirecting would overwrite the CSV with
the printed table.

The CSV files produced by our own run are included in the archive, so steps
3 to 6 can be run on their own to reproduce every number in the report
without waiting for steps 1 and 2.

**All experiments must be run on the same machine.** Runtime depends on the
hardware, and runtime is one of the three metrics being evaluated (see also
*Reproducibility*).

## Parameters

`main.py` takes optional arguments that narrow the run. **With no arguments
it runs every experiment and writes `results.csv`, which is how the results
in the report were produced.** All arguments are optional and introduced
with a flag; there are no positional arguments. The flags can be given in any
order, and a flag that takes several values lists them separated by spaces
(for example `--k 5 7`).

| Flag | Type | Valid values | Default |
| --- | --- | --- | --- |
| `--dataset` | one or more strings | `pima-indians`, `mx` | both |
| `--model` | one or more strings | `knn`, `svm`, `rf`, `tuned` | all four |
| `--distance` | one or more strings | `euclidean`, `manhattan`, `clark`, `canberra`, `ivdm` | all five |
| `--k` | one or more integers | `3`, `5`, `7` | all three |
| `--voting` | one or more strings | `majority`, `inverse_distance` | both |
| `--retention` | one or more strings | `NR`, `AR`, `DC`, `OG` | all four |
| `--kernel` | one or more strings | `linear`, `rbf` | both |
| `--folds` | integer | 1 to 10 | 10 |
| `--output` | file name | any | `results.csv` on a full run, `results_partial.csv` otherwise |

- `--folds N` runs the first N folds, numbered from 0: `--folds 1` runs only
  fold 0, and `--folds 10` runs folds 0 to 9.
- `--distance`, `--k`, `--voting` and `--retention` only affect the kNN, and
  `--kernel` only the SVM. `tuned` runs both the tuned SVM and the tuned
  Random Forest.
- `--output` only controls where the results are written, inside the `code`
  folder; all the other flags control which experiments are run.

To repeat a single experiment, for instance the kNN configuration the report
carries forward:

```bash
python main.py --dataset pima-indians --model knn --distance euclidean --k 7 --voting majority --retention OG
```

That command finishes in seconds and reproduces exactly the ten rows of
`results.csv` for that configuration.

A narrowed run never overwrites `results.csv`. Unless `--output` says
otherwise it writes `results_partial.csv`, so the full results the report is
built on cannot be lost by accident. The analysis scripts always read
`results.csv`. Every run prints which datasets, models and folds it is about
to run and which file it will write, before it starts.

`python main.py --help` prints the same list of options.

`reduction_main.py` takes the same kind of optional arguments: all of them
are flags, there are no positional arguments, and they can be given in any
order. With no arguments it runs every reduction experiment and writes
`reduction_results.csv`, as in the report. The kNN configuration is always
the one named in its `BEST_KNN` constant (Euclidean, k = 7, majority, OG).

| Flag | Type | Valid values | Default |
| --- | --- | --- | --- |
| `--dataset` | one or more strings | `pima-indians`, `mx` | both |
| `--reduction` | one or more strings | `FCNN`, `ENNTh`, `DROP3` | all three |
| `--model` | one or more strings | `knn`, `svm`, `rf` | all three |
| `--folds` | integer | 1 to 10 | 10 |
| `--output` | file name | any | `reduction_results.csv` on a full run, `reduction_results_partial.csv` otherwise |

`--model` chooses which models are trained on the reduced set; `--folds`
works as in `main.py`. For example, to apply only FCNN on the first fold of
Pima-Indians and train the three models on it:

```bash
python reduction_main.py --dataset pima-indians --reduction FCNN --folds 1
```

A narrowed run never overwrites `reduction_results.csv`, and the analysis
scripts always read `reduction_results.csv`. `python reduction_main.py --help`
prints the list of options.

**The four analysis scripts take no arguments.** `analysis.py`, `plots.py`,
`reduction_analysis.py` and `reduction_table.py` always read the two CSV
files and write their full output.

## Outputs

| Command | Produces | Location |
| --- | --- | --- |
| `main.py` | `results.csv`, 2,640 rows, one per model × configuration × fold (`results_partial.csv` on a narrowed run) | `code/` |
| `reduction_main.py` | `reduction_results.csv`, one row per dataset × technique × model × fold (`reduction_results_partial.csv` on a narrowed run) | `code/` |
| `plots.py` | `analysis_table.md`; `k_effect_*.png`, `accuracy_vs_time_*.png`, `storage_by_retention.png`, `cd_pima-indians_k.png` | `code/`, `code/figures/` |
| `analysis.py` | the statistical analysis, printed to the terminal (`analysis_output.txt` when redirected) | `code/` |
| `reduction_analysis.py` | the reduction analysis, printed (`reduction_output_analysis.txt` when redirected); `reduction_tradeoff_*.png` | `code/`, `code/figures/` |
| `reduction_table.py` | `reduction_table.csv` | `code/` |

The `figures/` folder is created automatically if it does not exist. No
other folder needs to exist before a run. Existing files with the same name
are overwritten.

## Reproducibility

The only randomness is inside scikit-learn, and it is fixed: the Random
Forest and the tuned SVM are built with `random_state=42`, set in `rf.py`
and `svm.py`. The kNN, the distance metrics and the three reduction
techniques contain no randomness.

On Pima-Indians, accuracy and storage are reproduced exactly on any machine.
On Mx, the binary attributes produce many tied distances, and the order in
which NumPy sorts tied values can differ between machines and NumPy
versions; as a result, some kNN and DROP3 results on Mx may differ slightly
from ours. Runtime differs with the hardware.

## What the experiments cover

`main.py`, for each of the 10 folds of each dataset:

- **kNN**, 120 configurations: 5 distances (euclidean, manhattan, clark,
  canberra, ivdm) × 3 values of k (3, 5, 7) × 2 voting schemes (majority,
  inverse distance) × 4 retention policies (NR, AR, DC, OG)
- **Random Forest**, 8 configurations: 4 tree counts (10, 50, 100, 200) ×
  2 criteria (gini, entropy)
- **SVM**, 2 kernels: linear and rbf, with the scikit-learn defaults
  (C = 1, gamma = 'scale')
- **SVM and RF tuned per fold** by a grid search whose inner
  cross-validation (5 folds) sees the training fold only, recorded as
  separate methods so that the cost of the search is never mixed with the
  cost of a single fit. SVM grid: C in {0.1, 1, 10, 100}, gamma in
  {0.001, 0.01, 0.1, 1}. RF grid: n_estimators in {100, 300}, max_depth in
  {None, 5, 10}, max_features in {sqrt, log2, None}.

`reduction_main.py` takes the best kNN configuration (Euclidean distance,
k = 7, majority voting, OG retention) and applies three reduction techniques
to the training set before classifying: FCNN (condensation), ENNTh (edition,
threshold 0.7) and DROP3 (hybrid). Each reduced training set is then also
given to the RBF SVM with default parameters and to a Random Forest with 100
trees and the Gini criterion.

## Datasets

| Dataset | Instances | Attributes | Notes |
| --- | --- | --- | --- |
| Pima-Indians | 768 | 8 numeric | Zeros in `plas`, `pres`, `skin`, `insu`, `mass` encode missing values |
| Mx | 2,048 | 11 nominal | 11-bit multiplexer: 3 address bits select one of 8 data bits |

The 10 folds come pre-generated with the datasets and are used unchanged, as
the assignment requires. They are included in `data/`, so nothing is
downloaded at run time.

## Preprocessing

- **Missing values.** On Pima, zeros in `plas`, `pres`, `skin`, `insu` and
  `mass` are physiologically impossible and are treated as missing, then
  imputed with the median of the training fold. `preg` is left alone, since
  zero previous pregnancies is a valid reading.
- **Normalisation.** Numeric attributes are scaled to [0,1] with min-max, as
  the assignment requires. Without it, `insu` (range 0–846) would swamp
  `pedi` (range 0.08–2.42) in every distance computation.
- **Categorical attributes.** Mapped to integer codes taken from the ARFF
  header rather than from the values observed in a fold, so every legal
  value is covered even when a fold happens not to contain one of them.
- **No data leakage.** Every statistic is learned from the training fold
  only and then applied to both folds. A fresh `Preprocessor` is built per
  fold.

## Statistical analysis

Following Demšar (2006), the paired t-test is avoided: it assumes normality,
it is sensitive to outliers, and it misbehaves under multiple comparisons.
Instead, two methods are compared with the Wilcoxon signed-rank test, and
three or more with the Friedman test (Friedman, 1937) followed by the
Nemenyi post-hoc test and its critical difference.

The assignment fixes the number of datasets at two, while Demšar's framework
compares classifiers over many. Following the instruction given in the
course, each cross-validation fold is treated as a separate dataset, giving
N = 10. This is a known weakness: any two training folds share about eight
ninths of their instances, so the scores are correlated across folds and the
p-values should be read as indicative rather than exact. The report states
this explicitly.

## References

- Aha, D. W., Kibler, D., & Albert, M. K. (1991). Instance-based learning
  algorithms. *Machine Learning, 6*(1), 37–66.
- Angiulli, F. (2007). Fast nearest neighbor condensation for large data sets
  classification. *IEEE Transactions on Knowledge and Data Engineering,
  19*(11), 1450–1464.
- Demšar, J. (2006). Statistical comparisons of classifiers over multiple data
  sets. *Journal of Machine Learning Research, 7*, 1–30.
- Friedman, M. (1937). The use of ranks to avoid the assumption of normality
  implicit in the analysis of variance. *Journal of the American Statistical
  Association, 32*(200), 675–701.
- Hsu, C.-W., Chang, C.-C., & Lin, C.-J. (2003). *A practical guide to support
  vector classification*. Technical report, National Taiwan University.
- Vázquez, F., Sánchez, J. S., & Pla, F. (2005). A stochastic approach to
  Wilson's editing algorithm. In *Pattern Recognition and Image Analysis
  (IbPRIA 2005)*, Lecture Notes in Computer Science, vol. 3523 (pp. 35–42).
  Springer.
- Wilson, D. R., & Martinez, T. R. (2000). Reduction techniques for
  instance-based learning algorithms. *Machine Learning, 38*(3), 257–286.
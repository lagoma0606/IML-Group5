# IML Group 5 — Work 2

k-Nearest Neighbour (K-IBL) implementation, compared against SVM and Random
Forest over two datasets with 10-fold cross-validation.

## Datasets

| Dataset | Instances | Attributes | Notes |
| --- | --- | --- | --- |
| Pima-Indians | 768 | 8 numeric | Zeros in `plas`, `pres`, `skin`, `insu`, `mass` encode missing values |
| Mx | 2,048 | 11 nominal | 11-bit multiplexer: 3 address bits select one of 8 data bits |

The 10 folds come pre-generated with the datasets and must not be modified,
so no additional partitioning is performed.

## Layout

```
.
├── data/
│   ├── pima-indians/   20 .arff files (10 train + 10 test)
│   └── mx/             20 .arff files
├── preprocessor.py     Preprocessor class: fit/transform
├── data_loader.py      Loads and prepares one fold
├── voting.py           Voting schemes for the kNN (majority, inverse distance)
├── main.py             Experiment harness, writes results.csv
│                       (holds the kNN placeholder for now)
├── analysis.py         Statistical analysis of results.csv
└── requirements.txt
```

Each file has one responsibility: `data_loader.py` answers "where does the
data come from", `voting.py` answers "how do the K neighbours decide the
class", `main.py` answers "what experiments do I run with it", and
`analysis.py` answers "what do the results say". Generating results and
analysing them are kept apart because `main.py` takes a long time to run
while `analysis.py` reads the CSV in seconds.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Python 3.11, as required by the assignment.

## Running

```bash
python main.py       # runs every configuration, writes results.csv
python analysis.py   # reads results.csv, prints the statistical analysis
```

`main.py` runs, for each of the 10 folds of each dataset:

- Random Forest with 8 fixed configurations (4 tree counts × 2 criteria)
- SVM with 2 kernels (linear, rbf)
- SVM and RF with hyperparameters tuned per fold by grid search
- kNN with 96 configurations (4 distances × 3 k values × 2 votings × 4
  retention policies)

Every run records accuracy and runtime; kNN runs also record the final case
base size, which is the storage metric.

**All experiments must be run on the same machine.** Accuracy is comparable
across machines, runtime is not, and runtime is one of the three metrics
being evaluated.

## Preprocessing

- **Missing values.** On Pima, zeros in `plas`, `pres`, `skin`, `insu` and
  `mass` are biologically impossible and are treated as missing. They are
  imputed with the median of the training fold. `preg` is left alone, since
  0 previous pregnancies is a valid reading.
- **Normalisation.** Numeric attributes are scaled to [0,1] with min-max, as
  the assignment requires. Needed because kNN and SVM sum differences across
  all columns: unnormalised, `insu` (range 0–846) would swamp `pedi`
  (range 0–2.3).
- **Categorical attributes.** Mapped to integer codes. The attribute types
  travel alongside the data so that HEOM and IVDM can still treat them as
  categories rather than quantities.
- **No data leakage.** Every statistic (medians, min/max, label maps) is
  learned from the training fold only and then applied to both train and
  test. A fresh `Preprocessor` is built per fold.

## Statistical analysis

Following Demšar (2006), the paired t-test is avoided: it assumes normality,
is sensitive to outliers, and misbehaves under multiple comparisons. Instead:

- **Two configurations** → Wilcoxon signed-rank test
- **More than two** → Friedman test, and if it rejects the null hypothesis, a
  post-hoc procedure (Nemenyi for all-against-all, Bonferroni-Dunn against a
  control configuration)

Since only two datasets are available and Demšar's framework compares
classifiers over multiple datasets, each cross-validation fold is treated as
a separate dataset, following the lecturer's instruction. This is not
methodologically correct and would not be acceptable in a publication; it is
accepted here because requiring ten datasets would multiply the number of
runs by a hundred. The report states this explicitly.

## References

- Cover, T. M. and Hart, P. E. (1967). *Nearest neighbor pattern
  classification.*
- Aha, D. W., Kibler, D. and Albert, M. K. (1991). *Instance-based learning
  algorithms.*
- Wilson, D. R. and Martinez, T. R. (1997). *Improved heterogeneous distance
  functions.*
- Wilson, D. R. and Martinez, T. R. (2000). *Reduction techniques for
  instance-based learning algorithms.*
- Demšar, J. (2006). *Statistical comparisons of classifiers over multiple
  data sets.*

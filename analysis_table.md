## Friedman tests

### accuracy

| Dataset | Model | Parameter | Level | Mean rank | Friedman p | CD | Within CD of best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pima-indians | kNN | distance | ivdm | 2.15 | 0.112 | 1.93 | yes |
| pima-indians | kNN | distance | euclidean | 2.6 | 0.112 | 1.93 | yes |
| pima-indians | kNN | distance | manhattan | 2.95 | 0.112 | 1.93 | yes |
| pima-indians | kNN | distance | clark | 3.4 | 0.112 | 1.93 | yes |
| pima-indians | kNN | distance | canberra | 3.9 | 0.112 | 1.93 | yes |
| pima-indians | kNN | k | 7 | 1.3 | 0.00745 | 1.05 | yes |
| pima-indians | kNN | k | 5 | 2.0 | 0.00745 | 1.05 | yes |
| pima-indians | kNN | k | 3 | 2.7 | 0.00745 | 1.05 | no |
| pima-indians | kNN | retention | AR | 2.15 | 0.194 | 1.48 | yes |
| pima-indians | kNN | retention | OG | 2.25 | 0.194 | 1.48 | yes |
| pima-indians | kNN | retention | NR | 2.35 | 0.194 | 1.48 | yes |
| pima-indians | kNN | retention | DC | 3.25 | 0.194 | 1.48 | yes |
| pima-indians | RF | n_estimators | 200 | 1.85 | 0.0301 | 1.48 | yes |
| pima-indians | RF | n_estimators | 100 | 2.1 | 0.0301 | 1.48 | yes |
| pima-indians | RF | n_estimators | 50 | 2.65 | 0.0301 | 1.48 | yes |
| pima-indians | RF | n_estimators | 10 | 3.4 | 0.0301 | 1.48 | no |
| pima-indians | SVM | config | rbf | 1.8 | 0.625 | 1.05 | yes |
| pima-indians | SVM | config | linear | 2.0 | 0.625 | 1.05 | yes |
| pima-indians | SVM | config | tuned | 2.2 | 0.625 | 1.05 | yes |
| mx | kNN | distance | canberra | 2.5 | 4.33e-08 | 1.93 | yes |
| mx | kNN | distance | clark | 2.5 | 4.33e-08 | 1.93 | yes |
| mx | kNN | distance | euclidean | 2.5 | 4.33e-08 | 1.93 | yes |
| mx | kNN | distance | manhattan | 2.5 | 4.33e-08 | 1.93 | yes |
| mx | kNN | distance | ivdm | 5.0 | 4.33e-08 | 1.93 | no |
| mx | kNN | k | 7 | 1.0 | 4.54e-05 | 1.05 | yes |
| mx | kNN | k | 5 | 2.0 | 4.54e-05 | 1.05 | yes |
| mx | kNN | k | 3 | 3.0 | 4.54e-05 | 1.05 | no |
| mx | kNN | retention | DC | 2.0 | 0.0954 | 1.48 | yes |
| mx | kNN | retention | AR | 2.1 | 0.0954 | 1.48 | yes |
| mx | kNN | retention | OG | 2.6 | 0.0954 | 1.48 | yes |
| mx | kNN | retention | NR | 3.3 | 0.0954 | 1.48 | yes |
| mx | RF | n_estimators | 200 | 1.75 | 1.18e-05 | 1.48 | yes |
| mx | RF | n_estimators | 100 | 1.9 | 1.18e-05 | 1.48 | yes |
| mx | RF | n_estimators | 50 | 2.35 | 1.18e-05 | 1.48 | yes |
| mx | RF | n_estimators | 10 | 4.0 | 1.18e-05 | 1.48 | no |
| mx | SVM | config | tuned | 1.35 | 8.58e-05 | 1.05 | yes |
| mx | SVM | config | rbf | 1.65 | 8.58e-05 | 1.05 | yes |
| mx | SVM | config | linear | 3.0 | 8.58e-05 | 1.05 | no |

### time

| Dataset | Model | Parameter | Level | Mean rank | Friedman p | CD | Within CD of best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pima-indians | kNN | distance | euclidean | 1.2 | 2.39e-07 | 1.93 | yes |
| pima-indians | kNN | distance | manhattan | 2.0 | 2.39e-07 | 1.93 | yes |
| pima-indians | kNN | distance | ivdm | 2.8 | 2.39e-07 | 1.93 | yes |
| pima-indians | kNN | distance | canberra | 4.1 | 2.39e-07 | 1.93 | no |
| pima-indians | kNN | distance | clark | 4.9 | 2.39e-07 | 1.93 | no |
| pima-indians | kNN | k | 7 | 1.4 | 0.0273 | 1.05 | yes |
| pima-indians | kNN | k | 5 | 2.0 | 0.0273 | 1.05 | yes |
| pima-indians | kNN | k | 3 | 2.6 | 0.0273 | 1.05 | no |
| pima-indians | kNN | retention | OG | 1.0 | 5.56e-06 | 1.48 | yes |
| pima-indians | kNN | retention | NR | 2.4 | 5.56e-06 | 1.48 | yes |
| pima-indians | kNN | retention | DC | 2.6 | 5.56e-06 | 1.48 | no |
| pima-indians | kNN | retention | AR | 4.0 | 5.56e-06 | 1.48 | no |
| pima-indians | RF | n_estimators | 10 | 1.0 | 1.38e-06 | 1.48 | yes |
| pima-indians | RF | n_estimators | 50 | 2.0 | 1.38e-06 | 1.48 | yes |
| pima-indians | RF | n_estimators | 100 | 3.0 | 1.38e-06 | 1.48 | no |
| pima-indians | RF | n_estimators | 200 | 4.0 | 1.38e-06 | 1.48 | no |
| pima-indians | SVM | config | linear | 1.1 | 0.000112 | 1.05 | yes |
| pima-indians | SVM | config | rbf | 1.9 | 0.000112 | 1.05 | yes |
| pima-indians | SVM | config | tuned | 3.0 | 0.000112 | 1.05 | no |
| mx | kNN | distance | ivdm | 1.0 | 1.08e-07 | 1.93 | yes |
| mx | kNN | distance | euclidean | 2.0 | 1.08e-07 | 1.93 | yes |
| mx | kNN | distance | manhattan | 3.0 | 1.08e-07 | 1.93 | no |
| mx | kNN | distance | canberra | 4.4 | 1.08e-07 | 1.93 | no |
| mx | kNN | distance | clark | 4.6 | 1.08e-07 | 1.93 | no |
| mx | kNN | k | 7 | 1.4 | 0.045 | 1.05 | yes |
| mx | kNN | k | 5 | 2.1 | 0.045 | 1.05 | yes |
| mx | kNN | k | 3 | 2.5 | 0.045 | 1.05 | no |
| mx | kNN | retention | OG | 1.0 | 4.67e-06 | 1.48 | yes |
| mx | kNN | retention | NR | 2.3 | 4.67e-06 | 1.48 | yes |
| mx | kNN | retention | DC | 2.7 | 4.67e-06 | 1.48 | no |
| mx | kNN | retention | AR | 4.0 | 4.67e-06 | 1.48 | no |
| mx | RF | n_estimators | 10 | 1.0 | 1.38e-06 | 1.48 | yes |
| mx | RF | n_estimators | 50 | 2.0 | 1.38e-06 | 1.48 | yes |
| mx | RF | n_estimators | 100 | 3.0 | 1.38e-06 | 1.48 | no |
| mx | RF | n_estimators | 200 | 4.0 | 1.38e-06 | 1.48 | no |
| mx | SVM | config | rbf | 1.2 | 0.000225 | 1.05 | yes |
| mx | SVM | config | linear | 1.8 | 0.000225 | 1.05 | yes |
| mx | SVM | config | tuned | 3.0 | 0.000225 | 1.05 | no |

### storage_pct

| Dataset | Model | Parameter | Level | Mean rank | Friedman p | CD | Within CD of best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pima-indians | kNN | distance | canberra | 1.5 | 0.000207 | 1.93 | yes |
| pima-indians | kNN | distance | clark | 2.2 | 0.000207 | 1.93 | yes |
| pima-indians | kNN | distance | manhattan | 3.05 | 0.000207 | 1.93 | yes |
| pima-indians | kNN | distance | euclidean | 4.0 | 0.000207 | 1.93 | no |
| pima-indians | kNN | distance | ivdm | 4.25 | 0.000207 | 1.93 | no |
| pima-indians | kNN | k | 7 | 1.0 | 4.54e-05 | 1.05 | yes |
| pima-indians | kNN | k | 5 | 2.0 | 4.54e-05 | 1.05 | yes |
| pima-indians | kNN | k | 3 | 3.0 | 4.54e-05 | 1.05 | no |
| pima-indians | kNN | retention | OG | 1.0 | 1.38e-06 | 1.48 | yes |
| pima-indians | kNN | retention | NR | 2.0 | 1.38e-06 | 1.48 | yes |
| pima-indians | kNN | retention | DC | 3.0 | 1.38e-06 | 1.48 | no |
| pima-indians | kNN | retention | AR | 4.0 | 1.38e-06 | 1.48 | no |
| mx | kNN | distance | ivdm | 1.0 | 4.33e-08 | 1.93 | yes |
| mx | kNN | distance | canberra | 3.5 | 4.33e-08 | 1.93 | no |
| mx | kNN | distance | clark | 3.5 | 4.33e-08 | 1.93 | no |
| mx | kNN | distance | euclidean | 3.5 | 4.33e-08 | 1.93 | no |
| mx | kNN | distance | manhattan | 3.5 | 4.33e-08 | 1.93 | no |
| mx | kNN | k | 7 | 1.0 | 4.54e-05 | 1.05 | yes |
| mx | kNN | k | 5 | 2.0 | 4.54e-05 | 1.05 | yes |
| mx | kNN | k | 3 | 3.0 | 4.54e-05 | 1.05 | no |
| mx | kNN | retention | OG | 1.0 | 1.38e-06 | 1.48 | yes |
| mx | kNN | retention | NR | 2.0 | 1.38e-06 | 1.48 | yes |
| mx | kNN | retention | DC | 3.0 | 1.38e-06 | 1.48 | no |
| mx | kNN | retention | AR | 4.0 | 1.38e-06 | 1.48 | no |

## Paired comparisons (Wilcoxon)

| Dataset | Model | Metric | Comparison | Mean first | Mean second | p-value |
| --- | --- | --- | --- | --- | --- | --- |
| pima-indians | kNN | accuracy | majority vs inverse_distance | 0.7347 | 0.7331 | 0.1289 |
| pima-indians | RF | accuracy | gini vs entropy | 0.758 | 0.756 | 0.7109 |
| mx | kNN | accuracy | majority vs inverse_distance | 0.8579 | 0.8565 | 0.1309 |
| mx | RF | accuracy | gini vs entropy | 0.9901 | 0.9904 | 0.5391 |
| pima-indians | kNN | time | majority vs inverse_distance | 0.5213 | 0.5204 | 0.625 |
| pima-indians | RF | time | gini vs entropy | 0.2111 | 0.2371 | 0.01953 |
| mx | kNN | time | majority vs inverse_distance | 3.2461 | 3.2385 | 0.9219 |
| mx | RF | time | gini vs entropy | 0.1987 | 0.1969 | 0.9219 |

## kNN vs SVM vs RF (best configuration of each)

### accuracy

| Dataset | Method | Mean | Std | Mean rank | Friedman p | CD | Within CD of best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pima-indians | RF | 0.7657 | 0.0403 | 1.75 | 0.549 | 1.05 | yes |
| pima-indians | SVM | 0.7646 | 0.0315 | 2.05 | 0.549 | 1.05 | yes |
| pima-indians | kNN | 0.7567 | 0.0448 | 2.2 | 0.549 | 1.05 | yes |
| mx | RF | 1.0 | 0.0 | 1.35 | 8.58e-05 | 1.05 | yes |
| mx | SVM | 0.9971 | 0.0053 | 1.65 | 8.58e-05 | 1.05 | yes |
| mx | kNN | 0.9814 | 0.0079 | 3.0 | 8.58e-05 | 1.05 | no |

### time

| Dataset | Method | Mean | Std | Mean rank | Friedman p | CD | Within CD of best |
| --- | --- | --- | --- | --- | --- | --- | --- |
| pima-indians | SVM | 0.0155 | 0.0065 | 1.0 | 0.000112 | 1.05 | yes |
| pima-indians | RF | 0.2399 | 0.1076 | 2.1 | 0.000112 | 1.05 | no |
| pima-indians | kNN | 0.2943 | 0.1572 | 2.9 | 0.000112 | 1.05 | no |
| mx | SVM | 0.0713 | 0.0043 | 1.0 | 4.54e-05 | 1.05 | yes |
| mx | RF | 0.2139 | 0.0259 | 2.0 | 4.54e-05 | 1.05 | yes |
| mx | kNN | 2.0989 | 0.1075 | 3.0 | 4.54e-05 | 1.05 | no |

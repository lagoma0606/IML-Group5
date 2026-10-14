"""Reduction experiments: the best kNN configuration against the three
sample reduction techniques.


"""

# Chosen from the Friedman and Nemenyi analysis over the 120 kNN
# configurations (see results/analysis_output.txt).
#
# k = 7 wins accuracy, time and storage in both datasets, significantly.
# Voting is never significant, so the simpler scheme is used.
# OG wins time and storage and is not significantly worse in accuracy.
# IVDM is significantly worse than the other four metrics on mx, and on
# pima-indians the four are statistically tied, so euclidean is used in
# both: it is the fastest and it keeps the two datasets comparable.
BEST_KNN = {
    'pima-indians': ('euclidean', 7, 'majority', 'OG'),
    'mx':           ('euclidean', 7, 'majority', 'OG'),
}
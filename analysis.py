import pandas as pd
from scipy.stats import wilcoxon

df = pd.read_csv('results.csv')

# Compare the two SVM kernels on Pima.
# Each fold is treated as a dataset, as the lecturer indicated.
svm = df[(df['model'] == 'svm') & (df['dataset'] == 'pima-indians')]

linear = svm[svm['config'] == 'linear'].sort_values('fold')['accuracy'].values
rbf = svm[svm['config'] == 'rbf'].sort_values('fold')['accuracy'].values

print(linear)
print(rbf)

# sort_values('fold') is essential: Wilcoxon is a paired test, it compares
# fold 0 against fold 0, fold 1 against fold 1. If the two vectors arrived
# in a different order, the pairing would be wrong.
stat, p = wilcoxon(linear, rbf)
print("p-value:", p)